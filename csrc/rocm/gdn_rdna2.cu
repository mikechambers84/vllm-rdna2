// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) kernels for Gated DeltaNet prefill.
//
// gdn_post_conv_rdna2: FLA's fused post-conv1d preparation (split of the conv
// output into q / k / v, l2 norm of q and k, g = -exp(A_log) *
// softplus(a + dt_bias), beta = sigmoid(b)). One 256-thread workgroup per
// token, each thread moving 8 fp16 values (16 bytes) per step along the token's
// row, so reads and writes are coalesced; a head of K dims is K / 8 adjacent
// lanes and is normalized with a lane reduction. ~425 GB/s on a V620 against
// ~80 GB/s for the Triton kernel at every launch config.
//
// gdn_wy_rdna2: the chunked delta rule's WY step, fusing FLA's
// chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril and
// recompute_w_u_fwd. One workgroup per (64-token chunk, value head):
//   g_cum = chunk-local cumsum(g)                               (written)
//   A     = strict_lower(beta_t (k_t . k_s) exp(g_t - g_s))     (fp32, LDS)
//   Ai    = (I + A)^-1: 16x16 diagonal blocks by row recurrence, then block
//           forward substitution (fp32), rounded to fp16 as in FLA
//   u     = Ai @ fp16(v beta),  w = Ai @ fp16(k beta exp(g_cum))
// as v_dot2_f32_f16 GEMMs in k2-major LDS; A never goes to memory. Heads vary
// fastest across workgroups, so concurrent workgroups read neighbouring heads
// of the same rows (4x faster than chunks-first). 6x the four FLA kernels at
// 8K tokens on a V620.
//
// gdn_fwd_h_rdna2: the chunk state recurrence (FLA
// chunk_gated_delta_rule_fwd_h) with one workgroup per (sequence, value head,
// 64 value rows), the fp32 state in registers; per chunk h[i] = fp16(S), v_new
// = u - w S^T and S = exp(g_last) S + fp16(v_new exp(g_last - g_t))^T k as dot2
// GEMMs, the next chunk's w / k / u / g loaded during the GEMMs.
// Bandwidth-bound: 1.8x Triton.
//
// gdn_fwd_o_rdna2: the chunk output (FLA chunk_fwd_o), one workgroup per (value
// head, chunk): P = fp16(causal(q k^T) exp(g_t - g_s)),
// o = scale (exp(g_t) q h^T + P v_new), in two 64-column halves. 2.2-2.7x the
// heads-first Triton kernel.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gdn_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BT = 64;    // chunk size
static constexpr int KD = 128;   // key / value head dim
static constexpr int H_BV = 64;  // value rows per fwd_h workgroup

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int K, bool L2NORM, bool G_EXP>
__global__ void __launch_bounds__(THREADS) gdn_post_conv_rdna2_kernel(
    const __half* __restrict__ x, const __half* __restrict__ a,
    const __half* __restrict__ b, const float* __restrict__ A_log,
    const void* __restrict__ dt_bias, const bool dt_f32, __half* __restrict__ q,
    __half* __restrict__ k, __half* __restrict__ v, float* __restrict__ g,
    float* __restrict__ beta, const int H, const int HV, const int V,
    const long sx, const long sa, const long sb, const float eps) {
  constexpr int LANES = K / 8;  // lanes per q/k head
  const int t = blockIdx.x, tid = threadIdx.x;
  const int HK = H * K, VD = HV * V;
  const __half* row = x + t * sx;
  for (int c = tid; c < 2 * HK / 8; c += THREADS) {
    uint4 raw = *reinterpret_cast<const uint4*>(row + c * 8);
    if constexpr (L2NORM) {
      __half2* h2 = reinterpret_cast<__half2*>(&raw);
      float f[8];
      float ss = 0.f;
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        const float2 p = __half22float2(h2[e]);
        f[2 * e] = p.x;
        f[2 * e + 1] = p.y;
        ss += p.x * p.x + p.y * p.y;
      }
  #pragma unroll
      for (int o = LANES / 2; o > 0; o >>= 1) ss += __shfl_xor(ss, o, LANES);
      const float inv = 1.f / sqrtf(ss + eps);
  #pragma unroll
      for (int e = 0; e < 4; e++)
        h2[e] = __floats2half2_rn(f[2 * e] * inv, f[2 * e + 1] * inv);
    }
    const int d = c * 8;
    __half* dst = (d < HK ? q + d : k + (d - HK)) + (long)t * HK;
    *reinterpret_cast<uint4*>(dst) = raw;
  }
  for (int c = tid; c < VD / 8; c += THREADS)
    *reinterpret_cast<uint4*>(v + (long)t * VD + c * 8) =
        *reinterpret_cast<const uint4*>(row + 2 * HK + c * 8);
  for (int h = tid; h < HV; h += THREADS) {
    const float dt = dt_f32
                         ? static_cast<const float*>(dt_bias)[h]
                         : __half2float(static_cast<const __half*>(dt_bias)[h]);
    const float xa = __half2float(a[t * sa + h]) + dt;
    const float sp = xa > 20.f ? xa : log1pf(expf(xa));
    const float gv = -expf(A_log[h]) * sp;
    g[(long)t * HV + h] = G_EXP ? expf(gv) : gv;
    beta[(long)t * HV + h] = 1.f / (1.f + expf(-__half2float(b[t * sb + h])));
  }
}

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}
__device__ __forceinline__ uint32_t as_u32(half2 h) {
  return *reinterpret_cast<uint32_t*>(&h);
}

// Two fp16x8 rows (keys/tokens 2s and 2s+1) -> 8 dwords pairing them per dim.
__device__ __forceinline__ void pair_rows(const uint4& a, const uint4& b,
                                          float sa, float sb, uint32_t* w) {
  const half2* ha = reinterpret_cast<const half2*>(&a);
  const half2* hb = reinterpret_cast<const half2*>(&b);
  #pragma unroll
  for (int e = 0; e < 4; e++) {
    const float2 fa = __half22float2(ha[e]);
    const float2 fb = __half22float2(hb[e]);
    w[2 * e] = as_u32(__floats2half2_rn(fa.x * sa, fb.x * sb));
    w[2 * e + 1] = as_u32(__floats2half2_rn(fa.y * sa, fb.y * sb));
  }
}

__global__ void __launch_bounds__(THREADS) gdn_wy_rdna2_kernel(
    const __half* __restrict__ k, const __half* __restrict__ v,
    const float* __restrict__ beta, const float* __restrict__ g,
    float* __restrict__ g_cum, __half* __restrict__ w, __half* __restrict__ u,
    const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_indices,
    const int H, const int Hg) {
  // R0: k2-major K tile [KD/2][BT] for A, then s2-major k*beta*exp(g)
  // [BT/2][KD].
  __shared__ __align__(16) uint32_t R0[(KD / 2) * BT];
  constexpr int LD =
      BT + 4;  // fp32 row stride: 16-byte aligned rows, banks shift by 4
  // R1: A fp32 [BT][LD]; then Ai fp16 s2-major [BT/2][BT] (dwords).
  __shared__ __align__(16) float R1[BT * LD];
  // R2: X fp32 [BT][LD] (zero above the diagonal); then s2-major v*beta
  // [BT/2][KD].
  __shared__ __align__(16) float R2[BT * LD];
  __shared__ float sT[16 * 48];
  __shared__ float sG[BT], sB[BT];
  __shared__ float wsum;

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  // Heads vary fastest across workgroups: concurrent workgroups read
  // neighbouring heads of the same rows (DRAM locality).
  const int h = blockIdx.x, hk = h / (H / Hg);
  const int n = chunk_indices[blockIdx.y * 2],
            c = chunk_indices[blockIdx.y * 2 + 1];
  const int bos = cu_seqlens[n], len = cu_seqlens[n + 1] - bos;
  const int t0 = c * BT;
  const int nv = min(BT, len - t0);  // valid rows
  const long row0 = bos + t0;

  // g (chunk-local inclusive cumsum) and beta.
  if (tid < BT) {
    float gv = tid < nv ? g[(row0 + tid) * H + h] : 0.f;
  #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up(gv, o, 32);
      if ((tid & 31) >= o) gv += y;
    }
    if (tid == 31) wsum = gv;
    sG[tid] = gv;
    sB[tid] = tid < nv ? beta[(row0 + tid) * H + h] : 0.f;
  }
  for (int i = tid; i < BT * LD; i += THREADS) R2[i] = 0.f;
  // All global loads up front, coalesced (16 lanes per 256-byte row segment).
  constexpr int P_ITEMS = (BT / 2) * (KD / 8) / THREADS;  // 2
  uint4 pk0[P_ITEMS], pk1[P_ITEMS], pv0[P_ITEMS], pv1[P_ITEMS];
  #pragma unroll
  for (int it = 0; it < P_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int s2 = idx / (KD / 8), cc = idx % (KD / 8);
    const int s = 2 * s2;
    const uint4 z = make_uint4(0, 0, 0, 0);
    pk0[it] = s < nv ? *reinterpret_cast<const uint4*>(
                           k + ((row0 + s) * Hg + hk) * KD + cc * 8)
                     : z;
    pk1[it] = s + 1 < nv ? *reinterpret_cast<const uint4*>(
                               k + ((row0 + s + 1) * Hg + hk) * KD + cc * 8)
                         : z;
    pv0[it] = s < nv ? *reinterpret_cast<const uint4*>(
                           v + ((row0 + s) * H + h) * KD + cc * 8)
                     : z;
    pv1[it] = s + 1 < nv ? *reinterpret_cast<const uint4*>(
                               v + ((row0 + s + 1) * H + h) * KD + cc * 8)
                         : z;
  }
  // K tile -> R0[k2][t ^ sw(k2)], sw = 4 * ((k2 / 4) % 8): the transposing
  // stores are at most 2-way conflicted and b128 reads stay aligned.
  #pragma unroll
  for (int it = 0; it < P_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int s2 = idx / (KD / 8), cc = idx % (KD / 8);
    const int sw = (cc & 7) * 4;
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&pk0[it]);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&pk1[it]);
  #pragma unroll
    for (int e = 0; e < 4; e++) {
      R0[(cc * 4 + e) * BT + ((2 * s2) ^ sw)] = a0[e];
      R0[(cc * 4 + e) * BT + ((2 * s2 + 1) ^ sw)] = a1[e];
    }
  }
  __syncthreads();
  if (tid >= 32 && tid < BT) sG[tid] += wsum;
  __syncthreads();
  if (tid < nv) g_cum[(row0 + tid) * H + h] = sG[tid];

  // A = k k^T: rows 4ty.., cols 4tx..
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++)
  #pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  #pragma unroll 4
  for (int k2 = 0; k2 < KD / 2; k2++) {
    const int sw = ((k2 >> 2) & 7) * 4;
    const uint4 a =
        *reinterpret_cast<const uint4*>(&R0[k2 * BT + ((ty * 4) ^ sw)]);
    const uint4 b =
        *reinterpret_cast<const uint4*>(&R0[k2 * BT + ((tx * 4) ^ sw)]);
    const half2 av[4] = {as_h2(a.x), as_h2(a.y), as_h2(a.z), as_h2(a.w)};
    const half2 bv[4] = {as_h2(b.x), as_h2(b.y), as_h2(b.z), as_h2(b.w)};
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 4; j++)
        acc[i][j] = __builtin_amdgcn_fdot2(av[i], bv[j], acc[i][j], false);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int r = ty * 4 + i;
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int s = tx * 4 + j;
      R1[r * LD + s] = r > s ? acc[i][j] * sB[r] * __expf(sG[r] - sG[s]) : 0.f;
    }
  }
  __syncthreads();  // A ready; R0 free

  // k * beta * exp(g) -> R0 as s2-major [BT/2][KD].
  #pragma unroll
  for (int it = 0; it < P_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int s2 = idx / (KD / 8), cc = idx % (KD / 8);
    uint32_t wv[8];
    pair_rows(pk0[it], pk1[it], sB[2 * s2] * __expf(sG[2 * s2]),
              sB[2 * s2 + 1] * __expf(sG[2 * s2 + 1]), wv);
    *reinterpret_cast<uint4*>(&R0[s2 * KD + cc * 8]) =
        make_uint4(wv[0], wv[1], wv[2], wv[3]);
    *reinterpret_cast<uint4*>(&R0[s2 * KD + cc * 8 + 4]) =
        make_uint4(wv[4], wv[5], wv[6], wv[7]);
  }

  // Diagonal 16x16 blocks of X = (I + A)^-1 by row recurrence (thread =
  // column).
  if (tid < BT) {
    const int b = tid / 16, cl = tid % 16, base = b * 16;
    float x[16];
  #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = 0.f;
    x[cl] = 1.f;
  #pragma unroll
    for (int i = 1; i < 16; i++) {
      float sacc = 0.f;
  #pragma unroll
      for (int j = 0; j < i; j++) sacc += R1[(base + i) * LD + base + j] * x[j];
      if (i > cl) x[i] = -sacc;
    }
  #pragma unroll
    for (int i = 0; i < 16; i++) R2[(base + i) * LD + base + cl] = x[i];
  }
  __syncthreads();
  // Block rows 1..3: X_rc = -D_r * sum_{m=c}^{r-1} A_rm X_mc, as
  // T = A[rows r, cols < 16r] @ X[< 16r, < 16r], X[rows r] = -D_r @ T; X is
  // zero above the diagonal, so the sums run over full, unrolled ranges.
  #pragma unroll
  for (int r = 1; r < 4; r++) {
    const int ncol = 16 * r;
    for (int e = tid; e < 16 * ncol; e += THREADS) {
      const int i = e / ncol, j = e % ncol;
      const float* arow = &R1[(16 * r + i) * LD];
      float sacc = 0.f;
  #pragma unroll
      for (int l = 0; l < 48; l += 4) {
        if (l < ncol) {
          const float4 a4 = *reinterpret_cast<const float4*>(arow + l);
          sacc += a4.x * R2[(l + 0) * LD + j] + a4.y * R2[(l + 1) * LD + j] +
                  a4.z * R2[(l + 2) * LD + j] + a4.w * R2[(l + 3) * LD + j];
        }
      }
      sT[i * 48 + j] = sacc;
    }
    __syncthreads();
    for (int e = tid; e < 16 * ncol; e += THREADS) {
      const int i = e / ncol, j = e % ncol;
      const float* drow = &R2[(16 * r + i) * LD + 16 * r];
      float sacc = 0.f;
  #pragma unroll
      for (int p = 0; p < 16; p += 4) {
        const float4 d4 = *reinterpret_cast<const float4*>(drow + p);
        sacc += d4.x * sT[(p + 0) * 48 + j] + d4.y * sT[(p + 1) * 48 + j] +
                d4.z * sT[(p + 2) * 48 + j] + d4.w * sT[(p + 3) * 48 + j];
      }
      R2[(16 * r + i) * LD + j] = -sacc;
    }
    __syncthreads();
  }
  // Ai (fp16) -> R1 as s2-major [BT/2][BT] dwords: (Ai[t][2s2], Ai[t][2s2+1]).
  uint32_t* sAi = reinterpret_cast<uint32_t*>(R1);
  for (int idx = tid; idx < (BT / 2) * BT; idx += THREADS) {
    const int s2 = idx / BT, t = idx % BT;
    const float a0 = 2 * s2 <= t ? R2[t * LD + 2 * s2] : 0.f;
    const float a1 = 2 * s2 + 1 <= t ? R2[t * LD + 2 * s2 + 1] : 0.f;
    sAi[s2 * BT + t] = as_u32(__floats2half2_rn(a0, a1));
  }
  __syncthreads();  // Ai ready; X (R2) free
  uint32_t* sVb = reinterpret_cast<uint32_t*>(R2);
  #pragma unroll
  for (int it = 0; it < P_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int s2 = idx / (KD / 8), cc = idx % (KD / 8);
    uint32_t wv[8];
    pair_rows(pv0[it], pv1[it], sB[2 * s2], sB[2 * s2 + 1], wv);
    *reinterpret_cast<uint4*>(&sVb[s2 * KD + cc * 8]) =
        make_uint4(wv[0], wv[1], wv[2], wv[3]);
    *reinterpret_cast<uint4*>(&sVb[s2 * KD + cc * 8 + 4]) =
        make_uint4(wv[4], wv[5], wv[6], wv[7]);
  }
  __syncthreads();

  // w = Ai @ kbg (B = R0), u = Ai @ vb (B = sVb); rows 4ty.., cols tx*4 + {0,
  // 64}.
  auto gemm_store = [&](const uint32_t* B, __half* out, const long ld,
                        const int head) {
    float o[4][8];
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 8; j++) o[i][j] = 0.f;
  #pragma unroll 4
    for (int s2 = 0; s2 < BT / 2; s2++) {
      const uint4 a = *reinterpret_cast<const uint4*>(&sAi[s2 * BT + ty * 4]);
      const uint4 b0 = *reinterpret_cast<const uint4*>(&B[s2 * KD + tx * 4]);
      const uint4 b1 =
          *reinterpret_cast<const uint4*>(&B[s2 * KD + 64 + tx * 4]);
      const half2 av[4] = {as_h2(a.x), as_h2(a.y), as_h2(a.z), as_h2(a.w)};
      const half2 bv[8] = {as_h2(b0.x), as_h2(b0.y), as_h2(b0.z), as_h2(b0.w),
                           as_h2(b1.x), as_h2(b1.y), as_h2(b1.z), as_h2(b1.w)};
  #pragma unroll
      for (int i = 0; i < 4; i++)
  #pragma unroll
        for (int j = 0; j < 8; j++)
          o[i][j] = __builtin_amdgcn_fdot2(av[i], bv[j], o[i][j], false);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int t = ty * 4 + i;
      if (t >= nv) continue;
      __half* dst = out + ((row0 + t) * ld + head) * KD;
  #pragma unroll
      for (int half = 0; half < 2; half++) {
        const uint2 pk = make_uint2(
            as_u32(__floats2half2_rn(o[i][4 * half], o[i][4 * half + 1])),
            as_u32(__floats2half2_rn(o[i][4 * half + 2], o[i][4 * half + 3])));
        *reinterpret_cast<uint2*>(dst + half * 64 + tx * 4) = pk;
      }
    }
  };
  gemm_store(R0, w, H, h);
  gemm_store(sVb, u, H, h);
}

// Barrier for LDS hand-offs only: no global-memory fence, so prefetch loads
// stay in flight across it.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template <int BV, bool H0_F32>
__global__ void __launch_bounds__(THREADS)
    gdn_fwd_h_rdna2_kernel(const __half* __restrict__ k,
                           const __half* __restrict__ w,
                           const __half* __restrict__ u,
                           const float* __restrict__ g,
                           const void* __restrict__ h0, __half* __restrict__ hs,
                           __half* __restrict__ vnew, float* __restrict__ ht,
                           const int* __restrict__ cu_seqlens,
                           const void* __restrict__ chunk_offsets,
                           const bool co64, const int H, const int Hg) {
  constexpr int HLD =
      BV + (BV >= 64 ? 4 : 2);  // k2-major S row stride (dwords)
  constexpr int VR = BV / 16;   // state value rows per thread
  constexpr int VC = BV / 16;   // v~ value columns per thread
  // sWK: w chunk (k2-major [64][64 t], swizzled) for v~, then k chunk
  // (t2-major [32][128]) for the state update.
  __shared__ __align__(16) uint32_t sWK[(KD / 2) * BT];
  __shared__ __align__(16)
      uint32_t sS[(KD / 2) * HLD];  // fp16 S, k2-major [64][BV]
  __shared__ __align__(16)
      uint32_t sVt[(BT / 2) * BV];  // fp16 v~ scaled, t2-major
  __shared__ float sG[BT];

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int nvb = 128 / BV;
  const int vb = blockIdx.x % nvb, h = blockIdx.x / nvb, hk = h / (H / Hg);
  const int n = blockIdx.y;
  const int bos = cu_seqlens[n], T = cu_seqlens[n + 1] - bos;
  const int NT = (T + BT - 1) / BT;
  const long boh = co64 ? static_cast<const int64_t*>(chunk_offsets)[n]
                        : static_cast<const int*>(chunk_offsets)[n];
  const int v0 = vb * BV;

  // State: rows v = VR*ty + i, cols 4tx + {0..3} and 64 + 4tx + {0..3}.
  float S[VR][8];
  #pragma unroll
  for (int i = 0; i < VR; i++)
  #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int kc = (j / 4) * 64 + tx * 4 + j % 4;
      float val = 0.f;
      if (h0 != nullptr) {
        const long off = ((long)(n * H + h) * KD + v0 + VR * ty + i) * KD + kc;
        val = H0_F32 ? static_cast<const float*>(h0)[off]
                     : __half2float(static_cast<const __half*>(h0)[off]);
      }
      S[i][j] = val;
    }

  const uint4 z = make_uint4(0, 0, 0, 0);
  uint4 rw[4], rk0[2], rk1[2];
  __half ru[4][VC];
  float rg;
  auto load_w = [&](int i) {
    const long r0 = bos + (long)i * BT;
    const int nv = min(BT, T - i * BT);
  #pragma unroll
    for (int it = 0; it < 4; it++) {
      const int idx = tid + it * THREADS, r = idx / 16, cc = idx % 16;
      rw[it] = r < nv ? *reinterpret_cast<const uint4*>(
                            w + ((r0 + r) * H + h) * KD + cc * 8)
                      : z;
    }
    rg = (tid < BT && tid < nv) ? g[(r0 + tid) * H + h] : 0.f;
  #pragma unroll
    for (int a = 0; a < 4; a++) {
      const int t = 4 * ty + a;
  #pragma unroll
      for (int b = 0; b < VC; b++)
        ru[a][b] = t < nv ? u[((r0 + t) * H + h) * KD + v0 + VC * tx + b]
                          : __float2half(0.f);
    }
  };
  auto load_k = [&](int i) {
    const long r0 = bos + (long)i * BT;
    const int nv = min(BT, T - i * BT);
  #pragma unroll
    for (int it = 0; it < 2; it++) {
      const int idx = tid + it * THREADS, t2 = idx / 16, cc = idx % 16,
                t = 2 * t2;
      rk0[it] = t < nv ? *reinterpret_cast<const uint4*>(
                             k + ((r0 + t) * Hg + hk) * KD + cc * 8)
                       : z;
      rk1[it] = t + 1 < nv ? *reinterpret_cast<const uint4*>(
                                 k + ((r0 + t + 1) * Hg + hk) * KD + cc * 8)
                           : z;
    }
  };
  if (NT > 0) {
    load_w(0);
    load_k(0);
  }

  for (int i = 0; i < NT; i++) {
    const long r0 = bos + (long)i * BT;
    const int nv = min(BT, T - i * BT);
    lds_barrier();  // previous update done with sWK, sVt
  #pragma unroll
    for (int it = 0; it < 4; it++) {
      const int idx = tid + it * THREADS, r = idx / 16, cc = idx % 16,
                sw = (cc & 7) * 4;
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&rw[it]);
  #pragma unroll
      for (int e = 0; e < 4; e++) sWK[(cc * 4 + e) * BT + (r ^ sw)] = a[e];
    }
    if (tid < BT) sG[tid] = rg;
    float uc[4][VC];
  #pragma unroll
    for (int a = 0; a < 4; a++)
  #pragma unroll
      for (int b = 0; b < VC; b++) uc[a][b] = __half2float(ru[a][b]);
    __half* hrow = hs + ((boh + i) * H + h) * (long)KD * KD;
  #pragma unroll
    for (int ii = 0; ii < VR; ii++) {
      const int v = VR * ty + ii;
      uint32_t p[4];
  #pragma unroll
      for (int jj = 0; jj < 4; jj++)
        p[jj] = as_u32(__floats2half2_rn(S[ii][2 * jj], S[ii][2 * jj + 1]));
      sS[(2 * tx) * HLD + v] = p[0];
      sS[(2 * tx + 1) * HLD + v] = p[1];
      sS[(32 + 2 * tx) * HLD + v] = p[2];
      sS[(33 + 2 * tx) * HLD + v] = p[3];
      __half* dst = hrow + (long)(v0 + v) * KD;
      *reinterpret_cast<uint2*>(dst + tx * 4) = make_uint2(p[0], p[1]);
      *reinterpret_cast<uint2*>(dst + 64 + tx * 4) = make_uint2(p[2], p[3]);
    }
    lds_barrier();
    if (i + 1 < NT) load_w(i + 1);

    // v~ = u - w @ S^T: rows t = 4ty + a, cols v = VC*tx + b.
    float va[4][VC];
  #pragma unroll
    for (int a = 0; a < 4; a++)
  #pragma unroll
      for (int b = 0; b < VC; b++) va[a][b] = 0.f;
  #pragma unroll 4
    for (int k2 = 0; k2 < KD / 2; k2++) {
      const int sw = ((k2 >> 2) & 7) * 4;
      const uint4 wa =
          *reinterpret_cast<const uint4*>(&sWK[k2 * BT + ((ty * 4) ^ sw)]);
      const half2 av[4] = {as_h2(wa.x), as_h2(wa.y), as_h2(wa.z), as_h2(wa.w)};
      half2 bv[VC];
      if constexpr (VC == 4) {
        const uint4 sb =
            *reinterpret_cast<const uint4*>(&sS[k2 * HLD + 4 * tx]);
        bv[0] = as_h2(sb.x);
        bv[1 % VC] = as_h2(sb.y);
        bv[2 % VC] = as_h2(sb.z);
        bv[3 % VC] = as_h2(sb.w);
      } else if constexpr (VC == 2) {
        const uint2 sb =
            *reinterpret_cast<const uint2*>(&sS[k2 * HLD + 2 * tx]);
        bv[0] = as_h2(sb.x);
        bv[1 % VC] = as_h2(sb.y);
      } else {
        bv[0] = as_h2(sS[k2 * HLD + tx]);
      }
  #pragma unroll
      for (int a = 0; a < 4; a++)
  #pragma unroll
        for (int b = 0; b < VC; b++)
          va[a][b] = __builtin_amdgcn_fdot2(av[a], bv[b], va[a][b], false);
    }
    const float g_last = sG[nv - 1];
    float vt[4][VC];
  #pragma unroll
    for (int a = 0; a < 4; a++) {
      const int t = 4 * ty + a;
      const bool ok = t < nv;
  #pragma unroll
      for (int b = 0; b < VC; b++) vt[a][b] = uc[a][b] - va[a][b];
      if (ok) {
        if constexpr (VC == 4) {
          *reinterpret_cast<uint2*>(vnew + ((r0 + t) * H + h) * KD + v0 +
                                    4 * tx) =
              make_uint2(
                  as_u32(__floats2half2_rn(vt[a][0], vt[a][1 % VC])),
                  as_u32(__floats2half2_rn(vt[a][2 % VC], vt[a][3 % VC])));
        } else if constexpr (VC == 2) {
          *reinterpret_cast<half2*>(vnew + ((r0 + t) * H + h) * KD + v0 +
                                    2 * tx) =
              __floats2half2_rn(vt[a][0], vt[a][1 % VC]);
        } else {
          vnew[((r0 + t) * H + h) * KD + v0 + tx] = __float2half_rn(vt[a][0]);
        }
      }
      const float sc = ok ? __expf(g_last - sG[t]) : 0.f;
  #pragma unroll
      for (int b = 0; b < VC; b++) vt[a][b] *= sc;
    }
  #pragma unroll
    for (int b = 0; b < VC; b++) {
      sVt[(2 * ty) * BV + VC * tx + b] =
          as_u32(__floats2half2_rn(vt[0][b], vt[1][b]));
      sVt[(2 * ty + 1) * BV + VC * tx + b] =
          as_u32(__floats2half2_rn(vt[2][b], vt[3][b]));
    }
    lds_barrier();  // v~ done reading sWK (w); sVt written
  #pragma unroll
    for (int it = 0; it < 2; it++) {
      const int idx = tid + it * THREADS, t2 = idx / 16, cc = idx % 16;
      const half2* ha = reinterpret_cast<const half2*>(&rk0[it]);
      const half2* hb = reinterpret_cast<const half2*>(&rk1[it]);
      uint32_t wv[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        wv[2 * e] = as_u32(__lows2half2(ha[e], hb[e]));
        wv[2 * e + 1] = as_u32(__highs2half2(ha[e], hb[e]));
      }
      *reinterpret_cast<uint4*>(&sWK[t2 * KD + cc * 8]) =
          make_uint4(wv[0], wv[1], wv[2], wv[3]);
      *reinterpret_cast<uint4*>(&sWK[t2 * KD + cc * 8 + 4]) =
          make_uint4(wv[4], wv[5], wv[6], wv[7]);
    }
    lds_barrier();
    if (i + 1 < NT) load_k(i + 1);
    // S = exp(g_last) S + v~^T k.
    const float eg = __expf(g_last);
  #pragma unroll
    for (int a = 0; a < VR; a++)
  #pragma unroll
      for (int j = 0; j < 8; j++) S[a][j] *= eg;
  #pragma unroll 4
    for (int t2 = 0; t2 < BT / 2; t2++) {
      half2 av[VR];
      if constexpr (VR == 4) {
        const uint4 va4 =
            *reinterpret_cast<const uint4*>(&sVt[t2 * BV + 4 * ty]);
        av[0] = as_h2(va4.x);
        av[1 % VR] = as_h2(va4.y);
        av[2 % VR] = as_h2(va4.z);
        av[3 % VR] = as_h2(va4.w);
      } else if constexpr (VR == 2) {
        const uint2 va2 =
            *reinterpret_cast<const uint2*>(&sVt[t2 * BV + 2 * ty]);
        av[0] = as_h2(va2.x);
        av[1 % VR] = as_h2(va2.y);
      } else {
        av[0] = as_h2(sVt[t2 * BV + ty]);
      }
      const uint4 kb0 = *reinterpret_cast<const uint4*>(&sWK[t2 * KD + tx * 4]);
      const uint4 kb1 =
          *reinterpret_cast<const uint4*>(&sWK[t2 * KD + 64 + tx * 4]);
      const half2 bv[8] = {as_h2(kb0.x), as_h2(kb0.y), as_h2(kb0.z),
                           as_h2(kb0.w), as_h2(kb1.x), as_h2(kb1.y),
                           as_h2(kb1.z), as_h2(kb1.w)};
  #pragma unroll
      for (int a = 0; a < VR; a++)
  #pragma unroll
        for (int j = 0; j < 8; j++)
          S[a][j] = __builtin_amdgcn_fdot2(av[a], bv[j], S[a][j], false);
    }
  }
  if (ht != nullptr) {
  #pragma unroll
    for (int a = 0; a < VR; a++)
  #pragma unroll
      for (int j = 0; j < 8; j++) {
        const int kc = (j / 4) * 64 + tx * 4 + j % 4;
        ht[((long)(n * H + h) * KD + v0 + VR * ty + a) * KD + kc] = S[a][j];
      }
  }
}

// Chunk idx (row idx / 16, 8 dims at idx % 16) of a rows x 128 fp16 tile ->
// k2-major [64][rows] dwords, 16-byte chunks XOR-swizzled by 4 * ((k2 / 4) %
// 8).
__device__ __forceinline__ void store_k2major(uint32_t* dst, int rows, int idx,
                                              const uint4& val) {
  const int r = idx / 16, cc = idx % 16;
  const int sw = (cc & 7) * 4;
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&val);
  #pragma unroll
  for (int e = 0; e < 4; e++) dst[(cc * 4 + e) * rows + (r ^ sw)] = a[e];
}

__global__ void __launch_bounds__(THREADS) gdn_fwd_o_rdna2_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ vn, const __half* __restrict__ hs,
    const float* __restrict__ g, __half* __restrict__ o,
    const int* __restrict__ cu_seqlens, const int* __restrict__ chunk_indices,
    const int H, const int Hg, const float scale) {
  __shared__ __align__(16) uint32_t sQ[(KD / 2) * BT];  // q, k2-major [64][64]
  __shared__ __align__(16)
      uint32_t sKH[(KD / 2) * BT];  // k, then h half [64 k2][64 v]
  __shared__ __align__(16) uint32_t sP[(BT / 2) * BT];  // P, s2-major [32][64]
  __shared__ __align__(16)
      uint32_t sV[(BT / 2) * 64];  // v_new half, s2-major [32][64]
  __shared__ float sG[BT];

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int h = blockIdx.x, hk = h / (H / Hg);
  const int ic = blockIdx.y;
  const int n = chunk_indices[ic * 2], c = chunk_indices[ic * 2 + 1];
  const int bos = cu_seqlens[n], len = cu_seqlens[n + 1] - bos;
  const int nv = min(BT, len - c * BT);
  const long row0 = bos + c * BT;
  const __half* hp = hs + ((long)ic * H + h) * KD * KD;  // [V][K] of this chunk

  // Loads: q, k rows (4 x 16 B per thread each), h half 0, v_new half 0.
  uint4 rq[4], rk[4], rh[4], rv0[2], rv1[2];
  const uint4 z = make_uint4(0, 0, 0, 0);
  #pragma unroll
  for (int it = 0; it < 4; it++) {
    const int idx = tid + it * THREADS, r = idx / 16, cc = idx % 16;
    rq[it] = r < nv ? *reinterpret_cast<const uint4*>(
                          q + ((row0 + r) * Hg + hk) * KD + cc * 8)
                    : z;
    rk[it] = r < nv ? *reinterpret_cast<const uint4*>(
                          k + ((row0 + r) * Hg + hk) * KD + cc * 8)
                    : z;
  }
  auto load_half = [&](int vh) {
  #pragma unroll
    for (int it = 0; it < 4;
         it++) {  // h rows v = vh*64 + idx/16, 16 chunks of K
      const int idx = tid + it * THREADS;
      rh[it] = *reinterpret_cast<const uint4*>(hp + (vh * 64 + idx / 16) * KD +
                                               (idx % 16) * 8);
    }
  #pragma unroll
    for (int it = 0; it < 2;
         it++) {  // v_new rows 2s2, 2s2+1, 8 chunks of the half
      const int idx = tid + it * THREADS, s2 = idx / 8, cc = idx % 8;
      const int s = 2 * s2;
      const long col = (long)h * KD + vh * 64 + cc * 8;
      rv0[it] =
          s < nv
              ? *reinterpret_cast<const uint4*>(vn + (row0 + s) * H * KD + col)
              : z;
      rv1[it] = s + 1 < nv ? *reinterpret_cast<const uint4*>(
                                 vn + (row0 + s + 1) * H * KD + col)
                           : z;
    }
  };
  load_half(0);
  if (tid < BT) sG[tid] = tid < nv ? g[(row0 + tid) * H + h] : 0.f;
  #pragma unroll
  for (int it = 0; it < 4; it++) {
    store_k2major(sQ, BT, tid + it * THREADS, rq[it]);
    store_k2major(sKH, BT, tid + it * THREADS, rk[it]);
  }
  __syncthreads();

  // S = q k^T: rows 4ty.., keys 4tx..
  float sacc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++)
  #pragma unroll
    for (int j = 0; j < 4; j++) sacc[i][j] = 0.f;
  #pragma unroll 4
  for (int k2 = 0; k2 < KD / 2; k2++) {
    const int sw = ((k2 >> 2) & 7) * 4;
    const uint4 a =
        *reinterpret_cast<const uint4*>(&sQ[k2 * BT + ((ty * 4) ^ sw)]);
    const uint4 b =
        *reinterpret_cast<const uint4*>(&sKH[k2 * BT + ((tx * 4) ^ sw)]);
    const half2 av[4] = {as_h2(a.x), as_h2(a.y), as_h2(a.z), as_h2(a.w)};
    const half2 bv[4] = {as_h2(b.x), as_h2(b.y), as_h2(b.z), as_h2(b.w)};
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 4; j++)
        sacc[i][j] = __builtin_amdgcn_fdot2(av[i], bv[j], sacc[i][j], false);
  }
  // P -> sP[s2][t ^ swz]: keys 4tx..4tx+3 are pairs 2tx, 2tx+1; row chunks
  // XOR-swizzled by key pair / 2 (= tx) so the stores are conflict-free.
  {
    uint32_t pw[4][2];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int t = ty * 4 + i;
      float p[4];
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        const int s = tx * 4 + j;
        p[j] = (s <= t && t < nv) ? sacc[i][j] * __expf(sG[t] - sG[s]) : 0.f;
      }
      pw[i][0] = as_u32(__floats2half2_rn(p[0], p[1]));
      pw[i][1] = as_u32(__floats2half2_rn(p[2], p[3]));
    }
    *reinterpret_cast<uint4*>(&sP[(tx * 2) * BT + (ty ^ tx) * 4]) =
        make_uint4(pw[0][0], pw[1][0], pw[2][0], pw[3][0]);
    *reinterpret_cast<uint4*>(&sP[(tx * 2 + 1) * BT + (ty ^ tx) * 4]) =
        make_uint4(pw[0][1], pw[1][1], pw[2][1], pw[3][1]);
  }
  float eg[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) eg[i] = __expf(sG[ty * 4 + i]);

  for (int vh = 0; vh < 2; vh++) {
    __syncthreads();  // S done with sKH / previous half done with sKH, sV
  #pragma unroll
    for (int it = 0; it < 4; it++)
      store_k2major(sKH, 64, tid + it * THREADS, rh[it]);
  #pragma unroll
    for (int it = 0; it < 2; it++) {
      const int idx = tid + it * THREADS, s2 = idx / 8, cc = idx % 8;
      const half2* ha = reinterpret_cast<const half2*>(&rv0[it]);
      const half2* hb = reinterpret_cast<const half2*>(&rv1[it]);
      uint32_t wv[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        wv[2 * e] = as_u32(__lows2half2(ha[e], hb[e]));
        wv[2 * e + 1] = as_u32(__highs2half2(ha[e], hb[e]));
      }
      *reinterpret_cast<uint4*>(&sV[s2 * 64 + cc * 8]) =
          make_uint4(wv[0], wv[1], wv[2], wv[3]);
      *reinterpret_cast<uint4*>(&sV[s2 * 64 + cc * 8 + 4]) =
          make_uint4(wv[4], wv[5], wv[6], wv[7]);
    }
    __syncthreads();
    if (vh == 0) load_half(1);  // in flight during this half's GEMMs

    float oacc[4][4];
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 4; j++) oacc[i][j] = 0.f;
    // O1 = q h^T over K (h half in sKH as [k2][v ^ swz]).
  #pragma unroll 4
    for (int k2 = 0; k2 < KD / 2; k2++) {
      const int sw = ((k2 >> 2) & 7) * 4;
      const uint4 a =
          *reinterpret_cast<const uint4*>(&sQ[k2 * BT + ((ty * 4) ^ sw)]);
      const uint4 b =
          *reinterpret_cast<const uint4*>(&sKH[k2 * 64 + ((tx * 4) ^ sw)]);
      const half2 av[4] = {as_h2(a.x), as_h2(a.y), as_h2(a.z), as_h2(a.w)};
      const half2 bv[4] = {as_h2(b.x), as_h2(b.y), as_h2(b.z), as_h2(b.w)};
  #pragma unroll
      for (int i = 0; i < 4; i++)
  #pragma unroll
        for (int j = 0; j < 4; j++)
          oacc[i][j] = __builtin_amdgcn_fdot2(av[i], bv[j], oacc[i][j], false);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 4; j++) oacc[i][j] *= eg[i];
    // O2 = P v_new over s.
  #pragma unroll 4
    for (int s2 = 0; s2 < BT / 2; s2++) {
      const uint4 a = *reinterpret_cast<const uint4*>(
          &sP[s2 * BT + (ty ^ ((s2 >> 1) & 15)) * 4]);
      const uint4 b = *reinterpret_cast<const uint4*>(&sV[s2 * 64 + tx * 4]);
      const half2 av[4] = {as_h2(a.x), as_h2(a.y), as_h2(a.z), as_h2(a.w)};
      const half2 bv[4] = {as_h2(b.x), as_h2(b.y), as_h2(b.z), as_h2(b.w)};
  #pragma unroll
      for (int i = 0; i < 4; i++)
  #pragma unroll
        for (int j = 0; j < 4; j++)
          oacc[i][j] = __builtin_amdgcn_fdot2(av[i], bv[j], oacc[i][j], false);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int t = ty * 4 + i;
      if (t >= nv) continue;
      const uint2 pk = make_uint2(
          as_u32(__floats2half2_rn(oacc[i][0] * scale, oacc[i][1] * scale)),
          as_u32(__floats2half2_rn(oacc[i][2] * scale, oacc[i][3] * scale)));
      *reinterpret_cast<uint2*>(o + ((row0 + t) * H + h) * KD + vh * 64 +
                                tx * 4) = pk;
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int K, bool L2NORM, bool G_EXP>
__global__ void gdn_post_conv_rdna2_kernel(
    const __half*, const __half*, const __half*, const float*, const void*,
    const bool, __half*, __half*, __half*, float*, float*, const int, const int,
    const int, const long, const long, const long, const float) {}

__global__ void gdn_wy_rdna2_kernel(const __half*, const __half*, const float*,
                                    const float*, float*, __half*, __half*,
                                    const int*, const int*, const int,
                                    const int) {}

template <int BV, bool H0_F32>
__global__ void gdn_fwd_h_rdna2_kernel(const __half*, const __half*,
                                       const __half*, const float*, const void*,
                                       __half*, __half*, float*, const int*,
                                       const void*, const bool, const int,
                                       const int) {}

__global__ void gdn_fwd_o_rdna2_kernel(const __half*, const __half*,
                                       const __half*, const __half*,
                                       const float*, __half*, const int*,
                                       const int*, const int, const int,
                                       const float) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gdn_rdna2
}  // namespace vllm

// conv_output [L, 2 * H * K + HV * V] fp16 (rows 16-byte aligned) -> q, k
// [L, H, K], v [L, HV, V] (contiguous fp16), g, beta [L, HV] (contiguous
// fp32); a, b [L, HV] fp16 with contiguous rows, A_log [HV] fp32, dt_bias
// [HV] fp16 or fp32. K in {64, 128, 256}, V % 8 == 0.
void gdn_post_conv_rdna2(const torch::Tensor& conv_output,
                         const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& A_log,
                         const torch::Tensor& dt_bias, torch::Tensor& q,
                         torch::Tensor& k, torch::Tensor& v, torch::Tensor& g,
                         torch::Tensor& beta, bool apply_l2norm,
                         bool output_g_exp, double eps) {
  using namespace vllm::gdn_rdna2;
  TORCH_CHECK(
      conv_output.dtype() == torch::kFloat16 && a.dtype() == torch::kFloat16 &&
          b.dtype() == torch::kFloat16 && A_log.dtype() == torch::kFloat32 &&
          (dt_bias.dtype() == torch::kFloat16 ||
           dt_bias.dtype() == torch::kFloat32) &&
          q.dtype() == torch::kFloat16 && k.dtype() == torch::kFloat16 &&
          v.dtype() == torch::kFloat16 && g.dtype() == torch::kFloat32 &&
          beta.dtype() == torch::kFloat32,
      "gdn_post_conv_rdna2: fp16 activations, fp32 A_log / g / beta");
  const int L = conv_output.size(0), H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  TORCH_CHECK(K == 64 || K == 128 || K == 256, "gdn_post_conv_rdna2: K");
  TORCH_CHECK(V % 8 == 0 && conv_output.size(1) == 2 * H * K + HV * V &&
                  conv_output.stride(1) == 1 &&
                  conv_output.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(conv_output.data_ptr()) % 16 == 0,
              "gdn_post_conv_rdna2: conv_output must be [L, 2*H*K + HV*V] "
              "with 16-byte aligned rows");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  g.is_contiguous() && beta.is_contiguous() &&
                  a.stride(1) == 1 && b.stride(1) == 1 &&
                  A_log.is_contiguous() && dt_bias.is_contiguous(),
              "gdn_post_conv_rdna2: contiguous outputs and rows");
  if (L == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(conv_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define VLLM_GDN_POST_CONV_LAUNCH(KD, NORM, GEXP)                              \
  vllm::gdn_rdna2::gdn_post_conv_rdna2_kernel<KD, NORM, GEXP>                  \
      <<<L, THREADS, 0, stream>>>(                                             \
          (const __half*)conv_output.data_ptr(), (const __half*)a.data_ptr(),  \
          (const __half*)b.data_ptr(), A_log.data_ptr<float>(),                \
          dt_bias.data_ptr(), dt_bias.dtype() == torch::kFloat32,              \
          (__half*)q.data_ptr(), (__half*)k.data_ptr(), (__half*)v.data_ptr(), \
          g.data_ptr<float>(), beta.data_ptr<float>(), H, HV, V,               \
          conv_output.stride(0), a.stride(0), b.stride(0), (float)eps)
#define VLLM_GDN_POST_CONV_BY_FLAGS(KD)          \
  if (apply_l2norm && !output_g_exp) {           \
    VLLM_GDN_POST_CONV_LAUNCH(KD, true, false);  \
  } else if (apply_l2norm) {                     \
    VLLM_GDN_POST_CONV_LAUNCH(KD, true, true);   \
  } else if (!output_g_exp) {                    \
    VLLM_GDN_POST_CONV_LAUNCH(KD, false, false); \
  } else {                                       \
    VLLM_GDN_POST_CONV_LAUNCH(KD, false, true);  \
  }
  if (K == 128) {
    VLLM_GDN_POST_CONV_BY_FLAGS(128);
  } else if (K == 64) {
    VLLM_GDN_POST_CONV_BY_FLAGS(64);
  } else {
    VLLM_GDN_POST_CONV_BY_FLAGS(256);
  }
#undef VLLM_GDN_POST_CONV_BY_FLAGS
#undef VLLM_GDN_POST_CONV_LAUNCH
}

// Varlen WY step of the chunked gated delta rule (chunk size 64): k [T, Hg,
// 128] and v [T, H, 128] fp16, beta and g [T, H] fp32 (g = per-token log
// decay) -> g_cum [T, H] fp32 (chunk-local cumsum), w [T, H, 128] and u
// [T, H, 128] fp16. cu_seqlens [N + 1] and chunk_indices [NT, 2] (sequence,
// chunk) int32 as from FLA's prepare_chunk_indices.
void gdn_wy_rdna2(const torch::Tensor& k, const torch::Tensor& v,
                  const torch::Tensor& beta, const torch::Tensor& g,
                  torch::Tensor& g_cum, torch::Tensor& w, torch::Tensor& u,
                  const torch::Tensor& cu_seqlens,
                  const torch::Tensor& chunk_indices) {
  using namespace vllm::gdn_rdna2;
  TORCH_CHECK(
      k.dtype() == torch::kFloat16 && v.dtype() == torch::kFloat16 &&
          w.dtype() == torch::kFloat16 && u.dtype() == torch::kFloat16 &&
          beta.dtype() == torch::kFloat32 && g.dtype() == torch::kFloat32 &&
          g_cum.dtype() == torch::kFloat32,
      "gdn_wy_rdna2: fp16 k / v / w / u, fp32 beta / g / g_cum");
  for (const torch::Tensor* t :
       {&k, &v, &beta, &g, static_cast<const torch::Tensor*>(&g_cum),
        static_cast<const torch::Tensor*>(&w),
        static_cast<const torch::Tensor*>(&u)})
    TORCH_CHECK(t->is_contiguous(), "gdn_wy_rdna2 needs contiguous tensors");
  TORCH_CHECK(k.size(-1) == KD && v.size(-1) == KD && w.size(-1) == KD,
              "gdn_wy_rdna2 supports head dim 128");
  const int H = v.size(-2), Hg = k.size(-2);
  const long T = v.numel() / (H * KD);
  TORCH_CHECK(H % Hg == 0 && k.numel() == T * Hg * KD &&
                  beta.numel() == T * H && g.numel() == T * H &&
                  g_cum.numel() == T * H && w.numel() == T * H * KD &&
                  u.numel() == v.numel(),
              "gdn_wy_rdna2: shape mismatch");
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32 &&
                  chunk_indices.dtype() == torch::kInt32 &&
                  cu_seqlens.is_contiguous() && chunk_indices.is_contiguous() &&
                  chunk_indices.dim() == 2 && chunk_indices.size(1) == 2,
              "gdn_wy_rdna2: int32 cu_seqlens and [NT, 2] chunk_indices");
  const int NT = chunk_indices.size(0);
  if (NT == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gdn_wy_rdna2_kernel<<<dim3(H, NT), THREADS, 0, stream>>>(
      (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
      beta.data_ptr<float>(), g.data_ptr<float>(), g_cum.data_ptr<float>(),
      (__half*)w.data_ptr(), (__half*)u.data_ptr(), cu_seqlens.data_ptr<int>(),
      chunk_indices.data_ptr<int>(), H, Hg);
}

// Chunk state recurrence: k [T, Hg, 128], w / u [T, H, 128] fp16, g_cum [T, H]
// fp32 (chunk-local cumsum) and optional initial state h0 [N, H, 128, 128]
// (fp32 or fp16, [value][key]) -> h [NT, H, 128, 128] fp16 (state at each chunk
// start), v_new [T, H, 128] fp16 and optional final state ht [N, H, 128, 128]
// fp32. chunk_offsets [N + 1] (int32 or int64) from FLA's
// prepare_chunk_offsets.
void gdn_fwd_h_rdna2(const torch::Tensor& k, const torch::Tensor& w,
                     const torch::Tensor& u, const torch::Tensor& g_cum,
                     const std::optional<torch::Tensor>& h0, torch::Tensor& h,
                     torch::Tensor& v_new,
                     const std::optional<torch::Tensor>& ht,
                     const torch::Tensor& cu_seqlens,
                     const torch::Tensor& chunk_offsets) {
  using namespace vllm::gdn_rdna2;
  TORCH_CHECK(
      k.dtype() == torch::kFloat16 && w.dtype() == torch::kFloat16 &&
          u.dtype() == torch::kFloat16 && h.dtype() == torch::kFloat16 &&
          v_new.dtype() == torch::kFloat16 && g_cum.dtype() == torch::kFloat32,
      "gdn_fwd_h_rdna2: fp16 k / w / u / h / v_new, fp32 g_cum");
  for (const torch::Tensor* t :
       {&k, &w, &u, &g_cum, static_cast<const torch::Tensor*>(&h),
        static_cast<const torch::Tensor*>(&v_new)})
    TORCH_CHECK(t->is_contiguous(), "gdn_fwd_h_rdna2 needs contiguous tensors");
  TORCH_CHECK(k.size(-1) == KD && u.size(-1) == KD && w.size(-1) == KD,
              "gdn_fwd_h_rdna2 supports head dim 128");
  const int H = u.size(-2), Hg = k.size(-2), N = cu_seqlens.size(0) - 1;
  TORCH_CHECK(H % Hg == 0 && cu_seqlens.dtype() == torch::kInt32 &&
                  (chunk_offsets.dtype() == torch::kInt32 ||
                   chunk_offsets.dtype() == torch::kInt64) &&
                  chunk_offsets.is_contiguous() &&
                  chunk_offsets.numel() == N + 1,
              "gdn_fwd_h_rdna2: heads / int32 metadata");
  const void* h0p = nullptr;
  bool h0_f32 = true;
  if (h0) {
    TORCH_CHECK(
        h0->is_contiguous() && h0->numel() == (long)N * H * KD * KD &&
            (h0->dtype() == torch::kFloat32 || h0->dtype() == torch::kFloat16),
        "gdn_fwd_h_rdna2: h0 must be contiguous [N, H, 128, 128] "
        "fp32 / fp16");
    h0p = h0->data_ptr();
    h0_f32 = h0->dtype() == torch::kFloat32;
  }
  float* htp = nullptr;
  if (ht) {
    TORCH_CHECK(ht->is_contiguous() && ht->dtype() == torch::kFloat32 &&
                    ht->numel() == (long)N * H * KD * KD,
                "gdn_fwd_h_rdna2: ht must be contiguous fp32 [N, H, 128, 128]");
    htp = ht->data_ptr<float>();
  }
  if (N == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(H * (KD / H_BV), N);
#define VLLM_GDN_FWD_H_LAUNCH(F32)                                 \
  gdn_fwd_h_rdna2_kernel<H_BV, F32><<<grid, THREADS, 0, stream>>>( \
      (const __half*)k.data_ptr(), (const __half*)w.data_ptr(),    \
      (const __half*)u.data_ptr(), g_cum.data_ptr<float>(), h0p,   \
      (__half*)h.data_ptr(), (__half*)v_new.data_ptr(), htp,       \
      cu_seqlens.data_ptr<int>(), chunk_offsets.data_ptr(),        \
      chunk_offsets.dtype() == torch::kInt64, H, Hg)
  if (h0_f32) {
    VLLM_GDN_FWD_H_LAUNCH(true);
  } else {
    VLLM_GDN_FWD_H_LAUNCH(false);
  }
#undef VLLM_GDN_FWD_H_LAUNCH
}

// Chunk output: q / k [T, Hg, 128], v_new [T, H, 128] fp16, h [NT, H, 128, 128]
// fp16, g_cum [T, H] fp32 -> o [T, H, 128] fp16 (contiguous).
void gdn_fwd_o_rdna2(const torch::Tensor& q, const torch::Tensor& k,
                     const torch::Tensor& v_new, const torch::Tensor& h,
                     const torch::Tensor& g_cum, torch::Tensor& o,
                     const torch::Tensor& cu_seqlens,
                     const torch::Tensor& chunk_indices, double scale) {
  using namespace vllm::gdn_rdna2;
  TORCH_CHECK(
      q.dtype() == torch::kFloat16 && k.dtype() == torch::kFloat16 &&
          v_new.dtype() == torch::kFloat16 && h.dtype() == torch::kFloat16 &&
          o.dtype() == torch::kFloat16 && g_cum.dtype() == torch::kFloat32,
      "gdn_fwd_o_rdna2: fp16 q / k / v_new / h / o, fp32 g_cum");
  for (const torch::Tensor* t :
       {&q, &k, &v_new, &h, &g_cum, static_cast<const torch::Tensor*>(&o)})
    TORCH_CHECK(t->is_contiguous(), "gdn_fwd_o_rdna2 needs contiguous tensors");
  TORCH_CHECK(
      q.size(-1) == KD && v_new.size(-1) == KD && o.numel() == v_new.numel(),
      "gdn_fwd_o_rdna2 supports head dim 128");
  const int H = v_new.size(-2), Hg = k.size(-2);
  const int NT = chunk_indices.size(0);
  TORCH_CHECK(H % Hg == 0 && h.numel() == (long)NT * H * KD * KD &&
                  cu_seqlens.dtype() == torch::kInt32 &&
                  chunk_indices.dtype() == torch::kInt32,
              "gdn_fwd_o_rdna2: shapes / int32 metadata");
  if (NT == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  gdn_fwd_o_rdna2_kernel<<<dim3(H, NT), THREADS, 0, stream>>>(
      (const __half*)q.data_ptr(), (const __half*)k.data_ptr(),
      (const __half*)v_new.data_ptr(), (const __half*)h.data_ptr(),
      g_cum.data_ptr<float>(), (__half*)o.data_ptr(),
      cu_seqlens.data_ptr<int>(), chunk_indices.data_ptr<int>(), H, Hg,
      (float)scale);
}
