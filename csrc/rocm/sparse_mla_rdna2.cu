// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) rope-free sparse MLA attention (GLM-5.3-Flash's DSA layers)
// over the indexer's ragged key lists: for each query token t and head h,
//   out[t, h] = softmax_k(scale * q[t, h] . kv[slot_k]) . kv[slot_k]
// with slot_k = indices[indptr[t] .. indptr[t + 1]) into kv [slots, 512] (the
// fp16 latent cache, K = V); slots outside [0, slots) are skipped.
//
// A workgroup (256 threads, 8 x 32) owns 32 rows, heads h0..h0+31 of one token,
// and walks the token's keys in 128-key tiles with the LDS scheme of
// attention_rdna2.cu (v_dot2_f32_f16 GEMMs on k2-major half2 tiles, every LDS
// read a ds_read_b128, row operands wave broadcasts):
//   S = Q K^T: each thread 4 rows x 4 keys; Q stays in LDS, K is staged 32
//              dims at a time;
//   O += P V : each thread 4 rows x 16 columns (4-column groups spaced 128
//              apart); V (the same rows) is staged 8 keys at a time.
// K and V stages share one double buffer with the next stage's gather loaded
// during the current GEMM. Online softmax in exp2 with fp32 statistics; row
// maxima and sums are reduced over the 32 lanes of a row group.
//
// With few tokens (decode, MTP verify) the key list is split into up to 16
// segments; each writes unnormalized fp32 partials and its (max, sum), and
// sparse_mla_combine_rdna2_kernel merges them. Prefill writes fp16 directly,
// longest key lists first.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace sparse_mla_rdna2 {

static constexpr int THREADS = 256;
static constexpr int D = 512;
static constexpr int D2 = D / 2;
static constexpr int BM = 32;   // rows (heads of one token) per workgroup
static constexpr int BN = 128;  // keys per tile
static constexpr int DC = 32;   // head dims per K stage
static constexpr int VC = 8;    // keys per V stage
static constexpr int KS = D / DC;
static constexpr int KL = DC / 8;  // lanes per key row in a K stage
static constexpr int K_ITEMS = BN * KL / THREADS;
static constexpr int K_STAGE = (DC / 2) * BN;
static constexpr int VS = BN / VC;
static constexpr int V_TOTAL = (VC / 2) * (D / 8);
static constexpr int V_STAGE = (VC / 2) * D;
static constexpr int U_STAGE = K_STAGE > V_STAGE ? K_STAGE : V_STAGE;
static constexpr int TN = D / 32;  // output columns per thread
static constexpr int Q_ITEMS = BM * (D / 8) / THREADS;
static constexpr int MAX_SEG = 16;
static_assert(K_ITEMS * THREADS == BN * KL && V_TOTAL == THREADS, "staging");

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}

__device__ __forceinline__ float row32_max(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}

__device__ __forceinline__ float row32_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// SPLIT: write fp32 partials + (max, sum) per segment instead of fp16 output.
template <bool SPLIT>
__global__ void __launch_bounds__(THREADS)
    sparse_mla_rdna2_kernel(const __half* __restrict__ Q,
                            const __half* __restrict__ KV,
                            const int* __restrict__ indices,
                            const int* __restrict__ indptr,
                            __half* __restrict__ O, float* __restrict__ part_o,
                            float* __restrict__ part_ml, const int H,
                            const int num_kv, const long q_st, const long q_sh,
                            const long kv_st, const long o_st, const long o_sh,
                            const float scale) {
  constexpr float LOG2E = 1.4426950408889634f;
  const float scale_log2e = scale * LOG2E;
  // sU holds K stages ([k2][key], 16-byte chunks XOR-swizzled by k2 so the
  // transposing stores are conflict-free) during S, then V stages ([k2][d]).
  __shared__ __align__(16) uint32_t sQ[D2][BM];
  __shared__ __align__(16) uint32_t sU[2][U_STAGE];
  __shared__ __align__(16) uint32_t sP[BN / 2][BM];

  const int tid = threadIdx.x, tx = tid % 32, ty = tid / 32;
  const int t = SPLIT ? blockIdx.x : gridDim.x - 1 - blockIdx.x;
  const int h0 = blockIdx.y * BM;
  const int seg = blockIdx.z, nseg = gridDim.z;
  const int kbeg = indptr[t], nkeys = indptr[t + 1] - kbeg;
  const int ntiles = (nkeys + BN - 1) / BN;
  const int tile_lo = seg * ntiles / nseg, tile_hi = (seg + 1) * ntiles / nseg;
  const int key_end = min(nkeys, tile_hi * BN);
  const int* idx = indices + kbeg;

  #pragma unroll
  for (int it = 0; it < Q_ITEMS; it++) {
    const int i = tid + it * THREADS;
    const int r = i % BM, c = i / BM;
    const uint4 v =
        *reinterpret_cast<const uint4*>(Q + t * q_st + (h0 + r) * q_sh + c * 8);
    sQ[c * 4 + 0][r] = v.x;
    sQ[c * 4 + 1][r] = v.y;
    sQ[c * 4 + 2][r] = v.z;
    sQ[c * 4 + 3][r] = v.w;
  }

  float acc[4][TN];
  #pragma unroll
  for (int i = 0; i < 4; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
  float m_i[4], l_i[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    m_i[i] = -INFINITY;
    l_i[i] = 0.f;
  }

  auto row_of = [&](int key) -> long {
    if (key >= key_end) return -1;
    const int s = idx[key];
    return s >= 0 && s < num_kv ? s * kv_st : -1;
  };
  long krow[K_ITEMS];
  uint4 kr[K_ITEMS];
  auto kaddr = [&](int kt) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++)
      krow[it] = row_of(kt + (tid + it * THREADS) / KL);
  };
  auto kload = [&](int dc) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int part = (tid + it * THREADS) % KL;
      kr[it] = krow[it] >= 0 ? *reinterpret_cast<const uint4*>(
                                   KV + krow[it] + dc * DC + part * 8)
                             : make_uint4(0, 0, 0, 0);
    }
  };
  auto kstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int i = tid + it * THREADS;
      const int key = i / KL, part = i % KL;
      uint32_t* d = &sU[buf][part * 4 * BN + (key ^ (part * (32 / KL)))];
      d[0] = kr[it].x;
      d[BN] = kr[it].y;
      d[2 * BN] = kr[it].z;
      d[3 * BN] = kr[it].w;
    }
  };
  uint4 va, vb;
  auto vload = [&](int kt, int vc) {
    const int kp = tid / (D / 8), c = tid % (D / 8);
    const int k0 = kt + vc * VC + kp * 2;
    const long ra = row_of(k0), rb = row_of(k0 + 1);
    va = ra >= 0 ? *reinterpret_cast<const uint4*>(KV + ra + c * 8)
                 : make_uint4(0, 0, 0, 0);
    vb = rb >= 0 ? *reinterpret_cast<const uint4*>(KV + rb + c * 8)
                 : make_uint4(0, 0, 0, 0);
  };
  auto vstore = [&](int buf) {
    const int kp = tid / (D / 8), c = tid % (D / 8);
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&va);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&vb);
    uint32_t w[8];
  #pragma unroll
    for (int e = 0; e < 4; e++) {
      w[2 * e] = (a[e] & 0xFFFFu) | (b[e] << 16);
      w[2 * e + 1] = (a[e] >> 16) | (b[e] & 0xFFFF0000u);
    }
    uint32_t* d = &sU[buf][kp * D + c * 8];
    *reinterpret_cast<uint4*>(d) = make_uint4(w[0], w[1], w[2], w[3]);
    *reinterpret_cast<uint4*>(d + 4) = make_uint4(w[4], w[5], w[6], w[7]);
  };

  const int key_begin = tile_lo * BN;
  if (key_begin < key_end) {
    kaddr(key_begin);
    kload(0);
  }
  for (int kt = key_begin; kt < key_end; kt += BN) {
    // sU[0] was last read two V stages ago; sP is rewritten after barriers.
    kstore(0);
    __syncthreads();
    float sacc[4][4];
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < 4; j++) sacc[i][j] = 0.f;
    for (int dc = 0; dc < KS; dc++) {
      const int cur = dc & 1;
      if (dc + 1 < KS)
        kload(dc + 1);
      else
        vload(kt, 0);
  #pragma unroll 4
      for (int k2 = 0; k2 < DC / 2; k2++) {
        const int sw = (k2 >> 2) * (32 / KL);
        const uint4 qa =
            *reinterpret_cast<const uint4*>(&sQ[dc * (DC / 2) + k2][ty * 4]);
        const uint4 kb = *reinterpret_cast<const uint4*>(
            &sU[cur][k2 * BN + ((tx * 4) ^ sw)]);
        const half2 a[4] = {as_h2(qa.x), as_h2(qa.y), as_h2(qa.z), as_h2(qa.w)};
        const half2 b[4] = {as_h2(kb.x), as_h2(kb.y), as_h2(kb.z), as_h2(kb.w)};
  #pragma unroll
        for (int i = 0; i < 4; i++)
  #pragma unroll
          for (int j = 0; j < 4; j++)
            sacc[i][j] = __builtin_amdgcn_fdot2(a[i], b[j], sacc[i][j], false);
      }
      if (dc + 1 < KS) {
        kstore(cur ^ 1);
        __syncthreads();
      }
    }

    bool valid[4];
  #pragma unroll
    for (int j = 0; j < 4; j++) valid[j] = row_of(kt + tx * 4 + j) >= 0;
    float alpha[4];
    uint32_t pw[4][2];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float sv[4];
      float mx = -INFINITY;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        sv[j] = valid[j] ? sacc[i][j] * scale_log2e : -INFINITY;
        mx = fmaxf(mx, sv[j]);
      }
      mx = row32_max(mx);
      // A row whose keys are all invalid so far keeps a finite floor.
      const float m_new = fmaxf(fmaxf(m_i[i], mx), -1e30f);
      alpha[i] = exp2f(m_i[i] - m_new);
      m_i[i] = m_new;
      float p[4], ps = 0.f;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        p[j] = exp2f(sv[j] - m_new);
        ps += p[j];
      }
      l_i[i] = l_i[i] * alpha[i] + ps;
      const half2 q0 = __floats2half2_rn(p[0], p[1]);
      const half2 q1 = __floats2half2_rn(p[2], p[3]);
      pw[i][0] = *reinterpret_cast<const uint32_t*>(&q0);
      pw[i][1] = *reinterpret_cast<const uint32_t*>(&q1);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] *= alpha[i];
    // P -> sP[key2][row]; 16-byte row chunks XOR-swizzled by key2 / 2 (= tx).
    const int pc = (ty ^ (tx & 7)) * 4;
    *reinterpret_cast<uint4*>(&sP[tx * 2][pc]) =
        make_uint4(pw[0][0], pw[1][0], pw[2][0], pw[3][0]);
    *reinterpret_cast<uint4*>(&sP[tx * 2 + 1][pc]) =
        make_uint4(pw[0][1], pw[1][1], pw[2][1], pw[3][1]);
    vstore(0);
    __syncthreads();  // sP and V stage 0 ready

    for (int vc = 0; vc < VS; vc++) {
      const int cur = vc & 1;
      if (vc + 1 < VS) {
        vload(kt, vc + 1);
      } else if (kt + BN < key_end) {
        kaddr(kt + BN);
        kload(0);
      }
  #pragma unroll
      for (int k2 = 0; k2 < VC / 2; k2++) {
        const int pk = vc * (VC / 2) + k2;
        const uint4 pa = *reinterpret_cast<const uint4*>(
            &sP[pk][(ty ^ ((pk >> 1) & 7)) * 4]);
        const half2 a[4] = {as_h2(pa.x), as_h2(pa.y), as_h2(pa.z), as_h2(pa.w)};
  #pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const uint4 vv = *reinterpret_cast<const uint4*>(
              &sU[cur][k2 * D + j * 32 + tx * 4]);
          const half2 b[4] = {as_h2(vv.x), as_h2(vv.y), as_h2(vv.z),
                              as_h2(vv.w)};
  #pragma unroll
          for (int i = 0; i < 4; i++)
  #pragma unroll
            for (int jj = 0; jj < 4; jj++)
              acc[i][j + jj] =
                  __builtin_amdgcn_fdot2(a[i], b[jj], acc[i][j + jj], false);
        }
      }
      if (vc + 1 < VS) {
        vstore(cur ^ 1);
        __syncthreads();
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int h = h0 + ty * 4 + i;
    const float l = row32_sum(l_i[i]);
    if constexpr (SPLIT) {
      const long pr = ((long)t * nseg + seg) * H + h;
      float* po = part_o + pr * D;
  #pragma unroll
      for (int j = 0; j < TN; j += 4)
        *reinterpret_cast<float4*>(po + j * 32 + tx * 4) =
            make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
      if (tx == 0) {
        part_ml[pr * 2] = m_i[i];
        part_ml[pr * 2 + 1] = l;
      }
    } else {
      const float inv = l > 0.f ? 1.f / l : 0.f;
      __half* o = O + t * o_st + h * o_sh;
  #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const half2 a0 =
            __floats2half2_rn(acc[i][j] * inv, acc[i][j + 1] * inv);
        const half2 a1 =
            __floats2half2_rn(acc[i][j + 2] * inv, acc[i][j + 3] * inv);
        *reinterpret_cast<uint2*>(o + j * 32 + tx * 4) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&a0),
                       *reinterpret_cast<const uint32_t*>(&a1));
      }
    }
  }
}

// One workgroup per (token, head): out = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s -
// M) l_s over the segments' partials.
__global__ void __launch_bounds__(THREADS)
    sparse_mla_combine_rdna2_kernel(const float* __restrict__ part_o,
                                    const float* __restrict__ part_ml,
                                    __half* __restrict__ O, const int H,
                                    const int nseg, const long o_st,
                                    const long o_sh) {
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  float w[MAX_SEG];
  float M = -INFINITY;
  for (int s = 0; s < nseg; s++) {
    const long pr = ((long)t * nseg + s) * H + h;
    w[s] = part_ml[pr * 2 + 1] > 0.f ? part_ml[pr * 2] : -INFINITY;
    M = fmaxf(M, w[s]);
  }
  float L = 0.f;
  for (int s = 0; s < nseg; s++) {
    const long pr = ((long)t * nseg + s) * H + h;
    w[s] = M == -INFINITY ? 0.f : exp2f(w[s] - M);
    L += w[s] * part_ml[pr * 2 + 1];
  }
  const float inv = L > 0.f ? 1.f / L : 0.f;
  const int c = tid * 2;
  float a0 = 0.f, a1 = 0.f;
  for (int s = 0; s < nseg; s++) {
    const float2 v = *reinterpret_cast<const float2*>(
        part_o + (((long)t * nseg + s) * H + h) * D + c);
    a0 += w[s] * v.x;
    a1 += w[s] * v.y;
  }
  *reinterpret_cast<half2*>(O + t * o_st + h * o_sh + c) =
      __floats2half2_rn(a0 * inv, a1 * inv);
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

template <bool SPLIT>
__global__ void sparse_mla_rdna2_kernel(const __half*, const __half*,
                                        const int*, const int*, __half*, float*,
                                        float*, const int, const int,
                                        const long, const long, const long,
                                        const long, const long, const float) {}

__global__ void sparse_mla_combine_rdna2_kernel(const float*, const float*,
                                                __half*, const int, const int,
                                                const long, const long) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace sparse_mla_rdna2
}  // namespace vllm

// q [T, H, 512] fp16 (contiguous rows), kv [slots, 512] fp16 (contiguous
// rows), indices / indptr int32 ragged key lists, out [T, H, 512] fp16
// (contiguous rows). H % 32 == 0. With nseg > 1 the key lists are split over
// nseg segments through part_o [T, nseg, H, 512] and part_ml [T, nseg, H, 2]
// (fp32 workspaces).
void sparse_mla_attn_rdna2(const torch::Tensor& q, const torch::Tensor& kv,
                           const torch::Tensor& indices,
                           const torch::Tensor& indptr, torch::Tensor& out,
                           const std::optional<torch::Tensor>& part_o,
                           const std::optional<torch::Tensor>& part_ml,
                           int64_t nseg, double scale) {
  using namespace vllm::sparse_mla_rdna2;
  TORCH_CHECK(q.scalar_type() == at::kHalf && kv.scalar_type() == at::kHalf &&
                  out.scalar_type() == at::kHalf,
              "sparse_mla_attn_rdna2: fp16 q / kv / out");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == D && q.stride(2) == 1 &&
                  out.dim() == 3 && out.size(0) == q.size(0) &&
                  out.size(1) == q.size(1) && out.size(2) == D &&
                  out.stride(2) == 1,
              "sparse_mla_attn_rdna2: q, out [T, H, 512] with contiguous rows");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == D && kv.stride(1) == 1,
              "sparse_mla_attn_rdna2: kv [slots, 512] with contiguous rows");
  const int T = q.size(0), H = q.size(1);
  TORCH_CHECK(H % BM == 0, "sparse_mla_attn_rdna2: heads % 32 == 0");
  TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 &&
                  kv.stride(0) % 8 == 0 && out.stride(1) % 4 == 0 &&
                  out.stride(0) % 4 == 0,
              "sparse_mla_attn_rdna2: 16-byte aligned rows");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.is_contiguous() &&
                  indptr.scalar_type() == at::kInt && indptr.is_contiguous() &&
                  indptr.numel() == T + 1,
              "sparse_mla_attn_rdna2: int32 indices, indptr [T + 1]");
  TORCH_CHECK(nseg >= 1 && nseg <= MAX_SEG,
              "sparse_mla_attn_rdna2: 1 <= nseg <= 16");
  if (nseg > 1)
    TORCH_CHECK(part_o && part_ml && part_o->scalar_type() == at::kFloat &&
                    part_ml->scalar_type() == at::kFloat &&
                    part_o->numel() >= (long)T * nseg * H * D &&
                    part_ml->numel() >= (long)T * nseg * H * 2,
                "sparse_mla_attn_rdna2: fp32 workspaces for nseg > 1");
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(T, H / BM, nseg);
  if (nseg == 1) {
    sparse_mla_rdna2_kernel<false><<<grid, THREADS, 0, stream>>>(
        (const __half*)q.data_ptr(), (const __half*)kv.data_ptr(),
        indices.data_ptr<int>(), indptr.data_ptr<int>(),
        (__half*)out.data_ptr(), nullptr, nullptr, H, kv.size(0), q.stride(0),
        q.stride(1), kv.stride(0), out.stride(0), out.stride(1), (float)scale);
  } else {
    sparse_mla_rdna2_kernel<true><<<grid, THREADS, 0, stream>>>(
        (const __half*)q.data_ptr(), (const __half*)kv.data_ptr(),
        indices.data_ptr<int>(), indptr.data_ptr<int>(), nullptr,
        part_o->data_ptr<float>(), part_ml->data_ptr<float>(), H, kv.size(0),
        q.stride(0), q.stride(1), kv.stride(0), out.stride(0), out.stride(1),
        (float)scale);
    sparse_mla_combine_rdna2_kernel<<<dim3(T, H), THREADS, 0, stream>>>(
        part_o->data_ptr<float>(), part_ml->data_ptr<float>(),
        (__half*)out.data_ptr(), H, (int)nseg, out.stride(0), out.stride(1));
  }
}
