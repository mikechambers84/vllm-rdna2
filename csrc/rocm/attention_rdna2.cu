// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) paged causal GQA attention for prefill, on vLLM's fp16 KV
// cache views ([num_blocks, block_size, num_kv_heads, head_size], any strides
// with contiguous heads) and the unified_attention varlen metadata.
//
// A workgroup (256 threads, 16 x 16) owns 64 rows: the (token, q-head) pairs
// of BQ = 64 / G consecutive tokens of one sequence and one KV head (GQA
// packing, G = q-heads per KV head), so K/V tiles are shared by all heads of
// the group. Each 64-key tile is two v_dot2_f32_f16 GEMMs in the k2-major
// LDS layout of the gfx1030 MoE GEMMs (a dword = half2 of two consecutive
// reduction indices; every LDS read is a ds_read_b128, row operands are wave
// broadcasts):
//   S = Q K^T: each thread 4 rows x 4 keys; Q stays in LDS, K is staged DC
//              head dims at a time;
//   O += P V : each thread 4 rows x D / 16 columns (4-column groups spaced 64
//              apart); V is staged 16 keys at a time.
// K and V stages share one double buffer; the next tile's first K stage is
// loaded during P V and the first V stage during the last K stage. Online
// softmax in exp2 with fp32 statistics; row maxima are reduced over the 16
// lanes of a DPP row, row sums once at the end. Tiles are scheduled longest
// first. Head 256 runs 14-15 TFLOPS on a V620 at 140 W (2.5-2.9x the Triton
// unified kernel), head 128 4.2-4.6x, head 64 3.1x.
//
// decode_attention_rdna2: the split-KV (3D) decode / spec-verify path for head
// 256: one workgroup per (sequence x 16-row group, KV head, segment) on the
// same 64-key LDS tiles (S: 1 row x 4 keys per thread, PV: 1 row x 16
// columns), writing the segment partials that unified_attention's
// reduce_segments combines. 1.4-1.9x the Triton 3D kernel (240-330 GB/s).

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <type_traits>

#include "rdna2_fp8.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace attention_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BM = 64;  // rows per workgroup
static constexpr int BN = 64;  // keys per tile
static constexpr int VC = 16;  // keys per V stage

// K/V caches hold fp16, fp8 e4m3fn (KV_FP8) widened exactly to fp16 x 2^-8
// on the way into LDS (the per-tensor scales, x 256, go into the softmax scale
// and the output), or bf16 (KV_BF16) converted to fp16 there (exact in fp16's
// normal range, saturating beyond it).
enum KvType { KV_FP16, KV_FP8, KV_BF16 };

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}

__device__ __forceinline__ float row16_max(float v) {
  #pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 16));
  return v;
}

__device__ __forceinline__ float row16_sum(float v) {
  #pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor(v, o, 16);
  return v;
}

template <int KV>
using KvRaw = std::conditional_t<KV == KV_FP8, uint2, uint4>;

template <int KV>
__device__ __forceinline__ KvRaw<KV> kv_load(const void* base, long elem) {
  if constexpr (KV == KV_FP8)
    return *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(base) +
                                           elem);
  else
    return *reinterpret_cast<const uint4*>(static_cast<const __half*>(base) +
                                           elem);
}

__device__ __forceinline__ uint32_t bf16x2_to_half2(const uint32_t u) {
  return __builtin_bit_cast(
      uint32_t, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(u << 16),
                                           __uint_as_float(u & 0xFFFF0000u)));
}

template <int KV>
__device__ __forceinline__ uint4 kv_widen(const KvRaw<KV> r) {
  if constexpr (KV == KV_FP8) {
    const half2 h[4] = {rdna2::fp8x2_to_half2(r.x, rdna2::FP8_LO),
                        rdna2::fp8x2_to_half2(r.x, rdna2::FP8_HI),
                        rdna2::fp8x2_to_half2(r.y, rdna2::FP8_LO),
                        rdna2::fp8x2_to_half2(r.y, rdna2::FP8_HI)};
    return *reinterpret_cast<const uint4*>(h);
  } else if constexpr (KV == KV_BF16) {
    return make_uint4(bf16x2_to_half2(r.x), bf16x2_to_half2(r.y),
                      bf16x2_to_half2(r.z), bf16x2_to_half2(r.w));
  } else {
    return r;
  }
}

template <int KV>
__device__ __forceinline__ float kv_scale(const float* scale) {
  return KV == KV_FP8 ? 256.f * scale[0] : 1.f;
}

// EXT: sliding window / softcap / sinks compiled in (plain causal otherwise).
// KV: cache element type (see KvType).
template <int D, int DC, bool EXT, int KV>
__global__ void __launch_bounds__(THREADS) unified_attention_rdna2_kernel(
    const __half* __restrict__ Q, const void* __restrict__ Kc,
    const void* __restrict__ Vc, __half* __restrict__ O,
    const int* __restrict__ cu_q, const int* __restrict__ seqused_k,
    const int* __restrict__ block_table, const int num_seqs, const int G,
    const int BQ, const int block_size, const long bt_stride, const long q_st,
    const long q_sh, const long k_sb, const long k_st, const long k_sh,
    const long v_sb, const long v_st, const long v_sh, const long o_st,
    const long o_sh, const float scale, const int window, const float softcap,
    const float* __restrict__ sinks, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale) {
  constexpr float LOG2E = 1.4426950408889634f;
  const float qk_scale = scale * kv_scale<KV>(k_scale);
  const float scale_log2e = qk_scale * LOG2E;
  constexpr int D2 = D / 2;
  constexpr int TN = D / 16;  // output columns per thread
  constexpr int Q_ITEMS = BM * (D / 8) / THREADS;
  constexpr int KS = D / DC;  // K stages per tile
  constexpr int KL = DC / 8;  // lanes per K row in a stage
  constexpr int K_ITEMS = BN * KL / THREADS;
  constexpr int K_STAGE = (DC / 2) * BN;
  constexpr int VS = BN / VC;  // V stages per tile
  constexpr int V_TOTAL = (VC / 2) * (D / 8);
  constexpr int V_ITEMS = (V_TOTAL + THREADS - 1) / THREADS;
  constexpr int V_STAGE = (VC / 2) * D;
  constexpr int U_STAGE = K_STAGE > V_STAGE ? K_STAGE : V_STAGE;
  static_assert(KS % 2 == 0 && K_ITEMS * THREADS == BN * KL, "K staging");
  // sU holds K stages ([k2][key], 16-byte chunks XOR-swizzled by k2 so the
  // transposing stores are conflict-free) during S, then V stages ([k2][d]).
  __shared__ __align__(16) uint32_t sQ[D2][BM];
  __shared__ __align__(16) uint32_t sU[2][U_STAGE];
  __shared__ __align__(16) uint32_t sP[BN / 2][BM];

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int kvh = blockIdx.y;
  const int bid = gridDim.x - 1 - blockIdx.x;  // longest tiles first
  // Sequence of this tile: the last s with cu_q[s] / BQ + s <= bid.
  int lo = 0, hi = num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (cu_q[mid] / BQ + mid <= bid)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int s = lo;
  const int q0 = cu_q[s], qlen = cu_q[s + 1] - q0;
  const int t0 = (bid - (q0 / BQ + s)) * BQ;
  if (t0 >= qlen) return;
  const int klen = seqused_k[s];
  const int ctx = klen - qlen;
  const int ntok = min(BQ, qlen - t0);
  const int key_end =
      ctx + t0 + ntok;  // keys [0, key_end) are visible to some row
  const int* bt = block_table + s * bt_stride;

  // Q tile -> sQ[d2][row] (lanes along rows: conflict-free stores).
  #pragma unroll
  for (int it = 0; it < Q_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int r = idx % BM, c = idx / BM;
    const int tq = r / G, g = r % G;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tq < ntok)
      v = *reinterpret_cast<const uint4*>(Q + (q0 + t0 + tq) * q_st +
                                          (kvh * G + g) * q_sh + c * 8);
    sQ[c * 4 + 0][r] = v.x;
    sQ[c * 4 + 1][r] = v.y;
    sQ[c * 4 + 2][r] = v.z;
    sQ[c * 4 + 3][r] = v.w;
  }

  // Visible keys of this thread's rows: klo[i] <= key <= lim[i] (causal, and
  // the sliding window when window > 0).
  int lim[4], klo[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    lim[i] = ctx + t0 + min((ty * 4 + i) / G, ntok - 1);
    klo[i] = EXT && window > 0 ? lim[i] - window + 1 : 0;
  }
  const int key_begin =
      EXT && window > 0 ? max(0, ctx + t0 - window + 1) / BN * BN : 0;
  const int lo_max = EXT && window > 0 ? ctx + t0 + ntok - window : 0;

  float acc[4][TN];
  #pragma unroll
  for (int i = 0; i < 4; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
  float m_i[4], l_i[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // A sink adds exp(sink) to the row's normalizer: start the running max
    // there with one unit of mass (on one lane; the row sums are per lane).
    const int r = ty * 4 + i;
    m_i[i] = EXT && sinks ? sinks[kvh * G + r % G] * LOG2E : -INFINITY;
    l_i[i] = (EXT && sinks && tx == 0) ? 1.f : 0.f;
  }

  auto k_off = [&](int key) -> long {
    return bt[key / block_size] * k_sb + key % block_size * k_st + kvh * k_sh;
  };
  auto v_off = [&](int key) -> long {
    return bt[key / block_size] * v_sb + key % block_size * v_st + kvh * v_sh;
  };

  long krow[K_ITEMS];
  KvRaw<KV> kr[K_ITEMS];
  auto kaddr = [&](int kt) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int key = kt + (tid + it * THREADS) / KL;
      krow[it] = key < klen ? k_off(key) : -1;
    }
  };
  auto kload = [&](int dc) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int part = (tid + it * THREADS) % KL;
      kr[it] = krow[it] >= 0 ? kv_load<KV>(Kc, krow[it] + dc * DC + part * 8)
                             : KvRaw<KV>{};
    }
  };
  auto kstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int key = idx / KL, part = idx % KL;
      uint32_t* d = &sU[buf][part * 4 * BN + (key ^ (part * (32 / KL)))];
      const uint4 kv = kv_widen<KV>(kr[it]);
      d[0] = kv.x;
      d[BN] = kv.y;
      d[2 * BN] = kv.z;
      d[3 * BN] = kv.w;
    }
  };
  KvRaw<KV> va[V_ITEMS], vb[V_ITEMS];
  auto vload = [&](int kt, int vc) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const int k0 = kt + vc * VC + kp * 2;
      va[it] = idx < V_TOTAL && k0 < klen ? kv_load<KV>(Vc, v_off(k0) + c * 8)
                                          : KvRaw<KV>{};
      vb[it] = idx < V_TOTAL && k0 + 1 < klen
                   ? kv_load<KV>(Vc, v_off(k0 + 1) + c * 8)
                   : KvRaw<KV>{};
    }
  };
  auto vstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx >= V_TOTAL) continue;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const uint4 av = kv_widen<KV>(va[it]), bv = kv_widen<KV>(vb[it]);
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&av);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&bv);
      uint32_t w[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        w[2 * e] = (a[e] & 0xFFFFu) | (b[e] << 16);
        w[2 * e + 1] = (a[e] >> 16) | (b[e] & 0xFFFF0000u);
      }
      uint32_t* d = &sU[buf][kp * D + c * 8];
      *reinterpret_cast<uint4*>(d) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(d + 4) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  };

  kaddr(key_begin);
  kload(0);
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
        const int sw = ((k2 >> 2) % KL) * (32 / KL);
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

    // Online softmax on this thread's 4 rows x 4 keys.
    const bool need_mask = kt + BN - 1 > ctx + t0 || (EXT && kt < lo_max);
    float alpha[4];
    uint32_t pw[4][2];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float sv[4];
      float mx = -INFINITY;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        if (EXT &&
            softcap > 0.f) {  // softcap * tanh(s / softcap), overflow-safe
          const float y = sacc[i][j] * qk_scale / softcap;
          sv[j] = softcap * (1.f - 2.f / (__expf(2.f * y) + 1.f)) * LOG2E;
        } else {
          sv[j] = sacc[i][j] * scale_log2e;
        }
        const int key = kt + tx * 4 + j;
        if (need_mask && (key > lim[i] || (EXT && key < klo[i])))
          sv[j] = -INFINITY;
        mx = fmaxf(mx, sv[j]);
      }
      mx = row16_max(mx);
      // A row with every key of the tile masked so far keeps m = -inf.
      const float m_new =
          EXT ? fmaxf(fmaxf(m_i[i], mx), -1e30f) : fmaxf(m_i[i], mx);
      alpha[i] = exp2f(m_i[i] - m_new);
      m_i[i] = m_new;
      float p[4], ps = 0.f;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        p[j] = exp2f(sv[j] - m_new);
        ps += p[j];
      }
      l_i[i] = l_i[i] * alpha[i] + ps;
      const half2 h0 = __floats2half2_rn(p[0], p[1]);
      const half2 h1 = __floats2half2_rn(p[2], p[3]);
      pw[i][0] = *reinterpret_cast<const uint32_t*>(&h0);
      pw[i][1] = *reinterpret_cast<const uint32_t*>(&h1);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] *= alpha[i];
    // P -> sP[key2][row]; 16-byte row chunks XOR-swizzled by key2 / 2 (= tx).
    *reinterpret_cast<uint4*>(&sP[tx * 2][(ty ^ tx) * 4]) =
        make_uint4(pw[0][0], pw[1][0], pw[2][0], pw[3][0]);
    *reinterpret_cast<uint4*>(&sP[tx * 2 + 1][(ty ^ tx) * 4]) =
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
  #pragma unroll 2
      for (int k2 = 0; k2 < VC / 2; k2++) {
        const int pk = vc * (VC / 2) + k2;
        const uint4 pa = *reinterpret_cast<const uint4*>(
            &sP[pk][(ty ^ ((pk >> 1) & 15)) * 4]);
        const half2 a[4] = {as_h2(pa.x), as_h2(pa.y), as_h2(pa.z), as_h2(pa.w)};
  #pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const uint4 vv = *reinterpret_cast<const uint4*>(
              &sU[cur][k2 * D + j * 16 + tx * 4]);
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
    const int r = ty * 4 + i;
    const int tq = r / G, g = r % G;
    const float l = row16_sum(l_i[i]);
    if (tq >= ntok) continue;
    const float inv = kv_scale<KV>(v_scale) / l;
    __half* o = O + (q0 + t0 + tq) * o_st + (kvh * G + g) * o_sh;
  #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const half2 h0 = __floats2half2_rn(acc[i][j] * inv, acc[i][j + 1] * inv);
      const half2 h1 =
          __floats2half2_rn(acc[i][j + 2] * inv, acc[i][j + 3] * inv);
      *reinterpret_cast<uint2*>(o + j * 16 + tx * 4) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                     *reinterpret_cast<const uint32_t*>(&h1));
    }
  }
}

// Split-KV decode / spec-verify attention: 16 rows (query token, q-head) per
// workgroup, the prefill kernel's 64-key tiles (S: 1 row x 4 keys per thread,
// PV: 1 row x W/16 columns), segment partials for reduce_segments. Heads
// narrower than W = 256 are packed: a workgroup takes PACK = W / D adjacent
// kv heads with 16 / PACK rows each, every row's query is zero outside its
// head's D-wide slice of the W-wide tiles, and each row writes only that
// slice of its output (so low GQA ratios still fill the 16 rows). EXT: sliding
// window / softcap / sinks as in the prefill kernel; segments wholly before the
// window write empty partials, segment 0 carries the sink.
template <int D, int DC, int KV, bool EXT>
__global__ void __launch_bounds__(THREADS) decode_attention_rdna2_kernel(
    const __half* __restrict__ Q, const void* __restrict__ Kc,
    const void* __restrict__ Vc, float* __restrict__ segm_out,
    float* __restrict__ segm_max, float* __restrict__ segm_sum,
    const int* __restrict__ cu_q, const int* __restrict__ seqused_k,
    const int* __restrict__ block_table, const int G, const int HQ,
    const int block_size, const long bt_stride, const long q_st,
    const long q_sh, const long k_sb, const long k_st, const long k_sh,
    const long v_sb, const long v_st, const long v_sh, const int tile,
    const int nseg, const int dpad, const int rgroups, const float scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int window, const float softcap, const float* __restrict__ sinks) {
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr int RB = 16;  // rows per workgroup
  constexpr int W = 256;  // tile width: PACK heads of D
  constexpr int PACK = W / D;
  constexpr int RH = RB / PACK;  // rows per head
  constexpr int W2 = W / 2;
  constexpr int TN = W / 16;
  constexpr int Q_ITEMS = (RB * (W / 8) + THREADS - 1) / THREADS;
  constexpr int KS = W / DC;
  constexpr int KL = DC / 8;
  constexpr int K_ITEMS = BN * KL / THREADS;
  constexpr int K_STAGE = (DC / 2) * BN;
  constexpr int VS = BN / VC;
  constexpr int V_TOTAL = (VC / 2) * (W / 8);
  constexpr int V_ITEMS = (V_TOTAL + THREADS - 1) / THREADS;
  constexpr int V_STAGE = (VC / 2) * W;
  static_assert(D % DC == 0 && PACK * D == W, "head packing");
  const int HKV = HQ / G;
  constexpr int U_STAGE = K_STAGE > V_STAGE ? K_STAGE : V_STAGE;
  constexpr int PLD =
      RB + 1;  // sP row stride: conflict-free transposing stores
  const float qk_scale = scale * kv_scale<KV>(k_scale);
  const float scale_log2e = qk_scale * LOG2E;
  __shared__ __align__(16) uint32_t sQ[W2][RB];
  __shared__ __align__(16) uint32_t sU[2][U_STAGE];
  __shared__ uint32_t sP[(BN / 2) * PLD];

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int s = blockIdx.x / rgroups, rg = blockIdx.x % rgroups;
  const int kvh = blockIdx.y * PACK, seg = blockIdx.z;  // first packed head
  const int q0 = cu_q[s], qlen = cu_q[s + 1] - q0;
  const int klen = seqused_k[s];
  const int ctx = klen - qlen;
  const int tps = (klen + nseg * tile - 1) / (nseg * tile);
  const int k0 = seg * tps * tile;
  if (k0 >= klen || qlen <= 0 || rg * RH >= qlen * G) return;
  const int k1 = min(k0 + tps * tile, klen);
  const int* bt = block_table + s * bt_stride;

  // Row r: head kvh + r / RH, query row rg * RH + r % RH of that head.
  #pragma unroll
  for (int it = 0; it < Q_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    if (idx >= RB * (W / 8)) break;
    const int r = idx % RB, c = idx / RB;
    const int hl = r / RH, row = rg * RH + r % RH, tq = row / G, g = row % G;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tq < qlen && c * 8 / D == hl && kvh + hl < HKV)
      v = *reinterpret_cast<const uint4*>(
          Q + (q0 + tq) * q_st + ((kvh + hl) * G + g) * q_sh + c * 8 % D);
    sQ[c * 4 + 0][r] = v.x;
    sQ[c * 4 + 1][r] = v.y;
    sQ[c * 4 + 2][r] = v.z;
    sQ[c * 4 + 3][r] = v.w;
  }
  const int my_hl = ty / RH, my_row = rg * RH + ty % RH;
  const int my_tq = my_row / G, my_g = my_row % G;
  const bool my_ok = my_tq < qlen && kvh + my_hl < HKV;
  const int lim = my_ok ? ctx + my_tq : -1;                  // key <= lim
  const int klo = EXT && window > 0 ? lim - window + 1 : 0;  // key >= klo
  // Tiles before the window of the workgroup's first query row are skipped.
  const int kt0 =
      EXT && window > 0 ? max(k0, max(0, ctx - window + 1) / BN * BN) : k0;

  float acc[TN];
  #pragma unroll
  for (int j = 0; j < TN; j++) acc[j] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;
  if (EXT && sinks && seg == 0 && my_ok) {
    // The sink's exp joins the normalizer once: one unit of mass on one lane.
    m_i = sinks[(kvh + my_hl) * G + my_g] * LOG2E;
    l_i = tx == 0 ? 1.f : 0.f;
  }

  auto k_off = [&](int key) -> long {
    return bt[key / block_size] * k_sb + key % block_size * k_st + kvh * k_sh;
  };
  auto v_off = [&](int key) -> long {
    return bt[key / block_size] * v_sb + key % block_size * v_st + kvh * v_sh;
  };
  long krow[K_ITEMS];
  KvRaw<KV> kr[K_ITEMS];
  auto kaddr = [&](int kt) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int key = kt + (tid + it * THREADS) / KL;
      krow[it] = key < k1 ? k_off(key) : -1;
    }
  };
  auto kload = [&](int dc) {
    // Stage dc covers dims dc * DC .. of packed head dc * DC / D.
    const int hl = dc * DC / D;
    const long hoff = hl * k_sh + dc * DC % D;
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int part = (tid + it * THREADS) % KL;
      kr[it] = krow[it] >= 0 && kvh + hl < HKV
                   ? kv_load<KV>(Kc, krow[it] + hoff + part * 8)
                   : KvRaw<KV>{};
    }
  };
  auto kstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int key = idx / KL, part = idx % KL;
      uint32_t* d = &sU[buf][part * 4 * BN + (key ^ (part * (32 / KL)))];
      const uint4 kv = kv_widen<KV>(kr[it]);
      d[0] = kv.x;
      d[BN] = kv.y;
      d[2 * BN] = kv.z;
      d[3 * BN] = kv.w;
    }
  };
  KvRaw<KV> va[V_ITEMS], vb[V_ITEMS];
  auto vload = [&](int kt, int vc) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int kp = idx / (W / 8), c = idx % (W / 8);
      const int key0 = kt + vc * VC + kp * 2;
      const int hl = c * 8 / D;
      const long hoff = hl * v_sh + c * 8 % D;
      const bool ok = idx < V_TOTAL && kvh + hl < HKV;
      va[it] =
          ok && key0 < k1 ? kv_load<KV>(Vc, v_off(key0) + hoff) : KvRaw<KV>{};
      vb[it] = ok && key0 + 1 < k1 ? kv_load<KV>(Vc, v_off(key0 + 1) + hoff)
                                   : KvRaw<KV>{};
    }
  };
  auto vstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx >= V_TOTAL) continue;
      const int kp = idx / (W / 8), c = idx % (W / 8);
      const uint4 av = kv_widen<KV>(va[it]), bv = kv_widen<KV>(vb[it]);
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&av);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&bv);
      uint32_t w[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        w[2 * e] = (a[e] & 0xFFFFu) | (b[e] << 16);
        w[2 * e + 1] = (a[e] >> 16) | (b[e] & 0xFFFF0000u);
      }
      uint32_t* d = &sU[buf][kp * W + c * 8];
      *reinterpret_cast<uint4*>(d) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(d + 4) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  };

  kaddr(kt0);
  kload(0);
  for (int kt = kt0; kt < k1; kt += BN) {
    kstore(0);
    __syncthreads();
    float sacc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int dc = 0; dc < KS; dc++) {
      const int cur = dc & 1;
      if (dc + 1 < KS)
        kload(dc + 1);
      else
        vload(kt, 0);
  #pragma unroll 8
      for (int k2 = 0; k2 < DC / 2; k2++) {
        const int sw = ((k2 >> 2) % KL) * (32 / KL);
        const half2 a = as_h2(sQ[dc * (DC / 2) + k2][ty]);
        const uint4 kb = *reinterpret_cast<const uint4*>(
            &sU[cur][k2 * BN + ((tx * 4) ^ sw)]);
        sacc[0] = __builtin_amdgcn_fdot2(a, as_h2(kb.x), sacc[0], false);
        sacc[1] = __builtin_amdgcn_fdot2(a, as_h2(kb.y), sacc[1], false);
        sacc[2] = __builtin_amdgcn_fdot2(a, as_h2(kb.z), sacc[2], false);
        sacc[3] = __builtin_amdgcn_fdot2(a, as_h2(kb.w), sacc[3], false);
      }
      if (dc + 1 < KS) {
        kstore(cur ^ 1);
        __syncthreads();
      }
    }
    float sv[4], mx = -INFINITY;
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int key = kt + tx * 4 + j;
      float x = sacc[j] * scale_log2e;
      if (EXT && softcap > 0.f) {  // softcap * tanh(s / softcap), overflow-safe
        const float y = sacc[j] * qk_scale / softcap;
        x = softcap * (1.f - 2.f / (__expf(2.f * y) + 1.f)) * LOG2E;
      }
      sv[j] = key < k1 && key <= lim && key >= klo ? x : -INFINITY;
      mx = fmaxf(mx, sv[j]);
    }
    mx = row16_max(mx);
    const float m_new = fmaxf(fmaxf(m_i, mx), -1e30f);
    const float alpha = exp2f(m_i - m_new);
    m_i = m_new;
    float p[4], ps = 0.f;
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      p[j] = exp2f(sv[j] - m_new);
      ps += p[j];
    }
    l_i = l_i * alpha + ps;
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[j] *= alpha;
    {
      const half2 h0 = __floats2half2_rn(p[0], p[1]);
      const half2 h1 = __floats2half2_rn(p[2], p[3]);
      sP[(tx * 2) * PLD + ty] = *reinterpret_cast<const uint32_t*>(&h0);
      sP[(tx * 2 + 1) * PLD + ty] = *reinterpret_cast<const uint32_t*>(&h1);
    }
    vstore(0);
    __syncthreads();
    for (int vc = 0; vc < VS; vc++) {
      const int cur = vc & 1;
      if (vc + 1 < VS) {
        vload(kt, vc + 1);
      } else if (kt + BN < k1) {
        kaddr(kt + BN);
        kload(0);
      }
  #pragma unroll
      for (int k2 = 0; k2 < VC / 2; k2++) {
        const half2 a = as_h2(sP[(vc * (VC / 2) + k2) * PLD + ty]);
  #pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const uint4 vv = *reinterpret_cast<const uint4*>(
              &sU[cur][k2 * W + j * 16 + tx * 4]);
          acc[j] = __builtin_amdgcn_fdot2(a, as_h2(vv.x), acc[j], false);
          acc[j + 1] =
              __builtin_amdgcn_fdot2(a, as_h2(vv.y), acc[j + 1], false);
          acc[j + 2] =
              __builtin_amdgcn_fdot2(a, as_h2(vv.z), acc[j + 2], false);
          acc[j + 3] =
              __builtin_amdgcn_fdot2(a, as_h2(vv.w), acc[j + 3], false);
        }
      }
      if (vc + 1 < VS) {
        vstore(cur ^ 1);
        __syncthreads();
      }
    }
  }
  const float l = row16_sum(l_i);
  if (!my_ok) return;
  const long base =
      ((long)(q0 + my_tq) * HQ + (kvh + my_hl) * G + my_g) * nseg + seg;
  float* o = segm_out + base * dpad - my_hl * D;  // dims of the row's head
  const float vs = kv_scale<KV>(v_scale);
  #pragma unroll
  for (int j = 0; j < TN; j += 4) {
    const int col = j * 16 + tx * 4;
    if (col / D == my_hl)
      *reinterpret_cast<float4*>(o + col) = make_float4(
          acc[j] * vs, acc[j + 1] * vs, acc[j + 2] * vs, acc[j + 3] * vs);
  }
  if (tx == 0) {
    segm_max[base] = m_i / LOG2E;
    segm_sum[base] = l;
  }
}

// QSA (Qwen4Exp sparse attention) decode / spec-verify, head 256: every query
// token attends to its own selection, so a workgroup takes one token's G <= 16
// heads of one kv head over 32-entry tiles of the selection. Latency first:
// 32 threads resolve a tile's selection -> block table -> cache offsets and
// every thread issues all of its K and V loads at once (a wave reads whole
// rows). While a tile is computed, the next tile's K / V, the block table
// entries of the one after and the selection entries of the third are in
// flight, issued earliest-needed first (loads complete in order).
// S = Q K^T: the two halves of the workgroup take the two halves of the head
// dims, a thread one key (lane) and HS = ceil(G / 4) heads (wave % 4), K in
// LDS as swizzled [dims / 8][key] 16-byte words; then one wave per head sums
// the halves and runs the online softmax in-wave. P V: a thread owns 2 dims
// of 2 * HS heads, so each V read serves them all. Q rows past G are zero
// (the loops have no head bounds). Workgroup
// (t, kvh, seg) runs tiles [seg * tps, (seg + 1) * tps) and writes the
// segment partials of decode_attention_rdna2 (unnormalized out, natural-log
// max, sum).
static constexpr int QD_TK = 32;    // selection entries per tile
static constexpr int QD_HMAX = 16;  // q heads per kv head
static constexpr int QD_MAXSEG = 256;

template <int KV, int HS>
__global__ void __launch_bounds__(THREADS) qsa_decode_rdna2_kernel(
    const __half* __restrict__ Q, const void* __restrict__ Kc,
    const void* __restrict__ Vc, float* __restrict__ segm_out,
    float* __restrict__ segm_max, float* __restrict__ segm_sum,
    const int* __restrict__ sel, const long sel_st, const int W,
    const int* __restrict__ tok2req, const int* __restrict__ block_table,
    const long bt_stride, const int bt_w, const int G, const int HQ,
    const int block_size, const long q_st, const long q_sh, const long k_sb,
    const long k_st, const long k_sh, const long v_sb, const long v_st,
    const long v_sh, const int nseg, const int tps, const float scale,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale) {
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr int D = 256, D8 = D / 8, TK = QD_TK;
  constexpr int ITEMS = TK * D8 / THREADS;  // rows (keys) per thread
  constexpr int VST = D / 2 + 4;  // sV row stride: conflict-free b128 stores
  constexpr int HV = 2 * HS, HP = 4 * HS;  // heads per P V thread, padded G
  static_assert(D8 == 32 && TK == 32, "a wave covers one row; lane = key");
  static_assert(HP <= QD_HMAX, "at most 16 q heads per kv head");
  __shared__ uint4 sQ[QD_HMAX][D8];
  __shared__ uint4 sK[D8][TK];  // key k of piece c at [c][k ^ c % 8]
  __shared__ __align__(16) uint32_t sV[TK][VST];
  __shared__ float sS[2][QD_HMAX][TK];
  __shared__ __align__(16) float sP[QD_HMAX][TK];
  __shared__ float sA[QD_HMAX];
  __shared__ long sOk[2][TK], sOv[2][TK];

  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int t = blockIdx.x, kvh = blockIdx.y, seg = blockIdx.z;
  const int tiles = (W + TK - 1) / TK;
  const int tile0 = seg * tps, tile1 = min(tile0 + tps, tiles);
  if (tile0 >= tiles) return;
  const int req = tok2req[t];
  const int* srow = sel + t * sel_st;
  const int* bt = block_table + (long)max(req, 0) * bt_stride;
  const float scale_log2e = scale * kv_scale<KV>(k_scale) * LOG2E;

  for (int i = tid; i < QD_HMAX * TK; i += THREADS) sP[i / TK][i % TK] = 0.f;
  for (int i = tid; i < HP * D8; i += THREADS)
    sQ[i / D8][i % D8] =
        i / D8 < G ? *reinterpret_cast<const uint4*>(
                         Q + t * q_st + (kvh * G + i / D8) * q_sh + i % D8 * 8)
                   : make_uint4(0, 0, 0, 0);

  // Threads < TK: selection entry tid of a tile, then its cache offsets.
  auto entry = [&](int tile) -> int {
    const int e = tile * TK + tid;
    return tid < TK && tile < tile1 && e < W ? srow[e] : -1;
  };
  long nk = -1, nv = -1;
  auto resolve = [&](int tok) {
    const bool ok = req >= 0 && tok >= 0 && tok / block_size < bt_w;
    const long page = ok ? bt[tok / block_size] : 0;
    nk = ok ? page * k_sb + tok % block_size * k_st + kvh * k_sh : -1;
    nv = ok ? page * v_sb + tok % block_size * v_st + kvh * v_sh : -1;
  };
  // Thread: piece `lane` (dims 8 * lane ..) of keys wave + 8 * it.
  KvRaw<KV> kr[ITEMS], vr[ITEMS];
  auto issue = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < ITEMS; it++) {
      const int key = wave + 8 * it;
      // Entries without a cache row read row 0 and are masked in S (their
      // P is 0, and V is zeroed so that 0 * garbage stays 0).
      const long ko = sOk[buf][key], vo = sOv[buf][key];
      kr[it] = kv_load<KV>(Kc, max(ko, 0L) + lane * 8);
      vr[it] = kv_load<KV>(Vc, max(vo, 0L) + lane * 8);
      if (vo < 0) vr[it] = KvRaw<KV>{};
    }
  };

  // S: dims half wave / 4 of heads (wave % 4) * hs ..; softmax: heads wave
  // and wave + 8; P V: dims 2 * (tid % 128) of heads (tid / 128) * hv ...
  const int sh = wave / 4, sh0 = wave % 4 * HS;
  const int dp = tid % 128, hv0 = tid / 128 * HV;
  float m_s[2] = {-INFINITY, -INFINITY}, l_s[2] = {0.f, 0.f};
  float acc[HV][2];
  #pragma unroll
  for (int r = 0; r < HV; r++) acc[r][0] = acc[r][1] = 0.f;

  int tok1 = entry(tile0 + 1), tok2 = entry(tile0 + 2);
  resolve(entry(tile0));
  if (tid < TK) {
    sOk[0][tid] = nk;
    sOv[0][tid] = nv;
  }
  __syncthreads();
  issue(0);
  resolve(tok1);
  for (int tile = tile0, buf = 0; tile < tile1; tile++, buf ^= 1) {
  #pragma unroll
    for (int it = 0; it < ITEMS; it++) {
      const int key = wave + 8 * it;
      sK[lane][key ^ lane % 8] = kv_widen<KV>(kr[it]);
      *reinterpret_cast<uint4*>(&sV[key][lane * 4]) = kv_widen<KV>(vr[it]);
    }
    if (tid < TK) {
      sOk[buf ^ 1][tid] = nk;
      sOv[buf ^ 1][tid] = nv;
    }
    __syncthreads();
    const int tok3 = entry(tile + 3);
    if (tile + 1 < tile1) issue(buf ^ 1);
    resolve(tok2);

    {
      float sc[HS];
  #pragma unroll
      for (int r = 0; r < HS; r++) sc[r] = 0.f;
  #pragma unroll 4
      for (int c = sh * (D8 / 2); c < (sh + 1) * (D8 / 2); c++) {
        const uint4 k = sK[c][lane ^ c % 8];
  #pragma unroll
        for (int r = 0; r < HS; r++) {
          const uint4 q = sQ[sh0 + r][c];
          sc[r] = __builtin_amdgcn_fdot2(as_h2(q.x), as_h2(k.x), sc[r], false);
          sc[r] = __builtin_amdgcn_fdot2(as_h2(q.y), as_h2(k.y), sc[r], false);
          sc[r] = __builtin_amdgcn_fdot2(as_h2(q.z), as_h2(k.z), sc[r], false);
          sc[r] = __builtin_amdgcn_fdot2(as_h2(q.w), as_h2(k.w), sc[r], false);
        }
      }
  #pragma unroll
      for (int r = 0; r < HS; r++) sS[sh][sh0 + r][lane] = sc[r];
    }
    __syncthreads();

  #pragma unroll
    for (int r = 0; r < 2; r++) {
      const int h = wave + 8 * r;
      if (h >= G) break;  // wave-uniform
      const float sc = sS[0][h][lane] + sS[1][h][lane];
      const float x = sOk[buf][lane] >= 0 ? sc * scale_log2e : -INFINITY;
      float mx = x;
  #pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
      const float m_new = fmaxf(fmaxf(m_s[r], mx), -1e30f);
      const float p = exp2f(x - m_new);
      float ps = p;
  #pragma unroll
      for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor(ps, o);
      const float alpha = exp2f(m_s[r] - m_new);
      l_s[r] = l_s[r] * alpha + ps;
      m_s[r] = m_new;
      sP[h][lane] = p;
      if (lane == 0) sA[h] = alpha;
    }
    __syncthreads();

  #pragma unroll
    for (int r = 0; r < HV; r++) {
      const float a = hv0 + r < G ? sA[hv0 + r] : 0.f;
      acc[r][0] *= a;
      acc[r][1] *= a;
    }
  #pragma unroll 2
    for (int j = 0; j < TK; j += 4) {
      float2 v[4];
  #pragma unroll
      for (int u = 0; u < 4; u++) v[u] = __half22float2(as_h2(sV[j + u][dp]));
  #pragma unroll
      for (int r = 0; r < HV; r++) {
        const float4 p = *reinterpret_cast<const float4*>(&sP[hv0 + r][j]);
        acc[r][0] = fmaf(p.x, v[0].x, acc[r][0]);
        acc[r][1] = fmaf(p.x, v[0].y, acc[r][1]);
        acc[r][0] = fmaf(p.y, v[1].x, acc[r][0]);
        acc[r][1] = fmaf(p.y, v[1].y, acc[r][1]);
        acc[r][0] = fmaf(p.z, v[2].x, acc[r][0]);
        acc[r][1] = fmaf(p.z, v[2].y, acc[r][1]);
        acc[r][0] = fmaf(p.w, v[3].x, acc[r][0]);
        acc[r][1] = fmaf(p.w, v[3].y, acc[r][1]);
      }
    }
    tok2 = tok3;  // here, so that tok3's load does not stall the compute
    __syncthreads();
  }

  const float vs = kv_scale<KV>(v_scale);
  #pragma unroll
  for (int r = 0; r < HV; r++) {
    if (hv0 + r >= G) break;
    const long base = ((long)t * HQ + kvh * G + hv0 + r) * nseg + seg;
    *reinterpret_cast<float2*>(segm_out + base * D + 2 * dp) =
        make_float2(acc[r][0] * vs, acc[r][1] * vs);
  }
  #pragma unroll
  for (int r = 0; r < 2; r++) {
    const int h = wave + 8 * r;
    if (h >= G || lane != 0) break;
    const long base = ((long)t * HQ + kvh * G + h) * nseg + seg;
    segm_max[base] = m_s[r] / LOG2E;
    segm_sum[base] = l_s[r];
  }
}

// QSA decode: the out [T, HQ, 256] (fp16) of the first act (<= QD_MAXSEG)
// segment partials of each (token, head) of qsa_decode_rdna2_kernel; weights
// in LDS, 4 dims per thread.
__global__ void __launch_bounds__(64)
    qsa_decode_merge_kernel(const float* __restrict__ segm_out,
                            const float* __restrict__ segm_max,
                            const float* __restrict__ segm_sum,
                            __half* __restrict__ out, const long o_st,
                            const long o_sh, const int HQ, const int nseg,
                            const int act) {
  __shared__ float sw[QD_MAXSEG];
  __shared__ float sred[2];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, d = tid * 4;
  const long base = ((long)t * HQ + h) * nseg;
  float m = -INFINITY;
  for (int sg = tid; sg < act; sg += 64) m = fmaxf(m, segm_max[base + sg]);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o));
  if (tid % 32 == 0) sred[tid / 32] = m;
  __syncthreads();
  m = fmaxf(sred[0], sred[1]);
  float l = 0.f;
  for (int sg = tid; sg < act; sg += 64) {
    const float w = __expf(segm_max[base + sg] - m);
    sw[sg] = w;
    l += w * segm_sum[base + sg];
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) l += __shfl_xor(l, o);
  __syncthreads();
  if (tid % 32 == 0) sred[tid / 32] = l;
  __syncthreads();
  l = sred[0] + sred[1];
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll 8
  for (int sg = 0; sg < act; sg++) {
    const float w = sw[sg];
    const float4 v =
        *reinterpret_cast<const float4*>(segm_out + (base + sg) * 256 + d);
    o[0] += w * v.x;
    o[1] += w * v.y;
    o[2] += w * v.z;
    o[3] += w * v.w;
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  const half2 h0 = __floats2half2_rn(o[0] * inv, o[1] * inv);
  const half2 h1 = __floats2half2_rn(o[2] * inv, o[3] * inv);
  *reinterpret_cast<uint2*>(out + t * o_st + h * o_sh + d) =
      make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                 *reinterpret_cast<const uint32_t*>(&h1));
}

// QSA (Qwen4Exp sparse attention) prefill. Every query token attends to its own
// selection: whole RATIO-token blocks plus the open group's causal tail, all
// at or before its position. Consecutive tokens select mostly the same blocks
// (the union of 5 is ~1.5x one selection), so tiles of BQ = 64 / G tokens run
// the prefill kernel above over the union of their selections, each row
// masked to its own blocks and position.
//
// qsa_union_rdna2_kernel: per tile, the ascending union of the blocks its
// tokens selected and, per block, the mask of those tokens (bit r = tile token
// r). Bitmaps of QU_CW words per token cover 32 * QU_CW blocks at a time.
static constexpr int QU_MAXBQ = 16;
static constexpr int QU_CW = 2 * THREADS;

__global__ void __launch_bounds__(THREADS)
    qsa_union_rdna2_kernel(const int* __restrict__ idx, const long idx_st,
                           const int width, const int* __restrict__ cu_q,
                           const int* __restrict__ seq_lens, const int num_seqs,
                           const int BQ, const int ratio,
                           int* __restrict__ ublk, uint16_t* __restrict__ umask,
                           int* __restrict__ ucount, const int cap) {
  __shared__ uint32_t bm[QU_MAXBQ][QU_CW];
  __shared__ int scan[THREADS];
  const int tid = threadIdx.x, bid = blockIdx.x;
  int lo = 0, hi = num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (cu_q[mid] / BQ + mid <= bid)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int q0 = cu_q[lo], qlen = cu_q[lo + 1] - q0;
  const int t0 = (bid - (q0 / BQ + lo)) * BQ;
  if (t0 >= qlen) {
    if (tid == 0) ucount[bid] = 0;
    return;
  }
  const int ntok = min(BQ, qlen - t0);
  const int nblk = (seq_lens[lo] - qlen + t0 + ntok - 1) / ratio + 1;
  int* out_b = ublk + (long)bid * cap;
  uint16_t* out_m = umask + (long)bid * cap;
  int total = 0;
  for (int c0 = 0; c0 < nblk; c0 += 32 * QU_CW) {
    for (int i = tid; i < ntok * QU_CW; i += THREADS)
      bm[i / QU_CW][i % QU_CW] = 0;
    __syncthreads();
    for (int r = 0; r < ntok; r++) {
      const int* row = idx + (long)(q0 + t0 + r) * idx_st;
      for (int j = tid; j < width; j += THREADS) {
        const int t = row[j];
        const int b = t / ratio - c0;
        if (t >= 0 && b >= 0 && b < 32 * QU_CW)
          atomicOr(&bm[r][b >> 5], 1u << (b & 31));
      }
    }
    __syncthreads();
    uint32_t u[2];
    int cnt = 0;
  #pragma unroll
    for (int k = 0; k < 2; k++) {
      uint32_t x = 0;
      for (int r = 0; r < ntok; r++) x |= bm[r][tid * 2 + k];
      u[k] = x;
      cnt += __popc(x);
    }
    scan[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < THREADS; off <<= 1) {
      const int v = tid >= off ? scan[tid - off] : 0;
      __syncthreads();
      scan[tid] += v;
      __syncthreads();
    }
    int pos = total + scan[tid] - cnt;
  #pragma unroll
    for (int k = 0; k < 2; k++) {
      const int w = tid * 2 + k;
      for (uint32_t x = u[k]; x; x &= x - 1) {
        const int bit = __ffs(x) - 1;
        uint32_t m = 0;
        for (int r = 0; r < ntok; r++) m |= ((bm[r][w] >> bit) & 1u) << r;
        if (pos < cap) {
          out_b[pos] = c0 + w * 32 + bit;
          out_m[pos] = (uint16_t)m;
        }
        pos++;
      }
    }
    total += scan[THREADS - 1];
    __syncthreads();
  }
  if (tid == 0) ucount[bid] = min(total, cap);
}

// qsa_attention_rdna2_kernel: unified_attention_rdna2_kernel over a tile's
// union blocks (key k of the tile is token ublk[k / ratio] * ratio + k %
// ratio). A row sees a key when its token selected the key's block and the key
// is at or before its position; rows can see no key of a whole 64-key tile,
// so the running max is clamped as on the sliding-window path.
template <int D, int DC, int KV>
__global__ void __launch_bounds__(THREADS) qsa_attention_rdna2_kernel(
    const __half* __restrict__ Q, const void* __restrict__ Kc,
    const void* __restrict__ Vc, __half* __restrict__ O,
    const int* __restrict__ cu_q, const int* __restrict__ seq_lens,
    const int* __restrict__ block_table, const int* __restrict__ ublk,
    const uint16_t* __restrict__ umask, const int* __restrict__ ucount,
    const int cap, const int ratio, const int num_seqs, const int G,
    const int BQ, const int block_size, const long bt_stride, const long q_st,
    const long q_sh, const long k_sb, const long k_st, const long k_sh,
    const long v_sb, const long v_st, const long v_sh, const long o_st,
    const long o_sh, const float scale, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale) {
  constexpr float LOG2E = 1.4426950408889634f;
  const float scale_log2e = scale * kv_scale<KV>(k_scale) * LOG2E;
  constexpr int D2 = D / 2;
  constexpr int TN = D / 16;
  constexpr int Q_ITEMS = BM * (D / 8) / THREADS;
  constexpr int KS = D / DC;
  constexpr int KL = DC / 8;
  constexpr int K_ITEMS = BN * KL / THREADS;
  constexpr int K_STAGE = (DC / 2) * BN;
  constexpr int VS = BN / VC;
  constexpr int V_TOTAL = (VC / 2) * (D / 8);
  constexpr int V_ITEMS = (V_TOTAL + THREADS - 1) / THREADS;
  constexpr int V_STAGE = (VC / 2) * D;
  constexpr int U_STAGE = K_STAGE > V_STAGE ? K_STAGE : V_STAGE;
  static_assert(KS % 2 == 0 && K_ITEMS * THREADS == BN * KL, "K staging");
  __shared__ __align__(16) uint32_t sQ[D2][BM];
  __shared__ __align__(16) uint32_t sU[2][U_STAGE];
  __shared__ __align__(16) uint32_t sP[BN / 2][BM];

  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int kvh = blockIdx.y;
  const int bid = gridDim.x - 1 - blockIdx.x;  // later (longer) tiles first
  int lo = 0, hi = num_seqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (cu_q[mid] / BQ + mid <= bid)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int s = lo;
  const int q0 = cu_q[s], qlen = cu_q[s + 1] - q0;
  const int t0 = (bid - (q0 / BQ + s)) * BQ;
  if (t0 >= qlen) return;
  const int ctx = seq_lens[s] - qlen;
  const int ntok = min(BQ, qlen - t0);
  const int key_end = ucount[bid] * ratio;
  const int* bt = block_table + s * bt_stride;
  const int* ub = ublk + (long)bid * cap;
  const uint16_t* um = umask + (long)bid * cap;

  #pragma unroll
  for (int it = 0; it < Q_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    const int r = idx % BM, c = idx / BM;
    const int tq = r / G, g = r % G;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (tq < ntok)
      v = *reinterpret_cast<const uint4*>(Q + (q0 + t0 + tq) * q_st +
                                          (kvh * G + g) * q_sh + c * 8);
    sQ[c * 4 + 0][r] = v.x;
    sQ[c * 4 + 1][r] = v.y;
    sQ[c * 4 + 2][r] = v.z;
    sQ[c * 4 + 3][r] = v.w;
  }

  // This thread's rows: tile token rt[i] at position lim[i].
  int lim[4], rt[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    rt[i] = min((ty * 4 + i) / G, ntok - 1);
    lim[i] = ctx + t0 + rt[i];
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

  auto token = [&](int key) { return ub[key / ratio] * ratio + key % ratio; };
  auto k_off = [&](int t) -> long {
    return bt[t / block_size] * k_sb + t % block_size * k_st + kvh * k_sh;
  };
  auto v_off = [&](int t) -> long {
    return bt[t / block_size] * v_sb + t % block_size * v_st + kvh * v_sh;
  };

  long krow[K_ITEMS];
  KvRaw<KV> kr[K_ITEMS];
  auto kaddr = [&](int kt) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int key = kt + (tid + it * THREADS) / KL;
      krow[it] = key < key_end ? k_off(token(key)) : -1;
    }
  };
  auto kload = [&](int dc) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int part = (tid + it * THREADS) % KL;
      kr[it] = krow[it] >= 0 ? kv_load<KV>(Kc, krow[it] + dc * DC + part * 8)
                             : KvRaw<KV>{};
    }
  };
  auto kstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int key = idx / KL, part = idx % KL;
      uint32_t* d = &sU[buf][part * 4 * BN + (key ^ (part * (32 / KL)))];
      const uint4 kv = kv_widen<KV>(kr[it]);
      d[0] = kv.x;
      d[BN] = kv.y;
      d[2 * BN] = kv.z;
      d[3 * BN] = kv.w;
    }
  };
  KvRaw<KV> va[V_ITEMS], vb[V_ITEMS];
  auto vload = [&](int kt, int vc) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const int k0 = kt + vc * VC + kp * 2;
      va[it] = idx < V_TOTAL && k0 < key_end
                   ? kv_load<KV>(Vc, v_off(token(k0)) + c * 8)
                   : KvRaw<KV>{};
      vb[it] = idx < V_TOTAL && k0 + 1 < key_end
                   ? kv_load<KV>(Vc, v_off(token(k0 + 1)) + c * 8)
                   : KvRaw<KV>{};
    }
  };
  auto vstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx >= V_TOTAL) continue;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const uint4 av = kv_widen<KV>(va[it]), bv = kv_widen<KV>(vb[it]);
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&av);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&bv);
      uint32_t w[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        w[2 * e] = (a[e] & 0xFFFFu) | (b[e] << 16);
        w[2 * e + 1] = (a[e] >> 16) | (b[e] & 0xFFFF0000u);
      }
      uint32_t* d = &sU[buf][kp * D + c * 8];
      *reinterpret_cast<uint4*>(d) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(d + 4) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  };

  kaddr(0);
  kload(0);
  for (int kt = 0; kt < key_end; kt += BN) {
    // This thread's 4 keys: tokens and the mask of tile tokens that see them.
    int ktok[4];
    uint32_t kmask[4];
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int key = kt + tx * 4 + j;
      const bool ok = key < key_end;
      ktok[j] = ok ? token(key) : 0;
      kmask[j] = ok ? um[key / ratio] : 0u;
    }
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
        const int sw = ((k2 >> 2) % KL) * (32 / KL);
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

    float alpha[4];
    uint32_t pw[4][2];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float sv[4];
      float mx = -INFINITY;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        const bool seen = ((kmask[j] >> rt[i]) & 1u) && ktok[j] <= lim[i];
        sv[j] = seen ? sacc[i][j] * scale_log2e : -INFINITY;
        mx = fmaxf(mx, sv[j]);
      }
      mx = row16_max(mx);
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
      const half2 h0 = __floats2half2_rn(p[0], p[1]);
      const half2 h1 = __floats2half2_rn(p[2], p[3]);
      pw[i][0] = *reinterpret_cast<const uint32_t*>(&h0);
      pw[i][1] = *reinterpret_cast<const uint32_t*>(&h1);
    }
  #pragma unroll
    for (int i = 0; i < 4; i++)
  #pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] *= alpha[i];
    *reinterpret_cast<uint4*>(&sP[tx * 2][(ty ^ tx) * 4]) =
        make_uint4(pw[0][0], pw[1][0], pw[2][0], pw[3][0]);
    *reinterpret_cast<uint4*>(&sP[tx * 2 + 1][(ty ^ tx) * 4]) =
        make_uint4(pw[0][1], pw[1][1], pw[2][1], pw[3][1]);
    vstore(0);
    __syncthreads();

    for (int vc = 0; vc < VS; vc++) {
      const int cur = vc & 1;
      if (vc + 1 < VS) {
        vload(kt, vc + 1);
      } else if (kt + BN < key_end) {
        kaddr(kt + BN);
        kload(0);
      }
  #pragma unroll 2
      for (int k2 = 0; k2 < VC / 2; k2++) {
        const int pk = vc * (VC / 2) + k2;
        const uint4 pa = *reinterpret_cast<const uint4*>(
            &sP[pk][(ty ^ ((pk >> 1) & 15)) * 4]);
        const half2 a[4] = {as_h2(pa.x), as_h2(pa.y), as_h2(pa.z), as_h2(pa.w)};
  #pragma unroll
        for (int j = 0; j < TN; j += 4) {
          const uint4 vv = *reinterpret_cast<const uint4*>(
              &sU[cur][k2 * D + j * 16 + tx * 4]);
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
    const int r = ty * 4 + i;
    const int tq = r / G, g = r % G;
    const float l = row16_sum(l_i[i]);
    if (tq >= ntok) continue;
    const float inv = l > 0.f ? kv_scale<KV>(v_scale) / l : 0.f;
    __half* o = O + (q0 + t0 + tq) * o_st + (kvh * G + g) * o_sh;
  #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const half2 h0 = __floats2half2_rn(acc[i][j] * inv, acc[i][j + 1] * inv);
      const half2 h1 =
          __floats2half2_rn(acc[i][j + 2] * inv, acc[i][j + 3] * inv);
      *reinterpret_cast<uint2*>(o + j * 16 + tx * 4) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&h0),
                     *reinterpret_cast<const uint32_t*>(&h1));
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

__global__ void qsa_union_rdna2_kernel(const int*, const long, const int,
                                       const int*, const int*, const int,
                                       const int, const int, int*, uint16_t*,
                                       int*, const int) {}

template <int D, int DC, int KV>
__global__ void qsa_attention_rdna2_kernel(
    const __half*, const void*, const void*, __half*, const int*, const int*,
    const int*, const int*, const uint16_t*, const int*, const int, const int,
    const int, const int, const int, const int, const long, const long,
    const long, const long, const long, const long, const long, const long,
    const long, const long, const long, const float, const float*,
    const float*) {}

template <int D, int DC, bool EXT, int KV>
__global__ void unified_attention_rdna2_kernel(
    const __half*, const void*, const void*, __half*, const int*, const int*,
    const int*, const int, const int, const int, const int, const long,
    const long, const long, const long, const long, const long, const long,
    const long, const long, const long, const long, const float, const int,
    const float, const float*, const float*, const float*) {}

template <int D, int DC, int KV, bool EXT>
__global__ void decode_attention_rdna2_kernel(
    const __half*, const void*, const void*, float*, float*, float*, const int*,
    const int*, const int*, const int, const int, const int, const long,
    const long, const long, const long, const long, const long, const long,
    const long, const long, const int, const int, const int, const int,
    const float, const float*, const float*, const int, const float,
    const float*) {}

template <int KV, int HS>
__global__ void qsa_decode_rdna2_kernel(
    const __half*, const void*, const void*, float*, float*, float*, const int*,
    const long, const int, const int*, const int*, const long, const int,
    const int, const int, const int, const long, const long, const long,
    const long, const long, const long, const long, const long, const int,
    const int, const float, const float*, const float*) {}

__global__ void qsa_decode_merge_kernel(const float*, const float*,
                                        const float*, __half*, const long,
                                        const long, const int, const int,
                                        const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace attention_rdna2
}  // namespace vllm

namespace {

// The caches' KvType; fp8 e4m3fn needs fp32 per-tensor k_scale / v_scale
// (only element 0 is read, so expanded views work).
int check_kv_dtype(const char* name, const torch::Tensor& k_cache,
                   const torch::Tensor& v_cache,
                   const std::optional<torch::Tensor>& k_scale,
                   const std::optional<torch::Tensor>& v_scale) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(k_cache.dtype() == v_cache.dtype(), name,
              " needs k_cache and v_cache of one dtype");
  if (k_cache.dtype() == torch::kFloat16) return KV_FP16;
  if (k_cache.dtype() == torch::kBFloat16) return KV_BF16;
  TORCH_CHECK(k_cache.dtype() == at::ScalarType::Float8_e4m3fn, name,
              " needs fp16, bf16 or float8_e4m3fn caches");
  TORCH_CHECK(k_scale && v_scale && k_scale->dtype() == torch::kFloat32 &&
                  v_scale->dtype() == torch::kFloat32,
              name, " needs fp32 k_scale and v_scale for fp8 caches");
  return KV_FP8;
}

const float* scale_ptr(const std::optional<torch::Tensor>& t) {
  return t ? t->data_ptr<float>() : nullptr;
}

}  // namespace

// Causal attention of q [num_tokens, num_q_heads, D] against the paged fp16
// (bf16, or fp8 e4m3fn dequantized with the per-tensor k_scale / v_scale)
// caches k_cache / v_cache [num_blocks, block_size, num_kv_heads, D] into out
// [num_tokens, num_q_heads, D]; D in {64, 128, 256}. Sequence s owns query
// tokens cu_seqlens_q[s] .. cu_seqlens_q[s + 1] - 1, which are the last ones
// of its seqused_k[s] keys, whose cache blocks are block_table[s]. window > 0
// limits each query to its last `window` keys (sliding window), softcap > 0
// applies softcap * tanh(score / softcap), sinks [num_q_heads] fp32 adds
// exp(sinks[h]) to each softmax normalizer.
void unified_attention_rdna2(torch::Tensor& out, const torch::Tensor& q,
                             const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache,
                             const torch::Tensor& cu_seqlens_q,
                             const torch::Tensor& seqused_k,
                             const torch::Tensor& block_table, double scale,
                             int64_t window, double softcap,
                             const std::optional<torch::Tensor>& sinks,
                             const std::optional<torch::Tensor>& k_scale,
                             const std::optional<torch::Tensor>& v_scale) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(q.dtype() == torch::kFloat16 && out.dtype() == torch::kFloat16,
              "unified_attention_rdna2 needs fp16 queries and output");
  const int kv = check_kv_dtype("unified_attention_rdna2", k_cache, v_cache,
                                k_scale, v_scale);
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                  out.sizes() == q.sizes(),
              "unified_attention_rdna2 needs q/out [tokens, heads, D] and "
              "caches [blocks, block_size, kv_heads, D]");
  const int D = q.size(2), HQ = q.size(1), HKV = k_cache.size(2);
  TORCH_CHECK(D == 64 || D == 128 || D == 256,
              "unified_attention_rdna2 supports head sizes 64, 128 and 256");
  TORCH_CHECK(k_cache.size(3) == D && v_cache.sizes() == k_cache.sizes() &&
                  HQ % HKV == 0 && HQ / HKV <= BM,
              "unified_attention_rdna2: cache shapes do not match q");
  for (const torch::Tensor* t :
       {&q, &k_cache, &v_cache, static_cast<const torch::Tensor*>(&out)}) {
    TORCH_CHECK(t->stride(-1) == 1 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "unified_attention_rdna2 needs 16-byte aligned, contiguous "
                "heads");
    for (int d = 0; d < t->dim() - 1; d++)
      TORCH_CHECK(t->stride(d) % 8 == 0,
                  "unified_attention_rdna2 needs strides that keep heads "
                  "16-byte aligned");
  }
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 &&
                  seqused_k.dtype() == torch::kInt32 &&
                  block_table.dtype() == torch::kInt32 &&
                  cu_seqlens_q.is_contiguous() && seqused_k.is_contiguous() &&
                  block_table.stride(1) == 1,
              "unified_attention_rdna2 needs int32 metadata with contiguous "
              "block-table rows");
  const int num_seqs = cu_seqlens_q.size(0) - 1;
  if (num_seqs <= 0 || q.size(0) == 0) return;

  const int G = HQ / HKV;
  const int BQ = BM / G;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Each sequence gets cu_seqlens_q[s + 1] / BQ - cu_seqlens_q[s] / BQ + 1 >=
  // ceil(query_len / BQ) tiles; the extra ones exit at once.
  const dim3 grid(q.size(0) / BQ + num_seqs, HKV);
  const float* sinks_ptr = nullptr;
  if (sinks) {
    TORCH_CHECK(sinks->dtype() == torch::kFloat32 && sinks->is_contiguous() &&
                    sinks->numel() == HQ,
                "unified_attention_rdna2: sinks must be fp32 [num_q_heads]");
    sinks_ptr = sinks->data_ptr<float>();
  }

#define VLLM_ATTN_RDNA2_LAUNCH(HD, DC, X, KV)                                  \
  unified_attention_rdna2_kernel<HD, DC, X, KV><<<grid, THREADS, 0, stream>>>( \
      (const __half*)q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),     \
      (__half*)out.data_ptr(), cu_seqlens_q.data_ptr<int>(),                   \
      seqused_k.data_ptr<int>(), block_table.data_ptr<int>(), num_seqs, G, BQ, \
      k_cache.size(1), block_table.stride(0), q.stride(0), q.stride(1),        \
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),                 \
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), out.stride(0),  \
      out.stride(1), (float)scale, (int)window, (float)softcap, sinks_ptr,     \
      scale_ptr(k_scale), scale_ptr(v_scale))
#define VLLM_ATTN_RDNA2_BY_KV(HD, DC, X)        \
  if (kv == KV_FP8) {                           \
    VLLM_ATTN_RDNA2_LAUNCH(HD, DC, X, KV_FP8);  \
  } else if (kv == KV_BF16) {                   \
    VLLM_ATTN_RDNA2_LAUNCH(HD, DC, X, KV_BF16); \
  } else {                                      \
    VLLM_ATTN_RDNA2_LAUNCH(HD, DC, X, KV_FP16); \
  }
#define VLLM_ATTN_RDNA2_BY_EXT(HD, DC)    \
  if (ext) {                              \
    VLLM_ATTN_RDNA2_BY_KV(HD, DC, true);  \
  } else {                                \
    VLLM_ATTN_RDNA2_BY_KV(HD, DC, false); \
  }
  const bool ext = window > 0 || softcap > 0 || sinks_ptr != nullptr;
  if (D == 256) {
    VLLM_ATTN_RDNA2_BY_EXT(256, 64);
  } else if (D == 128) {
    VLLM_ATTN_RDNA2_BY_EXT(128, 64);
  } else {
    VLLM_ATTN_RDNA2_BY_EXT(64, 32);
  }
#undef VLLM_ATTN_RDNA2_BY_EXT
#undef VLLM_ATTN_RDNA2_BY_KV
#undef VLLM_ATTN_RDNA2_LAUNCH
}

// Split-KV decode / spec-verify attention for head 64, 128 or 256: q
// [num_tokens, num_q_heads, D], paged caches (and scales) as for
// unified_attention_rdna2; writes the unnormalized per-segment outputs
// segm_out [>= num_tokens, num_q_heads, S, >= D] fp32 and their maxima
// (natural log) / exp sums segm_max, segm_sum [>= num_tokens, num_q_heads, S],
// segments of ceil(len / (S * tile)) * tile keys as in reduce_segments;
// window, softcap and sinks as for unified_attention_rdna2.
void decode_attention_rdna2(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& v_cache, torch::Tensor& segm_out,
    torch::Tensor& segm_max, torch::Tensor& segm_sum,
    const torch::Tensor& cu_seqlens_q, const torch::Tensor& seqused_k,
    const torch::Tensor& block_table, int64_t tile, int64_t max_seqlen_q,
    double scale, const std::optional<torch::Tensor>& k_scale,
    const std::optional<torch::Tensor>& v_scale, int64_t window, double softcap,
    const std::optional<torch::Tensor>& sinks) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(q.dtype() == torch::kFloat16 &&
                  segm_out.dtype() == torch::kFloat32 &&
                  segm_max.dtype() == torch::kFloat32 &&
                  segm_sum.dtype() == torch::kFloat32,
              "decode_attention_rdna2: fp16 q, fp32 segment scratch");
  const int kv = check_kv_dtype("decode_attention_rdna2", k_cache, v_cache,
                                k_scale, v_scale);
  const int HQ = q.size(1), D = q.size(2), HKV = k_cache.size(2);
  TORCH_CHECK((D == 64 || D == 128 || D == 256) && k_cache.size(3) == D &&
                  v_cache.sizes() == k_cache.sizes() && HQ % HKV == 0,
              "decode_attention_rdna2 supports head sizes 64, 128 and 256");
  for (const torch::Tensor* t : {&q, &k_cache, &v_cache}) {
    TORCH_CHECK(t->stride(-1) == 1 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "decode_attention_rdna2 needs 16-byte aligned heads");
    for (int d = 0; d < t->dim() - 1; d++)
      TORCH_CHECK(t->stride(d) % 8 == 0,
                  "decode_attention_rdna2 needs 16-byte aligned head strides");
  }
  const int nseg = segm_max.size(2);
  TORCH_CHECK(segm_out.is_contiguous() && segm_max.is_contiguous() &&
                  segm_sum.is_contiguous() && segm_out.size(1) == HQ &&
                  segm_out.size(2) == nseg && segm_out.size(3) >= D &&
                  segm_max.size(0) >= q.size(0),
              "decode_attention_rdna2: segment scratch shape");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 &&
                  seqused_k.dtype() == torch::kInt32 &&
                  block_table.dtype() == torch::kInt32 &&
                  block_table.stride(1) == 1,
              "decode_attention_rdna2: int32 metadata");
  const int num_seqs = cu_seqlens_q.size(0) - 1;
  if (num_seqs <= 0 || q.size(0) == 0) return;
  // Heads narrower than 256 are packed 256 / D per workgroup, 16 * D / 256
  // query rows each.
  const int G = HQ / HKV, pack = 256 / D, rows = 16 / pack;
  const int rgroups = (max_seqlen_q * G + rows - 1) / rows;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_seqs * rgroups, (HKV + pack - 1) / pack, nseg);
  const float* sinks_ptr = nullptr;
  if (sinks) {
    TORCH_CHECK(sinks->dtype() == torch::kFloat32 && sinks->is_contiguous() &&
                    sinks->numel() == HQ,
                "decode_attention_rdna2: sinks must be fp32 [num_q_heads]");
    sinks_ptr = sinks->data_ptr<float>();
  }
  const bool ext = window > 0 || softcap > 0 || sinks_ptr != nullptr;
  auto launch = [&](auto kernel) {
    kernel<<<grid, THREADS, 0, stream>>>(
        (const __half*)q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        segm_out.data_ptr<float>(), segm_max.data_ptr<float>(),
        segm_sum.data_ptr<float>(), cu_seqlens_q.data_ptr<int>(),
        seqused_k.data_ptr<int>(), block_table.data_ptr<int>(), G, HQ,
        k_cache.size(1), block_table.stride(0), q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), (int)tile,
        nseg, segm_out.size(3), rgroups, (float)scale, scale_ptr(k_scale),
        scale_ptr(v_scale), (int)window, (float)softcap, sinks_ptr);
  };
#define VLLM_DECODE_RDNA2_BY_EXT(HD, KV)                      \
  if (ext) {                                                  \
    launch(decode_attention_rdna2_kernel<HD, 64, KV, true>);  \
  } else {                                                    \
    launch(decode_attention_rdna2_kernel<HD, 64, KV, false>); \
  }
#define VLLM_DECODE_RDNA2_BY_KV(HD)        \
  if (kv == KV_FP8) {                      \
    VLLM_DECODE_RDNA2_BY_EXT(HD, KV_FP8);  \
  } else if (kv == KV_BF16) {              \
    VLLM_DECODE_RDNA2_BY_EXT(HD, KV_BF16); \
  } else {                                 \
    VLLM_DECODE_RDNA2_BY_EXT(HD, KV_FP16); \
  }
  if (D == 256) {
    VLLM_DECODE_RDNA2_BY_KV(256);
  } else if (D == 128) {
    VLLM_DECODE_RDNA2_BY_KV(128);
  } else {
    VLLM_DECODE_RDNA2_BY_KV(64);
  }
#undef VLLM_DECODE_RDNA2_BY_KV
#undef VLLM_DECODE_RDNA2_BY_EXT
}

// QSA (Qwen4Exp sparse attention) prefill, step 1: for tiles of tile_tokens
// consecutive query tokens of one sequence (cu_seqlens_q, seq_lens as for
// unified_attention_rdna2), the ascending union of the ratio-token blocks of
// the tokens' selections (indices [num_tokens, width] int32 request-relative
// tokens, -1 padded) into ublk [tiles, cap] int32 with the masks of the tile
// tokens selecting each block in umask [tiles, cap] int16 (bit r = token r)
// and the counts in ucount [tiles]; tiles >= num_tokens / tile_tokens +
// num_seqs, cap >= tile_tokens * (blocks one selection can touch).
void qsa_union_rdna2(const torch::Tensor& indices,
                     const torch::Tensor& cu_seqlens_q,
                     const torch::Tensor& seq_lens, int64_t tile_tokens,
                     int64_t ratio, torch::Tensor& ublk, torch::Tensor& umask,
                     torch::Tensor& ucount) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(indices.dtype() == torch::kInt32 && indices.dim() == 2 &&
                  indices.stride(1) == 1,
              "qsa_union_rdna2 needs int32 indices [tokens, width]");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 &&
                  seq_lens.dtype() == torch::kInt32 &&
                  cu_seqlens_q.is_contiguous() && seq_lens.is_contiguous(),
              "qsa_union_rdna2 needs contiguous int32 metadata");
  TORCH_CHECK(tile_tokens >= 1 && tile_tokens <= QU_MAXBQ && ratio >= 1,
              "qsa_union_rdna2: 1 <= tile_tokens <= 16, ratio >= 1");
  const int num_seqs = cu_seqlens_q.size(0) - 1;
  const int tiles = indices.size(0) / tile_tokens + num_seqs;
  TORCH_CHECK(ublk.dtype() == torch::kInt32 && umask.dtype() == torch::kInt16 &&
                  ucount.dtype() == torch::kInt32 && ublk.is_contiguous() &&
                  umask.is_contiguous() && ublk.sizes() == umask.sizes() &&
                  ublk.size(0) >= tiles && ucount.numel() >= tiles,
              "qsa_union_rdna2: output shapes");
  if (num_seqs <= 0 || indices.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(indices));
  qsa_union_rdna2_kernel<<<tiles, THREADS, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      indices.data_ptr<int>(), indices.stride(0), indices.size(1),
      cu_seqlens_q.data_ptr<int>(), seq_lens.data_ptr<int>(), num_seqs,
      tile_tokens, ratio, ublk.data_ptr<int>(),
      reinterpret_cast<uint16_t*>(umask.data_ptr()), ucount.data_ptr<int>(),
      ublk.size(1));
}

// QSA prefill, step 2: attention of fp16 q [num_tokens, num_q_heads, D] (D in
// 128, 256) over the qsa_union_rdna2 tiles, against caches laid out as for
// unified_attention_rdna2, into out (fp16, same shape as q).
void qsa_attention_rdna2(
    torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& v_cache, const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& seq_lens, const torch::Tensor& block_table,
    const torch::Tensor& ublk, const torch::Tensor& umask,
    const torch::Tensor& ucount, int64_t tile_tokens, int64_t ratio,
    double scale, const std::optional<torch::Tensor>& k_scale,
    const std::optional<torch::Tensor>& v_scale) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(q.dtype() == torch::kFloat16 && out.dtype() == torch::kFloat16,
              "qsa_attention_rdna2 needs fp16 queries and output");
  const int kv =
      check_kv_dtype("qsa_attention_rdna2", k_cache, v_cache, k_scale, v_scale);
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                  out.sizes() == q.sizes(),
              "qsa_attention_rdna2 needs q/out [tokens, heads, D] and caches "
              "[blocks, block_size, kv_heads, D]");
  const int D = q.size(2), HQ = q.size(1), HKV = k_cache.size(2);
  TORCH_CHECK((D == 128 || D == 256) && k_cache.size(3) == D &&
                  v_cache.sizes() == k_cache.sizes() && HQ % HKV == 0,
              "qsa_attention_rdna2 supports head sizes 128 and 256");
  for (const torch::Tensor* t :
       {&q, &k_cache, &v_cache, static_cast<const torch::Tensor*>(&out)}) {
    TORCH_CHECK(t->stride(-1) == 1 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "qsa_attention_rdna2 needs 16-byte aligned, contiguous heads");
    for (int d = 0; d < t->dim() - 1; d++)
      TORCH_CHECK(t->stride(d) % 8 == 0,
                  "qsa_attention_rdna2 needs strides that keep heads 16-byte "
                  "aligned");
  }
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 &&
                  seq_lens.dtype() == torch::kInt32 &&
                  block_table.dtype() == torch::kInt32 &&
                  cu_seqlens_q.is_contiguous() && seq_lens.is_contiguous() &&
                  block_table.stride(1) == 1,
              "qsa_attention_rdna2 needs int32 metadata with contiguous "
              "block-table rows");
  const int G = HQ / HKV;
  TORCH_CHECK(tile_tokens >= 1 && tile_tokens <= QU_MAXBQ &&
                  tile_tokens * G <= BM && ratio >= 1,
              "qsa_attention_rdna2: tile_tokens * G must fit 64 rows");
  const int num_seqs = cu_seqlens_q.size(0) - 1;
  if (num_seqs <= 0 || q.size(0) == 0) return;
  const int tiles = q.size(0) / tile_tokens + num_seqs;
  TORCH_CHECK(ublk.dtype() == torch::kInt32 && umask.dtype() == torch::kInt16 &&
                  ucount.dtype() == torch::kInt32 && ublk.size(0) >= tiles &&
                  ucount.numel() >= tiles && ublk.sizes() == umask.sizes(),
              "qsa_attention_rdna2: union shapes");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const dim3 grid(tiles, HKV);
  auto launch = [&](auto kernel) {
    kernel<<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        (__half*)out.data_ptr(), cu_seqlens_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), block_table.data_ptr<int>(),
        ublk.data_ptr<int>(),
        reinterpret_cast<const uint16_t*>(umask.data_ptr()),
        ucount.data_ptr<int>(), ublk.size(1), ratio, num_seqs, G, tile_tokens,
        k_cache.size(1), block_table.stride(0), q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), out.stride(0),
        out.stride(1), (float)scale, scale_ptr(k_scale), scale_ptr(v_scale));
  };
#define VLLM_QSA_RDNA2_BY_KV(HD)                         \
  if (kv == KV_FP8) {                                    \
    launch(qsa_attention_rdna2_kernel<HD, 64, KV_FP8>);  \
  } else if (kv == KV_BF16) {                            \
    launch(qsa_attention_rdna2_kernel<HD, 64, KV_BF16>); \
  } else {                                               \
    launch(qsa_attention_rdna2_kernel<HD, 64, KV_FP16>); \
  }
  if (D == 256) {
    VLLM_QSA_RDNA2_BY_KV(256);
  } else {
    VLLM_QSA_RDNA2_BY_KV(128);
  }
#undef VLLM_QSA_RDNA2_BY_KV
}

// QSA (Qwen4Exp sparse attention) decode / spec-verify for head 256: query
// token t of q [T, HQ, 256] fp16 (G = HQ / HKV <= 16) attends to its selection
// sel [T, W] int32 of positions (-1: none) in request tok2req[t] of the paged
// caches (and scales) as for unified_attention_rdna2, split into segments of
// 32-entry tiles and merged into out [T, HQ, 256] fp16.
void qsa_decode_rdna2(torch::Tensor& out, const torch::Tensor& q,
                      const torch::Tensor& k_cache,
                      const torch::Tensor& v_cache, const torch::Tensor& sel,
                      const torch::Tensor& tok2req,
                      const torch::Tensor& block_table, double scale,
                      const std::optional<torch::Tensor>& k_scale,
                      const std::optional<torch::Tensor>& v_scale) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(q.dtype() == torch::kFloat16 && out.dtype() == torch::kFloat16,
              "qsa_decode_rdna2: fp16 q and out");
  const int kv =
      check_kv_dtype("qsa_decode_rdna2", k_cache, v_cache, k_scale, v_scale);
  const int T = q.size(0), HQ = q.size(1), HKV = k_cache.size(2);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 256 && k_cache.size(3) == 256 &&
                  v_cache.sizes() == k_cache.sizes() && HQ % HKV == 0 &&
                  HQ / HKV <= QD_HMAX && out.sizes() == q.sizes() &&
                  out.stride(2) == 1 && out.stride(1) % 4 == 0 &&
                  out.stride(0) % 4 == 0,
              "qsa_decode_rdna2: head 256, at most 16 q heads per kv head");
  for (const torch::Tensor* t : {&q, &k_cache, &v_cache}) {
    TORCH_CHECK(t->stride(-1) == 1 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "qsa_decode_rdna2 needs 16-byte aligned heads");
    for (int d = 0; d < t->dim() - 1; d++)
      TORCH_CHECK(t->stride(d) % 8 == 0,
                  "qsa_decode_rdna2 needs 16-byte aligned head strides");
  }
  TORCH_CHECK(
      sel.dtype() == torch::kInt32 && sel.dim() == 2 && sel.size(0) == T &&
          sel.stride(1) == 1 && tok2req.dtype() == torch::kInt32 &&
          tok2req.numel() == T && tok2req.is_contiguous() &&
          block_table.dtype() == torch::kInt32 && block_table.stride(1) == 1,
      "qsa_decode_rdna2: int32 sel [T, W], tok2req [T], block_table");
  const int W = sel.size(1);
  if (T == 0 || W == 0) return;
  const int tiles = (W + QD_TK - 1) / QD_TK;
  // ~2 workgroups per CU (72-CU V620); more segments only add partials.
  const int nseg = std::max(
      1, std::min(std::min(tiles, QD_MAXSEG), (144 + T * HKV - 1) / (T * HKV)));
  const int tps = (tiles + nseg - 1) / nseg;
  const int act = (tiles + tps - 1) / tps;
  auto opts = q.options().dtype(torch::kFloat32);
  auto segm_out = torch::empty({T, HQ, nseg, 256}, opts);
  auto segm_max = torch::empty({T, HQ, nseg}, opts);
  auto segm_sum = torch::empty({T, HQ, nseg}, opts);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    kernel<<<dim3(T, HKV, nseg), THREADS, 0, stream>>>(
        (const __half*)q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        segm_out.data_ptr<float>(), segm_max.data_ptr<float>(),
        segm_sum.data_ptr<float>(), sel.data_ptr<int>(), sel.stride(0), W,
        tok2req.data_ptr<int>(), block_table.data_ptr<int>(),
        block_table.stride(0), (int)block_table.size(1), HQ / HKV, HQ,
        k_cache.size(1), q.stride(0), q.stride(1), k_cache.stride(0),
        k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),
        v_cache.stride(1), v_cache.stride(2), nseg, tps, (float)scale,
        scale_ptr(k_scale), scale_ptr(v_scale));
  };
  // Heads per S thread: ceil(G / 4).
  auto by_heads = [&](auto kv_tag) {
    constexpr int KVT = decltype(kv_tag)::value;
    switch ((HQ / HKV + 3) / 4) {
      case 1:
        launch(qsa_decode_rdna2_kernel<KVT, 1>);
        break;
      case 2:
        launch(qsa_decode_rdna2_kernel<KVT, 2>);
        break;
      case 3:
        launch(qsa_decode_rdna2_kernel<KVT, 3>);
        break;
      default:
        launch(qsa_decode_rdna2_kernel<KVT, 4>);
    }
  };
  if (kv == KV_FP8)
    by_heads(std::integral_constant<int, KV_FP8>{});
  else if (kv == KV_BF16)
    by_heads(std::integral_constant<int, KV_BF16>{});
  else
    by_heads(std::integral_constant<int, KV_FP16>{});
  qsa_decode_merge_kernel<<<dim3(T, HQ), 64, 0, stream>>>(
      segm_out.data_ptr<float>(), segm_max.data_ptr<float>(),
      segm_sum.data_ptr<float>(), (__half*)out.data_ptr(), out.stride(0),
      out.stride(1), HQ, nseg, act);
}
