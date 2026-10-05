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

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace attention_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BM = 64;  // rows per workgroup
static constexpr int BN = 64;  // keys per tile
static constexpr int VC = 16;  // keys per V stage

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

template <int D, int DC>
__global__ void __launch_bounds__(THREADS) unified_attention_rdna2_kernel(
    const __half* __restrict__ Q, const __half* __restrict__ Kc,
    const __half* __restrict__ Vc, __half* __restrict__ O,
    const int* __restrict__ cu_q, const int* __restrict__ seqused_k,
    const int* __restrict__ block_table, const int num_seqs, const int G,
    const int BQ, const int block_size, const long bt_stride, const long q_st,
    const long q_sh, const long k_sb, const long k_st, const long k_sh,
    const long v_sb, const long v_st, const long v_sh, const long o_st,
    const long o_sh, const float scale_log2e) {
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

  int lim[4];  // causal limit of this thread's rows (key index <= lim)
  #pragma unroll
  for (int i = 0; i < 4; i++)
    lim[i] = ctx + t0 + min((ty * 4 + i) / G, ntok - 1);

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

  auto k_off = [&](int key) -> long {
    return bt[key / block_size] * k_sb + key % block_size * k_st + kvh * k_sh;
  };
  auto v_off = [&](int key) -> long {
    return bt[key / block_size] * v_sb + key % block_size * v_st + kvh * v_sh;
  };

  long krow[K_ITEMS];
  uint4 kr[K_ITEMS];
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
      kr[it] = krow[it] >= 0 ? *reinterpret_cast<const uint4*>(
                                   Kc + krow[it] + dc * DC + part * 8)
                             : make_uint4(0, 0, 0, 0);
    }
  };
  auto kstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < K_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int key = idx / KL, part = idx % KL;
      uint32_t* d = &sU[buf][part * 4 * BN + (key ^ (part * (32 / KL)))];
      d[0] = kr[it].x;
      d[BN] = kr[it].y;
      d[2 * BN] = kr[it].z;
      d[3 * BN] = kr[it].w;
    }
  };
  uint4 va[V_ITEMS], vb[V_ITEMS];
  auto vload = [&](int kt, int vc) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const int k0 = kt + vc * VC + kp * 2;
      va[it] = idx < V_TOTAL && k0 < klen
                   ? *reinterpret_cast<const uint4*>(Vc + v_off(k0) + c * 8)
                   : make_uint4(0, 0, 0, 0);
      vb[it] = idx < V_TOTAL && k0 + 1 < klen
                   ? *reinterpret_cast<const uint4*>(Vc + v_off(k0 + 1) + c * 8)
                   : make_uint4(0, 0, 0, 0);
    }
  };
  auto vstore = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < V_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx >= V_TOTAL) continue;
      const int kp = idx / (D / 8), c = idx % (D / 8);
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&va[it]);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&vb[it]);
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
    const bool need_mask = kt + BN - 1 > ctx + t0;
    float alpha[4];
    uint32_t pw[4][2];
  #pragma unroll
    for (int i = 0; i < 4; i++) {
      float sv[4];
      float mx = -INFINITY;
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        sv[j] = sacc[i][j] * scale_log2e;
        if (need_mask && kt + tx * 4 + j > lim[i]) sv[j] = -INFINITY;
        mx = fmaxf(mx, sv[j]);
      }
      mx = row16_max(mx);
      const float m_new = fmaxf(m_i[i], mx);
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
    const float inv = 1.f / l;
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

template <int D, int DC>
__global__ void unified_attention_rdna2_kernel(
    const __half*, const __half*, const __half*, __half*, const int*,
    const int*, const int*, const int, const int, const int, const int,
    const long, const long, const long, const long, const long, const long,
    const long, const long, const long, const long, const long, const float) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace attention_rdna2
}  // namespace vllm

// Causal attention of q [num_tokens, num_q_heads, D] against the paged fp16
// caches k_cache / v_cache [num_blocks, block_size, num_kv_heads, D] into
// out [num_tokens, num_q_heads, D]; D in {64, 128, 256}. Sequence s owns query
// tokens cu_seqlens_q[s] .. cu_seqlens_q[s + 1] - 1, which are the last ones
// of its seqused_k[s] keys, whose cache blocks are block_table[s].
void unified_attention_rdna2(torch::Tensor& out, const torch::Tensor& q,
                             const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache,
                             const torch::Tensor& cu_seqlens_q,
                             const torch::Tensor& seqused_k,
                             const torch::Tensor& block_table, double scale) {
  using namespace vllm::attention_rdna2;
  TORCH_CHECK(
      q.dtype() == torch::kFloat16 && k_cache.dtype() == torch::kFloat16 &&
          v_cache.dtype() == torch::kFloat16 && out.dtype() == torch::kFloat16,
      "unified_attention_rdna2 needs fp16 queries, caches and output");
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
  const float scale_log2e = scale * 1.4426950408889634;

#define VLLM_ATTN_RDNA2_LAUNCH(HD, DC)                                    \
  unified_attention_rdna2_kernel<HD, DC><<<grid, THREADS, 0, stream>>>(   \
      (const __half*)q.data_ptr(), (const __half*)k_cache.data_ptr(),     \
      (const __half*)v_cache.data_ptr(), (__half*)out.data_ptr(),         \
      cu_seqlens_q.data_ptr<int>(), seqused_k.data_ptr<int>(),            \
      block_table.data_ptr<int>(), num_seqs, G, BQ, k_cache.size(1),      \
      block_table.stride(0), q.stride(0), q.stride(1), k_cache.stride(0), \
      k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),            \
      v_cache.stride(1), v_cache.stride(2), out.stride(0), out.stride(1), \
      scale_log2e)
  if (D == 256) {
    VLLM_ATTN_RDNA2_LAUNCH(256, 64);
  } else if (D == 128) {
    VLLM_ATTN_RDNA2_LAUNCH(128, 64);
  } else {
    VLLM_ATTN_RDNA2_LAUNCH(64, 32);
  }
#undef VLLM_ATTN_RDNA2_LAUNCH
}
