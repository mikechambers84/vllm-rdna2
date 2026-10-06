// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) grouped W4A16 GEMM for fused MoE at a few routed rows per
// expert (~2-28), on the Triton WNA16 tensors: w [E, N, K/2] uint8 (two
// values per byte, low nibble = even k), group scales [E, N, K/G] fp16, zero
// points [E, N/2, K/G] uint8 (two columns per byte) or 8, rows routed through
// moe_align_block_size(T) (row r of expert block b reads activation row
// sorted_ids[r] / top_k and writes output row sorted_ids[r]).
//
// moe_wna16_gemm_rdna2's 16 * RM x 256 tiles have RM = 1-2 here: every weight
// dword is read from LDS by 16 threads for one row each, and its K stages wait
// on global loads (60-110 GB/s). Here a workgroup takes one T-row expert block
// and a column slice: 256 lanes run along N with CT columns each and stream
// their columns' nibbles from global memory into registers (U x 16 bytes per
// batch, the next batch in flight), so each weight is read once per T rows;
// the T activation rows are gathered into LDS (in phases of K when T x K fp16
// exceeds 32 KB) and read as wave broadcasts; NW waves split K and are summed
// through LDS. Each 8-k group of activations is stored pair-permuted ((a0,
// a4), (a1, a5), ...) to match the nibble expansion (w >> 4i) & 0x000F000F |
// 0x64006400 = half2(1024 + q_i, 1024 + q_(i+4)); (q - z) is exact in fp16 and
// the group scale is applied per 32-k step in fp32. Qwen3.6-35B-A3B experts
// (E=256, top-8, random routing): 2.5x moe_wna16_gemm_rdna2 at 4 rows per
// expert, 1.9x at 8, 1.45x at 16, 1.4x at 24. (It also beats the fused
// moe_wna16_decode_rdna2 at 0.5-2 rows in isolation, but not in the engine
// once its routing alignment, SiLU and top-k sum are counted.)

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_wna16_skinny_rdna2 {

static constexpr int THREADS = 256;
static constexpr int LDS_ELEMS = 16384;  // 32 KB of activations per phase

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}

// T rows per workgroup, CT columns per lane, NW waves along K (8 / NW along
// N), U steps of 32 k per weight batch; kp: K per activation phase (a
// multiple of NW * 32 * U dividing K).
template <int T, int CT, int NW, int U, bool MUL_W, bool ZP>
__global__ void __launch_bounds__(THREADS) moe_wna16_skinny_rdna2_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    const __half* __restrict__ S, const uint8_t* __restrict__ Z,
    __half* __restrict__ C, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, const int num_valid, const int top_k,
    const int N, const int K, const int group_size, const int E, const int lda,
    const int ldc, const int kp) {
  constexpr int WC = 8 / NW;
  constexpr int NC = WC * 32 * CT;
  constexpr int KB = 32 * U;
  __shared__ __align__(16) __half sA[LDS_ELEMS];
  __shared__ float sR[NW > 1 ? NW - 1 : 1][T][NC];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int row0 = blockIdx.x * T;
  if (row0 >= num_post_padded[0]) return;
  const int e = expert_ids[blockIdx.x];
  if (e < 0 || e >= E) return;
  int sid[T];
  #pragma unroll
  for (int t = 0; t < T; t++) sid[t] = sorted_ids[row0 + t];

  const int wn = wave % WC, wk = wave / WC;
  const int col0 = blockIdx.y * NC + wn * 32 * CT + lane * CT;
  const int groups = K / group_size;
  const uint8_t* We = W + (long)e * N * (K / 2);
  const __half* Se = S + (long)e * N * groups;
  const uint8_t* Ze = ZP ? Z + (long)e * (N / 2) * groups : nullptr;
  int col[CT];
  #pragma unroll
  for (int c = 0; c < CT; c++) col[c] = min(col0 + c, N - 1);

  uint4 wq[U][CT];
  auto load = [&](int k) {
  #pragma unroll
    for (int u = 0; u < U; u++)
  #pragma unroll
      for (int c = 0; c < CT; c++)
        wq[u][c] = *reinterpret_cast<const uint4*>(We + (long)col[c] * (K / 2) +
                                                   (k + 32 * u) / 2);
  };
  float acc[T][CT];
  #pragma unroll
  for (int t = 0; t < T; t++)
  #pragma unroll
    for (int c = 0; c < CT; c++) acc[t][c] = 0.f;
  const int ks = kp / NW;

  for (int p0 = 0; p0 < K; p0 += kp) {
    const int kbeg = p0 + wk * ks;
    load(kbeg);  // in flight while the phase's activations are gathered
    if (p0 > 0) __syncthreads();
    for (int i = tid; i < T * (kp / 8); i += THREADS) {
      const int t = i / (kp / 8), c = i % (kp / 8);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (sid[t] < num_valid)
        v = *reinterpret_cast<const uint4*>(A + (long)(sid[t] / top_k) * lda +
                                            p0 + c * 8);
      uint4 q;
      q.x = __builtin_amdgcn_perm(v.z, v.x, 0x05040100u);
      q.y = __builtin_amdgcn_perm(v.z, v.x, 0x07060302u);
      q.z = __builtin_amdgcn_perm(v.w, v.y, 0x05040100u);
      q.w = __builtin_amdgcn_perm(v.w, v.y, 0x07060302u);
      *reinterpret_cast<uint4*>(&sA[t * kp + c * 8]) = q;
    }
    __syncthreads();
    for (int kb = kbeg; kb < kbeg + ks; kb += KB) {
      uint4 cur[U][CT];
  #pragma unroll
      for (int u = 0; u < U; u++)
  #pragma unroll
        for (int c = 0; c < CT; c++) cur[u][c] = wq[u][c];
      if (kb + KB < kbeg + ks) load(kb + KB);
  #pragma unroll
      for (int u = 0; u < U; u++) {
        const int k = kb + 32 * u, g = k / group_size;
        half2 w[CT][16];
        float sc[CT];
  #pragma unroll
        for (int c = 0; c < CT; c++) {
          sc[c] = __half2float(Se[(long)col[c] * groups + g]);
          uint32_t zb = 0x64086408u;  // 1024 + 8
          if constexpr (ZP)
            zb = (0x6400u |
                  ((Ze[(long)(col[c] / 2) * groups + g] >> ((col[c] & 1) * 4)) &
                   0xFu)) *
                 0x00010001u;
          const half2 bias = as_h2(zb);
          const uint32_t words[4] = {cur[u][c].x, cur[u][c].y, cur[u][c].z,
                                     cur[u][c].w};
  #pragma unroll
          for (int q = 0; q < 4; q++)
  #pragma unroll
            for (int i = 0; i < 4; i++)
              w[c][q * 4 + i] = __hsub2(
                  as_h2(((words[q] >> (4 * i)) & 0x000F000Fu) | 0x64006400u),
                  bias);
        }
  #pragma unroll
        for (int t = 0; t < T; t++) {
          half2 a[16];
  #pragma unroll
          for (int q = 0; q < 4; q++) {
            const uint4 v =
                *reinterpret_cast<const uint4*>(&sA[t * kp + (k - p0) + q * 8]);
            a[q * 4 + 0] = as_h2(v.x);
            a[q * 4 + 1] = as_h2(v.y);
            a[q * 4 + 2] = as_h2(v.z);
            a[q * 4 + 3] = as_h2(v.w);
          }
  #pragma unroll
          for (int c = 0; c < CT; c++) {
            float part = 0.f;
  #pragma unroll
            for (int j = 0; j < 16; j++)
              part = __builtin_amdgcn_fdot2(a[j], w[c][j], part, false);
            acc[t][c] += sc[c] * part;
          }
        }
      }
    }
  }

  const int lc = wn * 32 * CT + lane * CT;
  if constexpr (NW > 1) {
    if (wk > 0)
  #pragma unroll
      for (int t = 0; t < T; t++)
  #pragma unroll
        for (int c = 0; c < CT; c++) sR[wk - 1][t][lc + c] = acc[t][c];
    __syncthreads();
    if (wk > 0) return;
  #pragma unroll
    for (int s = 0; s < NW - 1; s++)
  #pragma unroll
      for (int t = 0; t < T; t++)
  #pragma unroll
        for (int c = 0; c < CT; c++) acc[t][c] += sR[s][t][lc + c];
  }
  #pragma unroll
  for (int t = 0; t < T; t++) {
    if (sid[t] >= num_valid) continue;
    const float wgt = MUL_W ? topk_w[sid[t]] : 1.f;
  #pragma unroll
    for (int c = 0; c < CT; c++)
      if (col0 + c < N)
        C[(long)sid[t] * ldc + col0 + c] = __float2half(acc[t][c] * wgt);
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int T, int CT, int NW, int U, bool MUL_W, bool ZP>
__global__ void moe_wna16_skinny_rdna2_kernel(const __half*, const uint8_t*,
                                              const __half*, const uint8_t*,
                                              __half*, const int*, const int*,
                                              const int*, const float*,
                                              const int, const int, const int,
                                              const int, const int, const int,
                                              const int, const int, const int) {
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// Largest K per phase that divides K, is a multiple of `step` and keeps the
// T x kp fp16 activation tile within LDS_ELEMS; 0 if none.
static int phase_k(int K, int T, int step) {
  for (int kp = K; kp >= step; kp -= step)
    if (K % kp == 0 && kp % step == 0 && T * kp <= LDS_ELEMS) return kp;
  return 0;
}

}  // namespace moe_wna16_skinny_rdna2
}  // namespace vllm

// output [num_valid, N] fp16 (num_valid = topk_weights.numel()) receives row
// sorted_ids[r] of every routed row r; a [rows, K] fp16 is read at row
// sorted_ids[r] / top_k; w, scales and zeros as for moe_wna16_gemm_rdna2 (G a
// multiple of 32 dividing K; K % 32 == 0). sorted_ids / expert_ids /
// num_tokens_post_padded come from moe_align_block_size with block_m in {4, 8,
// 16} (the workgroup's rows). mul_routed_weight scales each row by its fp32
// top-k weight.
void moe_wna16_skinny_rdna2(torch::Tensor& output, const torch::Tensor& a,
                            const torch::Tensor& w, const torch::Tensor& scales,
                            const std::optional<torch::Tensor>& zeros,
                            const torch::Tensor& sorted_ids,
                            const torch::Tensor& expert_ids,
                            const torch::Tensor& num_tokens_post_padded,
                            const torch::Tensor& topk_weights, int64_t top_k,
                            bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_wna16_skinny_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 &&
                  output.dtype() == torch::kFloat16 &&
                  scales.dtype() == torch::kFloat16,
              "moe_wna16_skinny_rdna2 needs fp16 activations, output and "
              "scales");
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous() &&
                  scales.dim() == 3 && scales.is_contiguous(),
              "moe_wna16_skinny_rdna2 needs contiguous [E, N, K/2] uint8 "
              "weights and [E, N, K/G] scales");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 &&
                  expert_ids.dtype() == torch::kInt32 &&
                  num_tokens_post_padded.dtype() == torch::kInt32 &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "moe_wna16_skinny_rdna2 needs int32 alignment tensors and fp32 "
              "top-k weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2) * 2;
  const int groups = scales.size(2);
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == N && groups > 0 &&
                  K % groups == 0 && (K / groups) % 32 == 0 && K % 32 == 0,
              "moe_wna16_skinny_rdna2 needs K % 32 == 0 and group sizes that "
              "are multiples of 32");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "moe_wna16_skinny_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows");
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_wna16_skinny_rdna2 needs a contiguous [num_valid, N] "
              "output");
  TORCH_CHECK(block_m == 4 || block_m == 8 || block_m == 16,
              "moe_wna16_skinny_rdna2 supports block_m 4, 8 and 16");
  const uint8_t* zp = nullptr;
  if (zeros) {
    TORCH_CHECK(zeros->dtype() == torch::kUInt8 && zeros->is_contiguous() &&
                    zeros->numel() == (long)E * (N / 2) * groups,
                "moe_wna16_skinny_rdna2: zeros must be contiguous uint8 "
                "[E, N/2, K/G]");
    zp = zeros->data_ptr<uint8_t>();
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int group_size = K / groups;

#define VLLM_SKINNY_LAUNCH(T, CT, NW, U, MW, Z)                               \
  {                                                                           \
    constexpr int nc = (8 / NW) * 32 * CT;                                    \
    const dim3 grid(expert_ids.numel(), (N + nc - 1) / nc);                   \
    moe_wna16_skinny_rdna2_kernel<T, CT, NW, U, MW, Z>                        \
        <<<grid, THREADS, 0, stream>>>(                                       \
            (const __half*)a.data_ptr(), w.data_ptr<uint8_t>(),               \
            (const __half*)scales.data_ptr(), zp, (__half*)output.data_ptr(), \
            sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),           \
            num_tokens_post_padded.data_ptr<int>(),                           \
            topk_weights.data_ptr<float>(), num_valid, top_k, N, K,           \
            group_size, E, a.stride(0), N, kp);                               \
  }
#define VLLM_SKINNY_BY_FLAGS(T, CT, NW, U)           \
  if (mul_routed_weight) {                           \
    if (zp) {                                        \
      VLLM_SKINNY_LAUNCH(T, CT, NW, U, true, true)   \
    } else {                                         \
      VLLM_SKINNY_LAUNCH(T, CT, NW, U, true, false)  \
    }                                                \
  } else {                                           \
    if (zp) {                                        \
      VLLM_SKINNY_LAUNCH(T, CT, NW, U, false, true)  \
    } else {                                         \
      VLLM_SKINNY_LAUNCH(T, CT, NW, U, false, false) \
    }                                                \
  }
  // Tiles by rows per workgroup, from a sweep on E=256 top-8 (H 2048, I 512);
  // K not divisible by the batch falls back to single-step batches.
  int kp;
  if (block_m == 4) {
    if ((kp = phase_k(K, 4, 256))) {
      VLLM_SKINNY_BY_FLAGS(4, 1, 1, 8)
    } else {
      kp = phase_k(K, 4, 32);
      VLLM_SKINNY_BY_FLAGS(4, 1, 1, 1)
    }
  } else if (block_m == 8) {
    if ((kp = phase_k(K, 8, 256))) {
      VLLM_SKINNY_BY_FLAGS(8, 1, 1, 8)
    } else {
      kp = phase_k(K, 8, 32);
      VLLM_SKINNY_BY_FLAGS(8, 1, 1, 1)
    }
  } else {
    if ((kp = phase_k(K, 16, 256))) {
      VLLM_SKINNY_BY_FLAGS(16, 2, 2, 4)
    } else {
      kp = phase_k(K, 16, 32);
      VLLM_SKINNY_BY_FLAGS(16, 1, 1, 1)
    }
  }
#undef VLLM_SKINNY_BY_FLAGS
#undef VLLM_SKINNY_LAUNCH
}
