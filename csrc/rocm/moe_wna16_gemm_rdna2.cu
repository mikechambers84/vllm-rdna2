// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) grouped W4A16 GEMM for fused-MoE prefill, on the Triton
// WNA16 tensors: w [E, N, K/2] uint8 (two values q + 8 per byte, low nibble =
// even k), scales [E, N, K/G] fp16, rows routed through moe_align_block_size's
// sorted token ids (row r of expert block b reads activation row
// sorted_ids[r] / top_k and writes output row sorted_ids[r]).
//
// 256 threads (16 x 16) per workgroup; each owns RM rows x 16 columns (four
// 4-column groups spaced 64 apart), so a workgroup computes 16 * RM rows x
// 256 columns. K is staged 32 at a time, double buffered, in a k2-major LDS
// layout ([16][rows] dwords, one half2 = two consecutive k), so every LDS read
// is a conflict-free ds_read_b128 (activation reads are wave broadcasts).
// Activation rows are gathered into LDS; int4 weights are dequantized exactly
// on their way into LDS (OR into the 1024.0 mantissa, subtract 1032, scale in
// fp16) and v_dot2_f32_f16 accumulates in fp32. Qwen3.6-35B-A3B's experts
// (E=256, top-8) run 1.4x Triton at 512 tokens, 1.8x at 2048, 2.1x at 8192.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_wna16_gemm_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BK = 32;
static constexpr int K2 = BK / 2;
static constexpr int TN = 16;
static constexpr int BN = 16 * TN;
static constexpr int CH = BK / 8;  // 8-element chunks per row per stage

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// UNR: unroll of the k2 loop; a full unroll lets the compiler hoist every
// LDS read and spill.
template <int RM, int UNR, bool MUL_W>
__global__ void __launch_bounds__(THREADS) moe_wna16_gemm_rdna2_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    const __half* __restrict__ S, __half* __restrict__ C,
    const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_post_padded, const float* __restrict__ topk_w,
    const int num_valid, const int top_k, const int N, const int K,
    const int group_size, const int E, const int lda, const int ldc) {
  constexpr int BM = 16 * RM;
  constexpr int A_ITEMS = (BM * CH + THREADS - 1) / THREADS;
  constexpr int W_ITEMS = BN * CH / THREADS;
  __shared__ __align__(16) uint32_t sA[2][K2][BM];
  __shared__ __align__(16) uint32_t sW[2][K2][BN];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int row0 = blockIdx.x * BM;
  if (row0 >= num_post_padded[0]) return;
  const int e = expert_ids[blockIdx.x];
  if (e < 0 || e >= E) return;
  const int n0 = blockIdx.y * BN;
  const uint8_t* We = W + (long)e * N * (K / 2);
  const __half* Se = S + (long)e * N * (K / group_size);

  // Element offset of each gathered activation chunk; -1 for padding rows.
  int a_off[A_ITEMS];
  #pragma unroll
  for (int it = 0; it < A_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    a_off[it] = -1;
    if (idx < BM * CH) {
      const int sid = sorted_ids[row0 + idx / CH];
      if (sid < num_valid) a_off[it] = (sid / top_k) * lda + idx % CH * 8;
    }
  }

  uint4 ra[A_ITEMS];
  uint32_t rw[W_ITEMS];
  __half rs[W_ITEMS];
  auto load = [&](int k0) {
  #pragma unroll
    for (int it = 0; it < A_ITEMS; it++)
      ra[it] = a_off[it] >= 0
                   ? *reinterpret_cast<const uint4*>(A + a_off[it] + k0)
                   : make_uint4(0, 0, 0, 0);
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int col = min(n0 + idx / CH, N - 1);
      const int k = k0 + idx % CH * 8;
      rw[it] =
          *reinterpret_cast<const uint32_t*>(We + (long)col * (K / 2) + k / 2);
      rs[it] = Se[(long)col * (K / group_size) + k / group_size];
    }
  };
  auto store = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < A_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx < BM * CH) {
        const int r = idx / CH, c = idx % CH;
        sA[buf][c * 4 + 0][r] = ra[it].x;
        sA[buf][c * 4 + 1][r] = ra[it].y;
        sA[buf][c * 4 + 2][r] = ra[it].z;
        sA[buf][c * 4 + 3][r] = ra[it].w;
      }
    }
    const half2 bias =
        __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int n = idx / CH, c = idx % CH;
      const half2 s2 = __halves2half2(rs[it], rs[it]);
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        // Byte j holds k = 2j (low nibble) and 2j + 1: one half2 pair.
        const uint32_t t = rw[it] >> (8 * j);
        uint32_t u = (t & 0xFu) | ((t & 0xF0u) << 12) | 0x64006400u;
        const half2 h =
            __hmul2(__hsub2(*reinterpret_cast<const half2*>(&u), bias), s2);
        sW[buf][c * 4 + j][n] = *reinterpret_cast<const uint32_t*>(&h);
      }
    }
  };

  float acc[RM][TN];
  #pragma unroll
  for (int i = 0; i < RM; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

  const int stages = K / BK;
  load(0);
  store(0);
  __syncthreads();
  for (int s = 0; s < stages; s++) {
    const int cur = s & 1;
    if (s + 1 < stages) load((s + 1) * BK);
  #pragma unroll UNR
    for (int k2 = 0; k2 < K2; k2++) {
      half2 a[RM], w[TN];
      if constexpr (RM % 4 == 0) {
  #pragma unroll
        for (int i = 0; i < RM; i += 4) {
          const uint4 v =
              *reinterpret_cast<const uint4*>(&sA[cur][k2][ty * RM + i]);
          a[i] = *reinterpret_cast<const half2*>(&v.x);
          a[i + 1] = *reinterpret_cast<const half2*>(&v.y);
          a[i + 2] = *reinterpret_cast<const half2*>(&v.z);
          a[i + 3] = *reinterpret_cast<const half2*>(&v.w);
        }
      } else if constexpr (RM == 2) {
        const uint2 v = *reinterpret_cast<const uint2*>(&sA[cur][k2][ty * 2]);
        a[0] = *reinterpret_cast<const half2*>(&v.x);
        a[RM - 1] = *reinterpret_cast<const half2*>(&v.y);
      } else {
        a[0] = *reinterpret_cast<const half2*>(&sA[cur][k2][ty]);
      }
  #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(&sW[cur][k2][j * 16 + tx * 4]);
        w[j] = *reinterpret_cast<const half2*>(&v.x);
        w[j + 1] = *reinterpret_cast<const half2*>(&v.y);
        w[j + 2] = *reinterpret_cast<const half2*>(&v.z);
        w[j + 3] = *reinterpret_cast<const half2*>(&v.w);
      }
  #pragma unroll
      for (int i = 0; i < RM; i++)
  #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = __builtin_amdgcn_fdot2(a[i], w[j], acc[i][j], false);
    }
    if (s + 1 < stages) {
      store(cur ^ 1);
      __syncthreads();
    }
  }

  #pragma unroll
  for (int i = 0; i < RM; i++) {
    const int sid = sorted_ids[row0 + ty * RM + i];
    if (sid >= num_valid) continue;
    const float wgt = MUL_W ? topk_w[sid] : 1.f;
  #pragma unroll
    for (int j = 0; j < TN; j += 2) {
      const int col = n0 + j / 4 * 64 + tx * 4 + j % 4;
      if (col < N)
        *reinterpret_cast<half2*>(C + (long)sid * ldc + col) =
            __floats2half2_rn(acc[i][j] * wgt, acc[i][j + 1] * wgt);
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int RM, int UNR, bool MUL_W>
__global__ void moe_wna16_gemm_rdna2_kernel(const __half*, const uint8_t*,
                                            const __half*, __half*, const int*,
                                            const int*, const int*,
                                            const float*, const int, const int,
                                            const int, const int, const int,
                                            const int, const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_wna16_gemm_rdna2
}  // namespace vllm

// output [num_valid, N] fp16 (num_valid = topk_weights.numel()) receives row
// sorted_ids[r] of every routed row r; a [rows, K] fp16 is read at row
// sorted_ids[r] / top_k; w [E, N, K/2] uint8 and scales [E, N, K/G] fp16 are
// symmetric int4 with G a multiple of 32 dividing K; K % 32 == 0, N % 8 == 0.
// sorted_ids / expert_ids / num_tokens_post_padded come from
// moe_align_block_size with block_m in {16, 32, 64, 128}. mul_routed_weight
// scales each row by its fp32 top-k weight.
void moe_wna16_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                          const torch::Tensor& w, const torch::Tensor& scales,
                          const torch::Tensor& sorted_ids,
                          const torch::Tensor& expert_ids,
                          const torch::Tensor& num_tokens_post_padded,
                          const torch::Tensor& topk_weights, int64_t top_k,
                          bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_wna16_gemm_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 &&
                  output.dtype() == torch::kFloat16 &&
                  scales.dtype() == torch::kFloat16,
              "moe_wna16_gemm_rdna2 needs fp16 activations, output and scales");
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous() &&
                  scales.dim() == 3 && scales.is_contiguous(),
              "moe_wna16_gemm_rdna2 needs contiguous [E, N, K/2] uint8 weights "
              "and [E, N, K/G] scales");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 &&
                  expert_ids.dtype() == torch::kInt32 &&
                  num_tokens_post_padded.dtype() == torch::kInt32 &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "moe_wna16_gemm_rdna2 needs int32 alignment tensors and fp32 "
              "topk weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2) * 2;
  const int groups = scales.size(2);
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "moe_wna16_gemm_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows");
  TORCH_CHECK(K % BK == 0 && N % 8 == 0 && groups > 0 && K % groups == 0 &&
                  (K / groups) % BK == 0 && scales.size(0) == E &&
                  scales.size(1) == N,
              "moe_wna16_gemm_rdna2 needs K % 32 == 0, N % 8 == 0 and a group "
              "size that is a multiple of 32 dividing K");
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_wna16_gemm_rdna2 needs a contiguous [num_valid, N] output");
  TORCH_CHECK(block_m == 16 || block_m == 32 || block_m == 64 || block_m == 128,
              "moe_wna16_gemm_rdna2 supports block_m 16, 32, 64 and 128");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(expert_ids.numel(), (N + BN - 1) / BN);

#define VLLM_MOE_GEMM_LAUNCH(RM, UNR, MW)                                     \
  moe_wna16_gemm_rdna2_kernel<RM, UNR, MW><<<grid, THREADS, 0, stream>>>(     \
      (const __half*)a.data_ptr(), w.data_ptr<uint8_t>(),                     \
      (const __half*)scales.data_ptr(), (__half*)output.data_ptr(),           \
      sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),                 \
      num_tokens_post_padded.data_ptr<int>(), topk_weights.data_ptr<float>(), \
      num_valid, top_k, N, K, K / groups, E, a.stride(0), N)
#define VLLM_MOE_GEMM_BY_W(RM, UNR)       \
  if (mul_routed_weight) {                \
    VLLM_MOE_GEMM_LAUNCH(RM, UNR, true);  \
  } else {                                \
    VLLM_MOE_GEMM_LAUNCH(RM, UNR, false); \
  }
  switch (block_m) {
    case 16:
      VLLM_MOE_GEMM_BY_W(1, 4);
      break;
    case 32:
      VLLM_MOE_GEMM_BY_W(2, 4);
      break;
    case 64:
      VLLM_MOE_GEMM_BY_W(4, 4);
      break;
    default:
      VLLM_MOE_GEMM_BY_W(8, 2);
      break;
  }
#undef VLLM_MOE_GEMM_BY_W
#undef VLLM_MOE_GEMM_LAUNCH
}
