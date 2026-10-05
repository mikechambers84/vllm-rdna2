// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) grouped W8A8 GEMM for fused-MoE prefill, on the Triton int8
// tensors: w [E, N, K] int8 with per-channel fp32 scales [E, N], int8
// activations with per-row fp32 scales, rows routed through
// moe_align_block_size's sorted token ids (row r of expert block b reads
// activation row sorted_ids[r] / top_k and writes output row sorted_ids[r]).
//
// 256 threads (16 x 16) per workgroup; each owns RM rows x 16 columns (four
// 4-column groups spaced 64 apart): 16 * RM rows x 256 columns per workgroup.
// K is staged 64 bytes at a time, double buffered, in a k4-major LDS layout
// ([16][rows] dwords, one dword = four consecutive k = one v_dot4_i32_i8
// operand), so LDS reads are conflict-free ds_read_b128. int32 accumulation
// is exact, so results equal the Triton W8A8 kernel's up to the fp16 output
// rounding. Qwen3.6-35B-A3B's experts (E=256, top-8): 1.3x Triton at 2048
// tokens, 1.3-1.45x at 8192 (~30 TOPS).

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_int8_gemm_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BK = 64;
static constexpr int K4 = BK / 4;
static constexpr int CH = BK / 16;  // 16-byte chunks per row per stage
static constexpr int TN = 16;
static constexpr int BN = 16 * TN;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// UNR: unroll of the k4 loop (a full unroll hoists every LDS read and
// spills).
template <int RM, int UNR, bool MUL_W>
__global__ void __launch_bounds__(THREADS) moe_int8_gemm_rdna2_kernel(
    const int8_t* __restrict__ A, const float* __restrict__ AS,
    const int8_t* __restrict__ W, const float* __restrict__ WS,
    __half* __restrict__ C, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, const int num_valid, const int top_k,
    const int N, const int K, const int E, const int lda, const int ldc) {
  constexpr int BM = 16 * RM;
  constexpr int A_ITEMS = (BM * CH + THREADS - 1) / THREADS;
  constexpr int W_ITEMS = BN * CH / THREADS;
  __shared__ __align__(16) int sA[2][K4][BM];
  __shared__ __align__(16) int sW[2][K4][BN];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int row0 = blockIdx.x * BM;
  if (row0 >= num_post_padded[0]) return;
  const int e = expert_ids[blockIdx.x];
  if (e < 0 || e >= E) return;
  const int n0 = blockIdx.y * BN;
  const int8_t* We = W + (long)e * N * K;

  // Byte offset of each gathered activation chunk; -1 for padding rows.
  int a_off[A_ITEMS];
  #pragma unroll
  for (int it = 0; it < A_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    a_off[it] = -1;
    if (idx < BM * CH) {
      const int sid = sorted_ids[row0 + idx / CH];
      if (sid < num_valid) a_off[it] = (sid / top_k) * lda + idx % CH * 16;
    }
  }

  int4 ra[A_ITEMS], rw[W_ITEMS];
  auto load = [&](int k0) {
  #pragma unroll
    for (int it = 0; it < A_ITEMS; it++)
      ra[it] = a_off[it] >= 0
                   ? *reinterpret_cast<const int4*>(A + a_off[it] + k0)
                   : make_int4(0, 0, 0, 0);
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int col = min(n0 + idx / CH, N - 1);
      rw[it] = *reinterpret_cast<const int4*>(We + (long)col * K + k0 +
                                              idx % CH * 16);
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
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int n = idx / CH, c = idx % CH;
      sW[buf][c * 4 + 0][n] = rw[it].x;
      sW[buf][c * 4 + 1][n] = rw[it].y;
      sW[buf][c * 4 + 2][n] = rw[it].z;
      sW[buf][c * 4 + 3][n] = rw[it].w;
    }
  };

  int acc[RM][TN];
  #pragma unroll
  for (int i = 0; i < RM; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0;

  const int stages = K / BK;
  load(0);
  store(0);
  __syncthreads();
  for (int s = 0; s < stages; s++) {
    const int cur = s & 1;
    if (s + 1 < stages) load((s + 1) * BK);
  #pragma unroll UNR
    for (int k4 = 0; k4 < K4; k4++) {
      int a[RM], w[TN];
  #pragma unroll
      for (int i = 0; i < RM; i += 4) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sA[cur][k4][ty * RM + i]);
        a[i] = v.x;
        a[i + 1] = v.y;
        a[i + 2] = v.z;
        a[i + 3] = v.w;
      }
  #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sW[cur][k4][j * 16 + tx * 4]);
        w[j] = v.x;
        w[j + 1] = v.y;
        w[j + 2] = v.z;
        w[j + 3] = v.w;
      }
  #pragma unroll
      for (int i = 0; i < RM; i++)
  #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = __builtin_amdgcn_sdot4(a[i], w[j], acc[i][j], false);
    }
    if (s + 1 < stages) {
      store(cur ^ 1);
      __syncthreads();
    }
  }

  float ws[TN];
  #pragma unroll
  for (int j = 0; j < TN; j++)
    ws[j] = WS[(long)e * N + min(n0 + j / 4 * 64 + tx * 4 + j % 4, N - 1)];
  #pragma unroll
  for (int i = 0; i < RM; i++) {
    const int sid = sorted_ids[row0 + ty * RM + i];
    if (sid >= num_valid) continue;
    const float sc = AS[sid / top_k] * (MUL_W ? topk_w[sid] : 1.f);
  #pragma unroll
    for (int j = 0; j < TN; j += 2) {
      const int col = n0 + j / 4 * 64 + tx * 4 + j % 4;
      if (col < N)
        *reinterpret_cast<half2*>(C + (long)sid * ldc + col) =
            __floats2half2_rn(acc[i][j] * sc * ws[j],
                              acc[i][j + 1] * sc * ws[j + 1]);
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int RM, int UNR, bool MUL_W>
__global__ void moe_int8_gemm_rdna2_kernel(const int8_t*, const float*,
                                           const int8_t*, const float*, __half*,
                                           const int*, const int*, const int*,
                                           const float*, const int, const int,
                                           const int, const int, const int,
                                           const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_int8_gemm_rdna2
}  // namespace vllm

// output [num_valid, N] fp16 (num_valid = topk_weights.numel()) receives row
// sorted_ids[r] of every routed row r; a [rows, K] int8 with row scales
// a_scale [rows] fp32 is read at row sorted_ids[r] / top_k; w [E, N, K] int8
// with w_scale [E, N] (or [E, N, 1]) fp32; K % 64 == 0, N % 8 == 0.
// sorted_ids / expert_ids / num_tokens_post_padded come from
// moe_align_block_size with block_m in {64, 128}. mul_routed_weight scales
// each row by its fp32 top-k weight.
void moe_int8_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                         const torch::Tensor& a_scale, const torch::Tensor& w,
                         const torch::Tensor& w_scale,
                         const torch::Tensor& sorted_ids,
                         const torch::Tensor& expert_ids,
                         const torch::Tensor& num_tokens_post_padded,
                         const torch::Tensor& topk_weights, int64_t top_k,
                         bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_int8_gemm_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8 &&
                  output.dtype() == torch::kFloat16,
              "moe_int8_gemm_rdna2 needs int8 activations and weights and an "
              "fp16 output");
  TORCH_CHECK(a_scale.dtype() == torch::kFloat32 && a_scale.is_contiguous() &&
                  w_scale.dtype() == torch::kFloat32 &&
                  w_scale.is_contiguous() &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "moe_int8_gemm_rdna2 needs contiguous fp32 scales and top-k "
              "weights");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 &&
                  expert_ids.dtype() == torch::kInt32 &&
                  num_tokens_post_padded.dtype() == torch::kInt32,
              "moe_int8_gemm_rdna2 needs int32 alignment tensors");
  TORCH_CHECK(w.dim() == 3 && w.is_contiguous(),
              "moe_int8_gemm_rdna2 needs contiguous [E, N, K] weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2);
  TORCH_CHECK(w_scale.numel() == (long)E * N,
              "moe_int8_gemm_rdna2 needs one weight scale per output channel");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  a_scale.numel() == a.size(0),
              "moe_int8_gemm_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows and one scale per row");
  TORCH_CHECK(K % BK == 0 && N % 8 == 0,
              "moe_int8_gemm_rdna2 needs K % 64 == 0 and N % 8 == 0");
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_int8_gemm_rdna2 needs a contiguous [num_valid, N] output");
  TORCH_CHECK(block_m == 64 || block_m == 128,
              "moe_int8_gemm_rdna2 supports block_m 64 and 128");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(expert_ids.numel(), (N + BN - 1) / BN);

#define VLLM_MOE_I8_LAUNCH(RM, UNR, MW)                                       \
  moe_int8_gemm_rdna2_kernel<RM, UNR, MW><<<grid, THREADS, 0, stream>>>(      \
      a.data_ptr<int8_t>(), a_scale.data_ptr<float>(), w.data_ptr<int8_t>(),  \
      w_scale.data_ptr<float>(), (__half*)output.data_ptr(),                  \
      sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),                 \
      num_tokens_post_padded.data_ptr<int>(), topk_weights.data_ptr<float>(), \
      num_valid, top_k, N, K, E, a.stride(0), N)
#define VLLM_MOE_I8_BY_W(RM, UNR)       \
  if (mul_routed_weight) {              \
    VLLM_MOE_I8_LAUNCH(RM, UNR, true);  \
  } else {                              \
    VLLM_MOE_I8_LAUNCH(RM, UNR, false); \
  }
  if (block_m == 64) {
    VLLM_MOE_I8_BY_W(4, 4);
  } else {
    VLLM_MOE_I8_BY_W(8, 4);
  }
#undef VLLM_MOE_I8_BY_W
#undef VLLM_MOE_I8_LAUNCH
}
