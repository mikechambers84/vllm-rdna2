// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) W8A8 GEMM for prefill: C[M,N] = (A[M,K] . W[N,K]^T) * sa *
// sb (+ bias), int8 A and W with int32 accumulation through v_dot4_i32_i8.
//
// 256 threads as a 16x16 grid; each thread owns 8 x TN outputs in 4x4 blocks
// spaced 64 apart, so the workgroup tile is 128 x (16 * TN) and every LDS read
// is a conflict-free ds_read_b128. K is staged BK bytes at a time, double
// buffered, in a "k4-major" LDS layout ([BK / 4][rows] dwords): a dword holds
// 4 consecutive K of one row, i.e. one dot4 operand. Global loads for the next
// stage are issued before the current stage is computed.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gemm_w8a8_rdna2 {

static constexpr int THREADS = 256;
static constexpr int TM = 8;
static constexpr int BM = 16 * TM;
static constexpr int BK = 64;
static constexpr int GROUP_M = 8;

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16 from_float<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename TOut, int TN>
__global__ void __launch_bounds__(THREADS)
    gemm_w8a8_rdna2_kernel(const int8_t* __restrict__ A,
                           const int8_t* __restrict__ W,
                           const float* __restrict__ sa,
                           const float* __restrict__ sb,
                           const TOut* __restrict__ bias, TOut* __restrict__ C,
                           const int M, const int N, const int K, const int lda,
                           const int ldw, const int ldc, const int sa_stride,
                           const int sb_stride) {
  constexpr int BN = 16 * TN;
  constexpr int K4 = BK / 4;
  constexpr int CH = K4 / 4;  // 16-byte chunks per row per stage
  constexpr int A_LOADS = BM * CH / THREADS;
  constexpr int W_LOADS = BN * CH / THREADS;
  // Full unrolling hoists every LDS read of a stage and runs out of VGPRs.
  constexpr int UNROLL = TN == 8 ? 4 : 2;
  __shared__ __align__(16) int sA[2][K4][BM];
  __shared__ __align__(16) int sW[2][K4][BN];

  // Grouped ordering: GROUP_M row blocks in a row share each weight block
  // while it is in L2.
  const int pid = blockIdx.x;
  const int num_m = (M + BM - 1) / BM;
  const int num_n = (N + BN - 1) / BN;
  const int width = GROUP_M * num_n;
  const int first_m = pid / width * GROUP_M;
  const int group_rows = min(num_m - first_m, GROUP_M);
  const int m0 = (first_m + pid % width % group_rows) * BM;
  const int n0 = pid % width / group_rows * BN;

  const int tid = threadIdx.x;
  const int tx = tid % 16;
  const int ty = tid / 16;

  // Thread loads chunk idx % CH of row idx / CH; tail rows are clamped and
  // their outputs are never stored.
  int4 ra[A_LOADS], rw[W_LOADS];
  auto load_stage = [&](int k0) {
  #pragma unroll
    for (int j = 0; j < A_LOADS; j++) {
      const int idx = tid + j * THREADS, row = min(m0 + idx / CH, M - 1);
      ra[j] = *reinterpret_cast<const int4*>(A + (long)row * lda + k0 +
                                             idx % CH * 16);
    }
  #pragma unroll
    for (int j = 0; j < W_LOADS; j++) {
      const int idx = tid + j * THREADS, row = min(n0 + idx / CH, N - 1);
      rw[j] = *reinterpret_cast<const int4*>(W + (long)row * ldw + k0 +
                                             idx % CH * 16);
    }
  };
  auto store_stage = [&](int buf) {
  #pragma unroll
    for (int j = 0; j < A_LOADS; j++) {
      const int idx = tid + j * THREADS, k4 = idx % CH * 4, r = idx / CH;
      sA[buf][k4][r] = ra[j].x;
      sA[buf][k4 + 1][r] = ra[j].y;
      sA[buf][k4 + 2][r] = ra[j].z;
      sA[buf][k4 + 3][r] = ra[j].w;
    }
  #pragma unroll
    for (int j = 0; j < W_LOADS; j++) {
      const int idx = tid + j * THREADS, k4 = idx % CH * 4, r = idx / CH;
      sW[buf][k4][r] = rw[j].x;
      sW[buf][k4 + 1][r] = rw[j].y;
      sW[buf][k4 + 2][r] = rw[j].z;
      sW[buf][k4 + 3][r] = rw[j].w;
    }
  };

  int acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0;

  load_stage(0);
  store_stage(0);
  __syncthreads();
  const int stages = K / BK;
  for (int s = 0; s < stages; s++) {
    const int buf = s & 1;
    if (s + 1 < stages) load_stage((s + 1) * BK);
  #pragma unroll UNROLL
    for (int k4 = 0; k4 < K4; k4++) {
      int av[TM], wv[TN];
  #pragma unroll
      for (int g = 0; g < TM / 4; g++) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sA[buf][k4][g * 64 + ty * 4]);
        av[g * 4] = v.x;
        av[g * 4 + 1] = v.y;
        av[g * 4 + 2] = v.z;
        av[g * 4 + 3] = v.w;
      }
  #pragma unroll
      for (int g = 0; g < TN / 4; g++) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sW[buf][k4][g * 64 + tx * 4]);
        wv[g * 4] = v.x;
        wv[g * 4 + 1] = v.y;
        wv[g * 4 + 2] = v.z;
        wv[g * 4 + 3] = v.w;
      }
  #pragma unroll
      for (int i = 0; i < TM; i++)
  #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = __builtin_amdgcn_sdot4(av[i], wv[j], acc[i][j], false);
    }
    if (s + 1 < stages) {
      store_stage(buf ^ 1);
      __syncthreads();
    }
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + i / 4 * 64 + ty * 4 + i % 4;
    if (row >= M) continue;
    const float s_row = sa[row * sa_stride];
  #pragma unroll
    for (int g = 0; g < TN / 4; g++) {
      const int col = n0 + g * 64 + tx * 4;
      if (col >= N) continue;
      alignas(8) TOut out[4];
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        const int n = min(col + j, N - 1);
        // Round, then add the bias in the output dtype, as triton_scaled_mm
        // does, so both give identical results.
        TOut v = from_float<TOut>((float)acc[i][g * 4 + j] * s_row *
                                  sb[n * sb_stride]);
        if (bias)
          v = from_float<TOut>(to_float<TOut>(v) + to_float<TOut>(bias[n]));
        out[j] = v;
      }
      TOut* dst = C + (long)row * ldc + col;
      if (col + 4 <= N && ldc % 4 == 0) {
        *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(out);
      } else {
        for (int j = 0; j < 4 && col + j < N; j++) dst[j] = out[j];
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename TOut, int TN>
__global__ void gemm_w8a8_rdna2_kernel(const int8_t*, const int8_t*,
                                       const float*, const float*, const TOut*,
                                       TOut*, const int, const int, const int,
                                       const int, const int, const int,
                                       const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gemm_w8a8_rdna2
}  // namespace vllm

// Requirements: a int8 [M, K] and w int8 [N, K] (an int8 linear's [K, N]
// weight view, transposed), both with contiguous, 16-byte aligned rows;
// K % 64 == 0; scale_a fp32 with 1 or M elements, scale_b fp32 with 1 or N
// elements; optional bias [N] in out_dtype. Returns fp16 or bf16 [M, N].
// Exactly matches triton_scaled_mm (int32 accumulation).
torch::Tensor w8a8_gemm_rdna2(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& scale_a,
                              const at::Tensor& scale_b,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype) {
  using namespace vllm::gemm_w8a8_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8,
              "w8a8_gemm_rdna2 needs int8 a and w");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1),
              "w8a8_gemm_rdna2 needs a [M, K] and w [N, K]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(0);
  TORCH_CHECK(M >= 1 && N >= 1, "w8a8_gemm_rdna2 needs non-empty a and w");
  TORCH_CHECK(K % BK == 0, "w8a8_gemm_rdna2 needs K % ", BK, " == 0");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1 && a.stride(0) % 16 == 0 &&
                  w.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "w8a8_gemm_rdna2 needs 16-byte aligned, K-contiguous rows");
  TORCH_CHECK(scale_a.dtype() == torch::kFloat32 &&
                  scale_b.dtype() == torch::kFloat32 &&
                  scale_a.is_contiguous() && scale_b.is_contiguous(),
              "w8a8_gemm_rdna2 needs contiguous fp32 scales");
  TORCH_CHECK(scale_a.numel() == 1 || scale_a.numel() == M,
              "w8a8_gemm_rdna2 scale_a must have 1 or M elements");
  TORCH_CHECK(scale_b.numel() == 1 || scale_b.numel() == N,
              "w8a8_gemm_rdna2 scale_b must have 1 or N elements");
  TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
              "w8a8_gemm_rdna2 writes fp16 or bf16");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == out_dtype && bias->numel() == N &&
                    bias->is_contiguous(),
                "w8a8_gemm_rdna2 bias must be a contiguous [N] out_dtype");
  }

  auto c = torch::empty({M, N}, a.options().dtype(out_dtype));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 128 x 256 tiles move ~25% fewer bytes per dot4 than 128 x 128, which
  // matters at the power cap, but only pay off with a few tiles per CU.
  const int cus = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long wide_tiles = (long)((M + BM - 1) / BM) * ((N + 255) / 256);
  const int TN = wide_tiles >= 4L * cus ? 16 : 8;
  const int num_tiles = ((M + BM - 1) / BM) * ((N + 16 * TN - 1) / (16 * TN));
  const int sa_stride = scale_a.numel() == 1 ? 0 : 1;
  const int sb_stride = scale_b.numel() == 1 ? 0 : 1;

#define VLLM_W8A8_GEMM_LAUNCH(TOUT, TNN)                                     \
  gemm_w8a8_rdna2_kernel<TOUT, TNN><<<num_tiles, THREADS, 0, stream>>>(      \
      a.data_ptr<int8_t>(), w.data_ptr<int8_t>(), scale_a.data_ptr<float>(), \
      scale_b.data_ptr<float>(),                                             \
      bias.has_value() ? (const TOUT*)bias->data_ptr() : nullptr,            \
      (TOUT*)c.data_ptr(), M, N, K, a.stride(0), w.stride(0), c.stride(0),   \
      sa_stride, sb_stride)
#define VLLM_W8A8_GEMM_BY_TN(TOUT)   \
  if (TN == 16)                      \
    VLLM_W8A8_GEMM_LAUNCH(TOUT, 16); \
  else                               \
    VLLM_W8A8_GEMM_LAUNCH(TOUT, 8);

  if (out_dtype == torch::kFloat16) {
    VLLM_W8A8_GEMM_BY_TN(__half);
  } else {
    VLLM_W8A8_GEMM_BY_TN(__hip_bfloat16);
  }

#undef VLLM_W8A8_GEMM_BY_TN
#undef VLLM_W8A8_GEMM_LAUNCH
  return c;
}
