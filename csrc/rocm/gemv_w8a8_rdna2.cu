// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) W8A8 GEMV for decode: C[M,N] = (A[M,K] . W[N,K]^T) * sa * sb
// (+ bias), int8 A and W with int32 accumulation through v_dot4_i32_i8, fp32
// per-token (or per-tensor) sa and per-channel (or per-tensor) sb. Each wave
// streams R weight rows with 16-byte loads; A is staged in LDS, in K chunks
// when M x K exceeds it, so it is read from global once per workgroup. Up to
// 24 tokens (batched decode, speculative-decoding verification).

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
namespace gemv_w8a8_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int WAVES = 8;  // 256 threads per workgroup
static constexpr int LDS_BYTES = 64 * 1024;
static constexpr int MAX_M = 24;

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

// One wave computes R consecutive output channels for all M tokens, of which
// the first m are real (M is padded above 8 to limit instantiations). kc_len
// is the number of K bytes per LDS chunk (K itself when A fits).
template <typename TOut, int M, int R>
__global__ void __launch_bounds__(WAVES* WARP32) gemv_w8a8_rdna2_kernel(
    const int8_t* __restrict__ W, const int8_t* __restrict__ A,
    const float* __restrict__ sa, const float* __restrict__ sb,
    const TOut* __restrict__ bias, TOut* __restrict__ C, const int m,
    const int N, const int K, const int kc_len, const int lda, const int ldw,
    const int ldc, const int sa_stride, const int sb_stride) {
  extern __shared__ __align__(16) int8_t sA[];  // [M][kc_len]
  const int tid = threadIdx.x;
  const int lane = tid & (WARP32 - 1);
  const int wave = tid / WARP32;
  const int n0 = (blockIdx.x * WAVES + wave) * R;

  int acc[M][R];
  #pragma unroll
  for (int i = 0; i < M; i++)
  #pragma unroll
    for (int r = 0; r < R; r++) acc[i][r] = 0;

  for (int kc = 0; kc < K; kc += kc_len) {
    const int klen = min(kc_len, K - kc);
    if (kc > 0) __syncthreads();
    for (int i = 0; i < M; i++)
      for (int k = tid * 16; k < klen; k += blockDim.x * 16)
        *reinterpret_cast<int4*>(sA + i * kc_len + k) =
            i < m ? *reinterpret_cast<const int4*>(A + (long)i * lda + kc + k)
                  : make_int4(0, 0, 0, 0);
    __syncthreads();

  #pragma unroll 2
    for (int k = lane * 16; k < klen; k += WARP32 * 16) {
      int4 w[R];
  #pragma unroll
      for (int r = 0; r < R; r++)
        w[r] = n0 + r < N ? *reinterpret_cast<const int4*>(
                                W + (long)(n0 + r) * ldw + kc + k)
                          : make_int4(0, 0, 0, 0);
  #pragma unroll
      for (int i = 0; i < M; i++) {
        const int4 a = *reinterpret_cast<const int4*>(sA + i * kc_len + k);
  #pragma unroll
        for (int r = 0; r < R; r++) {
          acc[i][r] = __builtin_amdgcn_sdot4(a.x, w[r].x, acc[i][r], false);
          acc[i][r] = __builtin_amdgcn_sdot4(a.y, w[r].y, acc[i][r], false);
          acc[i][r] = __builtin_amdgcn_sdot4(a.z, w[r].z, acc[i][r], false);
          acc[i][r] = __builtin_amdgcn_sdot4(a.w, w[r].w, acc[i][r], false);
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < M; i++)
  #pragma unroll
    for (int r = 0; r < R; r++)
  #pragma unroll
      for (int mask = WARP32 / 2; mask >= 1; mask >>= 1)
        acc[i][r] += __shfl_xor(acc[i][r], mask);

  if (lane == 0) {
  #pragma unroll
    for (int r = 0; r < R; r++) {
      const int n = n0 + r;
      if (n >= N) break;
      const float s = sb[n * sb_stride];
  #pragma unroll
      for (int i = 0; i < M; i++) {
        if (i < m) {
          // Round, then add the bias in the output dtype, as triton_scaled_mm
          // does, so both give identical results.
          TOut v = from_float<TOut>((float)acc[i][r] * sa[i * sa_stride] * s);
          if (bias)
            v = from_float<TOut>(to_float<TOut>(v) + to_float<TOut>(bias[n]));
          C[(long)i * ldc + n] = v;
        }
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename TOut, int M, int R>
__global__ void gemv_w8a8_rdna2_kernel(const int8_t*, const int8_t*,
                                       const float*, const float*, const TOut*,
                                       TOut*, const int, const int, const int,
                                       const int, const int, const int,
                                       const int, const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gemv_w8a8_rdna2
}  // namespace vllm

// Requirements: a int8 [M, K] with M in [1, 24] and contiguous rows; w int8
// [N, K] with contiguous rows (an int8 linear's [K, N] weight view,
// transposed); K % 16 == 0 and 16-byte aligned rows; scale_a fp32 with 1 or M
// elements, scale_b fp32 with 1 or N elements; optional bias [N] in out_dtype.
// Returns fp16 or bf16 [M, N]. Exactly matches triton_scaled_mm (int32
// accumulation).
torch::Tensor w8a8_gemv_rdna2(const at::Tensor& a, const at::Tensor& w,
                              const at::Tensor& scale_a,
                              const at::Tensor& scale_b,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType out_dtype) {
  using namespace vllm::gemv_w8a8_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8,
              "w8a8_gemv_rdna2 needs int8 a and w");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1),
              "w8a8_gemv_rdna2 needs a [M, K] and w [N, K]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_M, "w8a8_gemv_rdna2 supports M in [1, ", MAX_M,
              "]");
  TORCH_CHECK(K % 16 == 0, "w8a8_gemv_rdna2 needs K % 16 == 0");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(1) == 1 && a.stride(0) % 16 == 0 &&
                  w.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "w8a8_gemv_rdna2 needs 16-byte aligned, K-contiguous rows");
  TORCH_CHECK(scale_a.dtype() == torch::kFloat32 &&
                  scale_b.dtype() == torch::kFloat32 &&
                  scale_a.is_contiguous() && scale_b.is_contiguous(),
              "w8a8_gemv_rdna2 needs contiguous fp32 scales");
  TORCH_CHECK(scale_a.numel() == 1 || scale_a.numel() == M,
              "w8a8_gemv_rdna2 scale_a must have 1 or M elements");
  TORCH_CHECK(scale_b.numel() == 1 || scale_b.numel() == N,
              "w8a8_gemv_rdna2 scale_b must have 1 or N elements");
  TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
              "w8a8_gemv_rdna2 writes fp16 or bf16");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == out_dtype && bias->numel() == N &&
                    bias->is_contiguous(),
                "w8a8_gemv_rdna2 bias must be a contiguous [N] out_dtype");
  }

  auto c = torch::empty({M, N}, a.options().dtype(out_dtype));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Above 8 tokens M is padded to a multiple of 4 (the extra rows are zero).
  const int MP = M <= 8 ? M : (M + 3) / 4 * 4;
  // A in LDS: all of it when it fits, else K chunks (multiples of 512 bytes,
  // one wave-wide 16-byte load per lane) sized to fill the LDS.
  int kc_len = K;
  if ((size_t)MP * K > (size_t)LDS_BYTES) kc_len = LDS_BYTES / MP / 512 * 512;
  const size_t lds = (size_t)MP * kc_len;
  // More rows per wave reuse each staged A load across them; at one or two
  // tokens a single row keeps more waves in flight per weight byte.
  const int R = M <= 2 ? 1 : M <= 8 ? 2 : 4;
  const dim3 grid((N + WAVES * R - 1) / (WAVES * R));
  const dim3 block(WAVES * WARP32);
  const int sa_stride = scale_a.numel() == 1 ? 0 : 1;
  const int sb_stride = scale_b.numel() == 1 ? 0 : 1;

#define VLLM_W8A8_GEMV_LAUNCH(TOUT, MM, RR)                                  \
  gemv_w8a8_rdna2_kernel<TOUT, MM, RR><<<grid, block, lds, stream>>>(        \
      w.data_ptr<int8_t>(), a.data_ptr<int8_t>(), scale_a.data_ptr<float>(), \
      scale_b.data_ptr<float>(),                                             \
      bias.has_value() ? (const TOUT*)bias->data_ptr() : nullptr,            \
      (TOUT*)c.data_ptr(), M, N, K, kc_len, a.stride(0), w.stride(0),        \
      c.stride(0), sa_stride, sb_stride)
#define VLLM_W8A8_GEMV_CASE(TOUT, MM, RR) \
  case MM:                                \
    VLLM_W8A8_GEMV_LAUNCH(TOUT, MM, RR);  \
    break;
#define VLLM_W8A8_GEMV_BY_M(TOUT)    \
  switch (MP) {                      \
    VLLM_W8A8_GEMV_CASE(TOUT, 1, 1)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 2, 1)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 3, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 4, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 5, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 6, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 7, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 8, 2)  \
    VLLM_W8A8_GEMV_CASE(TOUT, 12, 4) \
    VLLM_W8A8_GEMV_CASE(TOUT, 16, 4) \
    VLLM_W8A8_GEMV_CASE(TOUT, 20, 4) \
    VLLM_W8A8_GEMV_CASE(TOUT, 24, 4) \
  }

  if (out_dtype == torch::kFloat16) {
    VLLM_W8A8_GEMV_BY_M(__half);
  } else {
    VLLM_W8A8_GEMV_BY_M(__hip_bfloat16);
  }

#undef VLLM_W8A8_GEMV_BY_M
#undef VLLM_W8A8_GEMV_CASE
#undef VLLM_W8A8_GEMV_LAUNCH
  return c;
}
