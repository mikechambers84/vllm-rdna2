// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) fp16/bf16 GEMV for 1-8 tokens: C[M,N] = A[M,K] . W[N,K]^T
// (+ bias). Each wave streams one weight row with 16-byte loads, four in
// flight per lane; A is staged in LDS (in K chunks when M x K exceeds it);
// fp32 accumulation, wave reduction.
// fp16 uses v_dot2_f32_f16; bf16, which has no dot instruction on gfx1030,
// widens to fp32 FMAs (free while the kernel is bandwidth-bound).

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
namespace gemv_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int WAVES = 8;
// LDS for the staged activations per workgroup: 16 KB keeps 4 workgroups per
// CU in flight (a 64 KB stage leaves one, and 8 tokens or long rows then go
// latency-bound); longer A is staged in K chunks.
static constexpr int LDS_BYTES = 16 * 1024;
static constexpr int MAX_M = 8;

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

// acc += dot(a[0:8], w[0:8]) for 8 packed fp16 or bf16 values in an int4.
template <typename T>
__device__ __forceinline__ float dot8(const int4& a, const int4& w, float acc);

template <>
__device__ __forceinline__ float dot8<__half>(const int4& a, const int4& w,
                                              float acc) {
  const half2* ah = reinterpret_cast<const half2*>(&a);
  const half2* wh = reinterpret_cast<const half2*>(&w);
  #pragma unroll
  for (int j = 0; j < 4; j++)
    acc = __builtin_amdgcn_fdot2(ah[j], wh[j], acc, false);
  return acc;
}

template <>
__device__ __forceinline__ float dot8<__hip_bfloat16>(const int4& a,
                                                      const int4& w,
                                                      float acc) {
  const uint32_t* au = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* wu = reinterpret_cast<const uint32_t*>(&w);
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    acc = fmaf(__uint_as_float(au[j] << 16), __uint_as_float(wu[j] << 16), acc);
    acc = fmaf(__uint_as_float(au[j] & 0xffff0000u),
               __uint_as_float(wu[j] & 0xffff0000u), acc);
  }
  return acc;
}

// One wave computes one output row for all M tokens. kc_len is the number of
// K elements per LDS chunk (K itself when A fits).
template <typename T, int M>
__global__ void __launch_bounds__(WAVES* WARP32)
    gemv_rdna2_kernel(const T* __restrict__ W, const T* __restrict__ A,
                      const T* __restrict__ bias, T* __restrict__ C,
                      const int N, const int K, const int kc_len, const int lda,
                      const int ldw, const int ldc) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* sA = reinterpret_cast<T*>(smem);  // [M][kc_len]
  const int tid = threadIdx.x;
  const int lane = tid % WARP32;
  const int n = blockIdx.x * WAVES + tid / WARP32;
  constexpr int STEP = WARP32 * 8;  // K elements per wave-wide 16-byte load

  float acc[M];
  #pragma unroll
  for (int i = 0; i < M; i++) acc[i] = 0.f;

  for (int kc = 0; kc < K; kc += kc_len) {
    const int klen = min(kc_len, K - kc);
    if (kc > 0) __syncthreads();
    for (int i = 0; i < M; i++)
      for (int k = tid * 8; k < klen; k += blockDim.x * 8)
        *reinterpret_cast<int4*>(sA + i * kc_len + k) =
            *reinterpret_cast<const int4*>(A + (long)i * lda + kc + k);
    __syncthreads();
    if (n >= N) continue;  // wave-uniform: keep the loads unpredicated

    const T* wrow = W + (long)n * ldw + kc;
    int k = lane * 8;
    // Issue eight (then four) loads before using any, so each wave keeps up to
    // 128 bytes per lane in flight; short rows (K = 2048-4096) are otherwise
    // latency-bound at M = 1. Above 4 tokens the eight-load batch costs more
    // in registers than it gains.
    if constexpr (M <= 4)
      for (; k + 7 * STEP < klen; k += 8 * STEP) {
        int4 w[8];
  #pragma unroll
        for (int u = 0; u < 8; u++)
          w[u] = *reinterpret_cast<const int4*>(wrow + k + u * STEP);
  #pragma unroll
        for (int u = 0; u < 8; u++)
  #pragma unroll
          for (int i = 0; i < M; i++)
            acc[i] = dot8<T>(
                *reinterpret_cast<const int4*>(sA + i * kc_len + k + u * STEP),
                w[u], acc[i]);
      }
    for (; k + 3 * STEP < klen; k += 4 * STEP) {
      int4 w[4];
  #pragma unroll
      for (int u = 0; u < 4; u++)
        w[u] = *reinterpret_cast<const int4*>(wrow + k + u * STEP);
  #pragma unroll
      for (int u = 0; u < 4; u++)
  #pragma unroll
        for (int i = 0; i < M; i++)
          acc[i] = dot8<T>(
              *reinterpret_cast<const int4*>(sA + i * kc_len + k + u * STEP),
              w[u], acc[i]);
    }
    for (; k < klen; k += STEP) {
      const int4 w = *reinterpret_cast<const int4*>(wrow + k);
  #pragma unroll
      for (int i = 0; i < M; i++)
        acc[i] = dot8<T>(*reinterpret_cast<const int4*>(sA + i * kc_len + k), w,
                         acc[i]);
    }
  }

  #pragma unroll
  for (int i = 0; i < M; i++)
  #pragma unroll
    for (int mask = WARP32 / 2; mask >= 1; mask >>= 1)
      acc[i] += __shfl_xor(acc[i], mask);

  if (lane == 0 && n < N) {
    const float b = bias ? to_float<T>(bias[n]) : 0.f;
  #pragma unroll
    for (int i = 0; i < M; i++)
      C[(long)i * ldc + n] = from_float<T>(acc[i] + b);
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, int M>
__global__ void gemv_rdna2_kernel(const T*, const T*, const T*, T*, const int,
                                  const int, const int, const int, const int,
                                  const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gemv_rdna2
}  // namespace vllm

// Requirements: a [M, K] with M in [1, 8] and w [N, K], fp16 or bf16 (same
// dtype), both with contiguous, 16-byte aligned rows (K % 8 == 0); optional
// contiguous bias [N]. Returns [M, N] in the input dtype.
torch::Tensor gemv_rdna2(const at::Tensor& a, const at::Tensor& w,
                         const std::optional<at::Tensor>& bias) {
  using namespace vllm::gemv_rdna2;
  TORCH_CHECK(a.dtype() == w.dtype() && (a.dtype() == torch::kFloat16 ||
                                         a.dtype() == torch::kBFloat16),
              "gemv_rdna2 needs fp16 or bf16 a and w of the same dtype");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1),
              "gemv_rdna2 needs a [M, K] and w [N, K]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_M, "gemv_rdna2 supports M in [1, ", MAX_M,
              "]");
  TORCH_CHECK(K % 8 == 0 && a.stride(1) == 1 && w.stride(1) == 1 &&
                  a.stride(0) % 8 == 0 && w.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "gemv_rdna2 needs 16-byte aligned, K-contiguous rows");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == a.dtype() && bias->numel() == N &&
                    bias->is_contiguous(),
                "gemv_rdna2 bias must be a contiguous [N] of the input dtype");
  }

  auto c = torch::empty({M, N}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // A in LDS: all of it when it fits, else K chunks (multiples of 256
  // elements, one wave-wide 16-byte load per lane) sized to the LDS budget.
  const int elem = a.element_size();
  int kc_len = K;
  if ((size_t)M * K * elem > (size_t)LDS_BYTES)
    kc_len = LDS_BYTES / (M * elem) / 256 * 256;
  const size_t lds = (size_t)M * kc_len * elem;
  const dim3 grid((N + WAVES - 1) / WAVES);
  const dim3 block(WAVES * WARP32);

#define VLLM_GEMV_RDNA2_CASE(T, MM)                               \
  case MM:                                                        \
    gemv_rdna2_kernel<T, MM><<<grid, block, lds, stream>>>(       \
        (const T*)w.data_ptr(), (const T*)a.data_ptr(),           \
        bias.has_value() ? (const T*)bias->data_ptr() : nullptr,  \
        (T*)c.data_ptr(), N, K, kc_len, a.stride(0), w.stride(0), \
        c.stride(0));                                             \
    break;
#define VLLM_GEMV_RDNA2_BY_M(T) \
  switch (M) {                  \
    VLLM_GEMV_RDNA2_CASE(T, 1)  \
    VLLM_GEMV_RDNA2_CASE(T, 2)  \
    VLLM_GEMV_RDNA2_CASE(T, 3)  \
    VLLM_GEMV_RDNA2_CASE(T, 4)  \
    VLLM_GEMV_RDNA2_CASE(T, 5)  \
    VLLM_GEMV_RDNA2_CASE(T, 6)  \
    VLLM_GEMV_RDNA2_CASE(T, 7)  \
    VLLM_GEMV_RDNA2_CASE(T, 8)  \
  }

  if (a.dtype() == torch::kFloat16) {
    VLLM_GEMV_RDNA2_BY_M(__half);
  } else {
    VLLM_GEMV_RDNA2_BY_M(__hip_bfloat16);
  }

#undef VLLM_GEMV_RDNA2_BY_M
#undef VLLM_GEMV_RDNA2_CASE
  return c;
}
