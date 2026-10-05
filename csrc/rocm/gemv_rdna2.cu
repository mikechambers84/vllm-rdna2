// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) fp16/bf16 GEMV for 1-8 tokens: C[M,N] = A[M,K] . W[N,K]^T
// (+ bias). Each wave streams one weight row with 16-byte loads, four in
// flight per lane; A is staged in LDS (in K chunks when M x K exceeds it);
// fp32 accumulation, wave reduction.
// fp16 uses v_dot2_f32_f16; bf16, which has no dot instruction on gfx1030,
// widens to fp32 FMAs (free while the kernel is bandwidth-bound).
// gemv_w8a16_rdna2 is the same GEMV on int8 weights with a per-row fp32
// scale: each 16-byte load carries 16 weights, converted exactly in registers.
// gemv_fp8_rdna2 does the same for fp8 e4m3fn weights (gfx1030 has no fp8
// instructions) with per-row or 2D block scales, and dequant_fp8_rdna2
// expands such weights to fp16/bf16 for the GEMM path of larger batches.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "rdna2_fp8.cuh"

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

// acc + dot(a[0:16], w[0:16]) for 16 int8 weights (one 16-byte load) and 16
// fp16 or bf16 activations at a. fp16: flip the sign bit, OR the byte into the
// mantissa of 1024.0 and subtract 1152 (exact), then v_dot2_f32_f16. bf16: the
// same trick into the mantissa of 2^23, then fp32 FMAs.
template <typename T>
__device__ __forceinline__ float dot16_i8(const int4& wv, const T* a,
                                          float acc);

template <>
__device__ __forceinline__ float dot16_i8<__half>(const int4& wv,
                                                  const __half* a, float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  const half2 bias =
      __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int w = 0; w < 2; w++) {
      const uint32_t u = q[c * 2 + w] ^ 0x80808080u;
      uint32_t lo = (u & 0xFFu) | ((u & 0xFF00u) << 8) | 0x64006400u;
      uint32_t hi = ((u >> 16) & 0xFFu) | ((u >> 8) & 0xFF0000u) | 0x64006400u;
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w], __hsub2(*reinterpret_cast<half2*>(&lo), bias), acc, false);
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w + 1], __hsub2(*reinterpret_cast<half2*>(&hi), bias), acc,
          false);
    }
  }
  return acc;
}

template <>
__device__ __forceinline__ float dot16_i8<__hip_bfloat16>(
    const int4& wv, const __hip_bfloat16* a, float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const uint32_t* au = reinterpret_cast<const uint32_t*>(&av);
  #pragma unroll
    for (int w = 0; w < 2; w++) {
      const uint32_t u = q[c * 2 + w] ^ 0x80808080u;
  #pragma unroll
      for (int b = 0; b < 4; b++) {
        const float wf =
            __uint_as_float(((u >> (8 * b)) & 0xFFu) | 0x4B000000u) - 8388736.f;
        const uint32_t ab = au[2 * w + b / 2];
        const float af = __uint_as_float(b % 2 ? ab & 0xffff0000u : ab << 16);
        acc = fmaf(af, wf, acc);
      }
    }
  }
  return acc;
}

using rdna2::FP8_HI;
using rdna2::FP8_LO;
using rdna2::fp8x2_to_half2;

// acc + 2^-8 * dot(a[0:16], w[0:16]) for 16 fp8 e4m3fn weights; fp16 uses
// v_dot2_f32_f16, bf16 widens both sides to fp32.
template <typename T>
__device__ __forceinline__ float dot16_fp8(const int4& wv, const T* a,
                                           float acc);

template <>
__device__ __forceinline__ float dot16_fp8<__half>(const int4& wv,
                                                   const __half* a, float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int w = 0; w < 2; w++) {
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w], fp8x2_to_half2(q[c * 2 + w], FP8_LO), acc, false);
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w + 1], fp8x2_to_half2(q[c * 2 + w], FP8_HI), acc, false);
    }
  }
  return acc;
}

template <>
__device__ __forceinline__ float dot16_fp8<__hip_bfloat16>(
    const int4& wv, const __hip_bfloat16* a, float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const uint32_t* au = reinterpret_cast<const uint32_t*>(&av);
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      const float2 wf = __half22float2(
          fp8x2_to_half2(q[c * 2 + j / 2], j % 2 ? FP8_HI : FP8_LO));
      acc = fmaf(__uint_as_float(au[j] << 16), wf.x, acc);
      acc = fmaf(__uint_as_float(au[j] & 0xffff0000u), wf.y, acc);
    }
  }
  return acc;
}

// 8-bit weight formats: int8 or fp8 with one scale per block_n rows, or fp8
// with 2D [block_n, block_k] block scales.
enum class W8 { kInt8, kFp8, kFp8Block };

template <W8 F, typename T>
__device__ __forceinline__ float dot16_w8(const int4& wv, const T* a,
                                          float acc) {
  if constexpr (F == W8::kInt8)
    return dot16_i8<T>(wv, a, acc);
  else
    return dot16_fp8<T>(wv, a, acc);
}

// acc[i] += a_i . w for all M tokens; block-scaled weights scale each
// 16-weight partial by its block's scale s.
template <W8 F, typename T, int M>
__device__ __forceinline__ void mac16(float (&acc)[M], const int4& w,
                                      const T* a, const int lda,
                                      const float s) {
  #pragma unroll
  for (int i = 0; i < M; i++) {
    if constexpr (F == W8::kFp8Block)
      acc[i] = fmaf(dot16_w8<F, T>(w, a + i * lda, 0.f), s, acc[i]);
    else
      acc[i] = dot16_w8<F, T>(w, a + i * lda, acc[i]);
  }
}

// One wave computes one output row (8-bit weights) for all M tokens; same
// structure as gemv_rdna2_kernel. Row n's scales start at
// scale[(n / bn) * lds_s]; block k-indices are k >> bk_shift.
template <typename T, int M, W8 F>
__global__ void __launch_bounds__(WAVES* WARP32)
    gemv_w8_rdna2_kernel(const int8_t* __restrict__ W,
                         const float* __restrict__ scale,
                         const T* __restrict__ A, const T* __restrict__ bias,
                         T* __restrict__ C, const int N, const int K,
                         const int kc_len, const int lda, const int ldw,
                         const int ldc, const int bn, const int bk_shift,
                         const int lds_s) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* sA = reinterpret_cast<T*>(smem);  // [M][kc_len]
  const int tid = threadIdx.x;
  const int lane = tid % WARP32;
  const int n = blockIdx.x * WAVES + tid / WARP32;
  constexpr int STEP = WARP32 * 16;  // K elements per wave-wide 16-byte load
  const float* srow = scale + (long)(min(n, N - 1) / bn) * lds_s;
  auto block_scale = [&](int k) {
    return F == W8::kFp8Block ? srow[k >> bk_shift] : 0.f;
  };

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
    if (n >= N) continue;  // wave-uniform

    const int8_t* wrow = W + (long)n * ldw + kc;
    int k = lane * 16;
    // Up to eight loads (128 bytes per lane) in flight, as in the fp16 GEMV.
    if constexpr (M <= 4)
      for (; k + 7 * STEP < klen; k += 8 * STEP) {
        int4 w[8];
  #pragma unroll
        for (int u = 0; u < 8; u++)
          w[u] = *reinterpret_cast<const int4*>(wrow + k + u * STEP);
  #pragma unroll
        for (int u = 0; u < 8; u++)
          mac16<F, T, M>(acc, w[u], sA + k + u * STEP, kc_len,
                         block_scale(kc + k + u * STEP));
      }
    if constexpr (M <= 4)
      for (; k + 3 * STEP < klen; k += 4 * STEP) {
        int4 w[4];
  #pragma unroll
        for (int u = 0; u < 4; u++)
          w[u] = *reinterpret_cast<const int4*>(wrow + k + u * STEP);
  #pragma unroll
        for (int u = 0; u < 4; u++)
          mac16<F, T, M>(acc, w[u], sA + k + u * STEP, kc_len,
                         block_scale(kc + k + u * STEP));
      }
    for (; k + STEP < klen; k += 2 * STEP) {
      const int4 w0 = *reinterpret_cast<const int4*>(wrow + k);
      const int4 w1 = *reinterpret_cast<const int4*>(wrow + k + STEP);
      mac16<F, T, M>(acc, w0, sA + k, kc_len, block_scale(kc + k));
      mac16<F, T, M>(acc, w1, sA + k + STEP, kc_len,
                     block_scale(kc + k + STEP));
    }
    for (; k < klen; k += STEP) {
      const int4 w0 = *reinterpret_cast<const int4*>(wrow + k);
      mac16<F, T, M>(acc, w0, sA + k, kc_len, block_scale(kc + k));
    }
  }

  #pragma unroll
  for (int i = 0; i < M; i++)
  #pragma unroll
    for (int mask = WARP32 / 2; mask >= 1; mask >>= 1)
      acc[i] += __shfl_xor(acc[i], mask);

  if (lane == 0 && n < N) {
    // fp8 partials carry the 2^-8 of the in-register conversion.
    float s = F == W8::kInt8 ? 1.f : 256.f;
    if constexpr (F != W8::kFp8Block) s *= srow[0];
    const float b = bias ? to_float<T>(bias[n]) : 0.f;
  #pragma unroll
    for (int i = 0; i < M; i++)
      C[(long)i * ldc + n] = from_float<T>(acc[i] * s + b);
  }
}

// out = W * scale for fp8 e4m3fn W [N, K]; each thread expands 16 weights.
// Scales are indexed [(n / bn) * lds_s + (k >> bk_shift)].
template <typename T>
__global__ void __launch_bounds__(256)
    dequant_fp8_rdna2_kernel(const int8_t* __restrict__ W,
                             const float* __restrict__ scale,
                             T* __restrict__ out, const int N, const int K,
                             const int bn, const int bk_shift,
                             const int lds_s) {
  const int kchunks = K / 16;
  const long idx = (long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= (long)N * kchunks) return;
  const int n = idx / kchunks;
  const int k = (idx - (long)n * kchunks) * 16;
  const int4 wv = *reinterpret_cast<const int4*>(W + (long)n * K + k);
  const float s = scale[(long)(n / bn) * lds_s + (k >> bk_shift)] * 256.f;
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  __align__(16) T o[16];
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const float2 f =
        __half22float2(fp8x2_to_half2(q[j / 2], j % 2 ? FP8_HI : FP8_LO));
    o[2 * j] = from_float<T>(f.x * s);
    o[2 * j + 1] = from_float<T>(f.y * s);
  }
  int4* dst = reinterpret_cast<int4*>(out + (long)n * K + k);
  dst[0] = reinterpret_cast<const int4*>(o)[0];
  dst[1] = reinterpret_cast<const int4*>(o)[1];
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, int M>
__global__ void gemv_rdna2_kernel(const T*, const T*, const T*, T*, const int,
                                  const int, const int, const int, const int,
                                  const int) {}
enum class W8 { kInt8, kFp8, kFp8Block };
template <typename T, int M, W8 F>
__global__ void gemv_w8_rdna2_kernel(const int8_t*, const float*, const T*,
                                     const T*, T*, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int) {}
template <typename T>
__global__ void dequant_fp8_rdna2_kernel(const int8_t*, const float*, T*,
                                         const int, const int, const int,
                                         const int, const int) {}

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

namespace {

// GEMV on 8-bit weights (validated by the callers): w [N, K] of bytes; row
// n's scales start at scale[(n / bn) * lds_s], block k-indices are
// k >> bk_shift.
template <vllm::gemv_rdna2::W8 F>
torch::Tensor gemv_w8(const char* name, const at::Tensor& a,
                      const at::Tensor& w, const at::Tensor& scale, int bn,
                      int bk_shift, int lds_s,
                      const std::optional<at::Tensor>& bias) {
  using namespace vllm::gemv_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16,
              name, " needs fp16 or bf16 activations");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1), name,
              " needs a [M, K] and w [N, K]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= MAX_M, name, " supports M in [1, ", MAX_M, "]");
  TORCH_CHECK(K % 16 == 0 && a.stride(1) == 1 && w.stride(1) == 1 &&
                  a.stride(0) % 8 == 0 && w.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              name, " needs 16-byte aligned, K-contiguous rows");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == a.dtype() && bias->numel() == N &&
                    bias->is_contiguous(),
                name, " bias must be a contiguous [N] of a's dtype");
  }

  auto c = torch::empty({M, N}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int elem = a.element_size();
  int kc_len = K;
  if ((size_t)M * K * elem > (size_t)LDS_BYTES)
    kc_len = LDS_BYTES / (M * elem) / 512 * 512;
  const size_t lds = (size_t)M * kc_len * elem;
  const dim3 grid((N + WAVES - 1) / WAVES);
  const dim3 block(WAVES * WARP32);

#define VLLM_GEMV_W8_CASE(T, MM)                                               \
  case MM:                                                                     \
    gemv_w8_rdna2_kernel<T, MM, F><<<grid, block, lds, stream>>>(              \
        (const int8_t*)w.data_ptr(), scale.data_ptr<float>(),                  \
        (const T*)a.data_ptr(),                                                \
        bias.has_value() ? (const T*)bias->data_ptr() : nullptr,               \
        (T*)c.data_ptr(), N, K, kc_len, a.stride(0), w.stride(0), c.stride(0), \
        bn, bk_shift, lds_s);                                                  \
    break;
#define VLLM_GEMV_W8_BY_M(T) \
  switch (M) {               \
    VLLM_GEMV_W8_CASE(T, 1)  \
    VLLM_GEMV_W8_CASE(T, 2)  \
    VLLM_GEMV_W8_CASE(T, 3)  \
    VLLM_GEMV_W8_CASE(T, 4)  \
    VLLM_GEMV_W8_CASE(T, 5)  \
    VLLM_GEMV_W8_CASE(T, 6)  \
    VLLM_GEMV_W8_CASE(T, 7)  \
    VLLM_GEMV_W8_CASE(T, 8)  \
  }

  if (a.dtype() == torch::kFloat16) {
    VLLM_GEMV_W8_BY_M(__half);
  } else {
    VLLM_GEMV_W8_BY_M(__hip_bfloat16);
  }

#undef VLLM_GEMV_W8_BY_M
#undef VLLM_GEMV_W8_CASE
  return c;
}

// Checks fp8 weights w [N, K] and their scales: fp32 [ceil(N / block_n), S]
// with S == 1 (one scale per block_n rows) or S == ceil(K / block_k) (2D
// blocks; block_k a power of two and a multiple of 16). Returns the shift
// mapping k to its scale column (31: per-row scales).
int check_fp8_weight(const char* name, const at::Tensor& w,
                     const at::Tensor& scale, int64_t block_n,
                     int64_t block_k) {
  TORCH_CHECK(w.dtype() == at::ScalarType::Float8_e4m3fn, name,
              " needs float8_e4m3fn weights");
  TORCH_CHECK(scale.dtype() == torch::kFloat32 && scale.dim() == 2 &&
                  scale.is_contiguous(),
              name, " needs contiguous 2D fp32 scales");
  const int64_t N = w.size(0), K = w.size(1);
  TORCH_CHECK(block_n >= 1 && scale.size(0) == (N + block_n - 1) / block_n,
              name, " needs ceil(N / block_n) scale rows");
  if (scale.size(1) == 1) return 31;
  TORCH_CHECK(block_k >= 16 && (block_k & (block_k - 1)) == 0 &&
                  scale.size(1) == (K + block_k - 1) / block_k,
              name,
              " needs block_k a power of two >= 16 and ceil(K / block_k) "
              "scale columns");
  return __builtin_ctzll(block_k);
}

}  // namespace

// Requirements: a [M, K] fp16 or bf16 with M in [1, 8]; w [N, K] int8 and
// scale [N] fp32 (W = w * scale per row); rows contiguous and 16-byte aligned
// (K % 16 == 0); optional contiguous bias [N] of a's dtype. Returns [M, N].
torch::Tensor gemv_w8a16_rdna2(const at::Tensor& a, const at::Tensor& w,
                               const at::Tensor& scale,
                               const std::optional<at::Tensor>& bias) {
  using vllm::gemv_rdna2::W8;
  TORCH_CHECK(w.dtype() == torch::kInt8 && scale.dtype() == torch::kFloat32 &&
                  scale.is_contiguous() && scale.numel() == w.size(0),
              "gemv_w8a16_rdna2 needs int8 weights and one contiguous fp32 "
              "scale per row");
  return gemv_w8<W8::kInt8>("gemv_w8a16_rdna2", a, w, scale, 1, 31, 1, bias);
}

// Requirements: a [M, K] fp16 or bf16 with M in [1, 8]; w [N, K]
// float8_e4m3fn with scales as in check_fp8_weight (W = w * scale); rows
// contiguous and 16-byte aligned (K % 16 == 0); optional contiguous bias [N]
// of a's dtype. Returns [M, N].
torch::Tensor gemv_fp8_rdna2(const at::Tensor& a, const at::Tensor& w,
                             const at::Tensor& scale, int64_t block_n,
                             int64_t block_k,
                             const std::optional<at::Tensor>& bias) {
  using vllm::gemv_rdna2::W8;
  const int shift =
      check_fp8_weight("gemv_fp8_rdna2", w, scale, block_n, block_k);
  if (scale.size(1) == 1)
    return gemv_w8<W8::kFp8>("gemv_fp8_rdna2", a, w, scale, block_n, 31, 1,
                             bias);
  return gemv_w8<W8::kFp8Block>("gemv_fp8_rdna2", a, w, scale, block_n, shift,
                                scale.size(1), bias);
}

// out [N, K] (fp16 or bf16, contiguous) = w * scale for float8_e4m3fn w
// [N, K] (contiguous, K % 16 == 0) with scales as in check_fp8_weight.
void dequant_fp8_rdna2(torch::Tensor& out, const at::Tensor& w,
                       const at::Tensor& scale, int64_t block_n,
                       int64_t block_k) {
  using namespace vllm::gemv_rdna2;
  const int shift =
      check_fp8_weight("dequant_fp8_rdna2", w, scale, block_n, block_k);
  const int N = w.size(0), K = w.size(1);
  TORCH_CHECK(K % 16 == 0 && w.is_contiguous() && out.is_contiguous() &&
                  out.size(0) == N && out.size(1) == K &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "dequant_fp8_rdna2 needs contiguous, 16-byte aligned w and out "
              "of the same [N, K] shape with K % 16 == 0");
  TORCH_CHECK(out.dtype() == torch::kFloat16 || out.dtype() == torch::kBFloat16,
              "dequant_fp8_rdna2 writes fp16 or bf16");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const long threads = (long)N * (K / 16);
  const dim3 grid((threads + 255) / 256);
  const int lds_s = scale.size(1);
  if (out.dtype() == torch::kFloat16) {
    dequant_fp8_rdna2_kernel<__half><<<grid, 256, 0, stream>>>(
        (const int8_t*)w.data_ptr(), scale.data_ptr<float>(),
        (__half*)out.data_ptr(), N, K, block_n, shift, lds_s);
  } else {
    dequant_fp8_rdna2_kernel<__hip_bfloat16><<<grid, 256, 0, stream>>>(
        (const int8_t*)w.data_ptr(), scale.data_ptr<float>(),
        (__hip_bfloat16*)out.data_ptr(), N, K, block_n, shift, lds_s);
  }
}
