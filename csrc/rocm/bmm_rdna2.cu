// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) batched GEMV: out[n, b, :] = x[n, b, :] @ w[n] for a few
// rows b (MLA's per-head absorption products q_nope @ W_UK^T and o @ W_UV in
// decode, where rocBLAS' 128 x 128 tiles stream the weights at a quarter of
// the bandwidth).
//
// A workgroup (256 threads) owns 256 output columns of one head: 32 lanes
// cover them 8 at a time (one 16-byte load per weight row, so a wave reads 512
// contiguous bytes), 8 waves split K, and the partial sums meet in LDS. x[n]
// is staged in LDS as fp32; accumulation is fp32.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace bmm_rdna2 {

static constexpr int THREADS = 256;
static constexpr int CB = 256;           // output columns per workgroup
static constexpr int KS = THREADS / 32;  // K slices (waves)
static constexpr int MAX_B = 8;
static constexpr int MAX_K = 512;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int B>
__global__ void __launch_bounds__(THREADS)
    bmm_skinny_rdna2_kernel(const __half* __restrict__ x,
                            const __half* __restrict__ w,
                            __half* __restrict__ out, const int K, const int C,
                            const long x_sn, const long x_sb, const long o_sn,
                            const long o_sb) {
  __shared__ float sx[B][MAX_K];
  __shared__ float4 sred[KS][32][2];
  const int n = blockIdx.x, c0 = blockIdx.y * CB, tid = threadIdx.x;
  const int lane = tid % 32, ks = tid / 32;
  for (int i = tid; i < B * K; i += THREADS) {
    const int b = i / K, k = i % K;
    sx[b][k] = __half2float(x[n * x_sn + b * x_sb + k]);
  }
  __syncthreads();
  const int c = c0 + lane * 8;
  const bool active = c < C;
  const int kper = (K + KS - 1) / KS, kb = ks * kper, ke = min(K, kb + kper);
  const __half* wn = w + ((long)n * K) * C + c;
  float acc[B][8];
  #pragma unroll
  for (int b = 0; b < B; b++)
  #pragma unroll
    for (int e = 0; e < 8; e++) acc[b][e] = 0.f;
  if (active) {
  #pragma unroll 4
    for (int k = kb; k < ke; k++) {
      const uint4 raw = *reinterpret_cast<const uint4*>(wn + (long)k * C);
      const __half2* h = reinterpret_cast<const __half2*>(&raw);
      float wv[8];
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        const float2 f = __half22float2(h[e]);
        wv[2 * e] = f.x;
        wv[2 * e + 1] = f.y;
      }
  #pragma unroll
      for (int b = 0; b < B; b++) {
        const float xv = sx[b][k];
  #pragma unroll
        for (int e = 0; e < 8; e++) acc[b][e] += xv * wv[e];
      }
    }
  }
  // One row at a time through LDS: the K slices' partial sums meet in wave 0.
  #pragma unroll
  for (int b = 0; b < B; b++) {
    sred[ks][lane][0] = make_float4(acc[b][0], acc[b][1], acc[b][2], acc[b][3]);
    sred[ks][lane][1] = make_float4(acc[b][4], acc[b][5], acc[b][6], acc[b][7]);
    __syncthreads();
    if (ks == 0 && active) {
      float r[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  #pragma unroll
      for (int s = 0; s < KS; s++) {
        const float4 u = sred[s][lane][0], v = sred[s][lane][1];
        r[0] += u.x;
        r[1] += u.y;
        r[2] += u.z;
        r[3] += u.w;
        r[4] += v.x;
        r[5] += v.y;
        r[6] += v.z;
        r[7] += v.w;
      }
      __half2 o[4];
  #pragma unroll
      for (int e = 0; e < 4; e++)
        o[e] = __floats2half2_rn(r[2 * e], r[2 * e + 1]);
      *reinterpret_cast<uint4*>(out + n * o_sn + b * o_sb + c) =
          *reinterpret_cast<const uint4*>(o);
    }
    __syncthreads();
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int B>
__global__ void bmm_skinny_rdna2_kernel(const __half*, const __half*, __half*,
                                        const int, const int, const long,
                                        const long, const long, const long) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace bmm_rdna2
}  // namespace vllm

// x [N, B, K] fp16 (last dim contiguous, any N / B strides), w [N, K, C] fp16
// contiguous, out [N, B, C] fp16 (last dim contiguous). B <= 8, K <= 512, C
// a multiple of 8.
void bmm_skinny_rdna2(const torch::Tensor& x, const torch::Tensor& w,
                      torch::Tensor& out) {
  using namespace vllm::bmm_rdna2;
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf &&
                  out.scalar_type() == at::kHalf,
              "bmm_skinny_rdna2: fp16 tensors");
  TORCH_CHECK(x.dim() == 3 && w.dim() == 3 && out.dim() == 3 &&
                  x.stride(2) == 1 && out.stride(2) == 1 && w.is_contiguous(),
              "bmm_skinny_rdna2: x [N, B, K], out [N, B, C] with contiguous "
              "rows, contiguous w [N, K, C]");
  const int N = x.size(0), B = x.size(1), K = x.size(2), C = w.size(2);
  TORCH_CHECK(w.size(0) == N && w.size(1) == K && out.size(0) == N &&
                  out.size(1) == B && out.size(2) == C,
              "bmm_skinny_rdna2: shape mismatch");
  TORCH_CHECK(B <= MAX_B && K <= MAX_K && C % 8 == 0 &&
                  out.stride(1) % 8 == 0 && out.stride(0) % 8 == 0,
              "bmm_skinny_rdna2: B <= 8, K <= 512, C and out strides % 8");
  if (N == 0 || B == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(N, (C + CB - 1) / CB);
#define VLLM_BMM_CASE(BB)                                         \
  case BB:                                                        \
    bmm_skinny_rdna2_kernel<BB><<<grid, THREADS, 0, stream>>>(    \
        (const __half*)x.data_ptr(), (const __half*)w.data_ptr(), \
        (__half*)out.data_ptr(), K, C, x.stride(0), x.stride(1),  \
        out.stride(0), out.stride(1));                            \
    break;
  switch (B) {
    VLLM_BMM_CASE(1)
    VLLM_BMM_CASE(2)
    VLLM_BMM_CASE(3)
    VLLM_BMM_CASE(4)
    VLLM_BMM_CASE(5)
    VLLM_BMM_CASE(6)
    VLLM_BMM_CASE(7)
    VLLM_BMM_CASE(8)
  }
#undef VLLM_BMM_CASE
}
