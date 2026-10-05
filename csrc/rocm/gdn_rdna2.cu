// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) kernels for Gated DeltaNet prefill.
//
// gdn_post_conv_rdna2: FLA's fused post-conv1d preparation (split of the conv
// output into q / k / v, l2 norm of q and k, g = -exp(A_log) *
// softplus(a + dt_bias), beta = sigmoid(b)). One 256-thread workgroup per
// token, each thread moving 8 fp16 values (16 bytes) per step along the token's
// row, so reads and writes are coalesced; a head of K dims is K / 8 adjacent
// lanes and is normalized with a lane reduction. ~425 GB/s on a V620 against
// ~80 GB/s for the Triton kernel at every launch config.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gdn_rdna2 {

static constexpr int THREADS = 256;

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

template <int K, bool L2NORM, bool G_EXP>
__global__ void __launch_bounds__(THREADS) gdn_post_conv_rdna2_kernel(
    const __half* __restrict__ x, const __half* __restrict__ a,
    const __half* __restrict__ b, const float* __restrict__ A_log,
    const void* __restrict__ dt_bias, const bool dt_f32, __half* __restrict__ q,
    __half* __restrict__ k, __half* __restrict__ v, float* __restrict__ g,
    float* __restrict__ beta, const int H, const int HV, const int V,
    const long sx, const long sa, const long sb, const float eps) {
  constexpr int LANES = K / 8;  // lanes per q/k head
  const int t = blockIdx.x, tid = threadIdx.x;
  const int HK = H * K, VD = HV * V;
  const __half* row = x + t * sx;
  for (int c = tid; c < 2 * HK / 8; c += THREADS) {
    uint4 raw = *reinterpret_cast<const uint4*>(row + c * 8);
    if constexpr (L2NORM) {
      __half2* h2 = reinterpret_cast<__half2*>(&raw);
      float f[8];
      float ss = 0.f;
  #pragma unroll
      for (int e = 0; e < 4; e++) {
        const float2 p = __half22float2(h2[e]);
        f[2 * e] = p.x;
        f[2 * e + 1] = p.y;
        ss += p.x * p.x + p.y * p.y;
      }
  #pragma unroll
      for (int o = LANES / 2; o > 0; o >>= 1) ss += __shfl_xor(ss, o, LANES);
      const float inv = 1.f / sqrtf(ss + eps);
  #pragma unroll
      for (int e = 0; e < 4; e++)
        h2[e] = __floats2half2_rn(f[2 * e] * inv, f[2 * e + 1] * inv);
    }
    const int d = c * 8;
    __half* dst = (d < HK ? q + d : k + (d - HK)) + (long)t * HK;
    *reinterpret_cast<uint4*>(dst) = raw;
  }
  for (int c = tid; c < VD / 8; c += THREADS)
    *reinterpret_cast<uint4*>(v + (long)t * VD + c * 8) =
        *reinterpret_cast<const uint4*>(row + 2 * HK + c * 8);
  for (int h = tid; h < HV; h += THREADS) {
    const float dt = dt_f32
                         ? static_cast<const float*>(dt_bias)[h]
                         : __half2float(static_cast<const __half*>(dt_bias)[h]);
    const float xa = __half2float(a[t * sa + h]) + dt;
    const float sp = xa > 20.f ? xa : log1pf(expf(xa));
    const float gv = -expf(A_log[h]) * sp;
    g[(long)t * HV + h] = G_EXP ? expf(gv) : gv;
    beta[(long)t * HV + h] = 1.f / (1.f + expf(-__half2float(b[t * sb + h])));
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int K, bool L2NORM, bool G_EXP>
__global__ void gdn_post_conv_rdna2_kernel(
    const __half*, const __half*, const __half*, const float*, const void*,
    const bool, __half*, __half*, __half*, float*, float*, const int, const int,
    const int, const long, const long, const long, const float) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace gdn_rdna2
}  // namespace vllm

// conv_output [L, 2 * H * K + HV * V] fp16 (rows 16-byte aligned) -> q, k
// [L, H, K], v [L, HV, V] (contiguous fp16), g, beta [L, HV] (contiguous
// fp32); a, b [L, HV] fp16 with contiguous rows, A_log [HV] fp32, dt_bias
// [HV] fp16 or fp32. K in {64, 128, 256}, V % 8 == 0.
void gdn_post_conv_rdna2(const torch::Tensor& conv_output,
                         const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& A_log,
                         const torch::Tensor& dt_bias, torch::Tensor& q,
                         torch::Tensor& k, torch::Tensor& v, torch::Tensor& g,
                         torch::Tensor& beta, bool apply_l2norm,
                         bool output_g_exp, double eps) {
  using namespace vllm::gdn_rdna2;
  TORCH_CHECK(
      conv_output.dtype() == torch::kFloat16 && a.dtype() == torch::kFloat16 &&
          b.dtype() == torch::kFloat16 && A_log.dtype() == torch::kFloat32 &&
          (dt_bias.dtype() == torch::kFloat16 ||
           dt_bias.dtype() == torch::kFloat32) &&
          q.dtype() == torch::kFloat16 && k.dtype() == torch::kFloat16 &&
          v.dtype() == torch::kFloat16 && g.dtype() == torch::kFloat32 &&
          beta.dtype() == torch::kFloat32,
      "gdn_post_conv_rdna2: fp16 activations, fp32 A_log / g / beta");
  const int L = conv_output.size(0), H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  TORCH_CHECK(K == 64 || K == 128 || K == 256, "gdn_post_conv_rdna2: K");
  TORCH_CHECK(V % 8 == 0 && conv_output.size(1) == 2 * H * K + HV * V &&
                  conv_output.stride(1) == 1 &&
                  conv_output.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(conv_output.data_ptr()) % 16 == 0,
              "gdn_post_conv_rdna2: conv_output must be [L, 2*H*K + HV*V] "
              "with 16-byte aligned rows");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  g.is_contiguous() && beta.is_contiguous() &&
                  a.stride(1) == 1 && b.stride(1) == 1 &&
                  A_log.is_contiguous() && dt_bias.is_contiguous(),
              "gdn_post_conv_rdna2: contiguous outputs and rows");
  if (L == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(conv_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define VLLM_GDN_POST_CONV_LAUNCH(KD, NORM, GEXP)                              \
  vllm::gdn_rdna2::gdn_post_conv_rdna2_kernel<KD, NORM, GEXP>                  \
      <<<L, THREADS, 0, stream>>>(                                             \
          (const __half*)conv_output.data_ptr(), (const __half*)a.data_ptr(),  \
          (const __half*)b.data_ptr(), A_log.data_ptr<float>(),                \
          dt_bias.data_ptr(), dt_bias.dtype() == torch::kFloat32,              \
          (__half*)q.data_ptr(), (__half*)k.data_ptr(), (__half*)v.data_ptr(), \
          g.data_ptr<float>(), beta.data_ptr<float>(), H, HV, V,               \
          conv_output.stride(0), a.stride(0), b.stride(0), (float)eps)
#define VLLM_GDN_POST_CONV_BY_FLAGS(KD)          \
  if (apply_l2norm && !output_g_exp) {           \
    VLLM_GDN_POST_CONV_LAUNCH(KD, true, false);  \
  } else if (apply_l2norm) {                     \
    VLLM_GDN_POST_CONV_LAUNCH(KD, true, true);   \
  } else if (!output_g_exp) {                    \
    VLLM_GDN_POST_CONV_LAUNCH(KD, false, false); \
  } else {                                       \
    VLLM_GDN_POST_CONV_LAUNCH(KD, false, true);  \
  }
  if (K == 128) {
    VLLM_GDN_POST_CONV_BY_FLAGS(128);
  } else if (K == 64) {
    VLLM_GDN_POST_CONV_BY_FLAGS(64);
  } else {
    VLLM_GDN_POST_CONV_BY_FLAGS(256);
  }
#undef VLLM_GDN_POST_CONV_BY_FLAGS
#undef VLLM_GDN_POST_CONV_LAUNCH
}
