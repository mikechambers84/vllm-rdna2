// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) MoE routing in one kernel (DeepSeek-V3 / GLM "noaux_tc"
// grouped top-k): scores = sigmoid / softmax of the router logits (or the
// logits as given), experts picked on scores + correction bias within the
// topk_group groups of highest top-2 (biased) or max (unbiased) score, routing
// weights = the unbiased scores of the picked experts, renormalized and
// scaled. Replaces ~8 torch launches (sorts and top-ks) per MoE layer.
//
// One wave32 per token: lane l holds experts l, l + 32, ...; every top-k round
// is a wave argmax (ties to the lower expert id). The scores also go to LDS so
// lanes can read any expert's score and scan groups.

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
namespace grouped_topk_rdna2 {

static constexpr int WAVE = 32;
static constexpr int WAVES = 4;  // tokens per workgroup
static constexpr int MAX_E = 512;
static constexpr int MAX_K = 32;
static constexpr int MAX_GROUPS = 32;

enum Scoring : int { kNone = 0, kSigmoid = 1, kSoftmax = 2 };

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(__hip_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float wave_max(float v) {
  #pragma unroll
  for (int s = WAVE / 2; s > 0; s /= 2) v = fmaxf(v, __shfl_xor(v, s, WAVE));
  return v;
}

__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int s = WAVE / 2; s > 0; s /= 2) v += __shfl_xor(v, s, WAVE);
  return v;
}

// Wave argmax of (v, i): the larger value, ties to the smaller index.
__device__ __forceinline__ void wave_argmax(float& v, int& i) {
  #pragma unroll
  for (int s = WAVE / 2; s > 0; s /= 2) {
    const float ov = __shfl_xor(v, s, WAVE);
    const int oi = __shfl_xor(i, s, WAVE);
    if (ov > v || (ov == v && oi < i)) {
      v = ov;
      i = oi;
    }
  }
}

template <typename T, int VPT>
__global__ void __launch_bounds__(WAVE* WAVES)
    grouped_topk_rdna2_kernel(const T* __restrict__ logits,
                              const float* __restrict__ bias,
                              float* __restrict__ out_w,
                              int* __restrict__ out_ids, const int num_tokens,
                              const int E, const long stride, const int K,
                              const int n_group, const int topk_group,
                              const int scoring, const bool renormalize,
                              const float scale) {
  __shared__ float s_score[WAVES][MAX_E];
  __shared__ float s_sel[WAVES][MAX_E];
  const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
  const int t = blockIdx.x * WAVES + w;
  const bool valid = t < num_tokens;
  const float NEG = -INFINITY;

  float sel[VPT];  // selection scores (biased), -inf when not selectable
  float sc[VPT];   // unbiased scores
  #pragma unroll
  for (int j = 0; j < VPT; j++) {
    const int e = lane + WAVE * j;
    sc[j] = (valid && e < E) ? to_f(logits[t * stride + e]) : NEG;
  }
  if (scoring == kSoftmax) {
    float m = NEG;
  #pragma unroll
    for (int j = 0; j < VPT; j++) m = fmaxf(m, sc[j]);
    m = wave_max(m);
    float sum = 0.f;
  #pragma unroll
    for (int j = 0; j < VPT; j++) {
      sc[j] = sc[j] == NEG ? 0.f : __expf(sc[j] - m);
      sum += sc[j];
    }
    sum = wave_sum(sum);
  #pragma unroll
    for (int j = 0; j < VPT; j++) sc[j] /= sum;
  }
  #pragma unroll
  for (int j = 0; j < VPT; j++) {
    const int e = lane + WAVE * j;
    if (e < E) {
      if (scoring == kSigmoid) sc[j] = 1.f / (1.f + __expf(-sc[j]));
      sel[j] = sc[j] + (bias != nullptr ? bias[e] : 0.f);
      s_score[w][e] = sc[j];
      s_sel[w][e] = sel[j];
    } else {
      sel[j] = NEG;
    }
  }
  __syncthreads();

  if (n_group > 1 && topk_group < n_group) {
    // Group score: sum of the top-2 selection scores (with bias), else max.
    const int G = E / n_group;
    float gs = NEG;
    if (lane < n_group) {
      float a = NEG, b = NEG;
      for (int i = 0; i < G; i++) {
        const float v = s_sel[w][lane * G + i];
        if (v > a) {
          b = a;
          a = v;
        } else if (v > b) {
          b = v;
        }
      }
      gs = bias != nullptr ? a + b : a;
    }
    unsigned gmask = 0;
    for (int r = 0; r < topk_group; r++) {
      float v = gs;
      int i = lane;
      wave_argmax(v, i);
      gmask |= 1u << i;
      if (lane == i) gs = NEG;
    }
  #pragma unroll
    for (int j = 0; j < VPT; j++) {
      const int e = lane + WAVE * j;
      if (e < E && !((gmask >> (e / G)) & 1u)) sel[j] = NEG;
    }
  }

  int my_id = 0;
  float my_w = 0.f;
  for (int k = 0; k < K; k++) {
    float v = NEG;
    int i = 0x7fffffff;
  #pragma unroll
    for (int j = 0; j < VPT; j++) {
      const int e = lane + WAVE * j;
      if (sel[j] > v) {
        v = sel[j];
        i = e;
      }
    }
    wave_argmax(v, i);
    if ((i % WAVE) == lane) {
  #pragma unroll
      for (int j = 0; j < VPT; j++)
        if (lane + WAVE * j == i) sel[j] = NEG;
    }
    if (lane == k && i < E) {
      my_id = i;
      my_w = s_score[w][i];
    }
  }
  if (renormalize) {
    const float sum = wave_sum(lane < K ? my_w : 0.f);
    my_w /= sum;
  }
  if (valid && lane < K) {
    out_w[t * K + lane] = my_w * scale;
    out_ids[t * K + lane] = my_id;
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, int VPT>
__global__ void grouped_topk_rdna2_kernel(const T*, const float*, float*, int*,
                                          const int, const int, const long,
                                          const int, const int, const int,
                                          const int, const bool, const float) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace grouped_topk_rdna2
}  // namespace vllm

// logits [T, E] (fp32 / fp16 / bf16, last dim contiguous), bias fp32 [E] or
// none -> topk_weights fp32 [T, K], topk_ids int32 [T, K] (both contiguous).
// E <= 512, K <= 32, n_group <= 32 dividing E; scoring 0 none, 1 sigmoid,
// 2 softmax.
void grouped_topk_rdna2(const torch::Tensor& logits,
                        const std::optional<torch::Tensor>& bias,
                        torch::Tensor& topk_weights, torch::Tensor& topk_ids,
                        int64_t n_group, int64_t topk_group, int64_t scoring,
                        bool renormalize, double scale) {
  using namespace vllm::grouped_topk_rdna2;
  TORCH_CHECK(logits.dim() == 2 && logits.stride(1) == 1,
              "grouped_topk_rdna2: logits [T, E] with contiguous rows");
  const int T = logits.size(0), E = logits.size(1), K = topk_weights.size(1);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                  topk_ids.scalar_type() == at::kInt &&
                  topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
                  topk_weights.size(0) == T && topk_ids.size(0) == T &&
                  topk_ids.size(1) == K,
              "grouped_topk_rdna2: fp32 / int32 [T, K] contiguous outputs");
  TORCH_CHECK(E >= 1 && E <= MAX_E && K >= 1 && K <= MAX_K && K <= E,
              "grouped_topk_rdna2: E <= 512, K <= 32");
  TORCH_CHECK(n_group >= 1 && n_group <= MAX_GROUPS && E % n_group == 0 &&
                  topk_group >= 1 && topk_group <= n_group &&
                  K <= topk_group * (E / n_group),
              "grouped_topk_rdna2: n_group <= 32 dividing E, enough experts");
  TORCH_CHECK(scoring >= kNone && scoring <= kSoftmax,
              "grouped_topk_rdna2: scoring 0 none, 1 sigmoid, 2 softmax");
  const float* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->is_contiguous() &&
                    bias->numel() == E,
                "grouped_topk_rdna2: fp32 bias [E]");
    bias_ptr = bias->data_ptr<float>();
  }
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((T + WAVES - 1) / WAVES);
  const int vpt = E <= 64 ? 2 : E <= 128 ? 4 : E <= 256 ? 8 : 16;
#define VLLM_TOPK_LAUNCH(TT, CT, V)                                    \
  grouped_topk_rdna2_kernel<CT, V><<<grid, WAVE * WAVES, 0, stream>>>( \
      reinterpret_cast<const CT*>(logits.data_ptr()), bias_ptr,        \
      topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(), T, E,  \
      logits.stride(0), K, n_group, topk_group, scoring, renormalize,  \
      (float)scale);
#define VLLM_TOPK_DTYPE(TT, CT)           \
  switch (vpt) {                          \
    case 2:                               \
      VLLM_TOPK_LAUNCH(TT, CT, 2) break;  \
    case 4:                               \
      VLLM_TOPK_LAUNCH(TT, CT, 4) break;  \
    case 8:                               \
      VLLM_TOPK_LAUNCH(TT, CT, 8) break;  \
    default:                              \
      VLLM_TOPK_LAUNCH(TT, CT, 16) break; \
  }
  switch (logits.scalar_type()) {
    case at::kFloat: {
      VLLM_TOPK_DTYPE(float, float)
      break;
    }
    case at::kHalf: {
      VLLM_TOPK_DTYPE(half, __half)
      break;
    }
    case at::kBFloat16: {
      VLLM_TOPK_DTYPE(bf16, __hip_bfloat16)
      break;
    }
    default:
      TORCH_CHECK(false, "grouped_topk_rdna2: fp32 / fp16 / bf16 logits");
  }
#undef VLLM_TOPK_DTYPE
#undef VLLM_TOPK_LAUNCH
}
