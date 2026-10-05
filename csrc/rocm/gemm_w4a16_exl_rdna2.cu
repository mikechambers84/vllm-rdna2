// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) 4-bit GPTQ GEMM for decode and small batches, on the weight
// layout Exllama's gptq_gemm uses, so its reconstruct + GEMM path keeps serving
// large batches from the same tensors: w [K/8, N] int32 with the nibbles of
// each word shuffled to k order 0,2,4,6,1,3,5,7 (gptq_shuffle), zeros
// [K/G, N/8] int32 (stored zero, +1 for GPTQv1), scales [K/G, N] fp16.
//
// Lanes run along N with four columns each (one 16-byte load = 4 columns x
// 8 k), so all lanes of a wave use the same activations, which come through
// scalar loads. The NW waves of a workgroup split K in 32-k blocks and reduce
// through LDS in a fixed order (deterministic, no atomics); T tokens share
// each dequantized weight. The shuffle makes (q >> 4j) & 0x000F000F the half2
// pair (2j, 2j + 1); OR into the 1024.0 mantissa gives exact integers, the
// zero is subtracted exactly and the group scale applied in fp16, then
// v_dot2_f32_f16 accumulates in fp32. Summed over the linear layers of eight
// models (Qwen2.5 1.5B-32B, Phi-3-mini, Llama-3 8B/70B, Mistral-Nemo,
// Qwen3.8-27B; g128 and g32) at steady clocks, against Exllama's kernels:
// 1.04-1.5x at 1 token (up to ~470 GB/s), 1.5-3x at 8-64 tokens, 1.06-1.25x
// at 512 tokens (~19 TFLOPS).

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace gemm_w4a16_exl_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int KBLOCK = 32;  // k per weight block: 4 words per column

struct Config {
  int t;   // tokens per workgroup
  int nw;  // waves per workgroup, splitting K
  int ct;  // 128-column tiles per wave
};

// Indexed by the cfg argument; the default for a shape is default_config().
static constexpr Config kConfigs[] = {
    {1, 16, 2}, {1, 4, 2},  {2, 16, 1}, {4, 8, 1},  {8, 8, 1},
    {8, 4, 1},  {16, 8, 1}, {16, 4, 1}, {32, 1, 1},
};
static constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// One word (8 k of one column) -> 4 half2 of (q - zero) * scale for the k
// pairs (2j, 2j + 1); bias holds 1024 + zero.
__device__ __forceinline__ void unpack8(uint32_t q, half2 bias, half2 scale,
                                        half2 (&h)[4]) {
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    uint32_t u = ((q >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
    h[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&u), bias), scale);
  }
}

template <int CT>
struct KBlock {
  uint4 w[4][CT];
  uint2 s[CT];
  uint32_t z[CT];
};

// grid (ceil(M / T), ceil(N / (128 * CT))). SYM: every zero is 8 (no zero
// loads), else zeros + zbias (1 for GPTQv1, 0 for v2).
template <int T, int NW, int CT, bool SYM>
__global__ void __launch_bounds__(NW* WARP32)
    gemm_w4a16_exl_rdna2_kernel(const uint4* __restrict__ W,
                                const uint32_t* __restrict__ Z,
                                const __half* __restrict__ S,
                                const __half* __restrict__ A,
                                __half* __restrict__ C, const int M,
                                const int N, const int K, const int group_size,
                                const int zbias, const int lda, const int ldc) {
  extern __shared__ __align__(16) float red[];
  const int lane = threadIdx.x % WARP32;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP32);
  const int t0 = blockIdx.x * T;
  const int blocks = K / KBLOCK;
  const int b_begin = blocks * wave / NW, b_end = blocks * (wave + 1) / NW;
  const int n4 = N / 4, n8 = N / 8;
  int col[CT];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
    col[ct] = min(blockIdx.y * WARP32 * CT + ct * WARP32 + lane, n4 - 1);

  float acc[CT][4][T];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
  #pragma unroll
    for (int c = 0; c < 4; c++)
  #pragma unroll
      for (int i = 0; i < T; i++) acc[ct][c][i] = 0.f;

  auto load = [&](int b, KBlock<CT>& d) {
    const int grp = b * KBLOCK / group_size;
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      d.s[ct] =
          *reinterpret_cast<const uint2*>(S + (long)grp * N + col[ct] * 4);
      if constexpr (!SYM)
        d.z[ct] = Z[(long)grp * n8 + col[ct] / 2] >> ((col[ct] % 2) * 16);
    }
  #pragma unroll
    for (int u = 0; u < 4; u++)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
        d.w[u][ct] = W[(long)(b * 4 + u) * n4 + col[ct]];
  };

  for (int b = b_begin; b < b_end; b++) {
    KBlock<CT> cur;
    load(b, cur);
    half2 scale[CT][4], bias[CT][4];
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      const half2 sa = *reinterpret_cast<const half2*>(&cur.s[ct].x);
      const half2 sb = *reinterpret_cast<const half2*>(&cur.s[ct].y);
      scale[ct][0] = __low2half2(sa);
      scale[ct][1] = __high2half2(sa);
      scale[ct][2] = __low2half2(sb);
      scale[ct][3] = __high2half2(sb);
  #pragma unroll
      for (int c = 0; c < 4; c++) {
        const unsigned short bb =
            SYM ? 0x6408 : 0x6400 + ((cur.z[ct] >> (4 * c)) & 0xF) + zbias;
        bias[ct][c] =
            __halves2half2(__ushort_as_half(bb), __ushort_as_half(bb));
      }
    }
  #pragma unroll
    for (int u = 0; u < 4; u++) {
      const int k0 = (b * 4 + u) * 8;
      half2 h[CT][4][4];
  #pragma unroll
      for (int ct = 0; ct < CT; ct++) {
        unpack8(cur.w[u][ct].x, bias[ct][0], scale[ct][0], h[ct][0]);
        unpack8(cur.w[u][ct].y, bias[ct][1], scale[ct][1], h[ct][1]);
        unpack8(cur.w[u][ct].z, bias[ct][2], scale[ct][2], h[ct][2]);
        unpack8(cur.w[u][ct].w, bias[ct][3], scale[ct][3], h[ct][3]);
      }
  #pragma unroll
      for (int i = 0; i < T; i++) {
        // Wave-uniform address: a scalar load.
        const uint4 av = *reinterpret_cast<const uint4*>(
            A + (long)min(t0 + i, M - 1) * lda + k0);
        const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
        for (int ct = 0; ct < CT; ct++)
  #pragma unroll
          for (int c = 0; c < 4; c++)
  #pragma unroll
            for (int j = 0; j < 4; j++)
              acc[ct][c][i] = __builtin_amdgcn_fdot2(ah[j], h[ct][c][j],
                                                     acc[ct][c][i], false);
      }
    }
  }

  // Tree reduction of the NW K slices: in each round the upper half of the
  // remaining waves hands its sums to the lower half. Rolled loops keep the
  // epilogue's registers (and so the kernel's occupancy) low.
  constexpr int PER_WAVE = CT * 4 * T * WARP32;
  #pragma unroll 1
  for (int half = NW / 2; half >= 1; half /= 2) {
    if (wave >= half && wave < 2 * half)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++)
  #pragma unroll
          for (int i = 0; i < T; i++)
            red[(wave - half) * PER_WAVE + ((ct * 4 + c) * T + i) * WARP32 +
                lane] = acc[ct][c][i];
    __syncthreads();
    if (wave < half)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++)
  #pragma unroll
          for (int i = 0; i < T; i++)
            acc[ct][c][i] +=
                red[wave * PER_WAVE + ((ct * 4 + c) * T + i) * WARP32 + lane];
    __syncthreads();
  }

  if (wave == 0) {
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      const int c4 = blockIdx.y * WARP32 * CT + ct * WARP32 + lane;
      if (c4 >= n4) continue;
  #pragma unroll
      for (int i = 0; i < T; i++) {
        if (t0 + i < M) {
          const half2 lo = __floats2half2_rn(acc[ct][0][i], acc[ct][1][i]);
          const half2 hi = __floats2half2_rn(acc[ct][2][i], acc[ct][3][i]);
          uint2 o;
          o.x = *reinterpret_cast<const uint32_t*>(&lo);
          o.y = *reinterpret_cast<const uint32_t*>(&hi);
          *reinterpret_cast<uint2*>(C + (long)(t0 + i) * ldc + c4 * 4) = o;
        }
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int T, int NW, int CT, bool SYM>
__global__ void gemm_w4a16_exl_rdna2_kernel(const uint4*, const uint32_t*,
                                            const __half*, const __half*,
                                            __half*, const int, const int,
                                            const int, const int, const int,
                                            const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <int T, int NW, int CT, bool SYM>
void launch(const at::Tensor& a, const at::Tensor& w, const at::Tensor& zeros,
            const at::Tensor& scales, at::Tensor& c, int group_size, int zbias,
            cudaStream_t stream) {
  const int M = a.size(0), K = a.size(1), N = w.size(1);
  const dim3 grid((M + T - 1) / T, (N / 4 + WARP32 * CT - 1) / (WARP32 * CT));
  const size_t lds = (size_t)(NW / 2) * CT * 4 * T * WARP32 * sizeof(float);
  gemm_w4a16_exl_rdna2_kernel<T, NW, CT, SYM>
      <<<grid, dim3(NW * WARP32), lds, stream>>>(
          (const uint4*)w.data_ptr(), (const uint32_t*)zeros.data_ptr(),
          (const __half*)scales.data_ptr(), (const __half*)a.data_ptr(),
          (__half*)c.data_ptr(), M, N, K, group_size, zbias, a.stride(0),
          c.stride(0));
}

// Default config, fitted on 38 layer shapes of 8 models (1.5B-70B, g32 and
// g128) at steady clocks: larger token tiles as M grows, but enough waves to
// fill the GPU on narrow layers.
static int default_config(int M, int N) {
  auto waves = [&](int id) {
    const Config& c = kConfigs[id];
    return (M + c.t - 1) / c.t * ((N + 128 * c.ct - 1) / (128 * c.ct)) * c.nw;
  };
  if (M == 1) return N >= 32768 ? 1 : 0;
  if (M == 2) return 2;
  if (M <= 4) return 3;
  if (M <= 32) return 4;
  if (M <= 128) return waves(5) >= 768 ? 5 : 4;
  if (M <= 256) return 6;
  return waves(8) >= 512 ? 8 : 7;
}

}  // namespace gemm_w4a16_exl_rdna2
}  // namespace vllm

// Requirements: a [M, K] fp16, K-contiguous, 16-byte aligned rows; w, zeros
// and scales contiguous as above, group_size a multiple of 32 dividing K, N a
// multiple of 8. symmetric ignores zeros (all 8); otherwise the zero of a
// weight is its stored zero plus 1 unless use_v2_format. cfg < 0 picks the
// default config for M. Returns [M, N] fp16.
torch::Tensor gemm_w4a16_exl_rdna2(const at::Tensor& a, const at::Tensor& w,
                                   const at::Tensor& zeros,
                                   const at::Tensor& scales, bool symmetric,
                                   bool use_v2_format, int64_t cfg) {
  using namespace vllm::gemm_w4a16_exl_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 && scales.dtype() == torch::kFloat16,
              "gemm_w4a16_exl_rdna2 needs fp16 activations and scales");
  TORCH_CHECK(w.dtype() == torch::kInt32 && zeros.dtype() == torch::kInt32 &&
                  w.is_contiguous() && zeros.is_contiguous() &&
                  scales.is_contiguous(),
              "gemm_w4a16_exl_rdna2 needs contiguous int32 weights and zeros "
              "and fp16 scales");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && w.size(0) * 8 == a.size(1),
              "gemm_w4a16_exl_rdna2 needs a [M, K] and w [K / 8, N]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(1);
  TORCH_CHECK(M >= 1 && N % 8 == 0,
              "gemm_w4a16_exl_rdna2 needs M >= 1 and N "
              "a multiple of 8");
  const int groups = scales.size(0);
  TORCH_CHECK(groups > 0 && K % groups == 0 && (K / groups) % KBLOCK == 0 &&
                  scales.size(1) == N && zeros.size(0) == groups &&
                  zeros.size(1) == N / 8,
              "gemm_w4a16_exl_rdna2 needs a group size that is a multiple of "
              "32, [K / G, N] scales and [K / G, N / 8] zeros");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "gemm_w4a16_exl_rdna2 needs 16-byte aligned, K-contiguous rows");
  TORCH_CHECK(cfg < kNumConfigs, "gemm_w4a16_exl_rdna2: cfg out of range");

  auto c = torch::empty({M, N}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int group_size = K / groups;
  const int zbias = use_v2_format ? 0 : 1;
  const int id = cfg < 0 ? default_config(M, N) : (int)cfg;

#define VLLM_EXL_RDNA2_CASE(ID, T, NW, CT)                                \
  case ID:                                                                \
    static_assert(kConfigs[ID].t == T && kConfigs[ID].nw == NW &&         \
                  kConfigs[ID].ct == CT);                                 \
    if (symmetric) {                                                      \
      launch<T, NW, CT, true>(a, w, zeros, scales, c, group_size, zbias,  \
                              stream);                                    \
    } else {                                                              \
      launch<T, NW, CT, false>(a, w, zeros, scales, c, group_size, zbias, \
                               stream);                                   \
    }                                                                     \
    break;
  switch (id) {
    VLLM_EXL_RDNA2_CASE(0, 1, 16, 2)
    VLLM_EXL_RDNA2_CASE(1, 1, 4, 2)
    VLLM_EXL_RDNA2_CASE(2, 2, 16, 1)
    VLLM_EXL_RDNA2_CASE(3, 4, 8, 1)
    VLLM_EXL_RDNA2_CASE(4, 8, 8, 1)
    VLLM_EXL_RDNA2_CASE(5, 8, 4, 1)
    VLLM_EXL_RDNA2_CASE(6, 16, 8, 1)
    VLLM_EXL_RDNA2_CASE(7, 16, 4, 1)
    VLLM_EXL_RDNA2_CASE(8, 32, 1, 1)
  }
#undef VLLM_EXL_RDNA2_CASE
  return c;
}
