// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) W4A8 GEMM on the 4-bit GPTQ layout gemm_w4a16_exl_rdna2 and
// Exllama read (w [K/8, N] int32 with the nibbles of each word in k order
// 0,2,4,6,1,3,5,7 (gptq_shuffle), zeros [K/G, N/8], scales [K/G, N] fp16),
// with activations quantized to int8 per token: for opt-in W4A8 on W4A16
// checkpoints, no weight re-quantization.
//
// quant_int8_exl_rdna2 quantizes each row (scale = amax / 127), stores the
// bytes of each 8-k group in the order 0,4,1,5,2,6,3,7 and the sum of each
// 32-k block. Then q & 0x0F0F0F0F and (q >> 4) & 0x0F0F0F0F of a weight word
// are v_dot4_i32_i8 operands against the two activation dwords as stored, and
// a 32-k block of one column sums exactly in int32 as
// dot(a, q) - zero * sum(a); the block's group scale applies in fp32. The
// rest follows gemm_w4a16_exl_rdna2: lanes along N with four columns each,
// activations through scalar loads, NW waves splitting K with a fixed-order
// LDS reduction, T tokens per workgroup. On the Qwen3.8-27B layers at steady
// clocks: 1.25-1.33x gemm_w4a16_exl_rdna2 at 8 rows, 1.4-1.9x at 16-128
// (up to ~30 TOPS), 1.5-1.7x at 512-2048.

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
namespace gemm_w4a8_exl_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int KBLOCK = 32;  // k per block (4 words per column)
static constexpr int QUANT_THREADS = 256;

struct Config {
  int t;   // tokens per workgroup
  int nw;  // waves per workgroup, splitting K
  int ct;  // 128-column tiles per wave
};

// Indexed by the cfg argument; the default for a shape is default_config().
static constexpr Config kConfigs[] = {
    {1, 16, 2}, {1, 8, 2}, {2, 8, 2},  {4, 8, 1},  {4, 4, 1},
    {8, 8, 1},  {8, 4, 1}, {16, 8, 1}, {16, 4, 1}, {16, 2, 1},
};
static constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

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

// One row per workgroup: amax, then 8 k per thread and step.
template <typename T>
__global__ void __launch_bounds__(QUANT_THREADS)
    quant_int8_exl_rdna2_kernel(const T* __restrict__ X, int8_t* __restrict__ Q,
                                float* __restrict__ S, int* __restrict__ SUM,
                                const int K, const int ldx) {
  __shared__ float wave_max[QUANT_THREADS / WARP32];
  const int row = blockIdx.x;
  const T* x = X + (long)row * ldx;
  float amax = 0.f;
  for (int k = threadIdx.x * 8; k < K; k += QUANT_THREADS * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(x + k);
    const T* e = reinterpret_cast<const T*>(&v);
  #pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(to_float<T>(e[j])));
  }
  #pragma unroll
  for (int mask = WARP32 / 2; mask >= 1; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, mask));
  if (threadIdx.x % WARP32 == 0) wave_max[threadIdx.x / WARP32] = amax;
  __syncthreads();
  amax = 0.f;
  #pragma unroll
  for (int w = 0; w < QUANT_THREADS / WARP32; w++)
    amax = fmaxf(amax, wave_max[w]);
  const float scale = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  if (threadIdx.x == 0) S[row] = scale;

  // Lanes 4j..4j+3 hold the four 8-k chunks of one 32-k block.
  for (int k = threadIdx.x * 8; k < K; k += QUANT_THREADS * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(x + k);
    const T* e = reinterpret_cast<const T*>(&v);
    int q[8], sum = 0;
  #pragma unroll
    for (int j = 0; j < 8; j++) {
      q[j] = min(max(__float2int_rn(to_float<T>(e[j]) * inv), -127), 127);
      sum += q[j];
    }
    auto pack = [](int a, int b, int c, int d) {
      return (uint32_t)(a & 0xFF) | (uint32_t)(b & 0xFF) << 8 |
             (uint32_t)(c & 0xFF) << 16 | (uint32_t)(d & 0xFF) << 24;
    };
    uint2 o;
    o.x = pack(q[0], q[4], q[1], q[5]);
    o.y = pack(q[2], q[6], q[3], q[7]);
    *reinterpret_cast<uint2*>(Q + (long)row * K + k) = o;
    sum += __shfl_xor(sum, 1);
    sum += __shfl_xor(sum, 2);
    if (threadIdx.x % 4 == 0) SUM[(long)row * (K / KBLOCK) + k / KBLOCK] = sum;
  }
}

__device__ __forceinline__ uint32_t dword(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// grid (ceil(M / T), ceil(N / (128 * CT))). SYM: every zero is 8 (no zero
// loads), else zeros + zbias (1 for GPTQv1, 0 for v2).
template <typename O, int T, int NW, int CT, bool SYM>
__global__ void __launch_bounds__(NW* WARP32) gemm_w4a8_exl_rdna2_kernel(
    const uint4* __restrict__ W, const uint32_t* __restrict__ Z,
    const __half* __restrict__ S, const int8_t* __restrict__ A,
    const float* __restrict__ SA, const int* __restrict__ SUM,
    O* __restrict__ C, const int M, const int N, const int K,
    const int group_size, const int zbias, const int ldc) {
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

  for (int b = b_begin; b < b_end; b++) {
    const int grp = b * KBLOCK / group_size;
    uint4 w[4][CT];
    uint2 sc[CT];
    uint32_t zw[CT];
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      sc[ct] = *reinterpret_cast<const uint2*>(S + (long)grp * N + col[ct] * 4);
      if constexpr (!SYM)
        zw[ct] = Z[(long)grp * n8 + col[ct] / 2] >> ((col[ct] % 2) * 16);
    }
  #pragma unroll
    for (int u = 0; u < 4; u++)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
        w[u][ct] = W[(long)(b * 4 + u) * n4 + col[ct]];
    uint32_t lo[4][CT][4], hi[4][CT][4];
    float s[CT][4];
    int z[CT][4];
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
  #pragma unroll
      for (int u = 0; u < 4; u++)
  #pragma unroll
        for (int c = 0; c < 4; c++) {
          const uint32_t q = dword(w[u][ct], c);
          lo[u][ct][c] = q & 0x0F0F0F0Fu;
          hi[u][ct][c] = (q >> 4) & 0x0F0F0F0Fu;
        }
      const half2 s01 = *reinterpret_cast<const half2*>(&sc[ct].x);
      const half2 s23 = *reinterpret_cast<const half2*>(&sc[ct].y);
      s[ct][0] = __low2float(s01);
      s[ct][1] = __high2float(s01);
      s[ct][2] = __low2float(s23);
      s[ct][3] = __high2float(s23);
  #pragma unroll
      for (int c = 0; c < 4; c++)
        z[ct][c] = SYM ? 8 : (int)((zw[ct] >> (4 * c)) & 0xF) + zbias;
    }
  #pragma unroll
    for (int i = 0; i < T; i++) {
      // Wave-uniform addresses: scalar loads of the block's 32 activation
      // bytes and their sum.
      const int row = min(t0 + i, M - 1);
      const int4 a0 =
          *reinterpret_cast<const int4*>(A + (long)row * K + b * KBLOCK);
      const int4 a1 =
          *reinterpret_cast<const int4*>(A + (long)row * K + b * KBLOCK + 16);
      const int sum = SUM[(long)row * blocks + b];
      const int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++) {
          int d = -z[ct][c] * sum;
  #pragma unroll
          for (int u = 0; u < 4; u++) {
            d = __builtin_amdgcn_sdot4(av[2 * u], lo[u][ct][c], d, false);
            d = __builtin_amdgcn_sdot4(av[2 * u + 1], hi[u][ct][c], d, false);
          }
          acc[ct][c][i] = fmaf((float)d, s[ct][c], acc[ct][c][i]);
        }
    }
  }

  // Tree reduction of the NW K slices, as in gemm_w4a16_exl_rdna2.
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
          const float sa = SA[t0 + i];
          __align__(8) O o[4];
  #pragma unroll
          for (int c = 0; c < 4; c++) o[c] = from_float<O>(acc[ct][c][i] * sa);
          *reinterpret_cast<uint2*>(C + (long)(t0 + i) * ldc + c4 * 4) =
              *reinterpret_cast<const uint2*>(o);
        }
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

template <typename T>
__global__ void quant_int8_exl_rdna2_kernel(const T*, int8_t*, float*, int*,
                                            const int, const int) {}
template <typename O, int T, int NW, int CT, bool SYM>
__global__ void gemm_w4a8_exl_rdna2_kernel(const uint4*, const uint32_t*,
                                           const __half*, const int8_t*,
                                           const float*, const int*, O*,
                                           const int, const int, const int,
                                           const int, const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <typename O, int T, int NW, int CT, bool SYM>
void launch(const at::Tensor& a, const at::Tensor& scale_a,
            const at::Tensor& sum_a, const at::Tensor& w,
            const at::Tensor& zeros, const at::Tensor& scales, at::Tensor& c,
            int group_size, int zbias, cudaStream_t stream) {
  const int M = a.size(0), K = a.size(1), N = w.size(1);
  const dim3 grid((M + T - 1) / T, (N / 4 + WARP32 * CT - 1) / (WARP32 * CT));
  const size_t lds = (size_t)(NW / 2) * CT * 4 * T * WARP32 * sizeof(float);
  gemm_w4a8_exl_rdna2_kernel<O, T, NW, CT, SYM>
      <<<grid, dim3(NW * WARP32), lds, stream>>>(
          (const uint4*)w.data_ptr(), (const uint32_t*)zeros.data_ptr(),
          (const __half*)scales.data_ptr(), a.data_ptr<int8_t>(),
          scale_a.data_ptr<float>(), sum_a.data_ptr<int>(), (O*)c.data_ptr(), M,
          N, K, group_size, zbias, c.stride(0));
}

// Default config, fitted on the Qwen3.8-27B layers (g32) at steady clocks
// (within 1% of the best config summed per row count, quantization
// included): small token tiles on narrow layers, 16-token tiles with 4-8
// waves from 12 rows on.
static int default_config(int M, int N) {
  const bool narrow = N <= 6144;
  if (M == 1) return 1;
  if (M == 2) return 2;
  if (M <= 4) return narrow ? 3 : 4;
  if (M <= 8) return narrow ? 3 : 5;
  if (M <= 12) return narrow ? 3 : 8;
  if (M <= 16) return narrow ? 5 : 8;
  if (M <= 24) return narrow ? 5 : 6;
  if (M <= 64) return 7;
  if (M <= 512) return narrow ? 7 : 8;
  return M <= 1024 ? 7 : 8;
}

}  // namespace gemm_w4a8_exl_rdna2
}  // namespace vllm

// x [M, K] fp16 or bf16 with K-contiguous, 16-byte aligned rows and
// K % 32 == 0 -> (int8 [M, K] in the byte order gemm_w4a8_exl_rdna2 reads,
// fp32 [M] scales, int32 [M, K / 32] sums of each 32-k block).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quant_int8_exl_rdna2(
    const at::Tensor& x) {
  using namespace vllm::gemm_w4a8_exl_rdna2;
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
              "quant_int8_exl_rdna2 needs fp16 or bf16 input");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % KBLOCK == 0 && x.stride(1) == 1 &&
                  x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "quant_int8_exl_rdna2 needs x [M, K] with K % 32 == 0 and "
              "16-byte aligned, K-contiguous rows");
  const int M = x.size(0), K = x.size(1);
  auto q = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
  auto sum = torch::empty({M, K / KBLOCK}, x.options().dtype(torch::kInt32));
  if (M == 0) return {q, s, sum};
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (x.dtype() == torch::kFloat16) {
    quant_int8_exl_rdna2_kernel<__half><<<M, QUANT_THREADS, 0, stream>>>(
        (const __half*)x.data_ptr(), q.data_ptr<int8_t>(), s.data_ptr<float>(),
        sum.data_ptr<int>(), K, x.stride(0));
  } else {
    quant_int8_exl_rdna2_kernel<__hip_bfloat16>
        <<<M, QUANT_THREADS, 0, stream>>>(
            (const __hip_bfloat16*)x.data_ptr(), q.data_ptr<int8_t>(),
            s.data_ptr<float>(), sum.data_ptr<int>(), K, x.stride(0));
  }
  return {q, s, sum};
}

// Requirements: a, scale_a, sum_a from quant_int8_exl_rdna2; w, zeros and
// scales as gemm_w4a16_exl_rdna2 takes them, with a group size that is a
// multiple of 32. symmetric ignores zeros (all 8); otherwise the zero of a
// weight is its stored zero plus 1 unless use_v2_format. cfg < 0 picks the
// default config for M. Returns [M, N] in out_dtype (fp16 or bf16).
torch::Tensor gemm_w4a8_exl_rdna2(
    const at::Tensor& a, const at::Tensor& scale_a, const at::Tensor& sum_a,
    const at::Tensor& w, const at::Tensor& zeros, const at::Tensor& scales,
    bool symmetric, bool use_v2_format, at::ScalarType out_dtype, int64_t cfg) {
  using namespace vllm::gemm_w4a8_exl_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && a.dim() == 2 && a.is_contiguous(),
              "gemm_w4a8_exl_rdna2 needs contiguous int8 a [M, K]");
  const int M = a.size(0);
  const int K = a.size(1);
  TORCH_CHECK(scale_a.dtype() == torch::kFloat32 && scale_a.numel() == M &&
                  scale_a.is_contiguous() && sum_a.dtype() == torch::kInt32 &&
                  sum_a.is_contiguous() && sum_a.numel() == (long)M * (K / 32),
              "gemm_w4a8_exl_rdna2 needs scale_a [M] and sum_a [M, K / 32] "
              "from quant_int8_exl_rdna2");
  TORCH_CHECK(scales.dtype() == torch::kFloat16 && w.dtype() == torch::kInt32 &&
                  zeros.dtype() == torch::kInt32 && w.is_contiguous() &&
                  zeros.is_contiguous() && scales.is_contiguous(),
              "gemm_w4a8_exl_rdna2 needs contiguous int32 weights and zeros "
              "and fp16 scales");
  TORCH_CHECK(w.dim() == 2 && w.size(0) * 8 == K,
              "gemm_w4a8_exl_rdna2 needs w [K / 8, N]");
  const int N = w.size(1);
  TORCH_CHECK(N % 8 == 0 && K % KBLOCK == 0,
              "gemm_w4a8_exl_rdna2 needs N % 8 == 0 and K % 32 == 0");
  const int groups = scales.size(0);
  TORCH_CHECK(groups > 0 && K % groups == 0 && (K / groups) % KBLOCK == 0 &&
                  scales.size(1) == N && zeros.size(0) == groups &&
                  zeros.size(1) == N / 8,
              "gemm_w4a8_exl_rdna2 needs a group size that is a multiple of "
              "32, [K / G, N] scales and [K / G, N / 8] zeros");
  TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kBFloat16,
              "gemm_w4a8_exl_rdna2 writes fp16 or bf16");
  TORCH_CHECK(cfg < kNumConfigs, "gemm_w4a8_exl_rdna2: cfg out of range");

  auto c = torch::empty({M, N}, a.options().dtype(out_dtype));
  if (M == 0) return c;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int group_size = K / groups;
  const int zbias = use_v2_format ? 0 : 1;
  const int id = cfg < 0 ? default_config(M, N) : (int)cfg;

#define VLLM_W4A8_RDNA2_LAUNCH(O, T, NW, CT)                            \
  if (symmetric)                                                        \
    launch<O, T, NW, CT, true>(a, scale_a, sum_a, w, zeros, scales, c,  \
                               group_size, zbias, stream);              \
  else                                                                  \
    launch<O, T, NW, CT, false>(a, scale_a, sum_a, w, zeros, scales, c, \
                                group_size, zbias, stream);
#define VLLM_W4A8_RDNA2_CASE(ID, T, NW, CT)                       \
  case ID:                                                        \
    static_assert(kConfigs[ID].t == T && kConfigs[ID].nw == NW && \
                  kConfigs[ID].ct == CT);                         \
    if (out_dtype == torch::kFloat16) {                           \
      VLLM_W4A8_RDNA2_LAUNCH(__half, T, NW, CT)                   \
    } else {                                                      \
      VLLM_W4A8_RDNA2_LAUNCH(__hip_bfloat16, T, NW, CT)           \
    }                                                             \
    break;
  switch (id) {
    VLLM_W4A8_RDNA2_CASE(0, 1, 16, 2)
    VLLM_W4A8_RDNA2_CASE(1, 1, 8, 2)
    VLLM_W4A8_RDNA2_CASE(2, 2, 8, 2)
    VLLM_W4A8_RDNA2_CASE(3, 4, 8, 1)
    VLLM_W4A8_RDNA2_CASE(4, 4, 4, 1)
    VLLM_W4A8_RDNA2_CASE(5, 8, 8, 1)
    VLLM_W4A8_RDNA2_CASE(6, 8, 4, 1)
    VLLM_W4A8_RDNA2_CASE(7, 16, 8, 1)
    VLLM_W4A8_RDNA2_CASE(8, 16, 4, 1)
    VLLM_W4A8_RDNA2_CASE(9, 16, 2, 1)
  }
#undef VLLM_W4A8_RDNA2_CASE
#undef VLLM_W4A8_RDNA2_LAUNCH
  return c;
}
