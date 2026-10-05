// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) GEMM on 8-bit weights, fp8 e4m3fn or int8, with fp16/bf16
// activations (weight-only), for decode and prefill alike:
// C[M, N] = A[M, K] . (W * scale)^T (+ bias). gfx1030 has no fp8
// instructions; both formats are widened exactly in registers.
//
// The weights are stored K-major, w [K / 4, N, 4] (four consecutive k of one
// column per dword; kmajor_w8 in rdna2_w8a16.py makes it at load time). As in
// gemm_w4a16_exl_rdna2, lanes run along N with four columns each (one 16-byte
// load = 4 columns x 4 k), so all lanes of a wave use the same activations,
// which come through scalar loads; the NW waves of a workgroup split K in
// 16- or 32-k blocks and reduce through LDS in a fixed order (deterministic, no
// atomics); T tokens share each widened weight. fp8 widens through
// rdna2::fp8x2_to_half2 (its 2^-8 is folded into the output scale), int8 by
// placing the byte in the mantissa of 1024.0; fp16 activations accumulate
// through v_dot2_f32_f16, bf16 (no dot instruction on gfx1030) through fp32
// FMAs. 2D block scales are split at load time into a scale per column (the
// largest of its blocks) and fp16 ratios <= 1 per (k-block, column) that
// multiply the widened weights.
//
// On the linear layers of Qwen3.8-27B at steady clocks: 440-490 GB/s at 1-8
// rows, 300-340 at 16, ~200 at 32, 16-18 TFLOPS at 128 rows and ~20 from 1024
// on, above rocBLAS on fp16 weights (15-18 TFLOPS); the per-row GEMV it
// replaces fell to ~210 GB/s at 8 rows and ~90 at 16, and larger batches
// dequantized for rocBLAS.

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
namespace gemm_w8_rdna2 {

static constexpr int WARP32 = 32;

struct Config {
  int t;   // tokens per workgroup
  int nw;  // waves per workgroup, splitting K
  int ct;  // 128-column tiles per wave
  int kb;  // k per weight block (KB / 4 16-byte loads per column tile)
};

// Indexed by the cfg argument; the default for a shape is default_config().
static constexpr Config kConfigs[] = {
    {1, 8, 2, 32},  {2, 8, 2, 32},  {4, 4, 1, 32},  {6, 4, 1, 32},
    {8, 4, 1, 32},  {12, 4, 1, 32}, {16, 4, 1, 32}, {24, 4, 1, 32},
    {16, 8, 1, 32}, {16, 2, 1, 32}, {16, 1, 1, 32}, {16, 1, 2, 16},
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

// The four weights of one column in a dword -> half2 pairs (k, k + 1) and
// (k + 2, k + 3): fp8 times 2^-8, int8 exact (the sign-flipped byte under the
// exponent of 1024.0 is 1152 + w).
template <bool INT8>
__device__ __forceinline__ void widen4(uint32_t q, half2& lo, half2& hi) {
  if constexpr (INT8) {
    const uint32_t u = q ^ 0x80808080u;
    const half2 bias =
        __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
    uint32_t l = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u);
    uint32_t h = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u);
    lo = __hsub2(*reinterpret_cast<half2*>(&l), bias);
    hi = __hsub2(*reinterpret_cast<half2*>(&h), bias);
  } else {
    lo = rdna2::fp8x2_to_half2(q, rdna2::FP8_LO);
    hi = rdna2::fp8x2_to_half2(q, rdna2::FP8_HI);
  }
}

// acc + a . w for 8 activations (one 16-byte load) and 8 widened weights.
template <typename T>
__device__ __forceinline__ float dot8(const uint4& a, const half2 (&w)[4],
                                      float acc);

template <>
__device__ __forceinline__ float dot8<__half>(const uint4& a,
                                              const half2 (&w)[4], float acc) {
  const half2* ah = reinterpret_cast<const half2*>(&a);
  #pragma unroll
  for (int j = 0; j < 4; j++)
    acc = __builtin_amdgcn_fdot2(ah[j], w[j], acc, false);
  return acc;
}

template <>
__device__ __forceinline__ float dot8<__hip_bfloat16>(const uint4& a,
                                                      const half2 (&w)[4],
                                                      float acc) {
  const uint32_t* au = reinterpret_cast<const uint32_t*>(&a);
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const float2 wf = __half22float2(w[j]);
    acc = fmaf(__uint_as_float(au[j] << 16), wf.x, acc);
    acc = fmaf(__uint_as_float(au[j] & 0xffff0000u), wf.y, acc);
  }
  return acc;
}

__device__ __forceinline__ uint32_t dword(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// grid (ceil(M / T), ceil(N / (128 * CT))). W holds ldw 16-byte column quads
// per dword row; R (BLOCK) the ratios of k-block k >> bk_shift at
// R[kb * ldr + n].
template <typename T, int TT, int NW, int CT, int KBLOCK, bool INT8, bool BLOCK>
__global__ void __launch_bounds__(NW* WARP32)
    gemm_w8_rdna2_kernel(const uint4* __restrict__ W,
                         const float* __restrict__ S,
                         const __half* __restrict__ R, const T* __restrict__ A,
                         const T* __restrict__ bias, T* __restrict__ C,
                         const int M, const int N, const int K, const int ldw,
                         const int ldr, const int bk_shift, const int lda,
                         const int ldc) {
  extern __shared__ __align__(16) float red[];
  const int lane = threadIdx.x % WARP32;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP32);
  const int t0 = blockIdx.x * TT;
  const int blocks = K / KBLOCK;
  const int b_begin = blocks * wave / NW, b_end = blocks * (wave + 1) / NW;
  const int n4 = N / 4;
  int col[CT];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
    col[ct] = min(blockIdx.y * WARP32 * CT + ct * WARP32 + lane, n4 - 1);

  float acc[CT][4][TT];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
  #pragma unroll
    for (int c = 0; c < 4; c++)
  #pragma unroll
      for (int i = 0; i < TT; i++) acc[ct][c][i] = 0.f;

  half2 ratio[CT][4];
  int kb_cur = -1;

  // Two dword rows (8 k from k0) of each column.
  auto mac = [&](const uint4(&w)[2][CT], const int k0) {
    if constexpr (BLOCK) {
      const int kb = k0 >> bk_shift;
      if (kb != kb_cur) {
        kb_cur = kb;
  #pragma unroll
        for (int ct = 0; ct < CT; ct++) {
          const uint2 r =
              *reinterpret_cast<const uint2*>(R + (long)kb * ldr + col[ct] * 4);
          const half2 r01 = *reinterpret_cast<const half2*>(&r.x);
          const half2 r23 = *reinterpret_cast<const half2*>(&r.y);
          ratio[ct][0] = __low2half2(r01);
          ratio[ct][1] = __high2half2(r01);
          ratio[ct][2] = __low2half2(r23);
          ratio[ct][3] = __high2half2(r23);
        }
      }
    }
    half2 h[CT][4][4];
  #pragma unroll
    for (int ct = 0; ct < CT; ct++)
  #pragma unroll
      for (int c = 0; c < 4; c++) {
        widen4<INT8>(dword(w[0][ct], c), h[ct][c][0], h[ct][c][1]);
        widen4<INT8>(dword(w[1][ct], c), h[ct][c][2], h[ct][c][3]);
        if constexpr (BLOCK) {
  #pragma unroll
          for (int j = 0; j < 4; j++)
            h[ct][c][j] = __hmul2(h[ct][c][j], ratio[ct][c]);
        }
      }
  #pragma unroll
    for (int i = 0; i < TT; i++) {
      // Wave-uniform address: a scalar load.
      const uint4 av = *reinterpret_cast<const uint4*>(
          A + (long)min(t0 + i, M - 1) * lda + k0);
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++)
          acc[ct][c][i] = dot8<T>(av, h[ct][c], acc[ct][c][i]);
    }
  };

  for (int b = b_begin; b < b_end; b++) {
    uint4 w[KBLOCK / 4][CT];
  #pragma unroll
    for (int u = 0; u < KBLOCK / 4; u++)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
        w[u][ct] = W[(long)(b * (KBLOCK / 4) + u) * ldw + col[ct]];
  #pragma unroll
    for (int p = 0; p < KBLOCK / 8; p++)
      mac(reinterpret_cast<const uint4(&)[2][CT]>(w[2 * p]),
          b * KBLOCK + p * 8);
  }
  // The k past the last whole block (K % KBLOCK).
  if (wave == NW - 1)
    for (int kw = blocks * (KBLOCK / 4); kw < K / 4; kw += 2) {
      uint4 w[2][CT];
  #pragma unroll
      for (int u = 0; u < 2; u++)
  #pragma unroll
        for (int ct = 0; ct < CT; ct++)
          w[u][ct] = W[(long)(kw + u) * ldw + col[ct]];
      mac(w, kw * 4);
    }

  // Tree reduction of the NW K slices: in each round the upper half of the
  // remaining waves hands its sums to the lower half. Rolled loops keep the
  // epilogue's registers (and so the kernel's occupancy) low.
  constexpr int PER_WAVE = CT * 4 * TT * WARP32;
  #pragma unroll 1
  for (int half = NW / 2; half >= 1; half /= 2) {
    if (wave >= half && wave < 2 * half)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++)
  #pragma unroll
          for (int i = 0; i < TT; i++)
            red[(wave - half) * PER_WAVE + ((ct * 4 + c) * TT + i) * WARP32 +
                lane] = acc[ct][c][i];
    __syncthreads();
    if (wave < half)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
  #pragma unroll
        for (int c = 0; c < 4; c++)
  #pragma unroll
          for (int i = 0; i < TT; i++)
            acc[ct][c][i] +=
                red[wave * PER_WAVE + ((ct * 4 + c) * TT + i) * WARP32 + lane];
    __syncthreads();
  }

  if (wave == 0) {
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      const int c4 = blockIdx.y * WARP32 * CT + ct * WARP32 + lane;
      if (c4 >= n4) continue;
      const float4 sv = *reinterpret_cast<const float4*>(S + c4 * 4);
      // fp8 sums carry the 2^-8 of the in-register widening.
      const float m = INT8 ? 1.f : 256.f;
      const float s[4] = {sv.x * m, sv.y * m, sv.z * m, sv.w * m};
      float bv[4] = {0.f, 0.f, 0.f, 0.f};
      if (bias != nullptr)
  #pragma unroll
        for (int c = 0; c < 4; c++) bv[c] = to_float<T>(bias[c4 * 4 + c]);
  #pragma unroll
      for (int i = 0; i < TT; i++) {
        if (t0 + i < M) {
          __align__(8) T o[4];
  #pragma unroll
          for (int c = 0; c < 4; c++)
            o[c] = from_float<T>(fmaf(acc[ct][c][i], s[c], bv[c]));
          *reinterpret_cast<uint2*>(C + (long)(t0 + i) * ldc + c4 * 4) =
              *reinterpret_cast<const uint2*>(o);
        }
      }
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <typename T, int TT, int NW, int CT, int KBLOCK, bool INT8, bool BLOCK>
__global__ void gemm_w8_rdna2_kernel(const uint4*, const float*, const __half*,
                                     const T*, const T*, T*, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

struct Args {
  const at::Tensor& a;
  const at::Tensor& w;
  const at::Tensor& scale;
  const std::optional<at::Tensor>& block_scale;
  const std::optional<at::Tensor>& bias;
  at::Tensor& c;
  int bk_shift;
};

template <typename T, int TT, int NW, int CT, int KB, bool INT8, bool BLOCK>
void launch(const Args& p, cudaStream_t stream) {
  const int M = p.a.size(0), K = p.a.size(1), N = p.w.size(1);
  const dim3 grid((M + TT - 1) / TT, (N / 4 + WARP32 * CT - 1) / (WARP32 * CT));
  const size_t lds = (size_t)(NW / 2) * CT * 4 * TT * WARP32 * sizeof(float);
  gemm_w8_rdna2_kernel<T, TT, NW, CT, KB, INT8, BLOCK>
      <<<grid, dim3(NW * WARP32), lds, stream>>>(
          (const uint4*)p.w.data_ptr(), p.scale.data_ptr<float>(),
          BLOCK ? (const __half*)p.block_scale->data_ptr() : nullptr,
          (const T*)p.a.data_ptr(),
          p.bias.has_value() ? (const T*)p.bias->data_ptr() : nullptr,
          (T*)p.c.data_ptr(), M, N, K, p.w.stride(0) / 16,
          BLOCK ? p.block_scale->stride(0) : 0, p.bk_shift, p.a.stride(0),
          p.c.stride(0));
}

template <int TT, int NW, int CT, int KB>
void launch_config(const Args& p, cudaStream_t stream) {
  const bool int8 = p.w.dtype() == torch::kInt8;
  const bool block = p.block_scale.has_value();
  auto by_format = [&](auto t) {
    using T = decltype(t);
    if (int8)
      launch<T, TT, NW, CT, KB, true, false>(p, stream);
    else if (block)
      launch<T, TT, NW, CT, KB, false, true>(p, stream);
    else
      launch<T, TT, NW, CT, KB, false, false>(p, stream);
  };
  if (p.a.dtype() == torch::kFloat16)
    by_format(__half{});
  else
    by_format(__hip_bfloat16{});
}

// Default config, fitted on the linear layers of Qwen3.8-27B and
// Qwen3.6-35B-A3B at steady clocks (within 1-3% of the best config summed per
// row count): a token tile that wastes few rows; on narrow layers (few column
// tiles) smaller tiles or more waves per workgroup to fill the GPU; on wide
// ones, two column tiles per wave from 33 rows on.
static int default_config(int M, int N) {
  if (N <= 4096) {
    if (M <= 16) return 2;
    if (M <= 64) return 4;
    return M <= 1024 ? 7 : 11;
  }
  const bool narrow = N <= (M <= 24 ? 10240 : 6144), wide = N >= 32768;
  if (M == 1) return 0;
  if (M == 2) return 1;
  if (M <= 4) return 2;
  if (M <= 6) return 3;
  if (M <= 8) return 4;
  if (M <= 12) return narrow ? 3 : 5;
  if (M <= 16) return narrow ? 4 : 6;
  if (M <= 24) return narrow ? 5 : 7;
  if (M <= 32) return narrow ? 4 : wide ? 11 : 6;
  if (M <= 48) return narrow ? 8 : 7;
  if (M <= 128) return narrow ? 8 : wide ? 11 : 7;
  if (M <= 256) return N >= 16384 ? 11 : 7;
  return 11;
}

}  // namespace gemm_w8_rdna2
}  // namespace vllm

// Requirements: a [M, K] fp16 or bf16 with K-contiguous, 16-byte aligned rows
// and K % 16 == 0; w [K / 4, N, 4] int8 or float8_e4m3fn (kmajor_w8), its
// dword rows 16-byte aligned (stride(0) % 16 == 0, so N % 4 == 0); scale [N]
// fp32 contiguous; optional block_scale [ceil(K / block_k), N] fp16 with
// N-contiguous rows (stride(0) % 4 == 0) multiplying the weights of each
// block_k slice (fp8 only; block_k a power of two >= 8); optional contiguous
// bias [N] of a's dtype. cfg < 0 picks the default config for M. Returns
// [M, N] in a's dtype.
torch::Tensor gemm_w8_rdna2(const at::Tensor& a, const at::Tensor& w,
                            const at::Tensor& scale,
                            const std::optional<at::Tensor>& block_scale,
                            int64_t block_k,
                            const std::optional<at::Tensor>& bias,
                            int64_t cfg) {
  using namespace vllm::gemm_w8_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16,
              "gemm_w8_rdna2 needs fp16 or bf16 activations");
  TORCH_CHECK(
      w.dtype() == torch::kInt8 || w.dtype() == at::ScalarType::Float8_e4m3fn,
      "gemm_w8_rdna2 needs int8 or float8_e4m3fn weights");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 3 && w.size(2) == 4 &&
                  w.size(0) * 4 == a.size(1),
              "gemm_w8_rdna2 needs a [M, K] and w [K / 4, N, 4]");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(1);
  TORCH_CHECK(K % 16 == 0 && a.stride(1) == 1 && a.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "gemm_w8_rdna2 needs K % 16 == 0 and 16-byte aligned, "
              "K-contiguous rows of a");
  TORCH_CHECK(N % 4 == 0 && w.stride(2) == 1 && w.stride(1) == 4 &&
                  w.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "gemm_w8_rdna2 needs N % 4 == 0 and 16-byte aligned dword rows "
              "of w");
  TORCH_CHECK(scale.dtype() == torch::kFloat32 && scale.numel() == N &&
                  scale.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(scale.data_ptr()) % 16 == 0,
              "gemm_w8_rdna2 needs a 16-byte aligned, contiguous fp32 scale "
              "per output channel");
  int bk_shift = 0;
  if (block_scale.has_value()) {
    TORCH_CHECK(w.dtype() != torch::kInt8 && block_k >= 8 &&
                    (block_k & (block_k - 1)) == 0,
                "gemm_w8_rdna2 block scales need fp8 weights and block_k a "
                "power of two >= 8");
    TORCH_CHECK(
        block_scale->dtype() == torch::kFloat16 && block_scale->dim() == 2 &&
            block_scale->size(0) == (K + block_k - 1) / block_k &&
            block_scale->size(1) == N && block_scale->stride(1) == 1 &&
            block_scale->stride(0) % 4 == 0 &&
            reinterpret_cast<uintptr_t>(block_scale->data_ptr()) % 8 == 0,
        "gemm_w8_rdna2 needs fp16 block scales [ceil(K / block_k), N] "
        "with 8-byte aligned, N-contiguous rows");
    bk_shift = __builtin_ctzll(block_k);
  }
  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == a.dtype() && bias->numel() == N &&
                    bias->is_contiguous(),
                "gemm_w8_rdna2 bias must be a contiguous [N] of a's dtype");
  }
  TORCH_CHECK(cfg < kNumConfigs, "gemm_w8_rdna2: cfg out of range");

  auto c = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0) return c;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Args p{a, w, scale, block_scale, bias, c, bk_shift};
  const int id = cfg < 0 ? default_config(M, N) : (int)cfg;

#define VLLM_W8_RDNA2_CASE(ID, T, NW, CT, KB)                      \
  case ID:                                                         \
    static_assert(kConfigs[ID].t == T && kConfigs[ID].nw == NW &&  \
                  kConfigs[ID].ct == CT && kConfigs[ID].kb == KB); \
    launch_config<T, NW, CT, KB>(p, stream);                       \
    break;
  switch (id) {
    VLLM_W8_RDNA2_CASE(0, 1, 8, 2, 32)
    VLLM_W8_RDNA2_CASE(1, 2, 8, 2, 32)
    VLLM_W8_RDNA2_CASE(2, 4, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(3, 6, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(4, 8, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(5, 12, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(6, 16, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(7, 24, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(8, 16, 8, 1, 32)
    VLLM_W8_RDNA2_CASE(9, 16, 2, 1, 32)
    VLLM_W8_RDNA2_CASE(10, 16, 1, 1, 32)
    VLLM_W8_RDNA2_CASE(11, 16, 1, 2, 16)
  }
#undef VLLM_W8_RDNA2_CASE
  return c;
}
