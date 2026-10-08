// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) GEMM on 8-bit weights for decode and prefill alike:
// C[M, N] = A[M, K] . (W * scale)^T (+ bias), with
//  - fp8 e4m3fn or int8 weights and fp16/bf16 activations (weight-only):
//    gfx1030 has no fp8 instructions; both formats are widened exactly in
//    registers;
//  - int8 weights and int8 activations (W8A8) with a per-token activation
//    scale: v_dot4_i32_i8 on the bytes as stored, int32 accumulation, and the
//    epilogue of triton_scaled_mm (identical results).
//
// The weights are stored K-major, w [K / 4, N, 4] (four consecutive k of one
// column per dword; kmajor_w8 in rdna2_w8a16.py makes it at load time). As in
// gemm_w4a16_exl_rdna2, lanes run along N with four columns each (one 16-byte
// load = 4 columns x 4 k), so all lanes of a wave use the same activations,
// which come through scalar loads; the NW waves of a workgroup split K in
// 16- to 64-k blocks and reduce through LDS in a fixed order (deterministic,
// no atomics); T tokens share each loaded weight. fp8 widens through
// rdna2::fp8x2_to_half2 (its 2^-8 is folded into the output scale), int8 by
// placing the byte in the mantissa of 1024.0; fp16 activations accumulate
// through v_dot2_f32_f16, bf16 (no dot instruction on gfx1030) through fp32
// FMAs. 2D block scales are split at load time into a scale per column (the
// largest of its blocks) and fp16 ratios <= 1 per block that
// multiply the widened weights.
//
// On the linear layers of Qwen3.8-27B at steady clocks: 440-490 GB/s at 1-8
// rows, 300-340 at 16, ~200 at 32, 16-18 TFLOPS at 128 rows and ~20 from 1024
// on, above rocBLAS on fp16 weights (15-18 TFLOPS); the per-row GEMV it
// replaces fell to ~210 GB/s at 8 rows and ~90 at 16, and larger batches
// dequantized for rocBLAS. W8A8: 430-480 GB/s up to 16 rows, 340-380 at 32,
// 32-36 TOPS at 128 rows and 38-40 from 1024 on (w8a8_gemv_rdna2: 270-350 at
// 16 rows; w8a8_gemm_rdna2: ~110 at 32, 24-30 TOPS at 128).
//
// fp16 weights from 768 rows on run a separate LDS-tiled kernel
// (gemm_w16_tiled_kernel): 256 threads own 8 x 16 outputs each in a 128 x 256
// tile, K staged 32 at a time (double buffered) in k2-major LDS; the K-major
// weight rows are already k2-major, so they go to LDS as coalesced 16-byte
// loads. ~1.08x rocBLAS at 1K-8K rows on the Qwen3.6-35B dense shapes, where
// the lanes-along-N tiles reach 0.96-1.05x.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <type_traits>

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
// fp16/bf16 activations:
static constexpr Config kConfigs[] = {
    {1, 8, 2, 32},  {2, 8, 2, 32},  {4, 4, 1, 32},  {6, 4, 1, 32},
    {8, 4, 1, 32},  {12, 4, 1, 32}, {16, 4, 1, 32}, {24, 4, 1, 32},
    {16, 8, 1, 32}, {16, 2, 1, 32}, {16, 1, 1, 32}, {16, 1, 2, 16},
    {1, 4, 1, 32},  {1, 16, 1, 16}, {4, 8, 1, 32},
};
// int8 activations, which need no widening and so take longer weight blocks
// and token tiles:
static constexpr Config kConfigsI8[] = {
    {1, 8, 2, 64},  {2, 8, 2, 64},  {4, 4, 1, 64},  {8, 4, 1, 32},
    {16, 2, 1, 64}, {16, 4, 1, 32}, {24, 4, 1, 32}, {32, 4, 1, 32},
    {16, 4, 1, 64}, {32, 2, 1, 32}, {32, 1, 1, 32}, {16, 1, 2, 32},
    {16, 2, 2, 32},
};
// fp16 weights (two k per dword, so a 16-k block is 8 loads per column tile):
static constexpr Config kConfigsW16[] = {
    {1, 8, 2, 16},  {1, 16, 1, 16}, {2, 8, 2, 16},  {4, 8, 1, 16},
    {8, 4, 1, 32},  {8, 8, 1, 16},  {16, 4, 1, 16}, {16, 8, 1, 16},
    {16, 2, 1, 16}, {16, 1, 1, 16}, {16, 1, 2, 8},  {24, 4, 1, 16},
    {32, 2, 1, 8},
};
static constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
static constexpr int kNumConfigsW16 =
    sizeof(kConfigsW16) / sizeof(kConfigsW16[0]);
// Config index of the LDS-tiled fp16 kernel (after the lanes-along-N ones).
static constexpr int kTiledW16 = kNumConfigsW16;
static constexpr int TILED_THREADS = 256;
static constexpr int kNumConfigsI8 = sizeof(kConfigsI8) / sizeof(kConfigsI8[0]);
// Workgroups a split-K launch aims for (2 per CU on a 72-CU V620).
static constexpr int kSplitKTarget = 144;

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

// grid (ceil(M / T), ceil(N / (128 * CT)), KS). T is the activation type
// (int8_t: W8A8), O the output type; W16: fp16 weights, two k per dword, no
// scale. SPLIT (float sums only): the NW waves of the KS workgroups of a tile
// take consecutive K slices and write raw sums to P [KS, M, N] for
// splitk_epilogue_kernel (a separate instantiation, so unsplit launches keep
// their code generation: a runtime slice count slowed them up to 3x). W
// holds ldw 16-byte column quads per dword row; R (BLOCK) the ratios of
// k-block k >> bk_shift at R[kb * ldr + n / bn]; SA (W8A8) the activation
// scale of row t at SA[t * sa_stride].
template <typename T, typename O, int TT, int NW, int CT, int KBLOCK, bool INT8,
          bool BLOCK, bool W16, bool SPLIT>
__global__ void __launch_bounds__(NW* WARP32)
    gemm_w8_rdna2_kernel(const uint4* __restrict__ W,
                         const float* __restrict__ S,
                         const float* __restrict__ SA, const int sa_stride,
                         const __half* __restrict__ R, const T* __restrict__ A,
                         const O* __restrict__ bias, O* __restrict__ C,
                         const int M, const int N, const int K, const int ldw,
                         const int ldr, const int bn, const int bk_shift,
                         const int lda, const int ldc, float* __restrict__ P) {
  constexpr bool DOT4 = std::is_same_v<T, int8_t>;
  static_assert(!(SPLIT && DOT4), "split K sums floats");
  using Acc = std::conditional_t<DOT4, int, float>;
  // k per weight dword, per activation load (16 bytes) and the dword rows the
  // latter covers.
  constexpr int KPD = W16 ? 2 : 4;
  constexpr int STEP = DOT4 ? 16 : 8;
  constexpr int ROWS = STEP / KPD;
  extern __shared__ __align__(16) char smem[];
  Acc* red = reinterpret_cast<Acc*>(smem);
  const int lane = threadIdx.x % WARP32;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP32);
  const int t0 = blockIdx.x * TT;
  const int blocks = K / KBLOCK;
  int b_begin, b_end;
  bool tail;  // the k past the last whole block (K % KBLOCK)
  if constexpr (SPLIT) {
    const int vw = blockIdx.z * NW + wave, vws = gridDim.z * NW;
    b_begin = blocks * vw / vws;
    b_end = blocks * (vw + 1) / vws;
    tail = vw == vws - 1;
  } else {
    b_begin = blocks * wave / NW;
    b_end = blocks * (wave + 1) / NW;
    tail = wave == NW - 1;
  }
  const int n4 = N / 4;
  int col[CT];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
    col[ct] = min(blockIdx.y * WARP32 * CT + ct * WARP32 + lane, n4 - 1);

  Acc acc[CT][4][TT];
  #pragma unroll
  for (int ct = 0; ct < CT; ct++)
  #pragma unroll
    for (int c = 0; c < 4; c++)
  #pragma unroll
      for (int i = 0; i < TT; i++) acc[ct][c][i] = 0;

  half2 ratio[CT][4];
  int kb_cur = -1;

  // ROWS dword rows (STEP k from k0) of each column.
  auto mac = [&](const uint4(&w)[ROWS][CT], const int k0) {
    if constexpr (DOT4) {
  #pragma unroll
      for (int i = 0; i < TT; i++) {
        // Wave-uniform address: a scalar load.
        const uint4 av = *reinterpret_cast<const uint4*>(
            A + (long)min(t0 + i, M - 1) * lda + k0);
  #pragma unroll
        for (int ct = 0; ct < CT; ct++)
  #pragma unroll
          for (int c = 0; c < 4; c++) {
            int s = acc[ct][c][i];
            s = __builtin_amdgcn_sdot4(av.x, dword(w[0][ct], c), s, false);
            s = __builtin_amdgcn_sdot4(av.y, dword(w[1][ct], c), s, false);
            s = __builtin_amdgcn_sdot4(av.z, dword(w[2][ct], c), s, false);
            s = __builtin_amdgcn_sdot4(av.w, dword(w[3][ct], c), s, false);
            acc[ct][c][i] = s;
          }
      }
    } else if constexpr (W16) {
  #pragma unroll
      for (int i = 0; i < TT; i++) {
        // Wave-uniform address: a scalar load.
        const uint4 av = *reinterpret_cast<const uint4*>(
            A + (long)min(t0 + i, M - 1) * lda + k0);
  #pragma unroll
        for (int ct = 0; ct < CT; ct++)
  #pragma unroll
          for (int c = 0; c < 4; c++) {
            half2 h[4];
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              const uint32_t q = dword(w[j][ct], c);
              h[j] = *reinterpret_cast<const half2*>(&q);
            }
            acc[ct][c][i] = dot8<T>(av, h, acc[ct][c][i]);
          }
      }
    } else {
      if constexpr (BLOCK) {
        const int kb = k0 >> bk_shift;
        if (kb != kb_cur) {
          kb_cur = kb;
  #pragma unroll
          for (int ct = 0; ct < CT; ct++)
  #pragma unroll
            for (int c = 0; c < 4; c++)
              ratio[ct][c] =
                  __half2half2(R[(long)kb * ldr + (col[ct] * 4 + c) / bn]);
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
    }
  };

  for (int b = b_begin; b < b_end; b++) {
    uint4 w[KBLOCK / KPD][CT];
  #pragma unroll
    for (int u = 0; u < KBLOCK / KPD; u++)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
        w[u][ct] = W[(long)(b * (KBLOCK / KPD) + u) * ldw + col[ct]];
  #pragma unroll
    for (int p = 0; p < KBLOCK / STEP; p++)
      mac(reinterpret_cast<const uint4(&)[ROWS][CT]>(w[ROWS * p]),
          b * KBLOCK + p * STEP);
  }
  if (tail)
    for (int kw = blocks * (KBLOCK / KPD); kw < K / KPD; kw += ROWS) {
      uint4 w[ROWS][CT];
  #pragma unroll
      for (int u = 0; u < ROWS; u++)
  #pragma unroll
        for (int ct = 0; ct < CT; ct++)
          w[u][ct] = W[(long)(kw + u) * ldw + col[ct]];
      mac(w, kw * KPD);
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

  if constexpr (SPLIT) {
    if (wave == 0)
  #pragma unroll
      for (int ct = 0; ct < CT; ct++) {
        const int c4 = blockIdx.y * WARP32 * CT + ct * WARP32 + lane;
        if (c4 >= n4) continue;
  #pragma unroll
        for (int i = 0; i < TT; i++)
          if (t0 + i < M)
            *reinterpret_cast<float4*>(P + ((long)blockIdx.z * M + t0 + i) * N +
                                       c4 * 4) =
                make_float4(acc[ct][0][i], acc[ct][1][i], acc[ct][2][i],
                            acc[ct][3][i]);
      }
    return;
  }

  if (wave == 0) {
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) {
      const int c4 = blockIdx.y * WARP32 * CT + ct * WARP32 + lane;
      if (c4 >= n4) continue;
      float s[4] = {1.f, 1.f, 1.f, 1.f};
      if constexpr (!W16) {
        const float4 sv = *reinterpret_cast<const float4*>(S + c4 * 4);
        // fp8 sums carry the 2^-8 of the in-register widening.
        const float m = INT8 ? 1.f : 256.f;
        s[0] = sv.x * m;
        s[1] = sv.y * m;
        s[2] = sv.z * m;
        s[3] = sv.w * m;
      }
      float bv[4] = {0.f, 0.f, 0.f, 0.f};
      if (bias != nullptr)
  #pragma unroll
        for (int c = 0; c < 4; c++) bv[c] = to_float<O>(bias[c4 * 4 + c]);
  #pragma unroll
      for (int i = 0; i < TT; i++) {
        if (t0 + i < M) {
          __align__(8) O o[4];
          if constexpr (DOT4) {
            // As triton_scaled_mm: round the scaled sum, then add the bias.
            const float sa = SA[(t0 + i) * sa_stride];
  #pragma unroll
            for (int c = 0; c < 4; c++) {
              o[c] = from_float<O>((float)acc[ct][c][i] * sa * s[c]);
              if (bias != nullptr)
                o[c] = from_float<O>(to_float<O>(o[c]) + bv[c]);
            }
          } else {
  #pragma unroll
            for (int c = 0; c < 4; c++)
              o[c] = from_float<O>(fmaf(acc[ct][c][i], s[c], bv[c]));
          }
          *reinterpret_cast<uint2*>(C + (long)(t0 + i) * ldc + c4 * 4) =
              *reinterpret_cast<const uint2*>(o);
        }
      }
    }
  }
}

// Split-K epilogue: C = the sum over the KS slices of P (in slice order: the
// result is deterministic) times the column scale (x 256 for fp8, none for
// fp16 weights) plus bias; one thread per 4 columns of one row.
template <typename O, bool INT8, bool W16>
__global__ void splitk_epilogue_kernel(const float* __restrict__ P,
                                       const float* __restrict__ S,
                                       const O* __restrict__ bias,
                                       O* __restrict__ C, const int M,
                                       const int N, const int KS,
                                       const int ldc) {
  const int n4 = N / 4;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)M * n4) return;
  const int t = idx / n4, c4 = idx % n4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int z = 0; z < KS; z++) {
    const float4 v =
        *reinterpret_cast<const float4*>(P + ((long)z * M + t) * N + c4 * 4);
    acc[0] += v.x;
    acc[1] += v.y;
    acc[2] += v.z;
    acc[3] += v.w;
  }
  float s[4] = {1.f, 1.f, 1.f, 1.f};
  if constexpr (!W16) {
    const float4 sv = *reinterpret_cast<const float4*>(S + c4 * 4);
    const float m = INT8 ? 1.f : 256.f;
    s[0] = sv.x * m;
    s[1] = sv.y * m;
    s[2] = sv.z * m;
    s[3] = sv.w * m;
  }
  __align__(8) O o[4];
  #pragma unroll
  for (int c = 0; c < 4; c++) {
    const float b = bias != nullptr ? to_float<O>(bias[c4 * 4 + c]) : 0.f;
    o[c] = from_float<O>(fmaf(acc[c], s[c], b));
  }
  *reinterpret_cast<uint2*>(C + (long)t * ldc + c4 * 4) =
      *reinterpret_cast<const uint2*>(o);
}

// Qwen4Exp hyper-connection gate (GatedResidual) for a few tokens, three ops
// in one: y = silu(lora / HC) (rounded to T as the unfused op stores it),
// g = y @ W^T * scale (int8 K-major W [R / 4, HC * HS, 4]; rounded to T) and
// out[t, h] = sum_s sigmoid(g[t, s * HS + h]) * xn[t, s * HS + h] / HC.
// HC = 4: lane l owns stream l / 8 and columns 4 * (l % 8) .. + 3 of a
// 32-column tile, so the stream sum is two shuffles. grid (HS / 32,
// ceil(M / TT)); the NW waves split R. Faster than the three ops up to 4
// rows (at 8 the GEMM's tiles win).
static constexpr int kHcStreams = 4;
static constexpr int kHcMaxRank = 512;

template <typename T, int TT, int NW>
__global__ void __launch_bounds__(NW* WARP32)
    hc_up_mix_kernel(const uint4* __restrict__ W, const float* __restrict__ S,
                     const T* __restrict__ lora, const int ldl,
                     const T* __restrict__ xn, const int ldx,
                     T* __restrict__ out, const int ldo, const int M,
                     const int HS, const int R, const int ldw,
                     const float inv_hc) {
  __shared__ __align__(16) T sy[TT * kHcMaxRank];
  __shared__ float red[NW - 1][4][TT][WARP32];
  const int lane = threadIdx.x % WARP32;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP32);
  const int t0 = blockIdx.y * TT;
  for (int idx = threadIdx.x; idx < TT * R; idx += NW * WARP32) {
    const int i = idx / R, k = idx % R;
    const float v =
        to_float<T>(lora[(long)min(t0 + i, M - 1) * ldl + k]) * inv_hc;
    sy[i * R + k] = from_float<T>(v / (1.f + __expf(-v)));
  }
  __syncthreads();

  const int s = lane / 8, h = blockIdx.x * 32 + lane % 8 * 4;
  const int col = (s * HS + h) / 4;
  float acc[4][TT];
  #pragma unroll
  for (int c = 0; c < 4; c++)
  #pragma unroll
    for (int i = 0; i < TT; i++) acc[c][i] = 0.f;
  const int steps = R / 8;
  for (int st = steps * wave / NW; st < steps * (wave + 1) / NW; st++) {
    const uint4 w0 = W[(long)(2 * st) * ldw + col];
    const uint4 w1 = W[(long)(2 * st + 1) * ldw + col];
    half2 hw[4][4];
  #pragma unroll
    for (int c = 0; c < 4; c++) {
      widen4<true>(dword(w0, c), hw[c][0], hw[c][1]);
      widen4<true>(dword(w1, c), hw[c][2], hw[c][3]);
    }
  #pragma unroll
    for (int i = 0; i < TT; i++) {
      const uint4 av = *reinterpret_cast<const uint4*>(sy + i * R + st * 8);
  #pragma unroll
      for (int c = 0; c < 4; c++) acc[c][i] = dot8<T>(av, hw[c], acc[c][i]);
    }
  }

  if (wave > 0)
  #pragma unroll
    for (int c = 0; c < 4; c++)
  #pragma unroll
      for (int i = 0; i < TT; i++) red[wave - 1][c][i][lane] = acc[c][i];
  __syncthreads();
  if (wave > 0) return;
  #pragma unroll
  for (int w = 0; w < NW - 1; w++)
  #pragma unroll
    for (int c = 0; c < 4; c++)
  #pragma unroll
      for (int i = 0; i < TT; i++) acc[c][i] += red[w][c][i][lane];

  const float4 sv = *reinterpret_cast<const float4*>(S + s * HS + h);
  const float sc[4] = {sv.x, sv.y, sv.z, sv.w};
  #pragma unroll
  for (int i = 0; i < TT; i++) {
    const int t = t0 + i;
    if (t >= M) break;  // wave-uniform: the shuffles below see every lane
    __align__(8) T x[4];
    *reinterpret_cast<uint2*>(x) =
        *reinterpret_cast<const uint2*>(xn + (long)t * ldx + s * HS + h);
    float v[4];
  #pragma unroll
    for (int c = 0; c < 4; c++) {
      const float g = to_float<T>(from_float<T>(acc[c][i] * sc[c]));
      v[c] = to_float<T>(x[c]) / (1.f + __expf(-g));
      v[c] += __shfl_xor(v[c], 8);
      v[c] += __shfl_xor(v[c], 16);
    }
    if (lane < 8) {
      __align__(8) T o[4];
  #pragma unroll
      for (int c = 0; c < 4; c++) o[c] = from_float<T>(v[c] * inv_hc);
      *reinterpret_cast<uint2*>(out + (long)t * ldo + h) =
          *reinterpret_cast<const uint2*>(o);
    }
  }
}

// Tiled K-major fp16 GEMM (large M): TM x TN outputs per thread in 4 x 4
// blocks spaced 64 apart, BK k per stage; W is the [K/2][N] dword view of the
// K-major weights.
template <int TM, int TN, int BK, int UNR>
__global__ void __launch_bounds__(TILED_THREADS)
    gemm_w16_tiled_kernel(const __half* __restrict__ A,
                          const uint32_t* __restrict__ W,
                          const __half* __restrict__ bias,
                          __half* __restrict__ C, const int M, const int N,
                          const int K, const int lda, const int ldw,
                          const int ldc) {
  constexpr int BM = 16 * TM, BN = 16 * TN, K2 = BK / 2;
  constexpr int A_CH = BK / 8;  // 16-byte chunks per row per stage
  constexpr int A_LOADS = BM * A_CH / TILED_THREADS;
  constexpr int W_LOADS = K2 * BN / 4 / TILED_THREADS;
  constexpr int GROUP_M = 8;
  __shared__ __align__(16) uint32_t sA[2][K2][BM];
  __shared__ __align__(16) uint32_t sW[2][K2][BN];
  // Grouped ordering: GROUP_M row blocks share each weight block in L2.
  const int pid = blockIdx.x;
  const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN;
  const int width = GROUP_M * num_n;
  const int first_m = pid / width * GROUP_M;
  const int group_rows = min(num_m - first_m, GROUP_M);
  const int m0 = (first_m + pid % width % group_rows) * BM;
  const int n0 = pid % width / group_rows * BN;
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;

  uint4 ra[A_LOADS], rw[W_LOADS];
  auto load = [&](int k0) {
  #pragma unroll
    for (int j = 0; j < A_LOADS; j++) {
      const int idx = tid + j * TILED_THREADS,
                row = min(m0 + idx / A_CH, M - 1);
      ra[j] = *reinterpret_cast<const uint4*>(A + (long)row * lda + k0 +
                                              idx % A_CH * 8);
    }
  #pragma unroll
    for (int j = 0; j < W_LOADS; j++) {
      const int idx = tid + j * TILED_THREADS, k2 = idx / (BN / 4);
      const int col = min(n0 + idx % (BN / 4) * 4, N - 4);
      rw[j] =
          *reinterpret_cast<const uint4*>(W + (long)(k0 / 2 + k2) * ldw + col);
    }
  };
  auto store = [&](int buf) {
  #pragma unroll
    for (int j = 0; j < A_LOADS; j++) {
      const int idx = tid + j * TILED_THREADS, r = idx / A_CH, c = idx % A_CH;
      sA[buf][c * 4 + 0][r] = ra[j].x;
      sA[buf][c * 4 + 1][r] = ra[j].y;
      sA[buf][c * 4 + 2][r] = ra[j].z;
      sA[buf][c * 4 + 3][r] = ra[j].w;
    }
  #pragma unroll
    for (int j = 0; j < W_LOADS; j++) {
      const int idx = tid + j * TILED_THREADS;
      *reinterpret_cast<uint4*>(&sW[buf][idx / (BN / 4)][idx % (BN / 4) * 4]) =
          rw[j];
    }
  };

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

  load(0);
  store(0);
  __syncthreads();
  const int stages = K / BK;
  for (int s = 0; s < stages; s++) {
    const int buf = s & 1;
    if (s + 1 < stages) load((s + 1) * BK);
  #pragma unroll UNR
    for (int k2 = 0; k2 < K2; k2++) {
      half2 av[TM], wv[TN];
  #pragma unroll
      for (int g = 0; g < TM / 4; g++) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(&sA[buf][k2][g * 64 + ty * 4]);
        av[g * 4] = *reinterpret_cast<const half2*>(&v.x);
        av[g * 4 + 1] = *reinterpret_cast<const half2*>(&v.y);
        av[g * 4 + 2] = *reinterpret_cast<const half2*>(&v.z);
        av[g * 4 + 3] = *reinterpret_cast<const half2*>(&v.w);
      }
  #pragma unroll
      for (int g = 0; g < TN / 4; g++) {
        const uint4 v =
            *reinterpret_cast<const uint4*>(&sW[buf][k2][g * 64 + tx * 4]);
        wv[g * 4] = *reinterpret_cast<const half2*>(&v.x);
        wv[g * 4 + 1] = *reinterpret_cast<const half2*>(&v.y);
        wv[g * 4 + 2] = *reinterpret_cast<const half2*>(&v.z);
        wv[g * 4 + 3] = *reinterpret_cast<const half2*>(&v.w);
      }
  #pragma unroll
      for (int i = 0; i < TM; i++)
  #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = __builtin_amdgcn_fdot2(av[i], wv[j], acc[i][j], false);
    }
    if (s + 1 < stages) {
      store(buf ^ 1);
      __syncthreads();
    }
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = m0 + i / 4 * 64 + ty * 4 + i % 4;
    if (row >= M) continue;
  #pragma unroll
    for (int g = 0; g < TN / 4; g++) {
      const int col = n0 + g * 64 + tx * 4;
      if (col >= N) continue;
      __half o[4];
  #pragma unroll
      for (int j = 0; j < 4; j++) {
        // Round, then add the bias in fp16 (as F.linear on fp16 does).
        o[j] = __float2half(acc[i][g * 4 + j]);
        if (bias) o[j] = __hadd(o[j], bias[col + j]);
      }
      *reinterpret_cast<uint2*>(C + (long)row * ldc + col) =
          *reinterpret_cast<const uint2*>(o);
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int TM, int TN, int BK, int UNR>
__global__ void gemm_w16_tiled_kernel(const __half*, const uint32_t*,
                                      const __half*, __half*, const int,
                                      const int, const int, const int,
                                      const int, const int) {}

template <typename T, typename O, int TT, int NW, int CT, int KBLOCK, bool INT8,
          bool BLOCK, bool W16, bool SPLIT>
__global__ void gemm_w8_rdna2_kernel(const uint4*, const float*, const float*,
                                     const int, const __half*, const T*,
                                     const O*, O*, const int, const int,
                                     const int, const int, const int, const int,
                                     const int, const int, const int, float*) {}

template <typename O, bool INT8, bool W16>
__global__ void splitk_epilogue_kernel(const float*, const float*, const O*, O*,
                                       const int, const int, const int,
                                       const int) {}

static constexpr int kHcStreams = 4;
static constexpr int kHcMaxRank = 512;

template <typename T, int TT, int NW>
__global__ void hc_up_mix_kernel(const uint4*, const float*, const T*,
                                 const int, const T*, const int, T*, const int,
                                 const int, const int, const int, const int,
                                 const float) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

struct Args {
  const at::Tensor& a;
  const at::Tensor& w;
  const std::optional<at::Tensor>& scale;
  const std::optional<at::Tensor>& block_scale;
  const std::optional<at::Tensor>& bias;
  const std::optional<at::Tensor>& scale_a;
  at::Tensor& c;
  int bn;
  int bk_shift;
};

template <typename T, typename O, int TT, int NW, int CT, int KB, bool INT8,
          bool BLOCK, bool W16 = false>
void launch(const Args& p, cudaStream_t stream) {
  constexpr bool DOT4 = std::is_same_v<T, int8_t>;
  using Acc = std::conditional_t<DOT4, int, float>;
  const int M = p.a.size(0), K = p.a.size(1), N = p.w.size(1);
  const int col_tiles = (N / 4 + WARP32 * CT - 1) / (WARP32 * CT);
  const int tiles = ((M + TT - 1) / TT) * col_tiles;
  // Narrow layers with long rows (e.g. a 10240 -> 336 projection: 3 column
  // tiles) leave most CUs idle: small token tiles split K over more
  // workgroups (fp32 partials + an epilogue kernel), ~2 per CU, >= 4 weight
  // blocks per wave. With >= 64 waves or short rows the epilogue costs more.
  int ksplit = 1;
  if constexpr (!DOT4 && TT <= 4)
    if (tiles * NW < 64 && K >= 4096)
      ksplit = std::max(
          1, std::min((kSplitKTarget + tiles - 1) / tiles, K / (KB * NW * 4)));
  const dim3 grid((M + TT - 1) / TT, col_tiles, ksplit);
  const size_t lds = (size_t)(NW / 2) * CT * 4 * TT * WARP32 * sizeof(Acc);
  const bool sa = p.scale_a.has_value();
  at::Tensor partials;
  if (ksplit > 1)
    partials = torch::empty({ksplit, M, N}, p.a.options().dtype(torch::kFloat));
  auto run = [&](auto kernel) {
    kernel<<<grid, dim3(NW * WARP32), lds, stream>>>(
        (const uint4*)p.w.data_ptr(),
        W16 ? nullptr : p.scale->data_ptr<float>(),
        sa ? p.scale_a->data_ptr<float>() : nullptr,
        sa && p.scale_a->numel() > 1 ? 1 : 0,
        BLOCK ? (const __half*)p.block_scale->data_ptr() : nullptr,
        (const T*)p.a.data_ptr(),
        p.bias.has_value() ? (const O*)p.bias->data_ptr() : nullptr,
        (O*)p.c.data_ptr(), M, N, K,
        (int)(p.w.stride(0) * p.w.element_size() / 16),
        BLOCK ? p.block_scale->stride(0) : 0, p.bn, p.bk_shift, p.a.stride(0),
        p.c.stride(0), ksplit > 1 ? partials.data_ptr<float>() : nullptr);
  };
  if constexpr (!DOT4 && TT <= 4) {
    if (ksplit > 1) {
      run(gemm_w8_rdna2_kernel<T, O, TT, NW, CT, KB, INT8, BLOCK, W16, true>);
      const long items = (long)M * (N / 4);
      splitk_epilogue_kernel<O, INT8, W16>
          <<<(items + 255) / 256, 256, 0, stream>>>(
              partials.data_ptr<float>(),
              W16 ? nullptr : p.scale->data_ptr<float>(),
              p.bias.has_value() ? (const O*)p.bias->data_ptr() : nullptr,
              (O*)p.c.data_ptr(), M, N, ksplit, p.c.stride(0));
      return;
    }
  }
  run(gemm_w8_rdna2_kernel<T, O, TT, NW, CT, KB, INT8, BLOCK, W16, false>);
}

// fp16/bf16 activations: weight format by w's dtype and block scales.
template <int TT, int NW, int CT, int KB>
void launch_config(const Args& p, cudaStream_t stream) {
  const bool int8 = p.w.dtype() == torch::kInt8;
  const bool block = p.block_scale.has_value();
  auto by_format = [&](auto t) {
    using T = decltype(t);
    if (int8)
      launch<T, T, TT, NW, CT, KB, true, false>(p, stream);
    else if (block)
      launch<T, T, TT, NW, CT, KB, false, true>(p, stream);
    else
      launch<T, T, TT, NW, CT, KB, false, false>(p, stream);
  };
  if (p.a.dtype() == torch::kFloat16)
    by_format(__half{});
  else
    by_format(__hip_bfloat16{});
}

// fp16 weights (and activations).
template <int TT, int NW, int CT, int KB>
void launch_config_w16(const Args& p, cudaStream_t stream) {
  launch<__half, __half, TT, NW, CT, KB, false, false, true>(p, stream);
}

// The tiled fp16 kernel (config kTiledW16): 128 x 256 tiles, K % 32 == 0.
void launch_tiled_w16(const Args& p, cudaStream_t stream) {
  constexpr int TM = 8, TN = 16, BK = 32, UNR = 2;
  const int M = p.a.size(0), K = p.a.size(1), N = p.w.size(1);
  const int grid =
      ((M + 16 * TM - 1) / (16 * TM)) * ((N + 16 * TN - 1) / (16 * TN));
  gemm_w16_tiled_kernel<TM, TN, BK, UNR><<<grid, TILED_THREADS, 0, stream>>>(
      (const __half*)p.a.data_ptr(), (const uint32_t*)p.w.data_ptr(),
      p.bias ? (const __half*)p.bias->data_ptr() : nullptr,
      (__half*)p.c.data_ptr(), M, N, K, p.a.stride(0), p.w.stride(0) / 2, N);
}

// int8 activations (W8A8): output type by c's dtype.
template <int TT, int NW, int CT, int KB>
void launch_config_i8(const Args& p, cudaStream_t stream) {
  if (p.c.dtype() == torch::kFloat16)
    launch<int8_t, __half, TT, NW, CT, KB, true, false>(p, stream);
  else
    launch<int8_t, __hip_bfloat16, TT, NW, CT, KB, true, false>(p, stream);
}

// Default config, fitted on the linear layers of Qwen3.8-27B and
// Qwen3.6-35B-A3B at steady clocks (within 1-3% of the best config summed per
// row count; up to 4 rows timed inside HIP graphs, with Qwen3.8-Flash-Next's
// layers): a token tile that wastes few rows; on narrow layers (few column
// tiles) smaller tiles or more waves per workgroup to fill the GPU; on wide
// ones, two column tiles per wave at 2 rows and from 33 rows on.
static int default_config(int M, int N, int K) {
  // At most 10 column tiles: 16 waves per workgroup split K (8 on long rows,
  // which also split K over workgroups).
  if (M <= 4 && N <= 1280) return K <= 4096 ? 13 : 14;
  const bool wide = N >= 32768;
  if (M == 1) return wide ? 0 : 12;
  if (M == 2 && wide) return 1;
  if (N <= 4096) {
    if (M <= 16) return 2;
    if (M <= 64) return 4;
    return M <= 1024 ? 7 : 11;
  }
  const bool narrow = N <= (M <= 24 ? 10240 : 6144);
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

// The same for int8 activations, fitted on the Qwen3.8-27B and
// Qwen3.6-35B-A3B layers (within 0-3% of the best config summed per row
// count): layers with few column tiles and long rows (N <= 2048, K >= 2048)
// take tiny token tiles split 8 ways along K up to 16 rows; short rows
// (K <= 2048) keep two waves per workgroup where long ones take one.
static int default_config_i8(int M, int N, int K) {
  if (N <= 2048 && K >= 2048) {
    if (M <= 16) return 1;
    if (M <= 512) return 5;
    return M <= 1024 ? 8 : 11;
  }
  const bool narrow = N <= 6144, short_k = K <= 2048;
  if (M == 1) return 0;
  if (M == 2) return 1;
  if (M <= 4) return 2;
  if (M <= 8) return 3;
  if (M <= 16) return narrow || short_k ? 5 : 4;
  if (M <= 24) return 6;
  if (M <= 32) return narrow ? 8 : short_k ? 4 : 7;
  if (M <= 48) return narrow ? 4 : 6;
  if (M <= 256) return narrow ? 4 : M <= 64 || short_k ? 9 : 10;
  return M <= 512 ? 12 : 11;
}

// The same for fp16 weights, fitted on the Qwen3.6-35B-A3B fp16 layers (and
// lm_head) and Llama-3-8B-sized ones (within 0-3% of the best config summed
// per row count).
static int default_config_w16(int M, int N, int K) {
  const bool tiny = N <= 1024, small = N <= 2048, narrow = N <= 4096,
             wide = N >= 16384;
  if (M == 1) return wide ? 0 : 3;
  if (M == 2) return wide ? 2 : 3;
  if (M <= 4) return small || K <= 2048 ? 3 : 4;
  if (M <= 8) return small ? 3 : K <= 2048 ? 5 : 4;
  if (M <= 16) return small ? 3 : narrow ? 5 : 8;
  if (M <= 24) return tiny ? 3 : small ? 5 : narrow ? 4 : 11;
  if (M <= 32) return tiny ? 3 : small ? 5 : narrow ? 6 : wide ? 12 : 8;
  if (M <= 48) return small ? 5 : 11;
  if (M <= 64) return tiny ? 5 : small ? 6 : N >= 12288 ? 12 : 8;
  if (M <= 128) return narrow ? 6 : 12;
  if (M <= 512) return small ? 8 : 11;
  return M >= 768 && K % 32 == 0 ? kTiledW16 : 10;
}

}  // namespace gemm_w8_rdna2
}  // namespace vllm

// Requirements: a [M, K] with K-contiguous, 16-byte aligned rows and
// K % 16 == 0; w [K / 4, N, 4] (kmajor_w8), its dword rows 16-byte aligned
// (stride(0) % 16 == 0, so N % 4 == 0); scale [N] fp32 contiguous; optional
// contiguous bias [N] of the output dtype. a fp16 or bf16 (output: a's dtype)
// with w int8 or float8_e4m3fn and optionally fp8 block_scale
// [ceil(K / block_k), ceil(N / block_n)] fp16 with contiguous rows
// multiplying the weights of each block (block_k a power of two >= 8); or a
// int8 with int8 w, fp32 scale_a of 1 or M elements and an fp16 or bf16
// out_dtype (W8A8). cfg < 0 picks the default config for M. Returns [M, N].
torch::Tensor gemm_w8_rdna2(const at::Tensor& a, const at::Tensor& w,
                            const std::optional<at::Tensor>& scale,
                            const std::optional<at::Tensor>& block_scale,
                            int64_t block_n, int64_t block_k,
                            const std::optional<at::Tensor>& bias, int64_t cfg,
                            const std::optional<at::Tensor>& scale_a,
                            std::optional<at::ScalarType> out_dtype) {
  using namespace vllm::gemm_w8_rdna2;
  const bool w8a8 = a.dtype() == torch::kInt8;
  const bool w16 = w.dtype() == torch::kFloat16;
  TORCH_CHECK(
      w8a8 || a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16,
      "gemm_w8_rdna2 needs fp16, bf16 or int8 activations");
  TORCH_CHECK(w16 || w.dtype() == torch::kInt8 ||
                  w.dtype() == at::ScalarType::Float8_e4m3fn,
              "gemm_w8_rdna2 needs int8, float8_e4m3fn or fp16 weights");
  const int kpd = w16 ? 2 : 4;
  TORCH_CHECK(a.dim() == 2 && w.dim() == 3 && w.size(2) == kpd &&
                  w.size(0) * kpd == a.size(1),
              "gemm_w8_rdna2 needs a [M, K] and w [K / 4, N, 4] (8-bit) or "
              "[K / 2, N, 2] (fp16)");
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = w.size(1);
  TORCH_CHECK(K % 16 == 0 && a.stride(1) == 1 &&
                  a.stride(0) * a.element_size() % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "gemm_w8_rdna2 needs K % 16 == 0 and 16-byte aligned, "
              "K-contiguous rows of a");
  TORCH_CHECK(N % 4 == 0 && w.stride(2) == 1 && w.stride(1) == kpd &&
                  w.stride(0) * w.element_size() % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "gemm_w8_rdna2 needs N % 4 == 0 and 16-byte aligned dword rows "
              "of w");
  if (w16) {
    TORCH_CHECK(a.dtype() == torch::kFloat16 && !scale.has_value() &&
                    !block_scale.has_value(),
                "gemm_w8_rdna2 with fp16 weights needs fp16 activations and "
                "no scales");
  } else {
    TORCH_CHECK(scale.has_value() && scale->dtype() == torch::kFloat32 &&
                    scale->numel() == N && scale->is_contiguous() &&
                    reinterpret_cast<uintptr_t>(scale->data_ptr()) % 16 == 0,
                "gemm_w8_rdna2 needs a 16-byte aligned, contiguous fp32 scale "
                "per output channel");
  }
  at::ScalarType c_dtype = a.scalar_type();
  if (w8a8) {
    TORCH_CHECK(w.dtype() == torch::kInt8 && !block_scale.has_value(),
                "gemm_w8_rdna2 with int8 activations needs int8 weights "
                "without block scales");
    TORCH_CHECK(scale_a.has_value() && scale_a->dtype() == torch::kFloat32 &&
                    scale_a->is_contiguous() &&
                    (scale_a->numel() == 1 || scale_a->numel() == M),
                "gemm_w8_rdna2 with int8 activations needs a contiguous fp32 "
                "scale_a of 1 or M elements");
    TORCH_CHECK(out_dtype.has_value() && (*out_dtype == torch::kFloat16 ||
                                          *out_dtype == torch::kBFloat16),
                "gemm_w8_rdna2 with int8 activations needs an fp16 or bf16 "
                "out_dtype");
    c_dtype = *out_dtype;
  }
  int bk_shift = 0;
  if (block_scale.has_value()) {
    TORCH_CHECK(w.dtype() != torch::kInt8 && block_n >= 1 && block_k >= 8 &&
                    (block_k & (block_k - 1)) == 0,
                "gemm_w8_rdna2 block scales need fp8 weights, block_n >= 1 and "
                "block_k a power of two >= 8");
    TORCH_CHECK(block_scale->dtype() == torch::kFloat16 &&
                    block_scale->dim() == 2 &&
                    block_scale->size(0) == (K + block_k - 1) / block_k &&
                    block_scale->size(1) == (N + block_n - 1) / block_n &&
                    block_scale->stride(1) == 1,
                "gemm_w8_rdna2 needs fp16 block scales [ceil(K / block_k), "
                "ceil(N / block_n)] with contiguous rows");
    bk_shift = __builtin_ctzll(block_k);
  }
  if (bias.has_value()) {
    TORCH_CHECK(
        bias->dtype() == c_dtype && bias->numel() == N && bias->is_contiguous(),
        "gemm_w8_rdna2 bias must be a contiguous [N] of the output "
        "dtype");
  }
  TORCH_CHECK(cfg < (w8a8  ? kNumConfigsI8
                     : w16 ? kNumConfigsW16 + 1
                           : kNumConfigs),
              "gemm_w8_rdna2: cfg out of range");
  TORCH_CHECK(!(w16 && cfg == kTiledW16) || K % 32 == 0,
              "gemm_w8_rdna2: the tiled fp16 config needs K % 32 == 0");

  auto c = torch::empty({M, N}, a.options().dtype(c_dtype));
  if (M == 0 || N == 0) return c;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Args p{a,       w, scale,        block_scale, bias,
               scale_a, c, (int)block_n, bk_shift};

  if (w8a8) {
    const int id = cfg < 0 ? default_config_i8(M, N, K) : (int)cfg;
#define VLLM_W8A8_RDNA2_CASE(ID, T, NW, CT, KB)                        \
  case ID:                                                             \
    static_assert(kConfigsI8[ID].t == T && kConfigsI8[ID].nw == NW &&  \
                  kConfigsI8[ID].ct == CT && kConfigsI8[ID].kb == KB); \
    launch_config_i8<T, NW, CT, KB>(p, stream);                        \
    break;
    switch (id) {
      VLLM_W8A8_RDNA2_CASE(0, 1, 8, 2, 64)
      VLLM_W8A8_RDNA2_CASE(1, 2, 8, 2, 64)
      VLLM_W8A8_RDNA2_CASE(2, 4, 4, 1, 64)
      VLLM_W8A8_RDNA2_CASE(3, 8, 4, 1, 32)
      VLLM_W8A8_RDNA2_CASE(4, 16, 2, 1, 64)
      VLLM_W8A8_RDNA2_CASE(5, 16, 4, 1, 32)
      VLLM_W8A8_RDNA2_CASE(6, 24, 4, 1, 32)
      VLLM_W8A8_RDNA2_CASE(7, 32, 4, 1, 32)
      VLLM_W8A8_RDNA2_CASE(8, 16, 4, 1, 64)
      VLLM_W8A8_RDNA2_CASE(9, 32, 2, 1, 32)
      VLLM_W8A8_RDNA2_CASE(10, 32, 1, 1, 32)
      VLLM_W8A8_RDNA2_CASE(11, 16, 1, 2, 32)
      VLLM_W8A8_RDNA2_CASE(12, 16, 2, 2, 32)
    }
#undef VLLM_W8A8_RDNA2_CASE
    return c;
  }

  if (w16) {
    const int id = cfg < 0 ? default_config_w16(M, N, K) : (int)cfg;
    if (id == kTiledW16) {
      launch_tiled_w16(p, stream);
      return c;
    }
#define VLLM_W16_RDNA2_CASE(ID, T, NW, CT, KB)                           \
  case ID:                                                               \
    static_assert(kConfigsW16[ID].t == T && kConfigsW16[ID].nw == NW &&  \
                  kConfigsW16[ID].ct == CT && kConfigsW16[ID].kb == KB); \
    launch_config_w16<T, NW, CT, KB>(p, stream);                         \
    break;
    switch (id) {
      VLLM_W16_RDNA2_CASE(0, 1, 8, 2, 16)
      VLLM_W16_RDNA2_CASE(1, 1, 16, 1, 16)
      VLLM_W16_RDNA2_CASE(2, 2, 8, 2, 16)
      VLLM_W16_RDNA2_CASE(3, 4, 8, 1, 16)
      VLLM_W16_RDNA2_CASE(4, 8, 4, 1, 32)
      VLLM_W16_RDNA2_CASE(5, 8, 8, 1, 16)
      VLLM_W16_RDNA2_CASE(6, 16, 4, 1, 16)
      VLLM_W16_RDNA2_CASE(7, 16, 8, 1, 16)
      VLLM_W16_RDNA2_CASE(8, 16, 2, 1, 16)
      VLLM_W16_RDNA2_CASE(9, 16, 1, 1, 16)
      VLLM_W16_RDNA2_CASE(10, 16, 1, 2, 8)
      VLLM_W16_RDNA2_CASE(11, 24, 4, 1, 16)
      VLLM_W16_RDNA2_CASE(12, 32, 2, 1, 8)
    }
#undef VLLM_W16_RDNA2_CASE
    return c;
  }

  const int id = cfg < 0 ? default_config(M, N, K) : (int)cfg;
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
    VLLM_W8_RDNA2_CASE(12, 1, 4, 1, 32)
    VLLM_W8_RDNA2_CASE(13, 1, 16, 1, 16)
    VLLM_W8_RDNA2_CASE(14, 4, 8, 1, 32)
  }
#undef VLLM_W8_RDNA2_CASE
  return c;
}

// out [M, HS] (xn's dtype): the hyper-connection gate of hc_up_mix_kernel.
// lora [M, R] fp16 or bf16 with K-contiguous rows, w [R / 4, 4 * HS, 4] int8
// (kmajor_w8), scale [4 * HS] fp32, xn [M, 4 * HS] of lora's dtype with
// contiguous rows; hc_count 4, HS % 32 == 0, R % 8 == 0 and R <= 512.
torch::Tensor hc_up_mix_rdna2(const at::Tensor& lora, const at::Tensor& w,
                              const at::Tensor& scale, const at::Tensor& xn,
                              int64_t hc_count) {
  using namespace vllm::gemm_w8_rdna2;
  TORCH_CHECK(hc_count == kHcStreams, "hc_up_mix_rdna2 needs 4 streams");
  TORCH_CHECK(
      (lora.dtype() == torch::kFloat16 || lora.dtype() == torch::kBFloat16) &&
          xn.dtype() == lora.dtype(),
      "hc_up_mix_rdna2 needs fp16 or bf16 lora and xn of one dtype");
  TORCH_CHECK(lora.dim() == 2 && lora.stride(1) == 1 && xn.dim() == 2 &&
                  xn.stride(1) == 1 && xn.size(0) == lora.size(0) &&
                  xn.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(xn.data_ptr()) % 8 == 0,
              "hc_up_mix_rdna2 needs lora [M, R] and xn [M, 4 * HS] with "
              "contiguous, 8-byte aligned rows of xn");
  const int M = lora.size(0), R = lora.size(1), N = xn.size(1);
  TORCH_CHECK(R % 8 == 0 && R <= kHcMaxRank && N % (32 * kHcStreams) == 0,
              "hc_up_mix_rdna2 needs R % 8 == 0, R <= 512 and HS % 32 == 0");
  TORCH_CHECK(w.dtype() == torch::kInt8 && w.dim() == 3 && w.size(0) * 4 == R &&
                  w.size(1) == N && w.size(2) == 4 && w.is_contiguous(),
              "hc_up_mix_rdna2 needs contiguous int8 w [R / 4, 4 * HS, 4]");
  TORCH_CHECK(scale.dtype() == torch::kFloat32 && scale.numel() == N &&
                  scale.is_contiguous(),
              "hc_up_mix_rdna2 needs a contiguous fp32 scale [4 * HS]");
  const int HS = N / kHcStreams;
  auto out = torch::empty({M, HS}, xn.options());
  if (M == 0) return out;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xn));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int NW = 8;
  auto run = [&](auto t, auto tt) {
    using T = decltype(t);
    constexpr int TT = decltype(tt)::value;
    hc_up_mix_kernel<T, TT, NW>
        <<<dim3(HS / 32, (M + TT - 1) / TT), NW * WARP32, 0, stream>>>(
            (const uint4*)w.data_ptr(), scale.data_ptr<float>(),
            (const T*)lora.data_ptr(), lora.stride(0), (const T*)xn.data_ptr(),
            xn.stride(0), (T*)out.data_ptr(), out.stride(0), M, HS, R, N / 4,
            1.f / kHcStreams);
  };
  auto by_rows = [&](auto t) {
    if (M == 1)
      run(t, std::integral_constant<int, 1>{});
    else if (M == 2)
      run(t, std::integral_constant<int, 2>{});
    else
      run(t, std::integral_constant<int, 4>{});
  };
  if (lora.dtype() == torch::kFloat16)
    by_rows(__half{});
  else
    by_rows(__hip_bfloat16{});
  return out;
}
