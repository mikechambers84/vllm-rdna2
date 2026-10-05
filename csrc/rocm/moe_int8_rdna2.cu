// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) int8-weight fused-MoE decode for a few tokens, on the
// layout vLLM's Triton int8 MoE path keeps: w13 [E, 2I, H] and w2 [E, H, I]
// int8 with per-channel fp32 scales [E, N, 1].
//
// Decode is bandwidth-bound, so the int8 weights are converted exactly to
// fp16 and dotted with the fp16 activations (v_dot2_f32_f16), with the
// channel scale applied once per row: no per-token activation quantization,
// so it is more accurate than the W8A8 path at the same speed. Same two
// kernels as moe_wna16_rdna2.cu: gate/up with SiLU-and-mul fused, then down
// with the top-k weighted sum in registers (deterministic, no atomics).
// moe_fp8_decode_rdna2 runs the same kernels on fp8 e4m3fn weights (widened
// exactly to fp16 in registers) with per-tensor, per-channel or 2D block
// scales.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rdna2_fp8.cuh"

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_int8_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int WAVES = 8;
static constexpr int CHUNK = 16;  // int8 weights per 16-byte load

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// 4 signed bytes of q -> 2 half2 holding the exact values: flip the sign bit
// (b + 128 in [0, 255]), OR into the mantissa of 1024.0 and subtract 1152.
__device__ __forceinline__ void unpack4(uint32_t q, half2 (&h)[2]) {
  q ^= 0x80808080u;
  const half2 bias =
      __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
  uint32_t lo = (q & 0xFFu) | ((q & 0xFF00u) << 8) | 0x64006400u;
  uint32_t hi = ((q >> 16) & 0xFFu) | ((q >> 8) & 0xFF0000u) | 0x64006400u;
  h[0] = __hsub2(*reinterpret_cast<half2*>(&lo), bias);
  h[1] = __hsub2(*reinterpret_cast<half2*>(&hi), bias);
}

// acc + 16 int8 weights (one 16-byte load) . 16 fp16 activations at a.
__device__ __forceinline__ float dot16(const int4 wv, const __half* a,
                                       float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int w = 0; w < 2; w++) {
      half2 h[2];
      unpack4(q[c * 2 + w], h);
      acc = __builtin_amdgcn_fdot2(ah[2 * w], h[0], acc, false);
      acc = __builtin_amdgcn_fdot2(ah[2 * w + 1], h[1], acc, false);
    }
  }
  return acc;
}

// acc + 2^-8 * (16 fp8 e4m3fn weights . 16 fp16 activations at a).
__device__ __forceinline__ float dot16_fp8(const int4 wv, const __half* a,
                                           float acc) {
  const uint32_t* q = reinterpret_cast<const uint32_t*>(&wv);
  #pragma unroll
  for (int c = 0; c < 2; c++) {
    const int4 av = *reinterpret_cast<const int4*>(a + c * 8);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int w = 0; w < 2; w++) {
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w], rdna2::fp8x2_to_half2(q[c * 2 + w], rdna2::FP8_LO), acc,
          false);
      acc = __builtin_amdgcn_fdot2(
          ah[2 * w + 1], rdna2::fp8x2_to_half2(q[c * 2 + w], rdna2::FP8_HI),
          acc, false);
    }
  }
  return acc;
}

// 8-bit weight formats: int8 or fp8 with one scale per block of rows, or fp8
// with 2D [block_n, block_k] block scales.
enum class W8 { kInt8, kFp8, kFp8Block };

template <W8 F>
__device__ __forceinline__ float dot16_w8(const int4 wv, const __half* a,
                                          float acc) {
  if constexpr (F == W8::kInt8)
    return dot16(wv, a, acc);
  else
    return dot16_fp8(wv, a, acc);
}

// Scales of an [E, N, K] weight: fp32 [E, ceil(N / bn), cols], indexed
// [e][n / bn][k >> bk_shift] (cols == 1 unless block-scaled).
struct Scales {
  const float* s;
  int bn, bk_shift, row_blocks, cols;
  __device__ __forceinline__ const float* row(long e, int n) const {
    return s + (e * row_blocks + n / bn) * cols;
  }
};

// fp8 partials carry the 2^-8 of the in-register conversion.
template <W8 F>
static constexpr float W8_UNIT = F == W8::kInt8 ? 1.f : 256.f;

__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int mask = WARP32 / 2; mask >= 1; mask >>= 1) v += __shfl_xor(v, mask);
  return v;
}

__device__ __forceinline__ void stage_row(__half* dst, const __half* src,
                                          int K) {
  for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
    *reinterpret_cast<int4*>(dst + k) = *reinterpret_cast<const int4*>(src + k);
}

// One workgroup per (token, expert) pair and WAVES output columns; each wave
// computes gate row j and up row I + j and writes silu(gate) * up.
template <W8 F>
__global__ void __launch_bounds__(WAVES* WARP32)
    gate_up_silu_kernel(const __half* __restrict__ x,
                        const int32_t* __restrict__ topk_ids,
                        const int8_t* __restrict__ w13, const Scales s13,
                        __half* __restrict__ act, const int E, const int H,
                        const int I, const int topk, const int ldx) {
  extern __shared__ __align__(16) __half sx[];
  const int pair = blockIdx.y;
  const long e = topk_ids[pair];
  // Ids outside [0, E) (routing garbage from dummy warmup inputs, which the
  // Triton path's alignment also drops) produce zero activations.
  const bool valid = e >= 0 && e < E;
  stage_row(sx, x + (long)(pair / topk) * ldx, H);
  __syncthreads();
  const int lane = threadIdx.x % WARP32;
  const int j = blockIdx.x * WAVES + threadIdx.x / WARP32;
  if (j >= I) return;
  float gate = 0.f, up = 0.f;
  const float* gate_s = valid ? s13.row(e, j) : nullptr;
  const float* up_s = valid ? s13.row(e, I + j) : nullptr;
  if (valid) {
    const int8_t* gate_row = w13 + (e * 2 * I + j) * H;
    const int8_t* up_row = gate_row + (long)I * H;
    for (int k = lane * CHUNK; k < H; k += WARP32 * CHUNK) {
      const int4 gw = *reinterpret_cast<const int4*>(gate_row + k);
      const int4 uw = *reinterpret_cast<const int4*>(up_row + k);
      if constexpr (F == W8::kFp8Block) {
        const int kb = k >> s13.bk_shift;
        gate = fmaf(dot16_w8<F>(gw, sx + k, 0.f), gate_s[kb], gate);
        up = fmaf(dot16_w8<F>(uw, sx + k, 0.f), up_s[kb], up);
      } else {
        gate = dot16_w8<F>(gw, sx + k, gate);
        up = dot16_w8<F>(uw, sx + k, up);
      }
    }
  }
  gate = wave_sum(gate);
  up = wave_sum(up);
  if (lane == 0) {
    if (valid) {
      gate *= W8_UNIT<F>;
      up *= W8_UNIT<F>;
      if constexpr (F != W8::kFp8Block) {
        gate *= gate_s[0];
        up *= up_s[0];
      }
    }
    act[(long)pair * I + j] = __float2half(gate / (1.f + __expf(-gate)) * up);
  }
}

// One workgroup per token and WAVES output rows; each wave sums
// topk_weight * scale * (W2[e] row . act) over the token's experts.
template <W8 F>
__global__ void __launch_bounds__(WAVES* WARP32)
    down_sum_kernel(const __half* __restrict__ act,
                    const int32_t* __restrict__ topk_ids,
                    const float* __restrict__ topk_weights,
                    const int8_t* __restrict__ w2, const Scales s2,
                    __half* __restrict__ out, const int E, const int H,
                    const int I, const int topk, const int ldo) {
  extern __shared__ __align__(16) __half sa[];  // [topk][I]
  const int t = blockIdx.y;
  for (int k = 0; k < topk; k++)
    stage_row(sa + k * I, act + ((long)t * topk + k) * I, I);
  __syncthreads();
  const int lane = threadIdx.x % WARP32;
  const int n = blockIdx.x * WAVES + threadIdx.x / WARP32;
  if (n >= H) return;
  const int chunks = I / CHUNK;
  float acc = 0.f;
  for (int idx = lane; idx < topk * chunks; idx += WARP32) {
    const int k = idx / chunks, c = idx % chunks;
    const long e = topk_ids[t * topk + k];
    if (e < 0 || e >= E) continue;
    const float* srow = s2.row(e, n);
    const float s =
        F == W8::kFp8Block ? srow[(c * CHUNK) >> s2.bk_shift] : srow[0];
    acc += dot16_w8<F>(
               *reinterpret_cast<const int4*>(w2 + (e * H + n) * I + c * CHUNK),
               sa + k * I + c * CHUNK, 0.f) *
           (s * topk_weights[t * topk + k]);
  }
  acc = wave_sum(acc);
  if (lane == 0) out[(long)t * ldo + n] = __float2half(acc * W8_UNIT<F>);
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

enum class W8 { kInt8, kFp8, kFp8Block };
struct Scales {
  const float* s;
  int bn, bk_shift, row_blocks, cols;
};
template <W8 F>
__global__ void gate_up_silu_kernel(const __half*, const int32_t*,
                                    const int8_t*, const Scales, __half*,
                                    const int, const int, const int, const int,
                                    const int) {}
template <W8 F>
__global__ void down_sum_kernel(const __half*, const int32_t*, const float*,
                                const int8_t*, const Scales, __half*, const int,
                                const int, const int, const int, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_int8_rdna2
}  // namespace vllm

namespace {

using vllm::moe_int8_rdna2::Scales;
using vllm::moe_int8_rdna2::W8;

// Validated weights [E, N, K] (bytes) and their scales; launches both kernels.
template <W8 F>
void moe_w8_decode(const char* name, torch::Tensor& output,
                   const torch::Tensor& x, const torch::Tensor& topk_ids,
                   const torch::Tensor& topk_weights, const torch::Tensor& w13,
                   const Scales& s13, const torch::Tensor& w2, const Scales& s2,
                   torch::Tensor& act) {
  using namespace vllm::moe_int8_rdna2;
  TORCH_CHECK(x.dtype() == torch::kFloat16 &&
                  output.dtype() == torch::kFloat16 &&
                  act.dtype() == torch::kFloat16,
              name, " needs fp16 activations");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32 &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              name, " needs contiguous int32 topk_ids and fp32 topk_weights");
  const int M = x.size(0), H = x.size(1), topk = topk_ids.size(1);
  const int E = w13.size(0), I = w2.size(2);
  TORCH_CHECK(H % 256 == 0 && I % 256 == 0, name,
              " needs H and I to be multiples of 256");
  TORCH_CHECK(w13.is_contiguous() && w2.is_contiguous() &&
                  w13.size(1) == 2 * I && w13.size(2) == H && w2.size(0) == E &&
                  w2.size(1) == H,
              name, ": unexpected weight shape");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  output.stride(1) == 1 && output.size(0) == M &&
                  output.size(1) == H && topk_ids.size(0) == M &&
                  act.is_contiguous() && act.numel() >= (long)M * topk * I,
              name, ": bad x, output, topk_ids or act");
  if (M == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 block(WAVES * WARP32);
  gate_up_silu_kernel<F>
      <<<dim3((I + WAVES - 1) / WAVES, M * topk), block, H * sizeof(__half),
         stream>>>((const __half*)x.data_ptr(), topk_ids.data_ptr<int32_t>(),
                   (const int8_t*)w13.data_ptr(), s13, (__half*)act.data_ptr(),
                   E, H, I, topk, x.stride(0));
  down_sum_kernel<F><<<dim3((H + WAVES - 1) / WAVES, M), block,
                       topk * I * sizeof(__half), stream>>>(
      (const __half*)act.data_ptr(), topk_ids.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(), (const int8_t*)w2.data_ptr(), s2,
      (__half*)output.data_ptr(), E, H, I, topk, output.stride(0));
}

// Scales of fp8 w [E, N, K]: fp32 [E, ceil(N / block_n), S] (or [E] for one
// scale per expert) with S == 1 (one scale per block_n rows) or
// S == ceil(K / block_k) (2D blocks, block_k a power of two >= 16).
Scales fp8_scales(const torch::Tensor& w, const torch::Tensor& s,
                  int64_t block_n, int64_t block_k) {
  TORCH_CHECK(w.dtype() == at::ScalarType::Float8_e4m3fn &&
                  s.dtype() == torch::kFloat32 && s.is_contiguous(),
              "moe_fp8_decode_rdna2 needs float8_e4m3fn weights and "
              "contiguous fp32 scales");
  const int64_t E = w.size(0), N = w.size(1), K = w.size(2);
  if (s.dim() == 1) {
    TORCH_CHECK(s.numel() == E, "moe_fp8_decode_rdna2: bad per-expert scales");
    return Scales{s.data_ptr<float>(), (int)N, 31, 1, 1};
  }
  TORCH_CHECK(s.dim() == 3 && s.size(0) == E && block_n >= 1 &&
                  s.size(1) == (N + block_n - 1) / block_n,
              "moe_fp8_decode_rdna2 needs [E, ceil(N / block_n), S] scales");
  if (s.size(2) == 1)
    return Scales{s.data_ptr<float>(), (int)block_n, 31, (int)s.size(1), 1};
  TORCH_CHECK(block_k >= 16 && (block_k & (block_k - 1)) == 0 &&
                  s.size(2) == (K + block_k - 1) / block_k,
              "moe_fp8_decode_rdna2 needs block_k a power of two >= 16 and "
              "ceil(K / block_k) scale columns");
  return Scales{s.data_ptr<float>(), (int)block_n, __builtin_ctzll(block_k),
                (int)s.size(1), (int)s.size(2)};
}

}  // namespace

// output [M, H] = sum_k topk_weights[:, k] * W2[e_k] silu_mul(W13[e_k] x),
// W = int8 weight * per-channel scale. x [M, H] fp16 (rows contiguous),
// topk_ids [M, topk] int32, topk_weights [M, topk] fp32, w13 [E, 2I, H] and
// w2 [E, H, I] int8, s13 [E, 2I, 1] and s2 [E, H, 1] fp32, act a
// [M * topk, I] fp16 workspace. H and I must be multiples of 256.
void moe_int8_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                           const torch::Tensor& topk_ids,
                           const torch::Tensor& topk_weights,
                           const torch::Tensor& w13, const torch::Tensor& s13,
                           const torch::Tensor& w2, const torch::Tensor& s2,
                           torch::Tensor& act) {
  TORCH_CHECK(w13.dtype() == torch::kInt8 && w2.dtype() == torch::kInt8 &&
                  s13.dtype() == torch::kFloat32 &&
                  s2.dtype() == torch::kFloat32 && s13.is_contiguous() &&
                  s2.is_contiguous() &&
                  s13.numel() == w13.size(0) * w13.size(1) &&
                  s2.numel() == w2.size(0) * w2.size(1),
              "moe_int8_decode_rdna2 needs int8 weights and per-channel fp32 "
              "scales");
  const int N13 = w13.size(1), N2 = w2.size(1);
  moe_w8_decode<W8::kInt8>("moe_int8_decode_rdna2", output, x, topk_ids,
                           topk_weights, w13,
                           Scales{s13.data_ptr<float>(), 1, 31, N13, 1}, w2,
                           Scales{s2.data_ptr<float>(), 1, 31, N2, 1}, act);
}

// Same as moe_int8_decode_rdna2 for float8_e4m3fn w13 / w2 with scales as in
// fp8_scales (one block_n / block_k pair for both).
void moe_fp8_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                          const torch::Tensor& topk_ids,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& w13, const torch::Tensor& s13,
                          const torch::Tensor& w2, const torch::Tensor& s2,
                          int64_t block_n, int64_t block_k,
                          torch::Tensor& act) {
  const Scales a = fp8_scales(w13, s13, block_n, block_k);
  const Scales b = fp8_scales(w2, s2, block_n, block_k);
  TORCH_CHECK((a.cols > 1) == (b.cols > 1),
              "moe_fp8_decode_rdna2 needs w13 and w2 scales of one kind");
  if (a.cols > 1)
    moe_w8_decode<W8::kFp8Block>("moe_fp8_decode_rdna2", output, x, topk_ids,
                                 topk_weights, w13, a, w2, b, act);
  else
    moe_w8_decode<W8::kFp8>("moe_fp8_decode_rdna2", output, x, topk_ids,
                            topk_weights, w13, a, w2, b, act);
}
