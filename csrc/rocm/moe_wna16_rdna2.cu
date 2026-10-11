// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) W4A16 fused-MoE decode for a few tokens: two kernels on
// the layout vLLM's Triton WNA16 MoE backend keeps, so no second weight copy:
// w13 [E, 2I, H/8] and w2 [E, H, I/8] int32 (8 nibbles of q + 8 along K, low
// nibble first, viewed as uint8), scales [E, N, K/G] fp16 (G a multiple of
// 32), optional zero points [E, N/2, K/G] uint8 (two columns per byte, low
// nibble = even column; symmetric = 8).
//
//   gate_up_silu: one workgroup per (token, expert) pair and 8*R output
//     columns; each wave computes gate row j and up row I + j and writes
//     silu(gate) * up, fusing away the [M * topk, 2I] intermediate.
//   down_sum: one workgroup per token and 8*R output rows; each wave sums
//     topk_weight * (W2[e] row . act) over the token's experts in registers
//     (deterministic; gfx1030 has no global float atomics).
//   v2 (no zero points, H and I multiples of 1024): gate_up_v2 as above with
//     software-pipelined loads, and down_v2 per (token, expert) pair into fp32
//     partials that sum_pairs adds up per token in a fixed order. Steady
//     clocks, GLM-5.3-Flash shapes, weights streamed from DRAM: 1 token 345 ->
//     303 us, 2-8 tokens -8 to -9% (down -22 to -25%).
//
// Weights are dequantized to exact integers (OR into the 1024.0 mantissa)
// and the group scale is applied to the fp32 partial dot, as Exllama does.
// A zero point z adds (8 - z) * sum(activations) per 32-weight chunk.
// swiglu_limit L > 0 clamps the activation: silu(min(gate, L)) *
// clamp(up, -L, L).

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_wna16_rdna2 {

static constexpr int WARP32 = 32;
static constexpr int WAVES = 8;
static constexpr int GROUP = 32;
static constexpr int R1 = 2;  // output columns per wave, gate_up_silu (v2)
static constexpr int R2 = 4;  // output rows per wave, down_sum / down_v2

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// 8 nibbles (q + 8) of one word -> 4 half2 holding exact q.
__device__ __forceinline__ void unpack8(uint32_t q, half2 (&h)[4]) {
  const uint32_t magic = 0x64006400u;
  const half2 bias =
      __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    uint32_t w =
        (((q >> (8 * j)) & 0x0F) | (((q >> (8 * j + 4)) & 0x0F) << 16)) | magic;
    h[j] = __hsub2(*reinterpret_cast<half2*>(&w), bias);
  }
}

// One 32-weight group (16 bytes) of a row times 32 activations staged in LDS
// with chunk c (8 halves) at a + c * cstride; the unscaled fp32 dot.
__device__ __forceinline__ float group_dot(const int4 wv, const __half* a,
                                           int cstride) {
  float part = 0.f;
  #pragma unroll
  for (int w = 0; w < 4; w++) {
    half2 wh[4];
    unpack8(reinterpret_cast<const uint32_t*>(&wv)[w], wh);
    const int4 av = *reinterpret_cast<const int4*>(a + w * cstride);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int j = 0; j < 4; j++)
      part = __builtin_amdgcn_fdot2(ah[j], wh[j], part, false);
  }
  return part;
}

// Sum of the 32 activations of a chunk staged as in group_dot.
__device__ __forceinline__ float group_asum(const __half* a, int cstride) {
  const half2 one = __halves2half2(__float2half(1.f), __float2half(1.f));
  float s = 0.f;
  #pragma unroll
  for (int w = 0; w < 4; w++) {
    const int4 av = *reinterpret_cast<const int4*>(a + w * cstride);
    const half2* ah = reinterpret_cast<const half2*>(&av);
  #pragma unroll
    for (int j = 0; j < 4; j++)
      s = __builtin_amdgcn_fdot2(ah[j], one, s, false);
  }
  return s;
}

// 8 - zero point of column n (within its expert) in scale group sg.
__device__ __forceinline__ float zp_shift(const uint8_t* z, long ze, long n,
                                          int sg, int sgroups) {
  const uint8_t b = z[ze + (n >> 1) * sgroups + sg];
  return 8.f - (float)((b >> ((n & 1) * 4)) & 0xF);
}

__device__ __forceinline__ float wave_sum(float v) {
  #pragma unroll
  for (int mask = WARP32 / 2; mask >= 1; mask >>= 1) v += __shfl_xor(v, mask);
  return v;
}

// Stage a [K] fp16 row in LDS with chunk c of group g at slot c * K/32 + g,
// so lanes that own consecutive groups read consecutive 16-byte words.
__device__ __forceinline__ void stage_row(__half* dst, const __half* src,
                                          int K) {
  const int groups = K / GROUP;
  for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8) {
    const int g = k / GROUP, c = k % GROUP / 8;
    *reinterpret_cast<int4*>(dst + (c * groups + g) * 8) =
        *reinterpret_cast<const int4*>(src + k);
  }
}

template <bool ZP>
__global__ void __launch_bounds__(WAVES* WARP32)
    gate_up_silu_kernel(const __half* __restrict__ x,
                        const int32_t* __restrict__ topk_ids,
                        const uint32_t* __restrict__ w13,
                        const __half* __restrict__ s13,
                        const uint8_t* __restrict__ z13,
                        __half* __restrict__ act, const int E, const int H,
                        const int I, const int topk, const int ldx,
                        const int gsub, const float lim) {
  extern __shared__ __align__(16) __half sx[];
  const int pair = blockIdx.y;
  const long e = topk_ids[pair];
  // Ids outside [0, E) (e.g. routing garbage from dummy warmup inputs, which
  // the Triton path's alignment also drops) produce zero activations.
  const int groups_used = e >= 0 && e < E ? H / GROUP : 0;
  stage_row(sx, x + (long)(pair / topk) * ldx, H);
  __syncthreads();
  const int lane = threadIdx.x % WARP32, wave = threadIdx.x / WARP32;
  const int j0 = (blockIdx.x * WAVES + wave) * R1;
  const int words = H / 8, groups = H / GROUP, sgroups = groups / gsub;
  float acc[2][R1] = {};
  for (int g = lane; g < groups_used; g += WARP32) {
    const int sg = g / gsub;
    const float asum = ZP ? group_asum(sx + g * 8, groups * 8) : 0.f;
  #pragma unroll
    for (int h = 0; h < 2; h++)
  #pragma unroll
      for (int r = 0; r < R1; r++) {
        if (j0 + r >= I) continue;
        const long n = h * I + j0 + r, row = e * 2 * I + n;
        const int4 wv =
            *reinterpret_cast<const int4*>(w13 + row * words + g * 4);
        float d = group_dot(wv, sx + g * 8, groups * 8);
        if constexpr (ZP)
          d += zp_shift(z13, e * I * sgroups, n, sg, sgroups) * asum;
        acc[h][r] += d * __half2float(s13[row * sgroups + sg]);
      }
  }
  #pragma unroll
  for (int r = 0; r < R1; r++) {
    float gate = wave_sum(acc[0][r]), up = wave_sum(acc[1][r]);
    if (lim > 0.f) {
      gate = fminf(gate, lim);
      up = fminf(fmaxf(up, -lim), lim);
    }
    if (lane == 0 && j0 + r < I)
      act[(long)pair * I + j0 + r] =
          __float2half(gate / (1.f + __expf(-gate)) * up);
  }
}

template <bool ZP>
__global__ void __launch_bounds__(WAVES* WARP32)
    down_sum_kernel(const __half* __restrict__ act,
                    const int32_t* __restrict__ topk_ids,
                    const float* __restrict__ topk_weights,
                    const uint32_t* __restrict__ w2,
                    const __half* __restrict__ s2,
                    const uint8_t* __restrict__ z2, __half* __restrict__ out,
                    const int E, const int H, const int I, const int topk,
                    const int ldo, const int gsub) {
  extern __shared__ __align__(16) __half sa[];  // [topk][I]
  const int t = blockIdx.y;
  for (int k = 0; k < topk; k++)
    stage_row(sa + k * I, act + ((long)t * topk + k) * I, I);
  __syncthreads();
  const int lane = threadIdx.x % WARP32, wave = threadIdx.x / WARP32;
  const int n0 = (blockIdx.x * WAVES + wave) * R2;
  const int words = I / 8, groups = I / GROUP, sgroups = groups / gsub;
  float acc[R2] = {};
  for (int idx = lane; idx < topk * groups; idx += WARP32) {
    const int k = idx / groups, g = idx % groups, sg = g / gsub;
    const long e = topk_ids[t * topk + k];
    if (e < 0 || e >= E) continue;
    const float wk = topk_weights[t * topk + k];
    const float asum = ZP ? group_asum(sa + k * I + g * 8, groups * 8) : 0.f;
  #pragma unroll
    for (int r = 0; r < R2; r++) {
      if (n0 + r >= H) continue;
      const long row = e * H + n0 + r;
      const int4 wv = *reinterpret_cast<const int4*>(w2 + row * words + g * 4);
      float d = group_dot(wv, sa + k * I + g * 8, groups * 8);
      if constexpr (ZP)
        d += zp_shift(z2, e * (H / 2) * sgroups, n0 + r, sg, sgroups) * asum;
      acc[r] += d * (__half2float(s2[row * sgroups + sg]) * wk);
    }
  }
  #pragma unroll
  for (int r = 0; r < R2; r++) {
    const float v = wave_sum(acc[r]);
    if (lane == 0 && n0 + r < H) out[(long)t * ldo + n0 + r] = __float2half(v);
  }
}

// v2 decode kernels: no per-row bounds branches (the launcher requires the
// rows to tile the workgroups), scales kept as raw 16-bit values in 32-bit
// registers (packing two halves forced a wait on every scale load), and the
// next 32-group step's loads issued before the current step's math, so a wave
// keeps 2 * rows weight words in flight instead of one. No zero points.
template <int ROWS>
struct RowLoads {
  int4 w[ROWS];
  uint32_t s[ROWS];
};

template <int ROWS>
__device__ __forceinline__ void load_rows(RowLoads<ROWS>& v,
                                          const uint32_t* const (&wrow)[ROWS],
                                          const __half* const (&srow)[ROWS],
                                          int g, int gsub) {
  #pragma unroll
  for (int r = 0; r < ROWS; r++) {
    v.w[r] = *reinterpret_cast<const int4*>(wrow[r] + g * 4);
    v.s[r] = reinterpret_cast<const uint16_t*>(srow[r])[g / gsub];
  }
}

__device__ __forceinline__ float raw_half(uint32_t s) {
  return __half2float(__ushort_as_half((unsigned short)s));
}

// gate_up_silu v2: wave rows = gate j0 .. j0 + R1 - 1 and the matching up
// rows.
__global__ void __launch_bounds__(WAVES* WARP32)
    gate_up_v2_kernel(const __half* __restrict__ x,
                      const int32_t* __restrict__ topk_ids,
                      const uint32_t* __restrict__ w13,
                      const __half* __restrict__ s13, __half* __restrict__ act,
                      const int E, const int H, const int I, const int topk,
                      const int ldx, const int gsub, const float lim) {
  extern __shared__ __align__(16) __half sx[];
  constexpr int ROWS = 2 * R1;
  const int pair = blockIdx.y;
  const long e = topk_ids[pair];
  const int steps = e >= 0 && e < E ? H / (GROUP * WARP32) : 0;
  stage_row(sx, x + (long)(pair / topk) * ldx, H);
  __syncthreads();
  const int lane = threadIdx.x % WARP32, wave = threadIdx.x / WARP32;
  const int j0 = (blockIdx.x * WAVES + wave) * R1;
  const int words = H / 8, groups = H / GROUP, sgroups = groups / gsub;
  const uint32_t* wrow[ROWS];
  const __half* srow[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; r++) {
    const long row = e * 2 * I + (r / R1) * I + j0 + r % R1;
    wrow[r] = w13 + row * words;
    srow[r] = s13 + row * sgroups;
  }
  float acc[ROWS] = {};
  RowLoads<ROWS> cur, nxt;
  if (steps > 0) load_rows(cur, wrow, srow, lane, gsub);
  for (int it = 0; it < steps; it++) {
    const int g = lane + WARP32 * it;
    if (it + 1 < steps) load_rows(nxt, wrow, srow, g + WARP32, gsub);
    // Keep the next step's loads ahead of this step's math.
    asm volatile("" ::: "memory");
  #pragma unroll
    for (int r = 0; r < ROWS; r++)
      acc[r] +=
          group_dot(cur.w[r], sx + g * 8, groups * 8) * raw_half(cur.s[r]);
    cur = nxt;
  }
  #pragma unroll
  for (int r = 0; r < R1; r++) {
    float gate = wave_sum(acc[r]), up = wave_sum(acc[R1 + r]);
    if (lim > 0.f) {
      gate = fminf(gate, lim);
      up = fminf(fmaxf(up, -lim), lim);
    }
    if (lane == 0)
      act[(long)pair * I + j0 + r] =
          __float2half(gate / (1.f + __expf(-gate)) * up);
  }
}

// down v2 per (token, expert) pair: part[pair, n] = topk_weight * (W2[e] row
// n . act[pair]) for the wave's R2 rows; summed per token by sum_pairs.
__global__ void __launch_bounds__(WAVES* WARP32)
    down_v2_kernel(const __half* __restrict__ act,
                   const int32_t* __restrict__ topk_ids,
                   const float* __restrict__ topk_weights,
                   const uint32_t* __restrict__ w2,
                   const __half* __restrict__ s2, float* __restrict__ part,
                   const int E, const int H, const int I, const int gsub) {
  extern __shared__ __align__(16) __half sa[];  // [I]
  const int pair = blockIdx.y;
  const long e = topk_ids[pair];
  const int steps = e >= 0 && e < E ? I / (GROUP * WARP32) : 0;
  stage_row(sa, act + (long)pair * I, I);
  __syncthreads();
  const int lane = threadIdx.x % WARP32, wave = threadIdx.x / WARP32;
  const int n0 = (blockIdx.x * WAVES + wave) * R2;
  const int words = I / 8, groups = I / GROUP, sgroups = groups / gsub;
  const uint32_t* wrow[R2];
  const __half* srow[R2];
  #pragma unroll
  for (int r = 0; r < R2; r++) {
    const long row = e * H + n0 + r;
    wrow[r] = w2 + row * words;
    srow[r] = s2 + row * sgroups;
  }
  float acc[R2] = {};
  RowLoads<R2> cur, nxt;
  if (steps > 0) load_rows(cur, wrow, srow, lane, gsub);
  for (int it = 0; it < steps; it++) {
    const int g = lane + WARP32 * it;
    if (it + 1 < steps) load_rows(nxt, wrow, srow, g + WARP32, gsub);
    // Keep the next step's loads ahead of this step's math.
    asm volatile("" ::: "memory");
  #pragma unroll
    for (int r = 0; r < R2; r++)
      acc[r] +=
          group_dot(cur.w[r], sa + g * 8, groups * 8) * raw_half(cur.s[r]);
    cur = nxt;
  }
  const float wk = topk_weights[pair];
  #pragma unroll
  for (int r = 0; r < R2; r++) {
    const float v = wave_sum(acc[r]);
    if (lane == 0) part[(long)pair * H + n0 + r] = v * wk;
  }
}

// out[t, n] = sum over the token's topk pairs of part[pair, n], in order.
__global__ void sum_pairs_kernel(const float* __restrict__ part,
                                 __half* __restrict__ out, const int H,
                                 const int topk, const int ldo) {
  const int t = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= H) return;
  float v = 0.f;
  for (int k = 0; k < topk; k++) v += part[((long)t * topk + k) * H + n];
  out[(long)t * ldo + n] = __float2half(v);
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

template <bool ZP>
__global__ void gate_up_silu_kernel(const __half*, const int32_t*,
                                    const uint32_t*, const __half*,
                                    const uint8_t*, __half*, const int,
                                    const int, const int, const int, const int,
                                    const int, const float) {}
template <bool ZP>
__global__ void down_sum_kernel(const __half*, const int32_t*, const float*,
                                const uint32_t*, const __half*, const uint8_t*,
                                __half*, const int, const int, const int,
                                const int, const int, const int) {}
__global__ void gate_up_v2_kernel(const __half*, const int32_t*,
                                  const uint32_t*, const __half*, __half*,
                                  const int, const int, const int, const int,
                                  const int, const int, const float) {}
__global__ void down_v2_kernel(const __half*, const int32_t*, const float*,
                               const uint32_t*, const __half*, float*,
                               const int, const int, const int, const int) {}
__global__ void sum_pairs_kernel(const float*, __half*, const int, const int,
                                 const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_wna16_rdna2
}  // namespace vllm

// output [M, H] = sum_k topk_weights[:, k] * W2[e_k] silu_mul(W13[e_k] x).
// x [M, H] fp16 (rows contiguous), topk_ids [M, topk] int32, topk_weights
// [M, topk] fp32, w13 [E, 2I, H/2] and w2 [E, H, I/2] uint8 (packed int4 q,
// low nibble = even k), s13 [E, 2I, H/G] and s2 [E, H, I/G] fp16 (G a multiple
// of 32), optional zero points z13 [E, I, H/G] and z2 [E, H/2, I/G] uint8
// (two columns per byte; absent = 8), act a [M * topk, I] fp16 workspace.
// The activations are staged in LDS: H and topk * I at most 32768.
// swiglu_limit > 0 clamps gate and up before silu(gate) * up.
void moe_wna16_decode_rdna2(torch::Tensor& output, const torch::Tensor& x,
                            const torch::Tensor& topk_ids,
                            const torch::Tensor& topk_weights,
                            const torch::Tensor& w13, const torch::Tensor& s13,
                            const torch::Tensor& w2, const torch::Tensor& s2,
                            const std::optional<torch::Tensor>& z13,
                            const std::optional<torch::Tensor>& z2,
                            torch::Tensor& act, double swiglu_limit) {
  using namespace vllm::moe_wna16_rdna2;
  TORCH_CHECK(
      x.dtype() == torch::kFloat16 && output.dtype() == torch::kFloat16 &&
          act.dtype() == torch::kFloat16 && s13.dtype() == torch::kFloat16 &&
          s2.dtype() == torch::kFloat16,
      "moe_wna16_decode_rdna2 needs fp16 activations and scales");
  TORCH_CHECK(w13.dtype() == torch::kUInt8 && w2.dtype() == torch::kUInt8,
              "moe_wna16_decode_rdna2 needs uint8-packed int4 weights");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32 &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_ids.is_contiguous() && topk_weights.is_contiguous(),
              "moe_wna16_decode_rdna2 needs contiguous int32 topk_ids and fp32 "
              "topk_weights");
  const int M = x.size(0), H = x.size(1), topk = topk_ids.size(1);
  const int E = w13.size(0), I = w2.size(2) * 2;
  TORCH_CHECK(H <= 32768 && (long)topk * I <= 32768,
              "moe_wna16_decode_rdna2 stages H and topk * I fp16 values in "
              "LDS: both must be at most 32768");
  TORCH_CHECK(w13.is_contiguous() && w2.is_contiguous() &&
                  s13.is_contiguous() && s2.is_contiguous() &&
                  w13.size(1) == 2 * I && w13.size(2) == H / 2 &&
                  w2.size(0) == E && w2.size(1) == H && s13.size(1) == 2 * I &&
                  s2.size(1) == H && s13.size(2) > 0 && s2.size(2) > 0,
              "moe_wna16_decode_rdna2: unexpected weight or scale shape");
  const int G = H / s13.size(2);
  TORCH_CHECK(G % GROUP == 0 && H % G == 0 && I % G == 0 && s2.size(2) == I / G,
              "moe_wna16_decode_rdna2 needs one group size, a multiple of 32 "
              "dividing H and I");
  TORCH_CHECK(z13.has_value() == z2.has_value(),
              "moe_wna16_decode_rdna2: zero points for both or neither");
  const bool zp = z13.has_value();
  if (zp)
    TORCH_CHECK(z13->dtype() == torch::kUInt8 && z2->dtype() == torch::kUInt8 &&
                    z13->is_contiguous() && z2->is_contiguous() &&
                    z13->numel() == (long)E * I * (H / G) &&
                    z2->numel() == (long)E * (H / 2) * (I / G),
                "moe_wna16_decode_rdna2: zero points must be contiguous uint8 "
                "[E, N/2, K/G]");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  output.stride(1) == 1 && output.size(0) == M &&
                  output.size(1) == H && topk_ids.size(0) == M &&
                  act.is_contiguous() && act.numel() >= (long)M * topk * I,
              "moe_wna16_decode_rdna2: bad x, output, topk_ids or act");
  if (M == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 block(WAVES * WARP32);
  const int gsub = G / GROUP;
  const uint8_t* z13p = zp ? z13->data_ptr<uint8_t>() : nullptr;
  const uint8_t* z2p = zp ? z2->data_ptr<uint8_t>() : nullptr;
#define VLLM_MOE_DECODE_LAUNCH(ZPV)                                           \
  gate_up_silu_kernel<ZPV>                                                    \
      <<<dim3((I + WAVES * R1 - 1) / (WAVES * R1), M * topk), block,          \
         H * sizeof(__half), stream>>>(                                       \
          (const __half*)x.data_ptr(), topk_ids.data_ptr<int32_t>(),          \
          (const uint32_t*)w13.data_ptr(), (const __half*)s13.data_ptr(),     \
          z13p, (__half*)act.data_ptr(), E, H, I, topk, x.stride(0), gsub,    \
          (float)swiglu_limit);                                               \
  down_sum_kernel<ZPV><<<dim3((H + WAVES * R2 - 1) / (WAVES * R2), M), block, \
                         topk * I * sizeof(__half), stream>>>(                \
      (const __half*)act.data_ptr(), topk_ids.data_ptr<int32_t>(),            \
      topk_weights.data_ptr<float>(), (const uint32_t*)w2.data_ptr(),         \
      (const __half*)s2.data_ptr(), z2p, (__half*)output.data_ptr(), E, H, I, \
      topk, output.stride(0), gsub)
  const int step = GROUP * WARP32;
  if (!zp && H % step == 0 && I % step == 0 && I % (WAVES * R1) == 0 &&
      H % (WAVES * R2) == 0) {
    torch::Tensor part =
        torch::empty({(long)M * topk, H}, x.options().dtype(torch::kFloat32));
    gate_up_v2_kernel<<<dim3(I / (WAVES * R1), M * topk), block,
                        H * sizeof(__half), stream>>>(
        (const __half*)x.data_ptr(), topk_ids.data_ptr<int32_t>(),
        (const uint32_t*)w13.data_ptr(), (const __half*)s13.data_ptr(),
        (__half*)act.data_ptr(), E, H, I, topk, x.stride(0), gsub,
        (float)swiglu_limit);
    down_v2_kernel<<<dim3(H / (WAVES * R2), M * topk), block,
                     I * sizeof(__half), stream>>>(
        (const __half*)act.data_ptr(), topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(), (const uint32_t*)w2.data_ptr(),
        (const __half*)s2.data_ptr(), part.data_ptr<float>(), E, H, I, gsub);
    sum_pairs_kernel<<<dim3((H + 255) / 256, M), 256, 0, stream>>>(
        part.data_ptr<float>(), (__half*)output.data_ptr(), H, topk,
        output.stride(0));
  } else if (zp) {
    VLLM_MOE_DECODE_LAUNCH(true);
  } else {
    VLLM_MOE_DECODE_LAUNCH(false);
  }
#undef VLLM_MOE_DECODE_LAUNCH
}
