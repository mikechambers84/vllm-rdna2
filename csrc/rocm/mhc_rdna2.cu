// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) manifold-constrained hyper-connection (mHC) kernels for 4
// residual streams. The torch reference spreads one mHC seam over ~150
// launches (an fp32 rocBLAS GEMM with N = 24 and the Sinkhorn loop as single
// elementwise ops); here it is two kernels:
//
// mhc_partial_rdna2_kernel, one workgroup per (64-channel slice, token tile):
//   POST: res[j] = fp16(sum_i comb[i, j] res_in[i] + post[j] x)   (written)
//   PRE:  partial[t, slice, k] = flat(res)[slice] . fn[k][slice] for the 24
//         mix rows, and the slice's square sum as entry 24
// The slice of fn (24 rows x 4 streams x 64 channels, fp32, 24 KB) is staged
// in LDS once per tile of up to 256 tokens; eight lanes cover a token's 64
// channels of all four streams (8 adjacent channels each) and reduce their
// partials with lane shuffles. Splitting the 16K-long dot products over slices
// keeps the whole GPU busy for a single decode token. Waves without a token
// skip the dot products, and up to 4 tokens (decode, MTP verify) every wave
// takes the same tokens and a third of the 24 mix rows.
//
// mhc_finalize_rdna2_kernel, one workgroup per token:
//   mixes = sum_slices partial * rsqrt(sum_slices sq / (4 H) + rms_eps)
//   pre   = sigmoid(mixes[:4] s0 + b[:4]) + pre_eps
//   post' = sigmoid(mixes[4:8] s1 + b[4:8]) * post_mult            (written)
//   comb' = sinkhorn(softmax(mixes[8:] s2 + b[8:]))                (written)
//           (16 lanes, DPP row / column sums, hardware reciprocals)
//   y     = fp16(sum_i pre[i] res[i]), then optionally
//   y     = fp16(y * rsqrt(mean(y^2) + norm_eps) * w)              (written)
// Matches mhc_post_torch + mhc_pre_torch + rms_norm up to fp32 summation
// order and reciprocal rounding. One H = 4096 seam (post + pre + norm) at 1-3
// tokens: 20 -> 9.5 us.

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
namespace mhc_rdna2 {

static constexpr int THREADS = 256;
static constexpr int MAX_WAVES = THREADS / 32;
static constexpr int HC = 4;
static constexpr int NMIX = 2 * HC + HC * HC;  // 24 mix rows
static constexpr int NPART = NMIX + 1;         // + square sum
static constexpr int CH = 64;                  // channels per slice
static constexpr int LPT = CH / 8;             // lanes per token
static constexpr int TPP = THREADS / LPT;      // tokens per pass
static constexpr int TILE = 256;               // tokens per workgroup

template <typename T>
__device__ __forceinline__ void load8(const T* p, float* f) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const uint32_t* u = reinterpret_cast<const uint32_t*>(&raw);
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if constexpr (std::is_same_v<T, __half>) {
      const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&u[e]));
      f[2 * e] = v.x;
      f[2 * e + 1] = v.y;
    } else {
      f[2 * e] = __uint_as_float(u[e] << 16);
      f[2 * e + 1] = __uint_as_float(u[e] & 0xffff0000u);
    }
  }
}

__device__ __forceinline__ uint32_t bf16_rne(float f) {
  const uint32_t x = __float_as_uint(f);
  return (x + 0x7fffu + ((x >> 16) & 1u)) >> 16;
}

// Rounds f to T in place and stores it.
template <typename T>
__device__ __forceinline__ void round_store8(T* p, float* f) {
  uint32_t u[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    if constexpr (std::is_same_v<T, __half>) {
      const __half2 h = __floats2half2_rn(f[2 * e], f[2 * e + 1]);
      u[e] = *reinterpret_cast<const uint32_t*>(&h);
      const float2 r = __half22float2(h);
      f[2 * e] = r.x;
      f[2 * e + 1] = r.y;
    } else {
      const uint32_t b0 = bf16_rne(f[2 * e]), b1 = bf16_rne(f[2 * e + 1]);
      f[2 * e] = __uint_as_float(b0 << 16);
      f[2 * e + 1] = __uint_as_float(b1 << 16);
      u[e] = b0 | (b1 << 16);
    }
  }
  *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(u);
}

template <typename T>
__device__ __forceinline__ float round1(float f) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(__float2half_rn(f));
  } else {
    return __uint_as_float(bf16_rne(f) << 16);
  }
}

// Lane exchange within a 16-lane DPP row (no LDS round trip, unlike
// __shfl_xor): quad_perm [1, 0, 3, 2] = lane ^ 1, [2, 3, 0, 1] = lane ^ 2,
// row_ror 4 / 8 rotate the row.
template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __int_as_float(
      __builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}
static constexpr int DPP_XOR1 = 0xB1, DPP_XOR2 = 0x4E;
static constexpr int DPP_ROR4 = 0x124, DPP_ROR8 = 0x128;

__device__ __forceinline__ float wave_sum(float v) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// SMALL (T <= 32 / LPT = 4 tokens: decode and MTP verify): every wave takes
// the same tokens and the mix rows k = wave, wave + 8, wave + 16, so the
// slice's 24 dot products run on all 8 waves instead of one; wave 0 stores the
// post-mixed residual and the square sums.
template <typename T, bool POST, bool PRE, bool SMALL = false>
__global__ void __launch_bounds__(THREADS) mhc_partial_rdna2_kernel(
    const T* __restrict__ res_in, const T* __restrict__ x,
    const float* __restrict__ post_in, const float* __restrict__ comb_in,
    const float* __restrict__ fn, T* __restrict__ res_out,
    float* __restrict__ partial, const int T_, const int H) {
  const int slice = blockIdx.x, tid = threadIdx.x;
  const int h0 = slice * CH;
  const int wave = tid / warpSize;
  const int g = tid % LPT, ts = SMALL ? tid % warpSize / LPT : tid / LPT;
  const int c0 = h0 + g * 8;
  __shared__ float4 sfn[PRE ? NMIX * HC * CH / 4 : 1];
  if constexpr (PRE) {
    // All of a thread's fn words are loaded before any LDS store, so the
    // workgroup waits for one DRAM round trip instead of six in a row.
    constexpr int PER = NMIX * HC * CH / 4 / THREADS;
    float4 w[PER];
  #pragma unroll
    for (int q = 0; q < PER; q++) {
      const int idx = tid + q * THREADS;
      const int k = idx / (HC * CH / 4), rem = idx % (HC * CH / 4);
      const int i = rem / (CH / 4), c = rem % (CH / 4);
      w[q] = *reinterpret_cast<const float4*>(fn + ((long)k * HC + i) * H + h0 +
                                              c * 4);
    }
  #pragma unroll
    for (int q = 0; q < PER; q++) sfn[tid + q * THREADS] = w[q];
    __syncthreads();
  }
  const int t_end = min(T_, (int)(blockIdx.y + 1) * TILE);
  for (int t0 = blockIdx.y * TILE; t0 < t_end; t0 += TPP) {
    // A wave covers warpSize / LPT tokens; waves past the last token (7 of 8
    // for a single decode token) skip the 24 dot products and their LDS reads.
    if (!SMALL && t0 + wave * (warpSize / LPT) >= t_end) break;
    const int t = t0 + ts;
    const bool active = t < t_end;
    float r[HC][8];
    if (active) {
      const T* src = res_in + (long)t * HC * H + c0;
  #pragma unroll
      for (int i = 0; i < HC; i++) load8(src + i * H, r[i]);
      if constexpr (POST) {
        float xv[8];
        load8(x + (long)t * H + c0, xv);
        float comb[HC][HC], post[HC];
  #pragma unroll
        for (int i = 0; i < HC; i++) {
          post[i] = post_in[t * HC + i];
  #pragma unroll
          for (int k = 0; k < HC; k++)
            comb[i][k] = comb_in[(t * HC + i) * HC + k];
        }
        float o[HC][8];
  #pragma unroll
        for (int c = 0; c < HC; c++)
  #pragma unroll
          for (int e = 0; e < 8; e++) {
            float a = 0.f;
  #pragma unroll
            for (int i = 0; i < HC; i++) a += comb[i][c] * r[i][e];
            o[c][e] = a + post[c] * xv[e];
          }
        T* dst = res_out + (long)t * HC * H + c0;
  #pragma unroll
        for (int c = 0; c < HC; c++) {
          if (SMALL && wave != 0) {
  #pragma unroll
            for (int e = 0; e < 8; e++) r[c][e] = round1<T>(o[c][e]);
          } else {
            round_store8(dst + c * H, o[c]);
  #pragma unroll
            for (int e = 0; e < 8; e++) r[c][e] = o[c][e];
          }
        }
      }
    } else {
  #pragma unroll
      for (int i = 0; i < HC; i++)
  #pragma unroll
        for (int e = 0; e < 8; e++) r[i][e] = 0.f;
    }
    if constexpr (PRE) {
      // The fn reads do not depend on the token: an opaque offset keeps the
      // compiler from hoisting all 192 of them out of the token loop (768
      // VGPRs, spilled).
      int fo = g * 2;
      asm volatile("" : "+v"(fo));
      float* p = partial + ((long)t * (H / CH) + slice) * NPART;
      float sq = 0.f;
  #pragma unroll
      for (int i = 0; i < HC; i++)
  #pragma unroll
        for (int e = 0; e < 8; e++) sq += r[i][e] * r[i][e];
      for (int o = LPT / 2; o > 0; o >>= 1) sq += __shfl_xor(sq, o, LPT);
      if (active && g == 0 && (!SMALL || wave == 0)) p[NMIX] = sq;
      constexpr int KSTEP = SMALL ? MAX_WAVES : 1;
  #pragma unroll
      for (int k = SMALL ? wave : 0; k < NMIX; k += KSTEP) {
        float a = 0.f;
  #pragma unroll
        for (int i = 0; i < HC; i++) {
          const float4 w0 = sfn[(k * HC + i) * (CH / 4) + fo];
          const float4 w1 = sfn[(k * HC + i) * (CH / 4) + fo + 1];
          a += w0.x * r[i][0] + w0.y * r[i][1] + w0.z * r[i][2] +
               w0.w * r[i][3] + w1.x * r[i][4] + w1.y * r[i][5] +
               w1.z * r[i][6] + w1.w * r[i][7];
        }
        for (int o = LPT / 2; o > 0; o >>= 1) a += __shfl_xor(a, o, LPT);
        if (active && g == 0) p[k] = a;
      }
    }
  }
}

template <typename T, bool NORM>
__global__ void __launch_bounds__(THREADS) mhc_finalize_rdna2_kernel(
    const T* __restrict__ res, const float* __restrict__ partial,
    const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
    const T* __restrict__ norm_w, float* __restrict__ post_out,
    float* __restrict__ comb_out, T* __restrict__ y_out, const int T_,
    const int H, const float rms_eps, const float pre_eps, const float sk_eps,
    const float post_mult, const int sk_repeat, const float norm_eps) {
  const int t = blockIdx.x, tid = threadIdx.x;
  const int lane = tid % warpSize, wave = tid / warpSize;
  const int nwaves = THREADS / warpSize, nslices = H / CH;
  constexpr int RJ = 10;                         // threads per partial entry
  constexpr int RQ = (8192 / CH + RJ - 1) / RJ;  // slices per thread
  __shared__ float s_red[NPART][RJ];
  __shared__ float s_mix[NPART];
  __shared__ float s_pre[HC];
  __shared__ float red[MAX_WAVES];
  // The residual and norm weights for y do not depend on the mixes: load them
  // first so they arrive during the reductions and Sinkhorn.
  const T* rt = res + (long)t * HC * H;
  uint4 rraw[4][HC], wraw[4];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const int c = (j * THREADS + tid) * 8;
    if (c < H) {
  #pragma unroll
      for (int i = 0; i < HC; i++)
        rraw[j][i] = *reinterpret_cast<const uint4*>(rt + i * H + c);
      if constexpr (NORM) wraw[j] = *reinterpret_cast<const uint4*>(norm_w + c);
    }
  }
  if (tid < NPART * RJ) {
    const int k = tid / RJ, j = tid % RJ;
    const float* pt = partial + (long)t * nslices * NPART + k;
    float v[RQ];
  #pragma unroll
    for (int q = 0; q < RQ; q++) {
      const int sl = j + q * RJ;
      v[q] = sl < nslices ? pt[sl * NPART] : 0.f;
    }
    float s = 0.f;
  #pragma unroll
    for (int q = 0; q < RQ; q++) s += v[q];
    s_red[k][j] = s;
  }
  __syncthreads();
  if (tid < NPART) {
    float s = 0.f;
  #pragma unroll
    for (int j = 0; j < RJ; j++) s += s_red[tid][j];
    s_mix[tid] = s;
  }
  __syncthreads();
  if (tid < HC * HC) {
    // Lane l holds comb element (l / 4, l % 4); row sums are shuffles over
    // lane bits 0-1, column sums over bits 2-3.
    const float rs = rsqrtf(s_mix[NMIX] / (HC * H) + rms_eps);
    if (tid < HC) {
      s_pre[tid] =
          1.f / (1.f + expf(-(s_mix[tid] * rs * hc_scale[0] + hc_base[tid]))) +
          pre_eps;
      post_out[t * HC + tid] =
          post_mult / (1.f + expf(-(s_mix[HC + tid] * rs * hc_scale[1] +
                                    hc_base[HC + tid])));
    }
    // Row sums over lanes l ^ 1, l ^ 2 and column sums over the row_ror 4 / 8
    // rotations of the 16 lanes, all through DPP.
    float c = s_mix[2 * HC + tid] * rs * hc_scale[2] + hc_base[2 * HC + tid];
    float mx = fmaxf(c, dpp<DPP_XOR1>(c));
    mx = fmaxf(mx, dpp<DPP_XOR2>(mx));
    c = expf(c - mx);
    float s = c + dpp<DPP_XOR1>(c);
    s += dpp<DPP_XOR2>(s);
    c = c * __builtin_amdgcn_rcpf(s) + sk_eps;
    for (int it = 0; it < sk_repeat; it++) {
      if (it > 0) {
        s = c + dpp<DPP_XOR1>(c);
        s += dpp<DPP_XOR2>(s);
        c *= __builtin_amdgcn_rcpf(s + sk_eps);
      }
      s = c + dpp<DPP_ROR4>(c);
      s += dpp<DPP_ROR8>(s);
      c *= __builtin_amdgcn_rcpf(s + sk_eps);
    }
    comb_out[t * HC * HC + tid] = c;
  }
  __syncthreads();
  float pre[HC];
  #pragma unroll
  for (int i = 0; i < HC; i++) pre[i] = s_pre[i];
  T* yt = y_out + (long)t * H;
  float sq = 0.f;
  // Each thread keeps at most 4 groups of 8 channels (H <= 8192).
  float y[4][8];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const int c = (j * THREADS + tid) * 8;
    if (c >= H) break;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  #pragma unroll
    for (int i = 0; i < HC; i++) {
      float rv[8];
      load8(reinterpret_cast<const T*>(&rraw[j][i]), rv);
  #pragma unroll
      for (int e = 0; e < 8; e++) acc[e] += pre[i] * rv[e];
    }
  #pragma unroll
    for (int e = 0; e < 8; e++) {
      y[j][e] = round1<T>(acc[e]);
      sq += y[j][e] * y[j][e];
    }
  }
  float inv = 1.f;
  if constexpr (NORM) {
    sq = wave_sum(sq);
    if (lane == 0) red[wave] = sq;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < nwaves; w++) s += red[w];
    inv = rsqrtf(s / H + norm_eps);
  }
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const int c = (j * THREADS + tid) * 8;
    if (c >= H) break;
    if constexpr (NORM) {
      float w[8];
      load8(reinterpret_cast<const T*>(&wraw[j]), w);
  #pragma unroll
      for (int e = 0; e < 8; e++) y[j][e] = y[j][e] * inv * w[e];
    }
    round_store8(yt + c, y[j]);
  }
}

#else  // non-RDNA2 device pass: empty stubs for symbol parity.

template <typename T, bool POST, bool PRE, bool SMALL = false>
__global__ void mhc_partial_rdna2_kernel(const T*, const T*, const float*,
                                         const float*, const float*, T*, float*,
                                         const int, const int) {}

template <typename T, bool NORM>
__global__ void mhc_finalize_rdna2_kernel(const T*, const float*, const float*,
                                          const float*, const T*, float*,
                                          float*, T*, const int, const int,
                                          const float, const float, const float,
                                          const float, const int, const float) {
}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

template <typename T, bool POST, bool PRE>
void launch_partial(int T_, int H, cudaStream_t stream, const void* res_in,
                    const void* x, const float* post_in, const float* comb_in,
                    const float* fn, void* res_out, float* partial) {
  const dim3 grid(H / CH, (T_ + TILE - 1) / TILE);
  if (T_ <= 32 / LPT) {
    mhc_partial_rdna2_kernel<T, POST, PRE, true><<<grid, THREADS, 0, stream>>>(
        (const T*)res_in, (const T*)x, post_in, comb_in, fn, (T*)res_out,
        partial, T_, H);
  } else {
    mhc_partial_rdna2_kernel<T, POST, PRE, false><<<grid, THREADS, 0, stream>>>(
        (const T*)res_in, (const T*)x, post_in, comb_in, fn, (T*)res_out,
        partial, T_, H);
  }
}

template <typename T, bool NORM>
void launch_finalize(int T_, int H, cudaStream_t stream, const void* res,
                     const float* partial, const float* hc_scale,
                     const float* hc_base, const void* norm_w, float* post_out,
                     float* comb_out, void* y_out, float rms_eps, float pre_eps,
                     float sk_eps, float post_mult, int sk_repeat,
                     float norm_eps) {
  mhc_finalize_rdna2_kernel<T, NORM><<<T_, THREADS, 0, stream>>>(
      (const T*)res, partial, hc_scale, hc_base, (const T*)norm_w, post_out,
      comb_out, (T*)y_out, T_, H, rms_eps, pre_eps, sk_eps, post_mult,
      sk_repeat, norm_eps);
}

}  // namespace mhc_rdna2
}  // namespace vllm

static void check_residual(const torch::Tensor& residual, const char* name) {
  const auto dt = residual.scalar_type();
  TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16, name,
              ": fp16 or bf16 residual");
  TORCH_CHECK(residual.dim() >= 2 && residual.is_contiguous() &&
                  residual.size(-2) == vllm::mhc_rdna2::HC &&
                  residual.size(-1) % 2048 == 0 && residual.size(-1) <= 8192,
              name, ": contiguous residual [..., 4, H], H in 2048 * {1..4}");
}

static void check_post(const torch::Tensor& residual, const torch::Tensor& x,
                       const torch::Tensor& post_layer_mix,
                       const torch::Tensor& comb_res_mix,
                       const torch::Tensor& residual_out, long T_, int H,
                       const char* name) {
  constexpr int HC = vllm::mhc_rdna2::HC;
  const auto dt = residual.scalar_type();
  TORCH_CHECK(x.scalar_type() == dt && x.is_contiguous() &&
                  x.numel() == T_ * H && residual_out.scalar_type() == dt &&
                  residual_out.is_contiguous() &&
                  residual_out.numel() == residual.numel(),
              name, ": x [T, H] and the output residual like residual");
  TORCH_CHECK(
      post_layer_mix.scalar_type() == at::kFloat &&
          post_layer_mix.is_contiguous() && post_layer_mix.numel() == T_ * HC &&
          comb_res_mix.scalar_type() == at::kFloat &&
          comb_res_mix.is_contiguous() && comb_res_mix.numel() == T_ * HC * HC,
      name, ": post_layer_mix [T, 4, 1], comb_res_mix [T, 4, 4] fp32");
}

// residual [T, 4, H] (fp16 or bf16, contiguous). With x / post_layer_mix /
// comb_res_mix the post step runs first and writes residual_out [T, 4, H];
// then the pre step on the (new) residual writes post_mix [T, 4, 1], comb_mix
// [T, 4, 4] (fp32) and layer_input [T, H], RMS-normalized by norm_weight [H]
// when given. fn [24, 4H], hc_scale [3], hc_base [24] fp32; partial is an fp32
// workspace of at least H / 64 * T * 25 elements.
void mhc_rdna2(
    const torch::Tensor& residual, const std::optional<torch::Tensor>& x,
    const std::optional<torch::Tensor>& post_layer_mix,
    const std::optional<torch::Tensor>& comb_res_mix, const torch::Tensor& fn,
    const torch::Tensor& hc_scale, const torch::Tensor& hc_base,
    const std::optional<torch::Tensor>& norm_weight,
    std::optional<torch::Tensor> residual_out, torch::Tensor& post_mix,
    torch::Tensor& comb_mix, torch::Tensor& layer_input, torch::Tensor& partial,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps) {
  using namespace vllm::mhc_rdna2;
  check_residual(residual, "mhc_rdna2");
  const auto dt = residual.scalar_type();
  const int H = residual.size(-1);
  const long T_ = residual.numel() / ((long)HC * H);
  const bool post = x.has_value();
  TORCH_CHECK(post == post_layer_mix.has_value() &&
                  post == comb_res_mix.has_value() &&
                  post == residual_out.has_value(),
              "mhc_rdna2: x, post_layer_mix, comb_res_mix and residual_out go "
              "together");
  if (post)
    check_post(residual, *x, *post_layer_mix, *comb_res_mix, *residual_out, T_,
               H, "mhc_rdna2");
  TORCH_CHECK(fn.scalar_type() == at::kFloat && fn.is_contiguous() &&
                  fn.size(0) == NMIX && fn.size(1) == (long)HC * H,
              "mhc_rdna2: fn fp32 [24, 4H]");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3 &&
                  hc_base.scalar_type() == at::kFloat &&
                  hc_base.numel() == NMIX,
              "mhc_rdna2: hc_scale [3], hc_base [24] fp32");
  TORCH_CHECK(post_mix.scalar_type() == at::kFloat &&
                  post_mix.is_contiguous() && post_mix.numel() == T_ * HC &&
                  comb_mix.scalar_type() == at::kFloat &&
                  comb_mix.is_contiguous() && comb_mix.numel() == T_ * HC * HC,
              "mhc_rdna2: post_mix [T, 4, 1] and comb_mix [T, 4, 4] fp32");
  TORCH_CHECK(layer_input.scalar_type() == dt && layer_input.is_contiguous() &&
                  layer_input.numel() == T_ * H,
              "mhc_rdna2: layer_input [T, H] like residual");
  TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.is_contiguous() &&
                  partial.numel() >= (long)(H / CH) * T_ * NPART,
              "mhc_rdna2: partial fp32 workspace of H / 64 * T * 25");
  if (norm_weight)
    TORCH_CHECK(norm_weight->scalar_type() == dt && norm_weight->numel() == H &&
                    norm_weight->is_contiguous(),
                "mhc_rdna2: norm_weight [H] like residual");
  if (T_ == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(residual));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const void* xp = post ? x->data_ptr() : nullptr;
  const float* pp = post ? post_layer_mix->data_ptr<float>() : nullptr;
  const float* cp = post ? comb_res_mix->data_ptr<float>() : nullptr;
  void* rop = post ? residual_out->data_ptr() : nullptr;
  const void* res_cur = post ? rop : residual.data_ptr();
  const void* wp = norm_weight ? norm_weight->data_ptr() : nullptr;
#define VLLM_MHC_RUN(TYPE)                                                   \
  if (post)                                                                  \
    launch_partial<TYPE, true, true>(T_, H, stream, residual.data_ptr(), xp, \
                                     pp, cp, fn.data_ptr<float>(), rop,      \
                                     partial.data_ptr<float>());             \
  else                                                                       \
    launch_partial<TYPE, false, true>(                                       \
        T_, H, stream, residual.data_ptr(), nullptr, nullptr, nullptr,       \
        fn.data_ptr<float>(), nullptr, partial.data_ptr<float>());           \
  if (wp)                                                                    \
    launch_finalize<TYPE, true>(                                             \
        T_, H, stream, res_cur, partial.data_ptr<float>(),                   \
        hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(), wp,           \
        post_mix.data_ptr<float>(), comb_mix.data_ptr<float>(),              \
        layer_input.data_ptr(), (float)rms_eps, (float)hc_pre_eps,           \
        (float)hc_sinkhorn_eps, (float)hc_post_mult_value,                   \
        (int)sinkhorn_repeat, (float)norm_eps);                              \
  else                                                                       \
    launch_finalize<TYPE, false>(                                            \
        T_, H, stream, res_cur, partial.data_ptr<float>(),                   \
        hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(), nullptr,      \
        post_mix.data_ptr<float>(), comb_mix.data_ptr<float>(),              \
        layer_input.data_ptr(), (float)rms_eps, (float)hc_pre_eps,           \
        (float)hc_sinkhorn_eps, (float)hc_post_mult_value,                   \
        (int)sinkhorn_repeat, (float)norm_eps);
  if (dt == at::kHalf) {
    VLLM_MHC_RUN(__half)
  } else {
    VLLM_MHC_RUN(__hip_bfloat16)
  }
#undef VLLM_MHC_RUN
}

// out [T, 4, H] = sum_i comb_res_mix[:, i, j] residual[:, i] +
// post_layer_mix[:, j] x (the mHC post step alone).
void mhc_post_rdna2(const torch::Tensor& x, const torch::Tensor& residual,
                    const torch::Tensor& post_layer_mix,
                    const torch::Tensor& comb_res_mix, torch::Tensor& out) {
  using namespace vllm::mhc_rdna2;
  check_residual(residual, "mhc_post_rdna2");
  const int H = residual.size(-1);
  const long T_ = residual.numel() / ((long)HC * H);
  check_post(residual, x, post_layer_mix, comb_res_mix, out, T_, H,
             "mhc_post_rdna2");
  if (T_ == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(residual));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (residual.scalar_type() == at::kHalf)
    launch_partial<__half, true, false>(
        T_, H, stream, residual.data_ptr(), x.data_ptr(),
        post_layer_mix.data_ptr<float>(), comb_res_mix.data_ptr<float>(),
        nullptr, out.data_ptr(), nullptr);
  else
    launch_partial<__hip_bfloat16, true, false>(
        T_, H, stream, residual.data_ptr(), x.data_ptr(),
        post_layer_mix.data_ptr<float>(), comb_res_mix.data_ptr<float>(),
        nullptr, out.data_ptr(), nullptr);
}
