// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) paged MQA logits for the sparse-attention indexer in decode
// / MTP verify (fp8_paged_mqa_logits_torch without host syncs, so it can be
// captured in a CUDA graph): for query row r = (b, j) and cached key t,
//   logits[r, t] = scale[t] * sum_h w[r, h] * relu(q[r, h] . k[t])
// for t < limit(r), -inf for the other columns up to max_len. q [B, next_n, H,
// 128] and the keys are fp8 e4m3fn; the paged cache holds per block
// block_size x 128 fp8 values followed by block_size fp32 scales.
//
// A workgroup (256 threads) takes 64 keys of one row: q and the keys are
// widened exactly to fp16 (x 2^-8) into k2-major LDS tiles, each thread dots
// one key with H / 4 heads by v_dot2_f32_f16 (head operands are wave
// broadcasts), and the four head groups are summed through LDS.

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
namespace indexer_rdna2 {

static constexpr int THREADS = 256;
static constexpr int D = 128;
static constexpr int D2 = D / 2;
static constexpr int TT = 64;             // keys per workgroup
static constexpr int HGS = THREADS / TT;  // head groups

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}

__device__ __forceinline__ uint32_t as_u32(half2 h) {
  return *reinterpret_cast<uint32_t*>(&h);
}

// CTX_PER_ROW: context_lens holds one limit per row [B, next_n]; otherwise
// one per sequence [B] and row j of next_n sees limit - next_n + j + 1 keys.
template <int H, bool CTX_PER_ROW>
__global__ void __launch_bounds__(THREADS) paged_mqa_logits_rdna2_kernel(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ cache,
    const float* __restrict__ w, const int* __restrict__ ctx,
    const int* __restrict__ block_tables, float* __restrict__ out,
    const int next_n, const int block_size, const long cache_sb,
    const long bt_stride, const long out_stride, const int max_len) {
  constexpr int HG = H / HGS;  // heads per thread
  __shared__ __align__(16) uint32_t sQ[D2][H];
  __shared__ __align__(16) uint32_t sK[D2][TT];
  __shared__ float sScale[TT];
  __shared__ float sRed[HGS][TT];

  const int r = blockIdx.x, t0 = blockIdx.y * TT, tid = threadIdx.x;
  const int b = r / next_n, j = r % next_n;
  const int limit = CTX_PER_ROW ? ctx[r] : ctx[b] - next_n + j + 1;
  float* orow = out + r * out_stride;
  if (t0 >= limit) {
    if (tid < TT && t0 + tid < max_len) orow[t0 + tid] = -INFINITY;
    return;
  }

  // q row: H x 128 fp8 -> sQ[d2][h].
  for (int i = tid; i < H * D / 16; i += THREADS) {
    const int h = i / (D / 16), d0 = (i % (D / 16)) * 16;
    const uint4 v =
        *reinterpret_cast<const uint4*>(q + (long)r * H * D + h * D + d0);
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
  #pragma unroll
    for (int e = 0; e < 4; e++) {
      sQ[d0 / 2 + 2 * e][h] =
          as_u32(rdna2::fp8x2_to_half2(u[e], rdna2::FP8_LO));
      sQ[d0 / 2 + 2 * e + 1][h] =
          as_u32(rdna2::fp8x2_to_half2(u[e], rdna2::FP8_HI));
    }
  }
  // 64 keys: 4 lanes per key, 32 fp8 each -> sK[d2][key].
  {
    const int key = tid / 4, part = tid % 4, t = t0 + key;
    uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
    float sc = 0.f;
    if (t < limit) {
      const long blk = block_tables[b * bt_stride + t / block_size];
      const uint8_t* base = cache + blk * cache_sb;
      const int off = t % block_size;
      const uint8_t* kp = base + (long)off * D + part * 32;
      v0 = *reinterpret_cast<const uint4*>(kp);
      v1 = *reinterpret_cast<const uint4*>(kp + 16);
      if (part == 0)
        sc = *reinterpret_cast<const float*>(base + (long)block_size * D +
                                             off * 4);
    }
    if (part == 0) sScale[key] = sc;
    const uint32_t u[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  #pragma unroll
    for (int e = 0; e < 8; e++) {
      sK[part * 16 + 2 * e][key] =
          as_u32(rdna2::fp8x2_to_half2(u[e], rdna2::FP8_LO));
      sK[part * 16 + 2 * e + 1][key] =
          as_u32(rdna2::fp8x2_to_half2(u[e], rdna2::FP8_HI));
    }
  }
  __syncthreads();

  const int key = tid % TT, hg = tid / TT;
  float acc[HG];
  #pragma unroll
  for (int h = 0; h < HG; h++) acc[h] = 0.f;
  #pragma unroll 8
  for (int k2 = 0; k2 < D2; k2++) {
    const half2 kv = as_h2(sK[k2][key]);
  #pragma unroll
    for (int h = 0; h < HG; h += 4) {
      const uint4 qa = *reinterpret_cast<const uint4*>(&sQ[k2][hg * HG + h]);
      acc[h] = __builtin_amdgcn_fdot2(as_h2(qa.x), kv, acc[h], false);
      acc[h + 1] = __builtin_amdgcn_fdot2(as_h2(qa.y), kv, acc[h + 1], false);
      acc[h + 2] = __builtin_amdgcn_fdot2(as_h2(qa.z), kv, acc[h + 2], false);
      acc[h + 3] = __builtin_amdgcn_fdot2(as_h2(qa.w), kv, acc[h + 3], false);
    }
  }
  // Both operands carry 2^-8.
  float s = 0.f;
  const float* wr = w + (long)r * H + hg * HG;
  #pragma unroll
  for (int h = 0; h < HG; h++) s += fmaxf(acc[h] * 65536.f, 0.f) * wr[h];
  sRed[hg][key] = s;
  __syncthreads();
  if (tid < TT) {
    const int t = t0 + tid;
    if (t < max_len) {
      float v = -INFINITY;
      if (t < limit) {
        v = 0.f;
  #pragma unroll
        for (int g = 0; g < HGS; g++) v += sRed[g][tid];
        v *= sScale[tid];
      }
      orow[t] = v;
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int H, bool CTX_PER_ROW>
__global__ void paged_mqa_logits_rdna2_kernel(const uint8_t*, const uint8_t*,
                                              const float*, const int*,
                                              const int*, float*, const int,
                                              const int, const long, const long,
                                              const long, const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace indexer_rdna2
}  // namespace vllm

// q [B, next_n, H, 128] fp8 e4m3fn (contiguous; H in {32, 64}), kv_cache
// [num_blocks, block_size, (1,) 132] uint8 (per block: block_size x 128 fp8
// values, then block_size fp32 scales), weights [B * next_n, H] fp32,
// context_lens int32 [B] or [B, next_n], block_tables [B, max_blocks] int32
// -> out [B * next_n, >= max_len] fp32.
void paged_mqa_logits_rdna2(const torch::Tensor& q,
                            const torch::Tensor& kv_cache,
                            const torch::Tensor& weights,
                            const torch::Tensor& context_lens,
                            const torch::Tensor& block_tables,
                            torch::Tensor& out, int64_t max_len) {
  using namespace vllm::indexer_rdna2;
  TORCH_CHECK(q.dim() == 4 && q.size(3) == D && q.is_contiguous() &&
                  q.element_size() == 1,
              "paged_mqa_logits_rdna2: contiguous fp8 q [B, next_n, H, 128]");
  const int B = q.size(0), next_n = q.size(1), H = q.size(2);
  TORCH_CHECK(H == 32 || H == 64, "paged_mqa_logits_rdna2: 32 or 64 heads");
  TORCH_CHECK(kv_cache.element_size() == 1 && kv_cache.is_contiguous() &&
                  kv_cache.size(-1) == D + 4,
              "paged_mqa_logits_rdna2: contiguous uint8 cache [..., 132]");
  const int block_size = kv_cache.size(1);
  TORCH_CHECK(block_size % 4 == 0,
              "paged_mqa_logits_rdna2: block_size % 4 == 0 (16-byte rows)");
  TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.is_contiguous() &&
                  weights.numel() == (long)B * next_n * H,
              "paged_mqa_logits_rdna2: weights fp32 [B * next_n, H]");
  TORCH_CHECK(
      context_lens.scalar_type() == at::kInt && context_lens.is_contiguous() &&
          block_tables.scalar_type() == at::kInt && block_tables.stride(1) == 1,
      "paged_mqa_logits_rdna2: int32 context_lens / block_tables");
  const bool per_row = context_lens.numel() == (long)B * next_n;
  TORCH_CHECK(per_row || context_lens.numel() == B,
              "paged_mqa_logits_rdna2: context_lens [B] or [B, next_n]");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 &&
                  out.size(0) == (long)B * next_n && out.size(1) >= max_len &&
                  out.stride(1) == 1,
              "paged_mqa_logits_rdna2: out fp32 [B * next_n, max_len]");
  const int rows = B * next_n;
  if (rows == 0 || max_len == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(rows, (max_len + TT - 1) / TT);
  const long cache_sb = (long)block_size * (D + 4);
#define VLLM_PMQA(HH, PR)                                                      \
  paged_mqa_logits_rdna2_kernel<HH, PR><<<grid, THREADS, 0, stream>>>(         \
      (const uint8_t*)q.data_ptr(), (const uint8_t*)kv_cache.data_ptr(),       \
      weights.data_ptr<float>(), context_lens.data_ptr<int>(),                 \
      block_tables.data_ptr<int>(), out.data_ptr<float>(), next_n, block_size, \
      cache_sb, block_tables.stride(0), out.stride(0), (int)max_len)
  if (H == 32) {
    if (per_row)
      VLLM_PMQA(32, true);
    else
      VLLM_PMQA(32, false);
  } else {
    if (per_row)
      VLLM_PMQA(64, true);
    else
      VLLM_PMQA(64, false);
  }
#undef VLLM_PMQA
}
