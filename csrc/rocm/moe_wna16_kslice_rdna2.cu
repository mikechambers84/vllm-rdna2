// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) grouped W4A16 GEMM for fused MoE at a few routed rows per
// expert, on the Triton WNA16 tensors (w [E, N, K/2] uint8, two values q + 8
// per byte, low nibble = even k; group scales [E, N, K/G] fp16; optional zero
// points [E, N/2, K/G] uint8), rows routed through moe_align_block_size(T).
//
// moe_wna16_skinny_rdna2 gives each lane whole weight rows, so a wave's 16-byte
// loads touch 32 rows at once: ~180 GB/s on a V620 against ~490 GB/s for loads
// where 8 lanes cover 128 contiguous bytes of one row. Here the 8 lanes of an
// octet split K for one output column (32 k each per load, 256 k per wave
// load); each lane covers NG columns 4 apart (every activation read from LDS
// feeds NG columns), and the octet sums its partial dots with lane shuffles
// at the end. The T activation rows are
// gathered into LDS (in phases of K up to 32 KB), each 8-element group stored
// pair-permuted ((a0, a4), (a1, a5), ...) to match the nibble expansion (w >>
// 4i) & 0x000F000F | 0x64006400 = half2(1024 + q_i, 1024 + q_(i+4)); (q - z)
// is exact in fp16 and the group scale is applied per 32-k chunk in fp32. The
// next chunk's weights are loaded while the current one is dotted; each
// phase's group scales for the workgroup's 256 columns come through LDS, and
// padding rows of a block are skipped (16-row blocks then cover most experts
// of a 384-token chunk in one pass over their weights).
//
// At the 140 W cap this kernel runs at the power limit, so its speed follows
// its energy per row: reading the experts from DRAM, re-gathering activations
// per workgroup and per-chunk scale loads each cost 10-30%.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_wna16_kslice_rdna2 {

static constexpr int THREADS = 512;
static constexpr int LDS_ELEMS = 16384;   // 32 KB of activations per phase
static constexpr int KW = 256;            // k per wave load (8 lanes x 32)
static constexpr int COLS = 256;          // columns per workgroup (NG = 4)
static constexpr int SCALE_ELEMS = 8192;  // COLS x up to 32 groups per phase

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ half2 as_h2(uint32_t u) {
  return *reinterpret_cast<half2*>(&u);
}

// LDS slot of 8-element group q (0-3) of 32-k chunk kq (0-7) within a 256-k
// span: the octet's 8 lanes read the same q of 8 chunks 64 bytes apart, which
// would fall into 2 of the 8 16-byte bank groups; rotating each chunk's groups
// by kq / 2 spreads them over all 8.
__device__ __forceinline__ int lds_group(int kq, int q) {
  return kq * 4 + ((q + (kq >> 1)) & 3);
}

// T rows per workgroup, NG groups of 4 columns per wave (COLS columns per
// workgroup); kp: K per activation phase (a multiple of 256 dividing K).
template <int T, int NG, bool MUL_W, bool ZP>
__global__ void __launch_bounds__(THREADS) moe_wna16_kslice_rdna2_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    const __half* __restrict__ S, const uint8_t* __restrict__ Z,
    __half* __restrict__ C, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, const int num_valid, const int top_k,
    const int N, const int K, const int group_size, const int E, const int lda,
    const int ldc, const int kp, const int num_sorted) {
  __shared__ __align__(16) __half sA[LDS_ELEMS];
  __shared__ __align__(16) __half sS[SCALE_ELEMS];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int row0 = blockIdx.x * T;
  // All routing loads at once; padding rows (sid >= num_valid) are skipped.
  const int padded = num_post_padded[0];
  const int e = expert_ids[blockIdx.x];
  int sid[T];
  #pragma unroll
  for (int t = 0; t < T; t++)
    sid[t] = sorted_ids[min(row0 + t, num_sorted - 1)];
  if (row0 >= padded || e < 0 || e >= E) return;
  uint32_t valid = 0;
  #pragma unroll
  for (int t = 0; t < T; t++) valid |= (uint32_t)(sid[t] < num_valid) << t;
  valid = __builtin_amdgcn_readfirstlane(valid);

  const int cg = lane / 8, kq = lane % 8;
  const int groups = K / group_size;
  const uint8_t* We = W + (long)e * N * (K / 2);
  const __half* Se = S + (long)e * N * groups;
  const uint8_t* Ze = ZP ? Z + (long)e * (N / 2) * groups : nullptr;
  int col[NG];
  #pragma unroll
  for (int g = 0; g < NG; g++)
    col[g] = min(blockIdx.y * COLS + (wave * NG + g) * 4 + cg, N - 1);

  float acc[NG][T];
  #pragma unroll
  for (int g = 0; g < NG; g++)
  #pragma unroll
    for (int t = 0; t < T; t++) acc[g][t] = 0.f;

  const int kpb = kp / KW;  // chunks per phase
  // Chunk c of a phase: k = p0 + c * 256 + kq * 32 for all NG column groups,
  // so every activation read from LDS feeds NG columns.
  auto wload = [&](int k, uint4(&wv)[NG], uint32_t (&zb)[NG]) {
  #pragma unroll
    for (int g = 0; g < NG; g++) {
      wv[g] =
          *reinterpret_cast<const uint4*>(We + (long)col[g] * (K / 2) + k / 2);
      zb[g] = 0x64086408u;  // 1024 + 8
      if constexpr (ZP)
        zb[g] = (0x6400u | ((Ze[(long)(col[g] / 2) * groups + k / group_size] >>
                             ((col[g] & 1) * 4)) &
                            0xFu)) *
                0x00010001u;
    }
  };

  // The group scales of the workgroup's columns go through LDS per phase.
  const int gp = kp / group_size, col0 = blockIdx.y * COLS;
  const int gshift = __builtin_ctz(group_size), cstep = KW >> gshift;
  int sbase[NG];
  #pragma unroll
  for (int g = 0; g < NG; g++)
    sbase[g] = ((wave * NG + g) * 4 + cg) * gp + ((kq * 32) >> gshift);
  const int rows = 32 - __builtin_clz(valid);
  for (int p0 = 0; p0 < K; p0 += kp) {
    uint4 wv[NG];
    uint32_t zb[NG];
    wload(p0 + kq * 32, wv, zb);  // in flight during the activation gather
    if (p0 > 0) __syncthreads();
    const int g0 = p0 / group_size;
    if (gp % 8 == 0 && groups % 8 == 0) {
      for (int i = tid; i < COLS * gp / 8; i += THREADS) {
        const int cl = i / (gp / 8), gi = (i % (gp / 8)) * 8;
        *reinterpret_cast<uint4*>(&sS[cl * gp + gi]) =
            *reinterpret_cast<const uint4*>(
                Se + (long)min(col0 + cl, N - 1) * groups + g0 + gi);
      }
    } else {
      for (int i = tid; i < COLS * gp; i += THREADS) {
        const int cl = i / gp, gi = i % gp;
        sS[cl * gp + gi] = Se[(long)min(col0 + cl, N - 1) * groups + g0 + gi];
      }
    }
    for (int i = tid; i < rows * (kp / 8); i += THREADS) {
      const int t = i / (kp / 8), c = i % (kp / 8);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (sid[t] < num_valid)
        v = *reinterpret_cast<const uint4*>(A + (long)(sid[t] / top_k) * lda +
                                            p0 + c * 8);
      uint4 q;
      q.x = __builtin_amdgcn_perm(v.z, v.x, 0x05040100u);
      q.y = __builtin_amdgcn_perm(v.z, v.x, 0x07060302u);
      q.z = __builtin_amdgcn_perm(v.w, v.y, 0x05040100u);
      q.w = __builtin_amdgcn_perm(v.w, v.y, 0x07060302u);
      const int span = c / 32, kq8 = (c % 32) / 4;  // c: 8-element group
      *reinterpret_cast<uint4*>(
          &sA[t * kp + span * KW + lds_group(kq8, c % 4) * 8]) = q;
    }
    __syncthreads();
    float snext[NG];
  #pragma unroll
    for (int g = 0; g < NG; g++) snext[g] = __half2float(sS[sbase[g]]);
    for (int c = 0; c < kpb; c++) {
      half2 w[NG][16];
      float s[NG];
  #pragma unroll
      for (int g = 0; g < NG; g++) {
        s[g] = snext[g];
        const half2 bias = as_h2(zb[g]);
        const uint32_t words[4] = {wv[g].x, wv[g].y, wv[g].z, wv[g].w};
  #pragma unroll
        for (int q = 0; q < 4; q++)
  #pragma unroll
          for (int j = 0; j < 4; j++)
            w[g][q * 4 + j] = __hsub2(
                as_h2(((words[q] >> (4 * j)) & 0x000F000Fu) | 0x64006400u),
                bias);
      }
      if (c + 1 < kpb) {
        wload(p0 + (c + 1) * KW + kq * 32, wv, zb);
  #pragma unroll
        for (int g = 0; g < NG; g++)
          snext[g] = __half2float(sS[sbase[g] + (c + 1) * cstep]);
      }
      const int ko = c * KW;
  #pragma unroll
      for (int t = 0; t < T; t++) {
        if (!((valid >> t) & 1u)) continue;
        half2 a[16];
  #pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4 v = *reinterpret_cast<const uint4*>(
              &sA[t * kp + ko + lds_group(kq, q) * 8]);
          a[q * 4 + 0] = as_h2(v.x);
          a[q * 4 + 1] = as_h2(v.y);
          a[q * 4 + 2] = as_h2(v.z);
          a[q * 4 + 3] = as_h2(v.w);
        }
  #pragma unroll
        for (int g = 0; g < NG; g++) {
          float part = 0.f;
  #pragma unroll
          for (int j = 0; j < 16; j++)
            part = __builtin_amdgcn_fdot2(a[j], w[g][j], part, false);
          acc[g][t] += s[g] * part;
        }
      }
    }
  }

  // Octet sums; afterwards every lane of the octet holds the column's dots
  // and lane kq writes rows kq, kq + 8, ...
  #pragma unroll
  for (int g = 0; g < NG; g++)
  #pragma unroll
    for (int t = 0; t < T; t++) {
      float v = acc[g][t];
      v += __shfl_xor(v, 1, 8);
      v += __shfl_xor(v, 2, 8);
      v += __shfl_xor(v, 4, 8);
      acc[g][t] = v;
    }
  #pragma unroll
  for (int g = 0; g < NG; g++) {
    const int c = blockIdx.y * COLS + (wave * NG + g) * 4 + cg;
    if (c >= N) continue;
  #pragma unroll
    for (int t = 0; t < T; t++) {
      if (t % 8 != kq || sid[t] >= num_valid) continue;
      const float wgt = MUL_W ? topk_w[sid[t]] : 1.f;
      C[(long)sid[t] * ldc + c] = __float2half(acc[g][t] * wgt);
    }
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int T, int NG, bool MUL_W, bool ZP>
__global__ void moe_wna16_kslice_rdna2_kernel(
    const __half*, const uint8_t*, const __half*, const uint8_t*, __half*,
    const int*, const int*, const int*, const float*, const int, const int,
    const int, const int, const int, const int, const int, const int, const int,
    const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

// Largest K per phase that divides K, is a multiple of 256 and keeps the T x
// kp fp16 activation tile within LDS_ELEMS; 0 if none.
static int phase_k(int K, int T, int group_size = 32) {
  for (int kp = K; kp >= KW; kp -= KW)
    if (K % kp == 0 && T * kp <= LDS_ELEMS &&
        COLS * (kp / group_size) <= SCALE_ELEMS)
      return kp;
  return 0;
}

}  // namespace moe_wna16_kslice_rdna2
}  // namespace vllm

static bool moe_wna16_kslice_rdna2_supported(int64_t K, int64_t block_m) {
  using namespace vllm::moe_wna16_kslice_rdna2;
  return (block_m == 8 || block_m == 16) && K % KW == 0 &&
         phase_k(K, block_m) > 0;
}

// Same contract as moe_wna16_skinny_rdna2 (output [num_valid, N] receives row
// sorted_ids[r] of each block; mul_routed_weight scales by the fp32 top-k
// weight), for block_m 8 or 16 and K % 256 == 0.
void moe_wna16_kslice_rdna2(torch::Tensor& output, const torch::Tensor& a,
                            const torch::Tensor& w, const torch::Tensor& scales,
                            const std::optional<torch::Tensor>& zeros,
                            const torch::Tensor& sorted_ids,
                            const torch::Tensor& expert_ids,
                            const torch::Tensor& num_tokens_post_padded,
                            const torch::Tensor& topk_weights, int64_t top_k,
                            bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_wna16_kslice_rdna2;
  TORCH_CHECK(a.dtype() == torch::kFloat16 &&
                  output.dtype() == torch::kFloat16 &&
                  scales.dtype() == torch::kFloat16,
              "moe_wna16_kslice_rdna2 needs fp16 activations, output and "
              "scales");
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous() &&
                  scales.dim() == 3 && scales.is_contiguous(),
              "moe_wna16_kslice_rdna2 needs contiguous [E, N, K/2] uint8 "
              "weights and [E, N, K/G] scales");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 &&
                  expert_ids.dtype() == torch::kInt32 &&
                  num_tokens_post_padded.dtype() == torch::kInt32 &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "moe_wna16_kslice_rdna2 needs int32 alignment tensors and fp32 "
              "top-k weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2) * 2;
  const int groups = scales.size(2);
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == N && groups > 0 &&
                  K % groups == 0 && (K / groups) % 32 == 0 &&
                  ((K / groups) & (K / groups - 1)) == 0 && K / groups <= KW,
              "moe_wna16_kslice_rdna2 needs power-of-two group sizes from 32 "
              "to 256");
  TORCH_CHECK(moe_wna16_kslice_rdna2_supported(K, block_m),
              "moe_wna16_kslice_rdna2 needs block_m 8 or 16 and K % 256 == 0");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "moe_wna16_kslice_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows");
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_wna16_kslice_rdna2 needs a contiguous [num_valid, N] "
              "output");
  const uint8_t* zp = nullptr;
  if (zeros) {
    TORCH_CHECK(zeros->dtype() == torch::kUInt8 && zeros->is_contiguous() &&
                    zeros->numel() == (long)E * (N / 2) * groups,
                "moe_wna16_kslice_rdna2: zeros must be contiguous uint8 "
                "[E, N/2, K/G]");
    zp = zeros->data_ptr<uint8_t>();
  }
  if (expert_ids.numel() == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int group_size = K / groups;
  constexpr int NG = 4;
  const int kp = phase_k(K, block_m, group_size);
  const dim3 grid(expert_ids.numel(), (N + COLS - 1) / COLS);
#define VLLM_KSLICE_LAUNCH(T, MW, Z)                                          \
  moe_wna16_kslice_rdna2_kernel<T, NG, MW, Z><<<grid, THREADS, 0, stream>>>(  \
      (const __half*)a.data_ptr(), w.data_ptr<uint8_t>(),                     \
      (const __half*)scales.data_ptr(), zp, (__half*)output.data_ptr(),       \
      sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),                 \
      num_tokens_post_padded.data_ptr<int>(), topk_weights.data_ptr<float>(), \
      num_valid, top_k, N, K, group_size, E, a.stride(0), N, kp,              \
      (int)sorted_ids.numel())
#define VLLM_KSLICE_BY_FLAGS(T)            \
  if (mul_routed_weight) {                 \
    if (zp)                                \
      VLLM_KSLICE_LAUNCH(T, true, true);   \
    else                                   \
      VLLM_KSLICE_LAUNCH(T, true, false);  \
  } else {                                 \
    if (zp)                                \
      VLLM_KSLICE_LAUNCH(T, false, true);  \
    else                                   \
      VLLM_KSLICE_LAUNCH(T, false, false); \
  }
  if (block_m == 16) {
    VLLM_KSLICE_BY_FLAGS(16)
  } else {
    VLLM_KSLICE_BY_FLAGS(8)
  }
#undef VLLM_KSLICE_BY_FLAGS
#undef VLLM_KSLICE_LAUNCH
}
