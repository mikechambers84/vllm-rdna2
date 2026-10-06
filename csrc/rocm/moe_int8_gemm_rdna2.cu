// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// RDNA2 (gfx1030) grouped W8A8 GEMM for fused-MoE prefill, on the Triton int8
// tensors: w [E, N, K] int8 with per-channel fp32 scales [E, N], int8
// activations with per-row fp32 scales, rows routed through
// moe_align_block_size's sorted token ids (row r of expert block b reads
// activation row sorted_ids[r] / top_k and writes output row sorted_ids[r]).
//
// 256 threads (16 x 16) per workgroup; each owns RM rows x 16 columns (four
// 4-column groups spaced 64 apart): 16 * RM rows x 256 columns per workgroup.
// K is staged 64 bytes at a time, double buffered, in a k4-major LDS layout
// ([16][rows] dwords, one dword = four consecutive k = one v_dot4_i32_i8
// operand), so LDS reads are conflict-free ds_read_b128. int32 accumulation
// is exact, so results equal the Triton W8A8 kernel's up to the fp16 output
// rounding. Qwen3.6-35B-A3B's experts (E=256, top-8): 1.3x Triton at 2048
// tokens, 1.3-1.45x at 8192 (~30 TOPS).
// moe_w4a8_gemm_rdna2 is the same GEMM on the Triton WNA16 int4 tensors (w
// [E, N, K/2] uint8, low nibble = even k, fp16 group scales [E, N, K/G], zero
// points [E, N/2, K/G] or 8) for a W4A8 opt-in: each workgroup derives a
// per-channel int8 scale (max over groups of scale * max|q - z| / 127) and
// re-quantizes the nibbles to int8 on their way into LDS, as the dense W4A8
// prefill path does. 35B experts: 1.6-1.8x moe_wna16_gemm_rdna2 at 1K-8K
// tokens.
// moe_int8_skinny_rdna2 is the int8 counterpart of moe_wna16_skinny_rdna2 for
// a few routed rows per expert: one moe_align_block_size(16) block and a
// column slice per workgroup, lanes along N streaming their columns' int8
// weights from global memory in 8 x 16-byte batches (16 k each, straight
// v_dot4 operands), the block's int8 rows in LDS (K phases over 32 KB), two
// waves along K. That range is weight-bandwidth bound with int8 weights (805
// MB of 35B experts per layer): 1.25-1.34x the Triton int8 kernel at 1-8 rows
// per expert, 1.14x at 16.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1030__)
  #define __HIP__RDNA2__
#endif

namespace vllm {
namespace moe_int8_gemm_rdna2 {

static constexpr int THREADS = 256;
static constexpr int BK = 64;
static constexpr int K4 = BK / 4;
static constexpr int CH = BK / 16;  // 16-byte chunks per row per stage
static constexpr int TN = 16;
static constexpr int BN = 16 * TN;
static constexpr int SKINNY_LDS_BYTES = 32768;  // int8 activations per phase

#if defined(__HIP__RDNA2__) || !defined(__HIP_DEVICE_COMPILE__)

// UNR: unroll of the k4 loop (a full unroll hoists every LDS read and
// spills). W4: W holds int4 [E, N, K/2] with group scales GS [E, N, K/G] and
// zero points Z [E, N/2, K/G] (ZP) or 8; WS is unused.
template <int RM, int UNR, bool MUL_W, bool W4, bool ZP>
__global__ void __launch_bounds__(THREADS) moe_int8_gemm_rdna2_kernel(
    const int8_t* __restrict__ A, const float* __restrict__ AS,
    const int8_t* __restrict__ W, const float* __restrict__ WS,
    const __half* __restrict__ GS, const uint8_t* __restrict__ Z,
    __half* __restrict__ C, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, const int num_valid, const int top_k,
    const int N, const int K, const int group_size, const int E, const int lda,
    const int ldc) {
  static_assert(BN == THREADS, "one column per thread for the W4 scales");
  constexpr int BM = 16 * RM;
  constexpr int A_ITEMS = (BM * CH + THREADS - 1) / THREADS;
  constexpr int W_ITEMS = BN * CH / THREADS;
  __shared__ __align__(16) int sA[2][K4][BM];
  __shared__ __align__(16) int sW[2][K4][BN];
  // W4: per-column int8 scale and its inverse.
  __shared__ float sCh[2][W4 ? BN : 1];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int row0 = blockIdx.x * BM;
  if (row0 >= num_post_padded[0]) return;
  const int e = expert_ids[blockIdx.x];
  if (e < 0 || e >= E) return;
  const int n0 = blockIdx.y * BN;
  const int8_t* We = W + (long)e * N * (W4 ? K / 2 : K);
  const int groups = W4 ? K / group_size : 1;
  const __half* GSe = W4 ? GS + (long)e * N * groups : nullptr;
  const uint8_t* Ze = ZP ? Z + (long)e * (N / 2) * groups : nullptr;
  auto zero = [&](int col, int g) -> uint32_t {
    if constexpr (ZP)
      return (Ze[(long)(col / 2) * groups + g] >> ((col & 1) * 4)) & 0xFu;
    return 8;
  };
  if constexpr (W4) {
    const int col = min(n0 + tid, N - 1);
    float mx = 0.f;
    for (int g = 0; g < groups; g++) {
      const float z = zero(col, g);
      mx = fmaxf(
          mx, __half2float(GSe[(long)col * groups + g]) * fmaxf(z, 15.f - z));
    }
    sCh[0][tid] = mx * (1.f / 127.f);
    sCh[1][tid] = mx > 0.f ? 127.f / mx : 0.f;
    __syncthreads();
  }

  // Byte offset of each gathered activation chunk; -1 for padding rows.
  int a_off[A_ITEMS];
  #pragma unroll
  for (int it = 0; it < A_ITEMS; it++) {
    const int idx = tid + it * THREADS;
    a_off[it] = -1;
    if (idx < BM * CH) {
      const int sid = sorted_ids[row0 + idx / CH];
      if (sid < num_valid) a_off[it] = (sid / top_k) * lda + idx % CH * 16;
    }
  }

  int4 ra[A_ITEMS], rw[W4 ? 1 : W_ITEMS];
  uint2 rq[W4 ? W_ITEMS : 1];  // W4: 16 nibbles of one column
  float rs[W4 ? W_ITEMS : 1];
  uint32_t rz[W4 ? W_ITEMS : 1];
  auto load = [&](int k0) {
  #pragma unroll
    for (int it = 0; it < A_ITEMS; it++)
      ra[it] = a_off[it] >= 0
                   ? *reinterpret_cast<const int4*>(A + a_off[it] + k0)
                   : make_int4(0, 0, 0, 0);
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int col = min(n0 + idx / CH, N - 1);
      if constexpr (W4) {
        const int k = k0 + idx % CH * 16;
        rq[it] =
            *reinterpret_cast<const uint2*>(We + (long)col * (K / 2) + k / 2);
        rs[it] = __half2float(GSe[(long)col * groups + k / group_size]);
        rz[it] = zero(col, k / group_size);
      } else {
        rw[it] = *reinterpret_cast<const int4*>(We + (long)col * K + k0 +
                                                idx % CH * 16);
      }
    }
  };
  auto store = [&](int buf) {
  #pragma unroll
    for (int it = 0; it < A_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      if (idx < BM * CH) {
        const int r = idx / CH, c = idx % CH;
        sA[buf][c * 4 + 0][r] = ra[it].x;
        sA[buf][c * 4 + 1][r] = ra[it].y;
        sA[buf][c * 4 + 2][r] = ra[it].z;
        sA[buf][c * 4 + 3][r] = ra[it].w;
      }
    }
  #pragma unroll
    for (int it = 0; it < W_ITEMS; it++) {
      const int idx = tid + it * THREADS;
      const int n = idx / CH, c = idx % CH;
      if constexpr (W4) {
        // (q - z) * s / s_ch rounds to an integer v in [-127, 127] as
        // 1536 + v in fp16, whose low byte is v.
        const half2 r = __float2half2_rn(rs[it] * sCh[1][n]);
        const __half zb = __ushort_as_half(0x6400 | rz[it]);  // 1024 + z
        const half2 bias = __halves2half2(zb, zb);
        const half2 magic = __float2half2_rn(1536.f);
        const uint32_t words[2] = {rq[it].x, rq[it].y};
  #pragma unroll
        for (int h = 0; h < 2; h++) {
          uint32_t y[4];
  #pragma unroll
          for (int j = 0; j < 4; j++) {
            // Byte j holds k = 2j (low nibble) and 2j + 1.
            const uint32_t t = words[h] >> (8 * j);
            uint32_t u = (t & 0xFu) | ((t & 0xF0u) << 12) | 0x64006400u;
            const half2 v = __hfma2(
                __hsub2(*reinterpret_cast<const half2*>(&u), bias), r, magic);
            y[j] = *reinterpret_cast<const uint32_t*>(&v);
          }
          sW[buf][c * 4 + h * 2][n] =
              __builtin_amdgcn_perm(y[1], y[0], 0x06040200u);
          sW[buf][c * 4 + h * 2 + 1][n] =
              __builtin_amdgcn_perm(y[3], y[2], 0x06040200u);
        }
        continue;
      }
      sW[buf][c * 4 + 0][n] = rw[it].x;
      sW[buf][c * 4 + 1][n] = rw[it].y;
      sW[buf][c * 4 + 2][n] = rw[it].z;
      sW[buf][c * 4 + 3][n] = rw[it].w;
    }
  };

  int acc[RM][TN];
  #pragma unroll
  for (int i = 0; i < RM; i++)
  #pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0;

  const int stages = K / BK;
  load(0);
  store(0);
  __syncthreads();
  for (int s = 0; s < stages; s++) {
    const int cur = s & 1;
    if (s + 1 < stages) load((s + 1) * BK);
  #pragma unroll UNR
    for (int k4 = 0; k4 < K4; k4++) {
      int a[RM], w[TN];
  #pragma unroll
      for (int i = 0; i < RM; i += 4) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sA[cur][k4][ty * RM + i]);
        a[i] = v.x;
        a[i + 1] = v.y;
        a[i + 2] = v.z;
        a[i + 3] = v.w;
      }
  #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const int4 v =
            *reinterpret_cast<const int4*>(&sW[cur][k4][j * 16 + tx * 4]);
        w[j] = v.x;
        w[j + 1] = v.y;
        w[j + 2] = v.z;
        w[j + 3] = v.w;
      }
  #pragma unroll
      for (int i = 0; i < RM; i++)
  #pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] = __builtin_amdgcn_sdot4(a[i], w[j], acc[i][j], false);
    }
    if (s + 1 < stages) {
      store(cur ^ 1);
      __syncthreads();
    }
  }

  float ws[TN];
  #pragma unroll
  for (int j = 0; j < TN; j++) {
    const int c = j / 4 * 64 + tx * 4 + j % 4;
    ws[j] = W4 ? sCh[0][c] : WS[(long)e * N + min(n0 + c, N - 1)];
  }
  #pragma unroll
  for (int i = 0; i < RM; i++) {
    const int sid = sorted_ids[row0 + ty * RM + i];
    if (sid >= num_valid) continue;
    const float sc = AS[sid / top_k] * (MUL_W ? topk_w[sid] : 1.f);
  #pragma unroll
    for (int j = 0; j < TN; j += 2) {
      const int col = n0 + j / 4 * 64 + tx * 4 + j % 4;
      if (col < N)
        *reinterpret_cast<half2*>(C + (long)sid * ldc + col) =
            __floats2half2_rn(acc[i][j] * sc * ws[j],
                              acc[i][j + 1] * sc * ws[j + 1]);
    }
  }
}

// Skinny variant: T rows per workgroup, NW waves along K (8 / NW along N, one
// column per lane), U 16-byte batches; kp: K per activation phase.
template <int T, int NW, int U, bool MUL_W>
__global__ void __launch_bounds__(THREADS) moe_int8_skinny_rdna2_kernel(
    const int8_t* __restrict__ A, const float* __restrict__ AS,
    const int8_t* __restrict__ W, const float* __restrict__ WS,
    __half* __restrict__ C, const int* __restrict__ sorted_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, const int num_valid, const int top_k,
    const int N, const int K, const int E, const int lda, const int ldc,
    const int kp) {
  constexpr int WC = 8 / NW;
  constexpr int NC = WC * 32;
  constexpr int KB = 16 * U;
  __shared__ __align__(16) int sA[SKINNY_LDS_BYTES / 4];
  __shared__ int sR[NW > 1 ? NW - 1 : 1][T][NC];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int row0 = blockIdx.x * T;
  if (row0 >= num_post_padded[0]) return;
  const int e = expert_ids[blockIdx.x];
  if (e < 0 || e >= E) return;
  int sid[T];
  #pragma unroll
  for (int t = 0; t < T; t++) sid[t] = sorted_ids[row0 + t];
  const int wn = wave % WC, wk = wave / WC;
  const int n = blockIdx.y * NC + wn * 32 + lane;
  const int col = min(n, N - 1);
  const int8_t* Wc = W + ((long)e * N + col) * K;
  uint4 wq[U];
  auto load = [&](int k) {
  #pragma unroll
    for (int u = 0; u < U; u++)
      wq[u] = *reinterpret_cast<const uint4*>(Wc + k + 16 * u);
  };
  int acc[T];
  #pragma unroll
  for (int t = 0; t < T; t++) acc[t] = 0;
  const int ks = kp / NW, kpd = kp / 4;
  for (int p0 = 0; p0 < K; p0 += kp) {
    const int kbeg = p0 + wk * ks;
    load(kbeg);  // in flight while the phase's activations are gathered
    if (p0 > 0) __syncthreads();
    for (int i = tid; i < T * (kp / 16); i += THREADS) {
      const int t = i / (kp / 16), c = i % (kp / 16);
      int4 v = make_int4(0, 0, 0, 0);
      if (sid[t] < num_valid)
        v = *reinterpret_cast<const int4*>(A + (long)(sid[t] / top_k) * lda +
                                           p0 + c * 16);
      *reinterpret_cast<int4*>(&sA[t * kpd + c * 4]) = v;
    }
    __syncthreads();
    for (int kb = kbeg; kb < kbeg + ks; kb += KB) {
      uint4 cur[U];
  #pragma unroll
      for (int u = 0; u < U; u++) cur[u] = wq[u];
      if (kb + KB < kbeg + ks) load(kb + KB);
  #pragma unroll
      for (int u = 0; u < U; u++) {
        const int k = kb + 16 * u;
  #pragma unroll
        for (int t = 0; t < T; t++) {
          const int4 a =
              *reinterpret_cast<const int4*>(&sA[t * kpd + (k - p0) / 4]);
          int v = acc[t];
          v = __builtin_amdgcn_sdot4(a.x, (int)cur[u].x, v, false);
          v = __builtin_amdgcn_sdot4(a.y, (int)cur[u].y, v, false);
          v = __builtin_amdgcn_sdot4(a.z, (int)cur[u].z, v, false);
          v = __builtin_amdgcn_sdot4(a.w, (int)cur[u].w, v, false);
          acc[t] = v;
        }
      }
    }
  }
  if constexpr (NW > 1) {
    const int lc = wn * 32 + lane;
    if (wk > 0)
  #pragma unroll
      for (int t = 0; t < T; t++) sR[wk - 1][t][lc] = acc[t];
    __syncthreads();
    if (wk > 0) return;
  #pragma unroll
    for (int s = 0; s < NW - 1; s++)
  #pragma unroll
      for (int t = 0; t < T; t++) acc[t] += sR[s][t][lc];
  }
  if (n >= N) return;
  const float ws = WS[(long)e * N + n];
  #pragma unroll
  for (int t = 0; t < T; t++) {
    if (sid[t] >= num_valid) continue;
    const float sc = AS[sid[t] / top_k] * (MUL_W ? topk_w[sid[t]] : 1.f);
    C[(long)sid[t] * ldc + n] = __float2half(acc[t] * sc * ws);
  }
}

#else  // non-RDNA2 device pass: empty stub for symbol parity.

template <int T, int NW, int U, bool MUL_W>
__global__ void moe_int8_skinny_rdna2_kernel(
    const int8_t*, const float*, const int8_t*, const float*, __half*,
    const int*, const int*, const int*, const float*, const int, const int,
    const int, const int, const int, const int, const int, const int) {}

template <int RM, int UNR, bool MUL_W, bool W4, bool ZP>
__global__ void moe_int8_gemm_rdna2_kernel(
    const int8_t*, const float*, const int8_t*, const float*, const __half*,
    const uint8_t*, __half*, const int*, const int*, const int*, const float*,
    const int, const int, const int, const int, const int, const int, const int,
    const int) {}

#endif  // __HIP__RDNA2__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_int8_gemm_rdna2
}  // namespace vllm

namespace {

// Shared launcher; w points at int8 [E, N, K] or (W4) int4 [E, N, K/2].
template <bool W4, bool ZP>
void launch_moe_int8_gemm(torch::Tensor& output, const torch::Tensor& a,
                          const torch::Tensor& a_scale, const int8_t* w,
                          const float* w_scale, const __half* group_scales,
                          const uint8_t* zeros, const torch::Tensor& sorted_ids,
                          const torch::Tensor& expert_ids,
                          const torch::Tensor& num_tokens_post_padded,
                          const torch::Tensor& topk_weights, int64_t top_k,
                          bool mul_routed_weight, int64_t block_m, int N, int K,
                          int group_size, int E) {
  using namespace vllm::moe_int8_gemm_rdna2;
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  a_scale.numel() == a.size(0),
              "moe_int8_gemm_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows and one scale per row");
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_int8_gemm_rdna2 needs a contiguous [num_valid, N] output");
  TORCH_CHECK(block_m == 64 || block_m == 128,
              "moe_int8_gemm_rdna2 supports block_m 64 and 128");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(expert_ids.numel(), (N + BN - 1) / BN);

#define VLLM_MOE_I8_LAUNCH(RM, UNR, MW)                                       \
  moe_int8_gemm_rdna2_kernel<RM, UNR, MW, W4, ZP>                             \
      <<<grid, THREADS, 0, stream>>>(                                         \
          a.data_ptr<int8_t>(), a_scale.data_ptr<float>(), w, w_scale,        \
          group_scales, zeros, (__half*)output.data_ptr(),                    \
          sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),             \
          num_tokens_post_padded.data_ptr<int>(),                             \
          topk_weights.data_ptr<float>(), num_valid, top_k, N, K, group_size, \
          E, a.stride(0), N)
#define VLLM_MOE_I8_BY_W(RM, UNR)       \
  if (mul_routed_weight) {              \
    VLLM_MOE_I8_LAUNCH(RM, UNR, true);  \
  } else {                              \
    VLLM_MOE_I8_LAUNCH(RM, UNR, false); \
  }
  if (block_m == 64) {
    VLLM_MOE_I8_BY_W(4, 4);
  } else {
    VLLM_MOE_I8_BY_W(8, 4);
  }
#undef VLLM_MOE_I8_BY_W
#undef VLLM_MOE_I8_LAUNCH
}

void check_routing(const torch::Tensor& a_scale,
                   const torch::Tensor& sorted_ids,
                   const torch::Tensor& expert_ids,
                   const torch::Tensor& num_tokens_post_padded,
                   const torch::Tensor& topk_weights) {
  TORCH_CHECK(a_scale.dtype() == torch::kFloat32 && a_scale.is_contiguous() &&
                  topk_weights.dtype() == torch::kFloat32 &&
                  topk_weights.is_contiguous(),
              "moe_int8_gemm_rdna2 needs contiguous fp32 activation scales and "
              "top-k weights");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 &&
                  expert_ids.dtype() == torch::kInt32 &&
                  num_tokens_post_padded.dtype() == torch::kInt32,
              "moe_int8_gemm_rdna2 needs int32 alignment tensors");
}

}  // namespace

// output [num_valid, N] fp16 (num_valid = topk_weights.numel()) receives row
// sorted_ids[r] of every routed row r; a [rows, K] int8 with row scales
// a_scale [rows] fp32 is read at row sorted_ids[r] / top_k; w [E, N, K] int8
// with w_scale [E, N] (or [E, N, 1]) fp32; K % 64 == 0, N % 8 == 0.
// sorted_ids / expert_ids / num_tokens_post_padded come from
// moe_align_block_size with block_m in {64, 128}. mul_routed_weight scales
// each row by its fp32 top-k weight.
void moe_int8_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                         const torch::Tensor& a_scale, const torch::Tensor& w,
                         const torch::Tensor& w_scale,
                         const torch::Tensor& sorted_ids,
                         const torch::Tensor& expert_ids,
                         const torch::Tensor& num_tokens_post_padded,
                         const torch::Tensor& topk_weights, int64_t top_k,
                         bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_int8_gemm_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8 &&
                  output.dtype() == torch::kFloat16,
              "moe_int8_gemm_rdna2 needs int8 activations and weights and an "
              "fp16 output");
  check_routing(a_scale, sorted_ids, expert_ids, num_tokens_post_padded,
                topk_weights);
  TORCH_CHECK(w.dim() == 3 && w.is_contiguous(),
              "moe_int8_gemm_rdna2 needs contiguous [E, N, K] weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2);
  TORCH_CHECK(w_scale.dtype() == torch::kFloat32 && w_scale.is_contiguous() &&
                  w_scale.numel() == (long)E * N,
              "moe_int8_gemm_rdna2 needs one contiguous fp32 weight scale per "
              "output channel");
  TORCH_CHECK(K % BK == 0 && N % 8 == 0,
              "moe_int8_gemm_rdna2 needs K % 64 == 0 and N % 8 == 0");
  launch_moe_int8_gemm<false, false>(
      output, a, a_scale, w.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
      nullptr, nullptr, sorted_ids, expert_ids, num_tokens_post_padded,
      topk_weights, top_k, mul_routed_weight, block_m, N, K, 1, E);
}

// moe_int8_gemm_rdna2 on int4 weights w [E, N, K/2] uint8 (two values per
// byte, low nibble = even k) with fp16 group scales [E, N, K/G] (G a multiple
// of 32 dividing K) and zero points zeros [E, N/2, K/G] uint8 (two columns per
// byte, low nibble = even column) or 8 when absent; K % 64 == 0, N % 8 == 0.
void moe_w4a8_gemm_rdna2(torch::Tensor& output, const torch::Tensor& a,
                         const torch::Tensor& a_scale, const torch::Tensor& w,
                         const torch::Tensor& scales,
                         const std::optional<torch::Tensor>& zeros,
                         const torch::Tensor& sorted_ids,
                         const torch::Tensor& expert_ids,
                         const torch::Tensor& num_tokens_post_padded,
                         const torch::Tensor& topk_weights, int64_t top_k,
                         bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_int8_gemm_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kUInt8 &&
                  scales.dtype() == torch::kFloat16 &&
                  output.dtype() == torch::kFloat16,
              "moe_w4a8_gemm_rdna2 needs int8 activations, uint8 int4 weights, "
              "fp16 scales and an fp16 output");
  check_routing(a_scale, sorted_ids, expert_ids, num_tokens_post_padded,
                topk_weights);
  TORCH_CHECK(w.dim() == 3 && w.is_contiguous() && scales.dim() == 3 &&
                  scales.is_contiguous(),
              "moe_w4a8_gemm_rdna2 needs contiguous [E, N, K/2] weights and "
              "[E, N, K/G] scales");
  const int E = w.size(0), N = w.size(1), K = w.size(2) * 2;
  const int groups = scales.size(2);
  TORCH_CHECK(scales.size(0) == E && scales.size(1) == N && groups > 0 &&
                  K % groups == 0 && (K / groups) % 32 == 0,
              "moe_w4a8_gemm_rdna2 needs group scales [E, N, K/G] with G a "
              "multiple of 32");
  TORCH_CHECK(K % BK == 0 && N % 8 == 0,
              "moe_w4a8_gemm_rdna2 needs K % 64 == 0 and N % 8 == 0");
  if (zeros) {
    TORCH_CHECK(zeros->dtype() == torch::kUInt8 && zeros->is_contiguous() &&
                    zeros->numel() == (long)E * (N / 2) * groups,
                "moe_w4a8_gemm_rdna2 needs contiguous [E, N/2, K/G] uint8 "
                "zero points");
    launch_moe_int8_gemm<true, true>(
        output, a, a_scale, reinterpret_cast<const int8_t*>(w.data_ptr()),
        nullptr, (const __half*)scales.data_ptr(), zeros->data_ptr<uint8_t>(),
        sorted_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k,
        mul_routed_weight, block_m, N, K, K / groups, E);
  } else {
    launch_moe_int8_gemm<true, false>(
        output, a, a_scale, reinterpret_cast<const int8_t*>(w.data_ptr()),
        nullptr, (const __half*)scales.data_ptr(), nullptr, sorted_ids,
        expert_ids, num_tokens_post_padded, topk_weights, top_k,
        mul_routed_weight, block_m, N, K, K / groups, E);
  }
}

// moe_int8_gemm_rdna2 for a few routed rows per expert (block_m 16 alignment):
// same operands and layout; K % 16 == 0, any N.
void moe_int8_skinny_rdna2(torch::Tensor& output, const torch::Tensor& a,
                           const torch::Tensor& a_scale, const torch::Tensor& w,
                           const torch::Tensor& w_scale,
                           const torch::Tensor& sorted_ids,
                           const torch::Tensor& expert_ids,
                           const torch::Tensor& num_tokens_post_padded,
                           const torch::Tensor& topk_weights, int64_t top_k,
                           bool mul_routed_weight, int64_t block_m) {
  using namespace vllm::moe_int8_gemm_rdna2;
  TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8 &&
                  output.dtype() == torch::kFloat16,
              "moe_int8_skinny_rdna2 needs int8 activations and weights and an "
              "fp16 output");
  check_routing(a_scale, sorted_ids, expert_ids, num_tokens_post_padded,
                topk_weights);
  TORCH_CHECK(w.dim() == 3 && w.is_contiguous(),
              "moe_int8_skinny_rdna2 needs contiguous [E, N, K] weights");
  const int E = w.size(0), N = w.size(1), K = w.size(2);
  TORCH_CHECK(
      w_scale.dtype() == torch::kFloat32 && w_scale.is_contiguous() &&
          w_scale.numel() == (long)E * N,
      "moe_int8_skinny_rdna2 needs one contiguous fp32 weight scale per "
      "output channel");
  TORCH_CHECK(K % 16 == 0, "moe_int8_skinny_rdna2 needs K % 16 == 0");
  TORCH_CHECK(a.dim() == 2 && a.size(1) == K && a.stride(1) == 1 &&
                  a.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 &&
                  a_scale.numel() == a.size(0),
              "moe_int8_skinny_rdna2 needs a [rows, K] with 16-byte aligned, "
              "K-contiguous rows and one scale per row");
  const int num_valid = topk_weights.numel();
  TORCH_CHECK(output.is_contiguous() && output.numel() == (long)num_valid * N,
              "moe_int8_skinny_rdna2 needs a contiguous [num_valid, N] output");
  TORCH_CHECK(block_m == 16, "moe_int8_skinny_rdna2 supports block_m 16");
  constexpr int T = 16;
  // Largest K per phase that divides K, is a multiple of the waves' batch
  // span and keeps the T x kp int8 tile within the LDS budget; 0 if none.
  auto phase_k = [&](int step) {
    for (int kp = K - K % step; kp >= step; kp -= step)
      if (K % kp == 0 && T * kp <= SKINNY_LDS_BYTES) return kp;
    return 0;
  };

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define VLLM_I8_SKINNY_LAUNCH(NW, U, MW)                                       \
  {                                                                            \
    constexpr int nc = (8 / NW) * 32;                                          \
    const dim3 grid(expert_ids.numel(), (N + nc - 1) / nc);                    \
    moe_int8_skinny_rdna2_kernel<T, NW, U, MW><<<grid, THREADS, 0, stream>>>(  \
        a.data_ptr<int8_t>(), a_scale.data_ptr<float>(), w.data_ptr<int8_t>(), \
        w_scale.data_ptr<float>(), (__half*)output.data_ptr(),                 \
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),                \
        num_tokens_post_padded.data_ptr<int>(),                                \
        topk_weights.data_ptr<float>(), num_valid, top_k, N, K, E,             \
        a.stride(0), N, kp);                                                   \
  }
  // Two waves along K, 128-byte batches; single-step batches when K is not a
  // multiple of their span.
  int kp = phase_k(2 * 16 * 8);
  if (kp) {
    if (mul_routed_weight) {
      VLLM_I8_SKINNY_LAUNCH(2, 8, true)
    } else {
      VLLM_I8_SKINNY_LAUNCH(2, 8, false)
    }
  } else {
    kp = phase_k(16);
    if (mul_routed_weight) {
      VLLM_I8_SKINNY_LAUNCH(1, 1, true)
    } else {
      VLLM_I8_SKINNY_LAUNCH(1, 1, false)
    }
  }
#undef VLLM_I8_SKINNY_LAUNCH
}
