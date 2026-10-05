// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// fp16/bf16 GEMMs through rocBLAS with an explicit Tensile solution for RDNA2
// (gfx1030), where rocBLAS's default choice is far from the best for some
// shapes (e.g. 3 instead of 17 TFLOPS for x [64, 5120] @ W [34816, 5120]^T).
// The solutions are picked by benchmarking (see rdna2_gemm.py); an index that
// cannot solve the problem falls back to rocBLAS's own choice.

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

namespace {

struct Problem {
  rocblas_handle handle;
  rocblas_operation op_w;
  int m, n, k, ldw, lda, ldc;
  const void* w;
  const void* a;
  void* c;
  rocblas_datatype type;
};

// out [M, N] = a [M, K] @ w^T for w [N, K] (w_kn false, as F.linear) or
// a @ w for w [K, N]. rocBLAS is column-major: out^T [N, M] = op(w) a^T.
Problem make_problem(const at::Tensor& a, const at::Tensor& w, bool w_kn,
                     at::Tensor& out) {
  auto handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  rocblas_set_stream(handle, at::cuda::getCurrentCUDAStream());
  const int M = a.size(0), K = a.size(1);
  const int N = w_kn ? w.size(1) : w.size(0);
  return Problem{handle,
                 w_kn ? rocblas_operation_none : rocblas_operation_transpose,
                 N,
                 M,
                 K,
                 (int)w.stride(0),
                 (int)a.stride(0),
                 (int)out.stride(0),
                 w.data_ptr(),
                 a.data_ptr(),
                 out.data_ptr(),
                 a.dtype() == torch::kFloat16 ? rocblas_datatype_f16_r
                                              : rocblas_datatype_bf16_r};
}

rocblas_status run(const Problem& p, int32_t solution, uint32_t flags) {
  const float alpha = 1.f, beta = 0.f;
  return rocblas_gemm_ex(
      p.handle, p.op_w, rocblas_operation_none, p.m, p.n, p.k, &alpha, p.w,
      p.type, p.ldw, p.a, p.type, p.lda, &beta, p.c, p.type, p.ldc, p.c, p.type,
      p.ldc, rocblas_datatype_f32_r,
      solution ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard,
      solution, flags);
}

void check_operands(const at::Tensor& a, const at::Tensor& w, bool w_kn) {
  TORCH_CHECK(a.dtype() == w.dtype() && (a.dtype() == torch::kFloat16 ||
                                         a.dtype() == torch::kBFloat16),
              "gemm_rocblas_rdna2 needs fp16 or bf16 a and w of one dtype");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.stride(1) == 1 &&
                  w.stride(1) == 1 && a.size(1) == w.size(w_kn ? 0 : 1),
              "gemm_rocblas_rdna2 needs row-major a [M, K] and w [N, K] (or "
              "[K, N] with w_kn)");
}

}  // namespace

// a [M, K] @ w^T (w [N, K]) or a @ w (w [K, N], w_kn) -> [M, N], fp32
// accumulation, with rocBLAS solution `solution` (0: rocBLAS's choice).
torch::Tensor gemm_rocblas_rdna2(const at::Tensor& a, const at::Tensor& w,
                                 int64_t solution, bool w_kn) {
  check_operands(a, w, w_kn);
  auto out = torch::empty({a.size(0), w.size(w_kn ? 1 : 0)}, a.options());
  if (out.numel() == 0) return out;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const Problem p = make_problem(a, w, w_kn, out);
  if (solution == 0 ||
      run(p, solution, rocblas_gemm_flags_check_solution_index) !=
          rocblas_status_success ||
      run(p, solution, rocblas_gemm_flags_none) != rocblas_status_success) {
    TORCH_CHECK(run(p, 0, rocblas_gemm_flags_none) == rocblas_status_success,
                "gemm_rocblas_rdna2: rocblas_gemm_ex failed");
  }
  return out;
}

// The rocBLAS solution indices that can compute gemm_rocblas_rdna2(a, w).
std::vector<int64_t> gemm_rocblas_solutions_rdna2(const at::Tensor& a,
                                                  const at::Tensor& w,
                                                  bool w_kn) {
  check_operands(a, w, w_kn);
  auto out = torch::empty({a.size(0), w.size(w_kn ? 1 : 0)}, a.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const Problem p = make_problem(a, w, w_kn, out);
  const float alpha = 1.f, beta = 0.f;
  auto list = [&](rocblas_int* ids, rocblas_int* size) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // beta API
    return rocblas_gemm_ex_get_solutions(
        p.handle, p.op_w, rocblas_operation_none, p.m, p.n, p.k, &alpha, p.w,
        p.type, p.ldw, p.a, p.type, p.lda, &beta, p.c, p.type, p.ldc, p.c,
        p.type, p.ldc, rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index,
        rocblas_gemm_flags_none, ids, size);
#pragma clang diagnostic pop
  };
  rocblas_int size = 0;
  TORCH_CHECK(list(nullptr, &size) == rocblas_status_success,
              "rocblas_gemm_ex_get_solutions failed");
  std::vector<rocblas_int> ids(size);
  TORCH_CHECK(list(ids.data(), &size) == rocblas_status_success,
              "rocblas_gemm_ex_get_solutions failed");
  return std::vector<int64_t>(ids.begin(), ids.begin() + size);
}

// The rocBLAS version string; solution indices are only valid for one build.
std::string rocblas_version_rdna2() {
  size_t size = 0;
  TORCH_CHECK(rocblas_get_version_string_size(&size) == rocblas_status_success,
              "rocblas_get_version_string_size failed");
  std::string version(size, '\0');
  TORCH_CHECK(rocblas_get_version_string(version.data(), size) ==
                  rocblas_status_success,
              "rocblas_get_version_string failed");
  version.resize(strlen(version.c_str()));
  return version;
}
