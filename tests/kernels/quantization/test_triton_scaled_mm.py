# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the triton_scaled_mm kernel.

Run `pytest tests/kernels/quantization/test_triton_scaled_mm.py`.
"""

import importlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

device = current_platform.device_type

triton_scaled_mm_module = importlib.import_module(
    "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
)
triton_scaled_mm = triton_scaled_mm_module.triton_scaled_mm


def torch_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def get_8bit_types():
    types = [torch.int8]
    if current_platform.supports_fp8():
        types.append(current_platform.fp8_dtype())
    return types


# This test is to check regressions for int8 support on ROCm.
@pytest.mark.parametrize(
    "model_path",
    [
        "neuralmagic/Llama-3.2-1B-quantized.w8a8",
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [10])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="Should only run on ROCm")
def test_rocm_compressed_tensors_w8a8(
    vllm_runner, example_prompts, model_path, max_tokens, num_logprobs
):
    dtype = "bfloat16"

    with vllm_runner(model_path, dtype=dtype) as vllm_model:
        vllm_model.generate_greedy_logprobs(example_prompts, max_tokens, num_logprobs)


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


@pytest.mark.parametrize("M,N,K", MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm(
    M, N, K, in_dtype, out_dtype, use_scalar_scale_a, use_scalar_scale_b, use_bias
):
    is_floating_point_type = lambda t: torch.tensor([1, 1], dtype=t).is_floating_point()

    set_random_seed(0)

    # NOTE: There are cases, where if the matrix is large enough, an output
    # like 65504.4 can be produced, and can easily turn into inf when
    # multiplied when using float16/bfloat16.  This means one function, e.g.,
    # testing function, and another function, e.g. golden function, can
    # produce a non-inf value while the other produces an inf value, and
    # will cause assert_close/allclose to fail, even though if overflow
    # wouldn't have occurred, the values would have been "close."
    #
    # So, the values here are kept small enough to avoid this situation.
    if is_floating_point_type(in_dtype):
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=device)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=device)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=device)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=device)

    if use_scalar_scale_a:
        scale_a = torch.rand((1, 1), device=device)
    else:
        scale_a = 0.25 * torch.rand((M, 1), device=device)

    if use_scalar_scale_b:
        scale_b = torch.rand((1, 1), device=device)
    else:
        scale_b = 0.25 * torch.rand((N, 1), device=device)

    bias = None
    if use_bias:
        bias = torch.rand((N,), device=device, dtype=out_dtype)

    c_check = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    c_actual = torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)


# TD operand loads must be bit-exact vs the plain masked-load path.
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Triton scaled_mm runs on CUDA-alike or XPU.",
)
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]
)
@pytest.mark.parametrize("in_dtype", get_8bit_types())
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_mm_td_matches_plain(M, N, K, in_dtype, use_scalar_scale_a, use_bias):
    dev = current_platform.device_type
    out_dtype = torch.bfloat16
    set_random_seed(0)

    is_fp = torch.tensor([1, 1], dtype=in_dtype).is_floating_point()
    if is_fp:
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device=dev)).to(in_dtype)
        b = (0.25 * torch.rand((K, N), dtype=torch.float32, device=dev)).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device=dev)
        b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device=dev)

    scale_a = (
        torch.rand((1, 1), device=dev)
        if use_scalar_scale_a
        else 0.25 * torch.rand((M, 1), device=dev)
    )
    scale_b = 0.25 * torch.rand((N, 1), device=dev)
    bias = torch.rand((N,), device=dev, dtype=out_dtype) if use_bias else None

    out_plain = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=False)
    out_td = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias, use_td=True)
    torch.testing.assert_close(out_td, out_plain, rtol=0, atol=0)


# Explicit tiles must work, and int8 must match the heuristic tiles bit-exactly
# (int32 accumulation is exact), whichever tile the heuristic picks.
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton scaled_mm on CUDA-alike."
)
@pytest.mark.parametrize("M", [1, 16, 64])
def test_scaled_mm_explicit_tiles_match_heuristic(M):
    dev = current_platform.device_type
    set_random_seed(0)
    N, K = 4096, 4096
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=dev)
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device=dev)
    scale_a = 0.25 * torch.rand((M, 1), device=dev)
    scale_b = 0.25 * torch.rand((N, 1), device=dev)

    heuristic = triton_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16)
    explicit = triton_scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        torch.bfloat16,
        block_size_m=32,
        block_size_n=64,
        block_size_k=64,
        use_heuristic=False,
    )
    torch.testing.assert_close(explicit, heuristic, rtol=0, atol=0)


# The int8 linear kernel calls triton_scaled_mm through this op so that
# torch.compile cannot bake in the tile picked for the traced M.
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton scaled_mm on CUDA-alike."
)
def test_triton_int8_scaled_mm_opcheck():
    from tests.kernels.utils import opcheck
    from vllm.model_executor.kernels.linear.scaled_mm import triton  # noqa: F401

    dev = current_platform.device_type
    M, N, K = 16, 256, 512
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device=dev)
    b = torch.randint(-32, 32, (N, K), dtype=torch.int8, device=dev).t()
    scale_a = torch.rand((M, 1), device=dev)
    scale_b = torch.rand((N, 1), device=dev)
    opcheck(
        torch.ops.vllm.triton_int8_scaled_mm,
        (a, b, scale_a, scale_b, torch.float16, None),
    )


def _rdna2_w8a8_ops_available() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, "w8a8_gemm_rdna2")


# The gfx1030 decode GEMV and prefill GEMM must reproduce triton_scaled_mm bit
# for bit: all accumulate exactly in int32 and scale in fp32 once.
@pytest.mark.skipif(not _rdna2_w8a8_ops_available(), reason="gfx1030 only")
@pytest.mark.parametrize(
    "op,m,n,k",
    [
        # GEMV: N not a multiple of the 16 rows per workgroup
        ("w8a8_gemv_rdna2", 1, 1000, 4096),
        ("w8a8_gemv_rdna2", 3, 1000, 4096),
        ("w8a8_gemv_rdna2", 8, 1000, 4096),
        # GEMV: M x K overflows LDS at M = 8, so A is staged in chunks
        ("w8a8_gemv_rdna2", 1, 512, 17408),
        ("w8a8_gemv_rdna2", 8, 512, 17408),
        # GEMV above 8 tokens: M padded to a multiple of 4, 4 rows per wave
        ("w8a8_gemv_rdna2", 9, 1000, 4096),
        ("w8a8_gemv_rdna2", 24, 512, 17408),
        # GEMM: partial row and column tiles
        ("w8a8_gemm_rdna2", 17, 1000, 320),
        ("w8a8_gemm_rdna2", 300, 4096, 5120),
        # GEMM: enough tiles for the 128 x 256 tile
        ("w8a8_gemm_rdna2", 2048, 17408, 5120),
    ],
)
@pytest.mark.parametrize("per_tensor_scales", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_rdna2_w8a8_matches_triton(op, m, n, k, per_tensor_scales, out_dtype):
    from vllm import _custom_ops as ops

    dev = current_platform.device_type
    set_random_seed(0)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=dev)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev)
    scale_a = 0.01 * torch.rand((1 if per_tensor_scales else m, 1), device=dev)
    scale_b = 0.01 * torch.rand((1 if per_tensor_scales else n, 1), device=dev)
    bias = torch.randn(n, device=dev, dtype=out_dtype)

    ref = triton_scaled_mm(a, w.t(), scale_a, scale_b, out_dtype, bias)
    out = getattr(ops, op)(a, w, scale_a, scale_b, out_dtype, bias)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
