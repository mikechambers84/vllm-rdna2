# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_cuda():
    pytest.skip(
        "ROCm skinny GEMM tests are not supported on CUDA.",
        allow_module_level=True,
    )

from vllm.model_executor.layers import utils
from vllm.platforms.rocm import on_gfx10 as _on_gfx10


def _on_gfx11() -> bool:
    """gfx11 check that is safe to evaluate at collection time on any platform."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx11

    return on_gfx11()


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_wvsplitk_path(monkeypatch, m):
    x = torch.randn(1, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "n, m, k, expected",
    [
        (1, 248320, 5120, "gemv"),  # lm_head
        (8, 96, 5120, "gemv"),  # tiny projection, rocBLAS is launch-bound
        (1, 1024, 2048, "gemv"),  # mid-size weight at narrow K
        (6, 4096, 14336, "wvsplitk"),  # long rows at 5-8 tokens
        (16, 248320, 5120, "rocblas"),  # too many tokens for a GEMV
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_gfx1030_rdna2_gemv_routing(
    monkeypatch, n, m, k, expected, dtype
):
    x = torch.randn(n, k, dtype=dtype)
    weight = torch.empty(m, k, dtype=dtype)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    mocks = {
        "gemv": MagicMock(return_value=torch.empty(n, m, dtype=dtype)),
        "wvsplitk": MagicMock(return_value=torch.empty(n, m, dtype=dtype)),
        "rocblas": MagicMock(return_value=torch.empty(n, m, dtype=dtype)),
    }
    monkeypatch.setattr(utils.ops, "gemv_rdna2", mocks["gemv"])
    monkeypatch.setattr(utils.ops, "wvSplitK_rdna2", mocks["wvsplitk"])
    monkeypatch.setattr(torch.nn.functional, "linear", mocks["rocblas"])

    utils.rocm_unquantized_gemm_impl(x, weight, None)

    assert {name for name, mock in mocks.items() if mock.called} == {expected}


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", range(1, 6))
def test_rocm_unquantized_gemm_single_output_real_kernel(dtype, n):
    torch.manual_seed(0)
    x = torch.randn(n, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(1, 2048, device="cuda", dtype=dtype)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_rocm_unquantized_gemm_makes_skinny_activation_contiguous(monkeypatch):
    x = torch.randn(64, 4, dtype=torch.float16).t()
    weight = torch.randn(128, 64, dtype=torch.float16)
    assert x.shape == (4, 64)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    x_view = wvsplitk_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_makes_llmm1_activation_contiguous(monkeypatch):
    x = torch.randn(1, 128, dtype=torch.float16)[:, ::2]
    weight = torch.randn(4, 64, dtype=torch.float16)
    assert x.shape == (1, 64)
    assert x.stride() == (128, 2)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    llmm1_mock.assert_called_once()
    x_view = llmm1_mock.call_args.args[1]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_rejects_unsupported_skinny_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(4, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def _row_stride_padded(weight: torch.Tensor, pad: int) -> torch.Tensor:
    """A weight whose row stride is offset by `pad` elements, as produced by
    `maybe_pad_weight_avoid_cache_aliasing`."""
    padded = torch.nn.functional.pad(weight, (0, pad))[..., :-pad]
    assert padded.shape == weight.shape
    assert padded.stride() == (weight.shape[1] + pad, 1)
    return padded


def test_rocm_unquantized_gemm_wvsplitk_accepts_row_stride_padded_weight(monkeypatch):
    # wvSplitK reads the weight row stride, so a row-stride-padded weight must
    # still reach it instead of falling back to torch.nn.functional.linear.
    x = torch.randn(4, 1024, dtype=torch.float16)
    weight = _row_stride_padded(torch.randn(128, 1024, dtype=torch.float16), 64)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    assert wvsplitk_mock.call_args.args[0].stride() == weight.stride()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "k,element_size,expected",
    [
        (4096, 2, 64),  # 8192 B row stride, aliases
        (12288, 2, 64),  # 24576 B row stride, aliases
        (1024, 2, 64),  # smallest aliasing bf16/fp16 K
        (2560, 2, 0),  # 5120 B, no aliasing
        (11008, 2, 0),  # 22016 B, no aliasing
        (2048, 1, 128),  # fp8: 2048 B, aliases, twice the pad
        (512, 4, 32),  # fp32: 2048 B, aliases, half the pad
    ],
)
def test_cache_aliasing_pad_elems(k, element_size, expected):
    assert utils.cache_aliasing_pad_elems(k, element_size) == expected


def test_maybe_pad_weight_avoid_cache_aliasing_skips_non_gfx11(monkeypatch):
    # Padding is gfx11-only, even for a K whose row stride would alias there.
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: False)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    assert utils.maybe_pad_weight_avoid_cache_aliasing(weight) is weight


def test_maybe_pad_weight_avoid_cache_aliasing(monkeypatch):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)

    weight = torch.randn(128, 4096, dtype=torch.bfloat16)
    padded = utils.maybe_pad_weight_avoid_cache_aliasing(weight)

    assert padded.shape == weight.shape
    assert padded.stride() == (4096 + 64, 1)
    assert torch.equal(padded, weight)
    # Idempotent: the padded stride no longer aliases, so a second call is a no-op
    # and preserves the pointer captured by any graph.
    repadded = utils.maybe_pad_weight_avoid_cache_aliasing(padded)
    assert repadded.data_ptr() == padded.data_ptr()


@pytest.mark.parametrize(
    "weight",
    [
        torch.randn(128, 2560, dtype=torch.bfloat16),  # no aliasing
        torch.randn(2560, 128, dtype=torch.bfloat16).t(),  # not contiguous
        torch.randn(2, 128, 4096, dtype=torch.bfloat16),  # not 2D
    ],
)
def test_maybe_pad_weight_avoid_cache_aliasing_skips(monkeypatch, weight):
    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: True)

    assert (
        utils.maybe_pad_weight_avoid_cache_aliasing(weight).data_ptr()
        == weight.data_ptr()
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.skipif(
    current_platform.is_rocm() and _on_gfx10(), reason="wvSplitK not used on gfx10"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel(monkeypatch, dtype):
    x = torch.randn(64, 4, device="cuda", dtype=dtype).t()
    weight = torch.randn(128, 64, device="cuda", dtype=dtype)
    assert x.stride() == (1, 4)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not _on_gfx11(), reason="cache-aliasing padding only applies on gfx11"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_unquantized_gemm_padded_weight_real_kernel(monkeypatch, dtype):
    x = torch.randn(4, 4096, device="cuda", dtype=dtype)
    weight = _row_stride_padded(torch.randn(128, 4096, device="cuda", dtype=dtype), 64)

    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    original_wvsplitk = utils.ops.wvSplitK
    wvsplitk_mock = MagicMock(side_effect=original_wvsplitk)
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_called_once()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def _check_rocm_unquantized_gemm_falls_back(monkeypatch, is_rdna4, n, m):
    x = torch.randn(n, 64, dtype=torch.float16)
    weight = torch.randn(m, 64, dtype=torch.float16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna4", lambda: is_rdna4)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock(side_effect=lambda w, x_view, _: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)
    swmmac_mock = MagicMock(
        side_effect=lambda x_view, w, _m, _cu_count, _bias: x_view @ w.t()
    )
    monkeypatch.setattr(utils.ops, "swmmac_gemm", swmmac_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    swmmac_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx1x_n_gt_5_falls_back(monkeypatch, m):
    # wvSplitK skinny GEMM handles n in [1, 5] (see PR #40687); n > 5 must
    # fall back to torch.nn.functional.linear.
    _check_rocm_unquantized_gemm_falls_back(monkeypatch, is_rdna4=False, n=6, m=m)


@pytest.mark.parametrize("m", [1, 128])
def test_rocm_unquantized_gemm_gfx120x_n_gt_8_falls_back(monkeypatch, m):
    # SWMMAC skinny GEMM handles n in [5, 8] on gfx120x (see PR #45559);
    # n > 8 must fall back to torch.nn.functional.linear.
    _check_rocm_unquantized_gemm_falls_back(monkeypatch, is_rdna4=True, n=9, m=m)


@pytest.mark.parametrize("noncontiguous_operand", ["weight", "bias"])
def test_rocm_unquantized_gemm_gfx120x_swmmac_rejects_unsupported_layouts(
    monkeypatch, noncontiguous_operand
):
    x = torch.randn(6, 64, dtype=torch.float16)
    weight = torch.randn(128, 64, dtype=torch.float16)
    bias = torch.randn(128, dtype=torch.float16)
    if noncontiguous_operand == "weight":
        weight = torch.randn(64, 128, dtype=torch.float16).t()
        assert not weight.is_contiguous()
    else:
        bias = torch.randn(256, dtype=torch.float16)[::2]
        assert not bias.is_contiguous()

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.rocm_aiter_ops, "is_tgemm_enabled", lambda: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna4", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx11", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: False)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    swmmac_mock = MagicMock(
        side_effect=lambda x_view, w, _m, _cu_count, _bias: x_view @ w.t()
    )
    monkeypatch.setattr(utils.ops, "swmmac_gemm", swmmac_mock)
    wvsplitk_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)
    llmm1_mock = MagicMock()
    monkeypatch.setattr(utils.ops, "LLMM1", llmm1_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, bias)
    ref = torch.nn.functional.linear(x, weight, bias)

    swmmac_mock.assert_not_called()
    wvsplitk_mock.assert_not_called()
    llmm1_mock.assert_not_called()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_rocm_unquantized_gemm_gfx950_wvsplitkrc_path(monkeypatch):
    x = torch.randn(1024, 16, dtype=torch.float16).t()
    weight = torch.randn(256, 1024, dtype=torch.float16)
    assert x.stride() == (1, 16)

    monkeypatch.setattr(utils, "use_aiter_triton_gemm", lambda *args: False)
    monkeypatch.setattr(utils.envs, "VLLM_ROCM_USE_SKINNY_GEMM", True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1x", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx9", lambda: False)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1250", lambda: True)
    monkeypatch.setattr(utils, "num_compute_units", lambda: 120)

    wvsplitkrc_mock = MagicMock(side_effect=lambda x_view, w, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitKrc", wvsplitkrc_mock)
    wvsplitk_mock = MagicMock(side_effect=lambda w, x_view, _, __: x_view @ w.t())
    monkeypatch.setattr(utils.ops, "wvSplitK", wvsplitk_mock)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)
    ref = torch.nn.functional.linear(x, weight, None)

    wvsplitkrc_mock.assert_called_once()
    wvsplitk_mock.assert_not_called()
    x_view = wvsplitkrc_mock.call_args.args[0]
    assert x_view.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
def test_rocm_unquantized_gemm_gfx1030_bf16_runs_in_fp16(monkeypatch):
    """gfx1030 has no bf16 dot instruction, so bf16 GEMMs past the GEMV row
    counts run rocBLAS on fp16 copies of the operands and return bf16."""
    from vllm.model_executor.kernels.linear import rdna2_gemm

    if not rdna2_gemm.available():
        pytest.skip("gfx1030 with gemm_rocblas_rdna2")
    monkeypatch.setattr(rdna2_gemm, "_table", lambda: {})
    linear = torch.nn.functional.linear
    dtypes = []

    def spy(x, w, b=None):
        dtypes.append((x.dtype, w.dtype))
        return linear(x, w, b)

    monkeypatch.setattr(torch.nn.functional, "linear", spy)
    torch.manual_seed(0)
    x = torch.randn(100, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3072, 2048, device="cuda", dtype=torch.bfloat16) * 0.02

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)

    assert out.dtype == torch.bfloat16
    assert dtypes == [(torch.float16, torch.float16)]
    ref = linear(x.float(), weight.float())
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only kernel test")
@pytest.mark.parametrize("valid", [True, False])
def test_rocm_unquantized_gemm_gfx1030_tuned_rocblas_solution(monkeypatch, valid):
    """On gfx1030 a weight shape in the tuned table runs rocBLAS with the
    table's solution for the row bucket; an index rocBLAS cannot use falls
    back to its own choice. Either way the result is F.linear's."""
    from vllm.model_executor.kernels.linear import rdna2_gemm

    if not rdna2_gemm.available():
        pytest.skip("gfx1030 with gemm_rocblas_rdna2")
    torch.manual_seed(0)
    x = torch.randn(100, 2048, device="cuda", dtype=torch.float16)
    weight = torch.randn(3072, 2048, device="cuda", dtype=torch.float16) * 0.02
    sols = torch.ops._rocm_C.gemm_rocblas_solutions_rdna2(x, weight, False)
    sol = sols[len(sols) // 2] if valid else 12345
    table = {rdna2_gemm._key(3072, 2048, torch.float16, False): [0, 0, 0, sol, 0, 0]}
    monkeypatch.setattr(rdna2_gemm, "_table", lambda: table)

    out = utils.rocm_unquantized_gemm_impl(x, weight, None)

    assert rdna2_gemm.solution(3072, 2048, 100, torch.float16) == sol
    ref = torch.nn.functional.linear(x.float(), weight.float())
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
