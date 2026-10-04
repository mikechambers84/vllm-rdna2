# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in int8 weight-only storage of unquantized linear layers on gfx1030
(VLLM_ROCM_W8A16_UNQUANTIZED)."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("gfx1030 only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1030  # noqa: E402


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 8, 24, 64])
def test_unquantized_linear_stores_int8_weights(monkeypatch, num_tokens):
    """The fp16 weight is replaced by int8 weights with per-channel scales;
    both the decode GEMV (<= 8 tokens) and the dequantize + GEMM path match
    the dequantized weight up to fp16 rounding, and the weight-only
    quantization error stays small."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setenv("VLLM_ROCM_W8A16_UNQUANTIZED", "1")
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randn(2048, 1024, dtype=torch.float16, device="cuda") * 0.02
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    method = UnquantizedLinearMethod()

    method.process_weights_after_loading(layer)

    assert layer.w8a16_weight.dtype == torch.int8 and layer.weight.numel() == 0
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    out = method.apply(layer, x)
    dequant = layer.w8a16_weight.float() * layer.w8a16_scale[:, None]
    ref = torch.nn.functional.linear(x.float(), dequant)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
    exact = torch.nn.functional.linear(x.float(), weight.float())
    assert ((out.float() - exact).norm() / exact.norm()).item() < 1e-2
