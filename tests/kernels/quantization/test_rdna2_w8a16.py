# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-bit weight-only linear layers on gfx1030: opt-in int8 storage of
unquantized layers (VLLM_ROCM_W8A16_UNQUANTIZED, VLLM_ROCM_W8A16_LM_HEAD) and
FP8 checkpoints (no FP8 instructions on gfx1030)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("gfx1030 only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1030  # noqa: E402


def _from_kmajor(w: torch.Tensor) -> torch.Tensor:
    """K-major 8-bit weights [K / 4, N, 4] back to [N, K]."""
    return w.transpose(0, 1).reshape(w.shape[1], -1)


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 8, 24, 64])
def test_unquantized_linear_stores_int8_weights(monkeypatch, num_tokens):
    """The fp16 weight is replaced by K-major int8 weights with per-channel
    scales; decode and larger batches match the dequantized weight up to fp16
    rounding, and the weight-only quantization error stays small."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setenv("VLLM_ROCM_W8A16_UNQUANTIZED", "1")
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randn(2048, 1024, dtype=torch.float16, device="cuda") * 0.02
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    method = UnquantizedLinearMethod()

    method.process_weights_after_loading(layer)

    assert layer.w8a16_weight.dtype == torch.int8 and layer.weight.numel() == 0
    assert layer.w8a16_weight.shape == (256, 2048, 4)
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    out = method.apply(layer, x)
    dequant = _from_kmajor(layer.w8a16_weight).float() * layer.w8a16_scale[:, None]
    ref = torch.nn.functional.linear(x.float(), dequant)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
    exact = torch.nn.functional.linear(x.float(), weight.float())
    assert ((out.float() - exact).norm() / exact.norm()).item() < 1e-2


def _lm_head(vocab: int, hidden: int):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    with torch.device("cuda"):
        head = ParallelLMHead(
            vocab, hidden, params_dtype=torch.float16, disable_tp=True
        )
    head.weight.data.copy_(torch.randn(head.weight.shape, device="cuda") * 0.02)
    return head


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 8, 40])
def test_lm_head_stores_int8_weights(monkeypatch, num_tokens):
    """VLLM_ROCM_W8A16_LM_HEAD: the untied lm_head goes int8; logits match the
    dequantized weight, also for a vocabulary prefix (MTP draft logits) that
    does not end on a whole dword of 4 channels."""
    from vllm.model_executor.kernels.linear import rdna2_w8a16

    monkeypatch.setenv("VLLM_ROCM_W8A16_LM_HEAD", "1")
    torch.manual_seed(0)
    head = _lm_head(8192, 1024)
    weight = head.weight.data.clone()

    head.quant_method.process_weights_after_loading(head)

    assert head.w8a16_weight.dtype == torch.int8 and head.weight.numel() == 0
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    dequant = _from_kmajor(head.w8a16_weight).float() * head.w8a16_scale[:, None]
    for rows in (None, 3001):
        if rows is None:
            out = head.quant_method.apply(head, x)
        else:
            out = rdna2_w8a16.apply_rdna2_w8a16(head, x, None, rows)
        ref = torch.nn.functional.linear(x.float(), dequant[:rows])
        assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
        exact = torch.nn.functional.linear(x.float(), weight[:rows].float())
        assert ((out.float() - exact).norm() / exact.norm()).item() < 1e-2


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
def test_tied_lm_head_stays_fp16(monkeypatch):
    """A tied lm_head shares the input embedding, which must stay fp16."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    monkeypatch.setenv("VLLM_ROCM_W8A16_LM_HEAD", "1")
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(
            4096, 512, params_dtype=torch.float16, disable_tp=True
        )
    head = _lm_head(4096, 512).tie_weights(embed)

    head.quant_method.process_weights_after_loading(head)

    assert not hasattr(head, "w8a16_weight") and head.weight is embed.weight


def _fp8_dequant(w, scale, block_n, block_k):
    s = scale.float().repeat_interleave(block_n, 0)[: w.shape[0]]
    if scale.shape[1] > 1:
        s = s.repeat_interleave(block_k, 1)[:, : w.shape[1]]
    return w.float() * s


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 8, 24, 64])
def test_fp8_linear_is_weight_only(dist_init, default_vllm_config, block, num_tokens):
    """An FP8 checkpoint's fused linear layer (per-shard tensor scales, or
    128x128 block scales) runs weight-only on gfx1030 and matches the
    checkpoint's dequantized weights at every batch size; per-shard scales
    are kept, not requantized."""
    from vllm.model_executor.kernels.linear.scaled_mm.rdna2 import (
        RDNA2FP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
    )

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.float16)
    torch.manual_seed(0)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128] if block else None,
    )
    k, widths = 512, [384, 640]
    with torch.device("cuda"):
        layer = MergedColumnParallelLinear(
            k, widths, bias=False, params_dtype=torch.float16, quant_config=config
        )
    method = layer.quant_method
    assert isinstance(method, Fp8LinearMethod)
    assert isinstance(method.fp8_linear, RDNA2FP8ScaledMMLinearKernel)
    dequant = []
    for shard, n in enumerate(widths):
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        if block:
            scale = torch.rand(n // 128, k // 128, device="cuda") * 0.01 + 1e-3
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale, shard)
            dequant.append(_fp8_dequant(w, scale, 128, 128))
        else:
            scale = torch.tensor(0.002 + 0.0013 * shard, device="cuda")
            layer.weight_scale.weight_loader(layer.weight_scale, scale, shard)
            dequant.append(w.float() * scale)
        layer.weight.weight_loader(layer.weight, w, shard)

    method.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, k, dtype=torch.float16, device="cuda")
    out = method.apply(layer, x)
    ref = x.float() @ torch.cat(dequant).t()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
def test_fp8_linear_partial_scale_blocks():
    """Block scales whose last row and column blocks are partial (N, K not
    multiples of 128) keep each weight's own block scale; bias is added."""
    from vllm.model_executor.kernels.linear.rdna2_w8a16 import (
        kmajor_w8,
        rdna2_w8_linear,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.rdna2 import (
        _split_block_scales,
    )

    torch.manual_seed(0)
    n, k = 1000, 1040
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    blocks = torch.rand(-(-n // 128), -(-k // 128), device="cuda") * 0.01 + 1e-3
    bias = torch.randn(n, dtype=torch.float16, device="cuda")
    x = torch.randn(40, k, dtype=torch.float16, device="cuda")
    scale, ratio = _split_block_scales(blocks, 128, n)

    out = rdna2_w8_linear(x, kmajor_w8(w), scale, ratio, 128, bias)

    ref = x.float() @ _fp8_dequant(w, blocks, 128, 128).t() + bias.float()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
