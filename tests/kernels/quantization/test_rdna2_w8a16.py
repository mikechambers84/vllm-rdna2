# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-bit linear layers on gfx1030: opt-in int8 weight-only storage of
unquantized layers (VLLM_ROCM_W8A16_UNQUANTIZED, VLLM_ROCM_W8A16_LM_HEAD), FP8
checkpoints (no FP8 instructions on gfx1030) and W8A8 int8 checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("gfx1030 only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1030  # noqa: E402


def _from_kmajor(w: torch.Tensor) -> torch.Tensor:
    """K-major weights [K / 4, N, 4] (8-bit) or [K / 2, N, 2] (fp16) back to
    [N, K]."""
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

    assert layer.kmajor_weight.dtype == torch.int8 and layer.weight.numel() == 0
    assert layer.kmajor_weight.shape == (256, 2048, 4)
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    out = method.apply(layer, x)
    dequant = _from_kmajor(layer.kmajor_weight).float() * layer.kmajor_scale[:, None]
    ref = torch.nn.functional.linear(x.float(), dequant)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
    exact = torch.nn.functional.linear(x.float(), weight.float())
    assert ((out.float() - exact).norm() / exact.norm()).item() < 1e-2


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 8, 24, 64, 600])
def test_unquantized_linear_stores_kmajor_fp16_weights(monkeypatch, num_tokens):
    """VLLM_ROCM_KMAJOR_UNQUANTIZED: the fp16 weight is stored K-major
    (lossless), so every batch size matches F.linear on the original weight
    up to fp16 rounding; small layers keep their weight."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setenv("VLLM_ROCM_KMAJOR_UNQUANTIZED", "1")
    torch.manual_seed(0)
    method = UnquantizedLinearMethod()
    layer = torch.nn.Module()
    weight = torch.randn(2048, 1024, dtype=torch.float16, device="cuda") * 0.02
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    small = torch.nn.Module()
    small.weight = torch.nn.Parameter(weight[:256].clone(), requires_grad=False)

    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(small)

    assert layer.kmajor_weight.shape == (512, 2048, 2) and layer.weight.numel() == 0
    assert torch.equal(_from_kmajor(layer.kmajor_weight), weight)
    assert not hasattr(small, "kmajor_weight")
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    bias = torch.randn(2048, dtype=torch.float16, device="cuda")
    out = method.apply(layer, x, bias)
    ref = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 40, 900])
def test_unquantized_linear_stores_kmajor_bf16_weights_as_fp16(monkeypatch, num_tokens):
    """VLLM_ROCM_KMAJOR_UNQUANTIZED with a bf16 layer: the weight is stored
    K-major in fp16 (exact in fp16's range) and bf16 activations run in fp16,
    so no batch size converts the weight per call; the output stays bf16."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    monkeypatch.setenv("VLLM_ROCM_KMAJOR_UNQUANTIZED", "1")
    torch.manual_seed(0)
    method = UnquantizedLinearMethod()
    layer = torch.nn.Module()
    weight = (torch.randn(2048, 1024, device="cuda") * 0.02).bfloat16()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

    method.process_weights_after_loading(layer)

    assert layer.kmajor_weight.dtype == torch.float16
    assert torch.equal(_from_kmajor(layer.kmajor_weight), weight.half())
    x = torch.randn(num_tokens, 1024, device="cuda").bfloat16()
    bias = torch.randn(2048, device="cuda").bfloat16()
    out = method.apply(layer, x, bias)
    assert out.dtype == torch.bfloat16
    ref = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize(
    "env", ["VLLM_ROCM_KMAJOR_UNQUANTIZED", "VLLM_ROCM_W8A16_UNQUANTIZED"]
)
def test_kmajor_layers_give_mla_their_weights(monkeypatch, env):
    """The weight helper MLA absorbs kv_b_proj with rebuilds a K-major layer's
    [N, K] weight: exactly for fp16, the dequantized int8 weight it computes
    with under W8A16."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_and_maybe_dequant_weights,
    )

    monkeypatch.setenv(env, "1")
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randn(2048, 1024, dtype=torch.float16, device="cuda") * 0.02
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.quant_method = UnquantizedLinearMethod()
    layer.quant_method.process_weights_after_loading(layer)
    assert hasattr(layer, "kmajor_weight") and layer.weight.numel() == 0

    out = get_and_maybe_dequant_weights(layer, torch.float32)

    if env == "VLLM_ROCM_KMAJOR_UNQUANTIZED":
        assert torch.equal(out, weight.float())
    else:
        err = ((out - weight.float()).norm() / weight.float().norm()).item()
        assert err < 1e-2


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

    assert head.kmajor_weight.dtype == torch.int8 and head.weight.numel() == 0
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    dequant = _from_kmajor(head.kmajor_weight).float() * head.kmajor_scale[:, None]
    for rows in (None, 3001):
        if rows is None:
            out = head.quant_method.apply(head, x)
        else:
            out = rdna2_w8a16.apply_rdna2_kmajor(head, x, None, rows)
        ref = torch.nn.functional.linear(x.float(), dequant[:rows])
        assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3
        exact = torch.nn.functional.linear(x.float(), weight[:rows].float())
        assert ((out.float() - exact).norm() / exact.norm()).item() < 1e-2


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("num_tokens", [1, 8, 40])
def test_lm_head_stores_kmajor_fp16(monkeypatch, num_tokens):
    """VLLM_ROCM_KMAJOR_UNQUANTIZED: the untied lm_head goes K-major fp16;
    logits match the original weight, also for a vocabulary prefix that does
    not end on a whole dword of 4 channels."""
    from vllm.model_executor.kernels.linear import rdna2_w8a16

    monkeypatch.setenv("VLLM_ROCM_KMAJOR_UNQUANTIZED", "1")
    torch.manual_seed(0)
    head = _lm_head(8192, 1024)
    weight = head.weight.data.clone()

    head.quant_method.process_weights_after_loading(head)

    assert head.kmajor_weight.dtype == torch.float16 and head.weight.numel() == 0
    x = torch.randn(num_tokens, 1024, dtype=torch.float16, device="cuda")
    for rows in (None, 3001):
        if rows is None:
            out = head.quant_method.apply(head, x)
        else:
            out = rdna2_w8a16.apply_rdna2_kmajor(head, x, None, rows)
        ref = torch.nn.functional.linear(x.float(), weight[:rows].float())
        assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
def test_tied_lm_head_stays_fp16(monkeypatch):
    """A tied lm_head shares the input embedding, which must stay fp16."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    monkeypatch.setenv("VLLM_ROCM_W8A16_LM_HEAD", "1")
    monkeypatch.setenv("VLLM_ROCM_KMAJOR_UNQUANTIZED", "1")
    with torch.device("cuda"):
        embed = VocabParallelEmbedding(
            4096, 512, params_dtype=torch.float16, disable_tp=True
        )
    head = _lm_head(4096, 512).tie_weights(embed)

    head.quant_method.process_weights_after_loading(head)

    assert not hasattr(head, "kmajor_weight") and head.weight is embed.weight


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

    out = rdna2_w8_linear(x, kmajor_w8(w), scale, ratio, 128, 128, bias)

    ref = x.float() @ _fp8_dequant(w, blocks, 128, 128).t() + bias.float()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-3


def _w8a8_layer(n: int, k: int, static: bool):
    torch.manual_seed(0)
    layer = torch.nn.Module()
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    scale = torch.rand(n, 1, device="cuda") * 0.01 + 1e-3
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    input_scale = torch.tensor([0.02], device="cuda")
    layer.input_scale = (
        torch.nn.Parameter(input_scale, requires_grad=False) if static else None
    )
    layer.input_zero_point = None
    layer.azp_adj = None
    layer.logical_widths = [n]
    return layer


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("n", [1024, 1030])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 40])
def test_w8a8_linear_matches_triton(n, static, num_tokens):
    """A symmetric W8A8 layer gives the Triton kernel's results exactly; one
    whose output size is not a multiple of 4 keeps the Triton layout."""
    from vllm.model_executor.kernels.linear import (
        RDNA2Int8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True, is_static_input_scheme=static, input_symmetric=True
    )
    names = ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"]
    rdna2 = RDNA2Int8ScaledMMLinearKernel(config, layer_param_names=names)
    triton = TritonInt8ScaledMMLinearKernel(config, layer_param_names=names)
    layer, ref_layer = _w8a8_layer(n, 512, static), _w8a8_layer(n, 512, static)
    bias = torch.randn(n, dtype=torch.float16, device="cuda")

    rdna2.process_weights_after_loading(layer)
    triton.process_weights_after_loading(ref_layer)

    assert layer.weight.dim() == (3 if n % 4 == 0 else 2)
    x = torch.randn(num_tokens, 512, dtype=torch.float16, device="cuda")
    out = rdna2.apply_weights(layer, x, bias)
    ref = triton.apply_weights(ref_layer, x, bias)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")
@pytest.mark.parametrize("weights", ["int8", "fp8", "fp16"])
@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_narrow_long_k_layers_split_k(weights, act_dtype, num_tokens):
    """Few columns with long rows (Flash-Next's 10240 -> 336 hyper-connection
    projection) split K over extra workgroups; the split sums and their
    epilogue (scale, bias) must match the reference."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.rdna2_w8a16 import (
        kmajor_w8,
        kmajor_w16,
    )

    if weights == "fp16" and act_dtype == torch.bfloat16:
        pytest.skip("fp16 weights take fp16 activations")
    torch.manual_seed(0)
    n, k = 336, 10240
    x = torch.randn(num_tokens, k, device="cuda", dtype=act_dtype)
    w = torch.randn(n, k, device="cuda") * 0.05
    bias = torch.randn(n, device="cuda", dtype=act_dtype)
    if weights == "fp16":
        ref_w, scale = w.half().float(), None
        weight = kmajor_w16(w.half())
        bias = None
    elif weights == "int8":
        scale = w.abs().amax(1) / 127
        q = torch.round(w / scale[:, None]).to(torch.int8)
        ref_w, weight = q.float() * scale[:, None], kmajor_w8(q)
    else:
        scale = w.abs().amax(1) / 448
        q = (w / scale[:, None]).to(torch.float8_e4m3fn)
        ref_w, weight = q.float() * scale[:, None], kmajor_w8(q)
    out = ops.gemm_w8_rdna2(x, weight, scale, None, 1, 0, bias, -1)
    ref = x.float() @ ref_w.t() + (0 if bias is None else bias.float())
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
