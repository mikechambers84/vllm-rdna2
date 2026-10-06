# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K-major linear layers on gfx1030 (RDNA2): the weight layouts and the op
shared by FP8 and W8A8 int8 checkpoints (scaled_mm/rdna2.py) and by the
opt-in storage of unquantized linear layers and the lm_head, as int8
weight-only (VLLM_ROCM_W8A16_UNQUANTIZED, VLLM_ROCM_W8A16_LM_HEAD) or as fp16
(VLLM_ROCM_KMAJOR_UNQUANTIZED, lossless).

Checkpoints that quantize only part of a model (e.g. Qwen3.6-35B-A3B W4A16
keeps its attention, GDN and shared-expert projections in fp16) spend much of
their decode time on those fp16 weights. As int8 with one fp32 scale per
output channel (weight-only, so the activations keep full precision) they
stream half the bytes; K-major fp16 keeps them exact and runs batches of
8-64 rows 1.6-4x faster than the GEMV / rocBLAS.

All of these run ``gemm_w8_rdna2`` at every batch size: it widens 8-bit
weights in registers for fp16/bf16 activations (outrunning rocBLAS on fp16
weights from decode to prefill), runs v_dot4 on int8 activations and
v_dot2 on fp16 weights.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# Small weights stay in fp16: they are cheap and some (router, gates) are
# accuracy-sensitive.
_MIN_NUMEL = 1 << 20


def _supported(weight: torch.Tensor) -> bool:
    from vllm.platforms.rocm import on_gfx1030

    return (
        on_gfx1030()
        and hasattr(torch.ops._rocm_C, "gemm_w8_rdna2")
        and weight.dtype in (torch.float16, torch.bfloat16)
        and weight.dim() == 2
        and weight.shape[0] % 4 == 0
        and weight.shape[1] % 16 == 0
    )


def use_rdna2_w8a16(weight: torch.Tensor) -> bool:
    if not (envs.VLLM_ROCM_W8A16_UNQUANTIZED and current_platform.is_rocm()):
        return False
    return _supported(weight) and weight.numel() >= _MIN_NUMEL


def use_rdna2_w8a16_lm_head(weight: torch.Tensor) -> bool:
    if not (envs.VLLM_ROCM_W8A16_LM_HEAD and current_platform.is_rocm()):
        return False
    return _supported(weight)


def use_rdna2_kmajor_fp16(weight: torch.Tensor, lm_head: bool = False) -> bool:
    if not (envs.VLLM_ROCM_KMAJOR_UNQUANTIZED and current_platform.is_rocm()):
        return False
    return (
        _supported(weight)
        and weight.dtype == torch.float16
        and (lm_head or weight.numel() >= _MIN_NUMEL)
    )


def kmajor_w8(weight: torch.Tensor) -> torch.Tensor:
    """8-bit weights [N, K] in the K-major layout of ``gemm_w8_rdna2``:
    [K / 4, N, 4], four consecutive k of one output channel per dword."""
    n, k = weight.shape
    w = weight.view(torch.int8).reshape(n, k // 4, 4).transpose(0, 1)
    return w.contiguous().view(weight.dtype)


def kmajor_w16(weight: torch.Tensor) -> torch.Tensor:
    """fp16 weights [N, K] in the K-major layout of ``gemm_w8_rdna2``:
    [K / 2, N, 2], two consecutive k of one output channel per dword."""
    n, k = weight.shape
    w = weight.view(torch.int32).t().contiguous()
    return w.view(torch.float16).view(k // 2, n, 2)


def _drop_weight(layer: torch.nn.Module, weight: torch.Tensor) -> None:
    # The [N, K] copy is freed; anything still reading layer.weight fails loudly.
    layer.weight = torch.nn.Parameter(
        torch.empty(0, dtype=weight.dtype, device=weight.device), requires_grad=False
    )


def relayout_weight(layer: torch.nn.Module) -> None:
    """Replace layer.weight with its K-major fp16 copy."""
    weight = layer.weight.data
    layer.kmajor_weight = kmajor_w16(weight)
    layer.kmajor_scale = None
    _drop_weight(layer, weight)


def quantize_weight(layer: torch.nn.Module) -> None:
    """Replace layer.weight with K-major int8 weights and per-channel fp32
    scales."""
    weight = layer.weight.data
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    # Row chunks keep the fp32 temporaries small for vocabulary-sized weights.
    for r0 in range(0, weight.shape[0], 8192):
        w = weight[r0 : r0 + 8192].float()
        s = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q[r0 : r0 + 8192] = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
        scale[r0 : r0 + 8192] = s
    layer.kmajor_weight = kmajor_w8(q)
    layer.kmajor_scale = scale
    _drop_weight(layer, weight)


def _rdna2_w8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    block_scale: torch.Tensor | None,
    block_n: int,
    block_k: int,
    bias: torch.Tensor | None,
    scale_a: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    if (
        x_2d.stride(-1) != 1
        or x_2d.stride(0) * x_2d.element_size() % 16
        or x_2d.data_ptr() % 16
    ):
        x_2d = x_2d.clone(memory_format=torch.contiguous_format)
    out = ops.gemm_w8_rdna2(
        x_2d,
        weight,
        scale,
        block_scale,
        block_n,
        block_k,
        bias,
        -1,
        scale_a,
        out_dtype,
    )
    return out.reshape(*x.shape[:-1], weight.shape[1])


def _rdna2_w8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    block_scale: torch.Tensor | None,
    block_n: int,
    block_k: int,
    bias: torch.Tensor | None,
    scale_a: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[1]), dtype=out_dtype or x.dtype)


direct_register_custom_op(
    "rdna2_w8_linear", _rdna2_w8_linear, fake_impl=_rdna2_w8_linear_fake
)


def rdna2_w8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    block_scale: torch.Tensor | None = None,
    block_n: int = 1,
    block_k: int = 0,
    bias: torch.Tensor | None = None,
    scale_a: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """X @ W^T (+ bias) for K-major 8-bit weights, as ``gemm_w8_rdna2`` (int8
    X with its scale_a and out_dtype: W8A8)."""
    return torch.ops.vllm.rdna2_w8_linear(
        x, weight, scale, block_scale, block_n, block_k, bias, scale_a, out_dtype
    )


def apply_rdna2_kmajor(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int | None = None,
) -> torch.Tensor:
    """Layer's K-major weight (its first ``rows`` output rows if given)
    applied to x."""
    weight, scale = layer.kmajor_weight, layer.kmajor_scale
    if rows is None:
        return rdna2_w8_linear(x, weight, scale, bias=bias)
    # The kernel takes whole dwords of 4 output channels.
    n = min(-(-rows // 4) * 4, weight.shape[1])
    if bias is not None:
        bias = bias[:n]
    if scale is not None:
        scale = scale[:n]
    out = rdna2_w8_linear(x, weight[:, :n], scale, bias=bias)
    return out[..., :rows]
