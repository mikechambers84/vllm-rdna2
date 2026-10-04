# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in int8 weight-only storage for unquantized linear layers on gfx1030
(VLLM_ROCM_W8A16_UNQUANTIZED).

Checkpoints that quantize only part of a model (e.g. Qwen3.6-35B-A3B W4A16
keeps its attention, GDN and shared-expert projections in fp16) spend most of
their decode time streaming those fp16 weights. Each such weight is stored as
int8 with one fp32 scale per output channel (weight-only, so the activations
keep full precision), which halves the bytes streamed and the memory used.
Decode runs the int8 GEMV (in 8-token chunks up to 32 tokens, where it still
streams fewer bytes than the fp16 GEMM); larger batches dequantize into a
shared workspace and use the regular fp16/bf16 GEMM.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Small weights stay in fp16: they are cheap and some (router, gates) are
# accuracy-sensitive.
_MIN_NUMEL = 1 << 20
_GEMV_TOKENS = 8
_MAX_GEMV_TOKENS = 32
_dequant_workspaces: dict[torch.device, torch.Tensor] = {}


def use_rdna2_w8a16(weight: torch.Tensor) -> bool:
    if not (envs.VLLM_ROCM_W8A16_UNQUANTIZED and current_platform.is_rocm()):
        return False
    from vllm.platforms.rocm import on_gfx1030

    return (
        on_gfx1030()
        and hasattr(torch.ops._rocm_C, "gemv_w8a16_rdna2")
        and weight.dtype in (torch.float16, torch.bfloat16)
        and weight.dim() == 2
        and weight.shape[1] % 16 == 0
        and weight.numel() >= _MIN_NUMEL
    )


def quantize_weight(layer: torch.nn.Module) -> None:
    """Replace layer.weight with int8 weights and per-row fp32 scales."""
    weight = layer.weight.data
    scale = weight.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
    layer.w8a16_weight = (
        torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
    )
    layer.w8a16_scale = scale.contiguous()
    layer.w8a16_dtype = weight.dtype
    # The fp16 copy is freed; anything still reading layer.weight fails loudly.
    layer.weight = torch.nn.Parameter(
        torch.empty(0, dtype=weight.dtype, device=weight.device), requires_grad=False
    )
    workspace = _dequant_workspaces.get(weight.device)
    if workspace is None or workspace.numel() < weight.numel():
        _dequant_workspaces[weight.device] = torch.empty(
            weight.numel(), dtype=torch.float16, device=weight.device
        )


@triton.jit
def _dequant_kernel(q_ptr, s_ptr, out_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    offs_k = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    offs = offs_n[:, None] * K + offs_k[None, :]
    q = tl.load(q_ptr + offs, mask=mask)
    s = tl.load(s_ptr + offs_n, mask=offs_n < N)
    w = q.to(tl.float32) * s[:, None]
    tl.store(out_ptr + offs, w.to(out_ptr.dtype.element_ty), mask=mask)


def _rdna2_w8a16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor,
) -> torch.Tensor:
    n, k = weight.shape
    x_2d = x.reshape(-1, k)
    if x_2d.shape[0] <= _MAX_GEMV_TOKENS:
        x_2d = x_2d.contiguous()
        if x_2d.data_ptr() % 16 == 0:
            out = torch.cat(
                [
                    ops.gemv_w8a16_rdna2(chunk, weight, scale, bias)
                    for chunk in x_2d.split(_GEMV_TOKENS)
                ]
            )
            return out.reshape(*x.shape[:-1], n)
    dense = workspace.view(x.dtype)[: n * k].view(n, k)
    grid = (triton.cdiv(n, 64), triton.cdiv(k, 128))
    _dequant_kernel[grid](weight, scale, dense, n, k, BN=64, BK=128)
    return torch.nn.functional.linear(x, dense, bias)


def _rdna2_w8a16_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


# Opaque to Dynamo, so the GEMV/GEMM split follows the runtime token count.
direct_register_custom_op(
    "rdna2_w8a16_linear", _rdna2_w8a16_linear, fake_impl=_rdna2_w8a16_linear_fake
)


def apply_rdna2_w8a16(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    return torch.ops.vllm.rdna2_w8a16_linear(
        x,
        layer.w8a16_weight,
        layer.w8a16_scale,
        bias,
        _dequant_workspaces[layer.w8a16_weight.device],
    )
