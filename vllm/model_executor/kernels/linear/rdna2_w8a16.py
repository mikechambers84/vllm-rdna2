# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in int8 weight-only storage for unquantized linear layers
(VLLM_ROCM_W8A16_UNQUANTIZED) and the lm_head (VLLM_ROCM_W8A16_LM_HEAD) on
gfx1030.

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
# An lm_head dequantizes for big batches in row chunks of at most this many
# elements instead of sizing the shared workspace to the whole vocabulary.
_MAX_WORKSPACE_NUMEL = 32 << 20
_dequant_workspaces: dict[torch.device, torch.Tensor] = {}


def _supported(weight: torch.Tensor) -> bool:
    from vllm.platforms.rocm import on_gfx1030

    return (
        on_gfx1030()
        and hasattr(torch.ops._rocm_C, "gemv_w8a16_rdna2")
        and weight.dtype in (torch.float16, torch.bfloat16)
        and weight.dim() == 2
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


def quantize_weight(layer: torch.nn.Module, chunked: bool = False) -> None:
    """Replace layer.weight with int8 weights and per-row fp32 scales. With
    ``chunked`` (an lm_head) the shared dequant workspace stays bounded and
    big batches dequantize in row chunks."""
    weight = layer.weight.data
    q = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    scale = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    # Row chunks keep the fp32 temporaries small for vocabulary-sized weights.
    for r0 in range(0, weight.shape[0], 8192):
        w = weight[r0 : r0 + 8192].float()
        s = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q[r0 : r0 + 8192] = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
        scale[r0 : r0 + 8192] = s
    layer.w8a16_weight = q
    layer.w8a16_scale = scale
    layer.w8a16_dtype = weight.dtype
    # The fp16 copy is freed; anything still reading layer.weight fails loudly.
    layer.weight = torch.nn.Parameter(
        torch.empty(0, dtype=weight.dtype, device=weight.device), requires_grad=False
    )
    numel = weight.numel()
    if chunked:
        numel = min(numel, max(_MAX_WORKSPACE_NUMEL, weight.shape[1]))
    reserve_dequant_workspace(weight.device, numel)


def reserve_dequant_workspace(device: torch.device, numel: int) -> torch.Tensor:
    """The 16-bit dequantization workspace shared by all layers on device,
    grown to at least numel elements."""
    workspace = _dequant_workspaces.get(device)
    if workspace is None or workspace.numel() < numel:
        workspace = torch.empty(numel, dtype=torch.float16, device=device)
        _dequant_workspaces[device] = workspace
    return workspace


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
    if n * k <= workspace.numel():
        dense = workspace.view(x.dtype)[: n * k].view(n, k)
        grid = (triton.cdiv(n, 64), triton.cdiv(k, 128))
        _dequant_kernel[grid](weight, scale, dense, n, k, BN=64, BK=128)
        return torch.nn.functional.linear(x, dense, bias)
    # Weight larger than the workspace: dequantize and multiply in row chunks.
    rows = workspace.numel() // k
    out = x_2d.new_empty(x_2d.shape[0], n)
    for n0 in range(0, n, rows):
        nr = min(rows, n - n0)
        dense = workspace.view(x.dtype)[: nr * k].view(nr, k)
        grid = (triton.cdiv(nr, 64), triton.cdiv(k, 128))
        _dequant_kernel[grid](weight[n0:], scale[n0:], dense, nr, k, BN=64, BK=128)
        out[:, n0 : n0 + nr] = x_2d @ dense.t()
    if bias is not None:
        out += bias
    return out.reshape(*x.shape[:-1], n)


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
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    rows: int | None = None,
) -> torch.Tensor:
    """Layer's int8 weight (its first ``rows`` output rows if given) applied
    to x."""
    weight, scale = layer.w8a16_weight, layer.w8a16_scale
    if rows is not None:
        weight, scale = weight[:rows], scale[:rows]
    return torch.ops.vllm.rdna2_w8a16_linear(
        x, weight, scale, bias, _dequant_workspaces[weight.device]
    )
