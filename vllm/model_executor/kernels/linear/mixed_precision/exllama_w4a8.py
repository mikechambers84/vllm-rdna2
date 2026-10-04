# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in W4A8 prefill for Exllama 4-bit symmetric weights on gfx1030.

Prefill GEMMs (above Exllama's fused-kernel row limit) quantize activations to
int8 per token, re-quantize the int4 group-quantized weight to int8 with one
scale per output channel, and run the W8A8 int8 GEMM on v_dot4. On a V620 the
Qwen3.8-27B prefill GEMMs run ~2x faster than dequant + fp16 GEMM, at the cost
of per-token activation quantization (as in W8A8 checkpoints) and a per-channel
weight scale.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    _triton_int8_scaled_mm_func,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _requant_int4_to_int8_kernel(
    q_ptr,
    s_ptr,
    inv_ptr,
    out_ptr,
    K,
    N,
    GROUP: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    # q: [K/8, N] int32 in Exllama's shuffled nibble order; s: [K/GROUP, N]
    # fp16 group scales; inv: [N] fp32 inverse channel scales; out: [N, K] int8,
    # the layout of W8A8 checkpoint weights.
    offs_k = tl.program_id(0) * BK + tl.arange(0, BK)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    words = tl.load(q_ptr + (offs_k[:, None] // 8) * N + offs_n[None, :], mask=mask)
    # Shuffled position of nibble k % 8: 0, 16, 4, 20, 8, 24, 12, 28.
    shift = ((offs_k % 8) // 2) * 4 + (offs_k % 2) * 16
    nib = (words >> shift[:, None]) & 0xF
    scale = tl.load(s_ptr + (offs_k[:, None] // GROUP) * N + offs_n[None, :], mask=mask)
    inv = tl.load(inv_ptr + offs_n, mask=offs_n < N)
    value = (nib - 8).to(tl.float32) * scale.to(tl.float32) * inv[None, :]
    tl.store(
        out_ptr + offs_n[None, :] * K + offs_k[:, None],
        tl.floor(value + 0.5).to(tl.int8),
        mask=mask,
    )


def channel_scales(group_scales: torch.Tensor) -> torch.Tensor:
    """Per-output-channel int8 scale covering every group: |q - 8| <= 8."""
    return (group_scales.float().amax(0) * 8 / 127).contiguous()


def w4a8_gemm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    channel_scale: torch.Tensor,
    group_size: int,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Int8 GEMM of x [M, K] and the re-quantized w_q; workspace >= K*N bytes."""
    k, n = x.shape[1], w_q.shape[1]
    w8 = workspace.view(torch.int8)[: k * n].view(n, k)
    grid = (triton.cdiv(k, 64), triton.cdiv(n, 128))
    _requant_int4_to_int8_kernel[grid](
        w_q, w_s, 1.0 / channel_scale, w8, k, n, GROUP=group_size, BK=64, BN=128
    )
    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
    return _triton_int8_scaled_mm_func(
        x_q, w8.t(), x_s, channel_scale.view(-1, 1), x.dtype
    )


def _exllama_w4a8_gemm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    channel_scale: torch.Tensor,
    workspace: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    # Above Exllama's fused-kernel limit (50 rows for 4-bit) gptq_gemm would
    # dequantize to fp16 and run an fp16 GEMM; run the int8 GEMM instead.
    if x.shape[0] > 50:
        return w4a8_gemm(x, w_q, w_s, channel_scale, group_size, workspace)
    return ops.gptq_gemm(x, w_q, w_zp, w_s, True, False, 4, workspace)


def _exllama_w4a8_gemm_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    channel_scale: torch.Tensor,
    workspace: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], w_q.shape[1]), dtype=x.dtype, device=x.device)


# Opaque to Dynamo, so the prefill/decode split follows the runtime row count.
direct_register_custom_op(
    "exllama_w4a8_gemm", _exllama_w4a8_gemm, fake_impl=_exllama_w4a8_gemm_fake
)
