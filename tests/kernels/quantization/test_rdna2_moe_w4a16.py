# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1030 W4A16 fused-MoE decode kernel (``moe_wna16_decode_rdna2``)
against an fp32 reference, on the uint8-packed ``[E, N, K/2]`` weights the
Triton WNA16 backend keeps (Qwen3.6-35B-A3B dims, E capped at 16)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
    rdna2_moe_kernel_available,
)

pytestmark = pytest.mark.skipif(
    not rdna2_moe_kernel_available(), reason="gfx1030 with moe_wna16_decode_rdna2"
)

E, H, INTER, TOPK, GROUP = 16, 2048, 512, 8, 32


def _experts(n, k):
    q = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, device="cuda")
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    scales = (torch.rand(E, n, k // GROUP, device="cuda") * 0.004 + 1e-4).half()
    ref = (q.float() - 8) * scales.float().repeat_interleave(GROUP, -1)
    return packed, scales, ref


@pytest.mark.parametrize("num_tokens", [1, 3, 16])
def test_moe_wna16_decode_rdna2_matches_reference(num_tokens):
    """Exact integer dequant with the group scale applied in fp32 keeps the
    error at fp16-rounding level (a scale-folded fp16 dequant costs ~2.5%)."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    w13, s13, w13_ref = _experts(2 * INTER, H)
    w2, s2, w2_ref = _experts(H, INTER)
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    # Out-of-range ids (routing garbage from dummy warmup inputs, which the
    # Triton path's alignment drops) must contribute nothing, not fault.
    topk_ids[0, -1] = E
    topk_weights[0, -1] = 0.0
    out = torch.empty(num_tokens, H, dtype=torch.float16, device="cuda")
    act = torch.empty(num_tokens * TOPK, INTER, dtype=torch.float16, device="cuda")

    ops.moe_wna16_decode_rdna2(out, x, topk_ids, topk_weights, w13, s13, w2, s2, act)

    topk_ids[0, -1] = 0
    gate_up = torch.einsum("tk,tjnk->tjn", x.float(), w13_ref[topk_ids])
    hidden = F.silu(gate_up[..., :INTER]) * gate_up[..., INTER:]
    ref = torch.einsum("tj,tjhn,tjn->th", topk_weights, w2_ref[topk_ids], hidden)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3
