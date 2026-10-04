# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1030 fused-MoE kernels against fp32 references, on the weights the
Triton paths keep (Qwen3.6-35B-A3B dims, E capped at 16): the decode kernels
``moe_wna16_decode_rdna2`` on uint8-packed ``[E, N, K/2]`` int4 g32 weights and
``moe_int8_decode_rdna2`` on ``[E, N, K]`` int8 weights with channel scales, and
the prefill GEMM ``moe_wna16_gemm_rdna2``."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
    rdna2_moe_kernel_available,
)

pytestmark = pytest.mark.skipif(
    not rdna2_moe_kernel_available("moe_int8_decode_rdna2"),
    reason="gfx1030 with the RDNA2 MoE decode ops",
)

E, H, INTER, TOPK, GROUP = 16, 2048, 512, 8, 32


def _experts(n, k, group=GROUP):
    q = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, device="cuda")
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    scales = (torch.rand(E, n, k // group, device="cuda") * 0.004 + 1e-4).half()
    ref = (q.float() - 8) * scales.float().repeat_interleave(group, -1)
    return packed, scales, ref


def _reference(x, topk_weights, topk_ids, w13_ref, w2_ref):
    gate_up = torch.einsum("tk,tjnk->tjn", x.float(), w13_ref[topk_ids])
    hidden = F.silu(gate_up[..., :INTER]) * gate_up[..., INTER:]
    return torch.einsum("tj,tjhn,tjn->th", topk_weights, w2_ref[topk_ids], hidden)


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
    ref = _reference(x, topk_weights, topk_ids, w13_ref, w2_ref)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("num_tokens", [1, 3, 32])
def test_moe_int8_decode_rdna2_matches_reference(num_tokens):
    """int8 weights are converted exactly to fp16 and the activations stay
    fp16, so the error is fp16 rounding (the W8A8 path also quantizes the
    activations, ~2e-2)."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    w13 = torch.randint(-127, 128, (E, 2 * INTER, H), dtype=torch.int8, device="cuda")
    w2 = torch.randint(-127, 128, (E, H, INTER), dtype=torch.int8, device="cuda")
    s13 = torch.rand(E, 2 * INTER, 1, device="cuda") * 2e-4 + 1e-5
    s2 = torch.rand(E, H, 1, device="cuda") * 2e-4 + 1e-5
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    out = torch.empty(num_tokens, H, dtype=torch.float16, device="cuda")
    act = torch.empty(num_tokens * TOPK, INTER, dtype=torch.float16, device="cuda")

    ops.moe_int8_decode_rdna2(out, x, topk_ids, topk_weights, w13, s13, w2, s2, act)

    ref = _reference(x, topk_weights, topk_ids, w13.float() * s13, w2.float() * s2)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_moe_wna16_gemm_rdna2_prefill_matches_reference(block_m, group):
    """The prefill path: both routed GEMMs over moe_align_block_size rows
    (gathered activations, scattered outputs, top-k weights on the second),
    with SiLU-and-mul and the top-k sum in between, for every tile height."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    num_tokens = 96
    w13, s13, w13_ref = _experts(2 * INTER, H, group)
    w2, s2, w2_ref = _experts(H, INTER, group)
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(topk_ids, block_m, E)
    gate_up = torch.empty(num_tokens * TOPK, 2 * INTER, dtype=x.dtype, device="cuda")
    act = torch.empty(num_tokens * TOPK, INTER, dtype=x.dtype, device="cuda")
    down = torch.empty(num_tokens, TOPK, H, dtype=x.dtype, device="cuda")
    args = (sorted_ids, expert_ids, num_post_padded, topk_weights)

    ops.moe_wna16_gemm_rdna2(gate_up, x, w13, s13, *args, TOPK, False, block_m)
    torch.ops._C.silu_and_mul(act, gate_up)
    ops.moe_wna16_gemm_rdna2(down, act, w2, s2, *args, 1, True, block_m)

    ref = _reference(x, topk_weights, topk_ids.long(), w13_ref, w2_ref)
    out = down.float().sum(1)
    assert ((out - ref).norm() / ref.norm()).item() < 2e-3
