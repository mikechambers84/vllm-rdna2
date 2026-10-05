# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1030 fused-MoE kernels against fp32 references, on the weights the
Triton paths keep (Qwen3.6-35B-A3B dims, E capped at 16): the decode kernels
``moe_wna16_decode_rdna2`` on uint8-packed ``[E, N, K/2]`` int4 weights (group
size 32 or 128, symmetric or with zero points) and
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


def _experts(n, k, group=GROUP, zp=False):
    """int4 weights in the Triton WNA16 layout; with zp, zero points packed two
    columns per byte ([E, N/2, K/G]), else None (8)."""
    q = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, device="cuda")
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    scales = (torch.rand(E, n, k // group, device="cuda") * 0.004 + 1e-4).half()
    shape = (E, n, k // group)
    zeros, z = None, torch.full(shape, 8.0, device="cuda")
    if zp:
        zq = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
        zeros = (zq[:, 0::2] | (zq[:, 1::2] << 4)).contiguous()
        z = zq.float()
    z, scale = (
        z.repeat_interleave(group, -1),
        scales.float().repeat_interleave(group, -1),
    )
    ref = (q.float() - z) * scale
    return packed, scales, ref, zeros


def _reference(x, topk_weights, topk_ids, w13_ref, w2_ref):
    gate_up = torch.einsum("tk,tjnk->tjn", x.float(), w13_ref[topk_ids])
    hidden = F.silu(gate_up[..., :INTER]) * gate_up[..., INTER:]
    return torch.einsum("tj,tjhn,tjn->th", topk_weights, w2_ref[topk_ids], hidden)


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
def test_moe_wna16_decode_rdna2_matches_reference(num_tokens, group, zp):
    """Exact integer dequant with the group scale applied in fp32 keeps the
    error at fp16-rounding level (a scale-folded fp16 dequant costs ~2.5%),
    for both group sizes and with zero points."""
    from vllm import _custom_ops as ops

    torch.manual_seed(0)
    w13, s13, w13_ref, z13 = _experts(2 * INTER, H, group, zp)
    w2, s2, w2_ref, z2 = _experts(H, INTER, group, zp)
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

    ops.moe_wna16_decode_rdna2(
        out, x, topk_ids, topk_weights, w13, s13, w2, s2, act, z13, z2
    )

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


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_moe_wna16_gemm_rdna2_prefill_matches_reference(block_m, group, zp):
    """The prefill path: both routed GEMMs over moe_align_block_size rows
    (gathered activations, scattered outputs, top-k weights on the second),
    with SiLU-and-mul and the top-k sum in between, for every tile height,
    symmetric and with zero points."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    num_tokens = 96
    w13, s13, w13_ref, z13 = _experts(2 * INTER, H, group, zp)
    w2, s2, w2_ref, z2 = _experts(H, INTER, group, zp)
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

    ops.moe_wna16_gemm_rdna2(gate_up, x, w13, s13, z13, *args, TOPK, False, block_m)
    torch.ops._C.silu_and_mul(act, gate_up)
    ops.moe_wna16_gemm_rdna2(down, act, w2, s2, z2, *args, 1, True, block_m)

    ref = _reference(x, topk_weights, topk_ids.long(), w13_ref, w2_ref)
    out = down.float().sum(1)
    assert ((out - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("block_m", [64, 128])
def test_moe_int8_gemm_rdna2_prefill_matches_triton(block_m):
    """The W8A8 prefill GEMM runs the same integer math as the Triton int8
    kernel (int8 rows gathered by moe_align_block_size, per-row activation and
    per-channel weight scales, top-k weights on the second GEMM), so both
    routed GEMMs match it to fp16 output rounding."""
    import triton.language as tl

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        dispatch_fused_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    num_tokens = 128
    w13 = torch.randint(-127, 128, (E, 2 * INTER, H), dtype=torch.int8, device="cuda")
    w2 = torch.randint(-127, 128, (E, H, INTER), dtype=torch.int8, device="cuda")
    s13 = torch.rand(E, 2 * INTER, 1, device="cuda") * 2e-4 + 1e-5
    s2 = torch.rand(E, H, 1, device="cuda") * 2e-4 + 1e-5
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    act = torch.randn(num_tokens * TOPK, INTER, dtype=x.dtype, device="cuda") * 0.5
    x_q, x_s, _ = ops.scaled_int8_quant(x, None, None, True)
    act_q, act_s, _ = ops.scaled_int8_quant(act, None, None, True)
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(topk_ids, block_m, E)
    flags = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
    )
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    align = (sorted_ids, expert_ids, num_post_padded)
    for a, a_s, w, w_s, top_k, mul in (
        (x_q, x_s, w13, s13, TOPK, False),
        (act_q, act_s, w2, s2, 1, True),
    ):
        ref = torch.empty(num_tokens, TOPK, w.size(1), dtype=x.dtype, device="cuda")
        dispatch_fused_moe_kernel(
            a,
            w,
            ref,
            a_s,
            w_s,
            None,
            topk_weights,
            *align,
            mul,
            top_k,
            config,
            compute_type=tl.float16,
            **flags,
        )
        out = torch.empty_like(ref)
        ops.moe_int8_gemm_rdna2(
            out, a, a_s.view(-1), w, w_s, *align, topk_weights, top_k, mul, block_m
        )
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=1e-3)
