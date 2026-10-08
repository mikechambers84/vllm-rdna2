# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1030 fused-MoE kernels against fp32 references, on the weights the
Triton paths keep (Qwen3.6-35B-A3B dims, E capped at 16): the decode kernels
``moe_wna16_decode_rdna2`` on uint8-packed ``[E, N, K/2]`` int4 weights (group
size 32 or 128, symmetric or with zero points) and
``moe_int8_decode_rdna2`` on ``[E, N, K]`` int8 weights with channel scales,
the prefill GEMMs ``moe_wna16_gemm_rdna2``, ``moe_wna16_skinny_rdna2`` and
``moe_w4a8_gemm_rdna2``, and the weight-only FP8 experts (``Rdna2Fp8Experts``)
on both kinds of kernels."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
    rdna2_moe_kernel_available,
)

pytestmark = pytest.mark.skipif(
    not rdna2_moe_kernel_available("moe_int8_decode_rdna2"),
    reason="gfx1030 with the RDNA2 MoE decode ops",
)

E, H, INTER, TOPK, GROUP = 16, 2048, 512, 8, 32


def _experts(n, k, group=GROUP, zp=False, requant=False):
    """int4 weights in the Triton WNA16 layout; with zp, zero points packed two
    columns per byte ([E, N/2, K/G]), else None (8). With requant, the
    reference is the per-channel int8 re-quantization of the W4A8 GEMM."""
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
    if requant:
        zg, sg = z[..., ::group], scales.float()
        mx = (sg * torch.maximum(zg, 15 - zg)).amax(-1, keepdim=True)
        r = (sg * (127 / mx)).half().float().repeat_interleave(group, -1)
        ref = torch.round((q.float() - z) * r) * (mx / 127)
    return packed, scales, ref, zeros


def _reference(x, topk_weights, topk_ids, w13_ref, w2_ref):
    gate_up = torch.einsum("tk,tjnk->tjn", x.float(), w13_ref[topk_ids])
    inter = gate_up.shape[-1] // 2
    hidden = F.silu(gate_up[..., :inter]) * gate_up[..., inter:]
    return torch.einsum("tj,tjhn,tjn->th", topk_weights, w2_ref[topk_ids], hidden)


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
@pytest.mark.parametrize("dims", [(H, INTER), (1152, 384)])
def test_moe_wna16_decode_rdna2_matches_reference(num_tokens, group, zp, dims):
    """Exact integer dequant with the group scale applied in fp32 keeps the
    error at fp16-rounding level (a scale-folded fp16 dequant costs ~2.5%),
    for both group sizes and with zero points; H and I need not be multiples
    of 256 (or of the 8 * R rows a workgroup covers)."""
    from vllm import _custom_ops as ops

    hidden, inter = dims
    torch.manual_seed(0)
    w13, s13, w13_ref, z13 = _experts(2 * inter, hidden, group, zp)
    w2, s2, w2_ref, z2 = _experts(hidden, inter, group, zp)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    # Out-of-range ids (routing garbage from dummy warmup inputs, which the
    # Triton path's alignment drops) must contribute nothing, not fault.
    topk_ids[0, -1] = E
    topk_weights[0, -1] = 0.0
    out = torch.empty(num_tokens, hidden, dtype=torch.float16, device="cuda")
    act = torch.empty(num_tokens * TOPK, inter, dtype=torch.float16, device="cuda")

    ops.moe_wna16_decode_rdna2(
        out, x, topk_ids, topk_weights, w13, s13, w2, s2, act, z13, z2
    )

    topk_ids[0, -1] = 0
    ref = _reference(x, topk_weights, topk_ids, w13_ref, w2_ref)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("num_tokens", [1, 3, 32])
@pytest.mark.parametrize("dims", [(H, INTER), (1040, 176)])
def test_moe_int8_decode_rdna2_matches_reference(num_tokens, dims):
    """int8 weights are converted exactly to fp16 and the activations stay
    fp16, so the error is fp16 rounding (the W8A8 path also quantizes the
    activations, ~2e-2); H and I need only be multiples of 16."""
    from vllm import _custom_ops as ops

    hidden, inter = dims
    torch.manual_seed(0)
    w13 = torch.randint(
        -127, 128, (E, 2 * inter, hidden), dtype=torch.int8, device="cuda"
    )
    w2 = torch.randint(-127, 128, (E, hidden, inter), dtype=torch.int8, device="cuda")
    s13 = torch.rand(E, 2 * inter, 1, device="cuda") * 2e-4 + 1e-5
    s2 = torch.rand(E, hidden, 1, device="cuda") * 2e-4 + 1e-5
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    out = torch.empty(num_tokens, hidden, dtype=torch.float16, device="cuda")
    act = torch.empty(num_tokens * TOPK, inter, dtype=torch.float16, device="cuda")

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


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("block_m", [4, 8, 16])
@pytest.mark.parametrize("dims", [(H, INTER), (1152, 384)])
def test_moe_wna16_skinny_rdna2_matches_reference(block_m, group, zp, dims):
    """The few-rows-per-expert GEMM: both routed GEMMs over
    moe_align_block_size rows for each block height, symmetric and with zero
    points; K not a multiple of 256 (1152 / 384) takes the single-step
    batches."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    hidden, inter = dims
    torch.manual_seed(0)
    num_tokens = 24
    w13, s13, w13_ref, z13 = _experts(2 * inter, hidden, group, zp)
    w2, s2, w2_ref, z2 = _experts(hidden, inter, group, zp)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    sorted_ids, expert_ids, num_post_padded = moe_align_block_size(topk_ids, block_m, E)
    gate_up = torch.empty(num_tokens * TOPK, 2 * inter, dtype=x.dtype, device="cuda")
    act = torch.empty(num_tokens * TOPK, inter, dtype=x.dtype, device="cuda")
    down = torch.empty(num_tokens, TOPK, hidden, dtype=x.dtype, device="cuda")
    args = (sorted_ids, expert_ids, num_post_padded, topk_weights)

    ops.moe_wna16_skinny_rdna2(gate_up, x, w13, s13, z13, *args, TOPK, False, block_m)
    torch.ops._C.silu_and_mul(act, gate_up)
    ops.moe_wna16_skinny_rdna2(down, act, w2, s2, z2, *args, 1, True, block_m)

    ref = _reference(x, topk_weights, topk_ids.long(), w13_ref, w2_ref)
    out = down.float().sum(1)
    assert ((out - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize(
    "op,block_m",
    [
        ("moe_int8_gemm_rdna2", 64),
        ("moe_int8_gemm_rdna2", 128),
        ("moe_int8_skinny_rdna2", 16),
    ],
)
@pytest.mark.parametrize("dims", [(H, INTER), (1040, 176)])
def test_moe_int8_gemm_rdna2_prefill_matches_triton(op, block_m, dims):
    """The W8A8 routed GEMMs (tiled, and skinny for a few rows per expert) run
    the same integer math as the Triton int8 kernel (int8 rows gathered by
    moe_align_block_size, per-row activation and per-channel weight scales,
    top-k weights on the second GEMM), so both routed GEMMs match it to fp16
    output rounding; the skinny kernel also on K off its 256-wide batches."""
    import triton.language as tl

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        dispatch_fused_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    hidden, inter = dims
    if op == "moe_int8_gemm_rdna2" and (hidden % 64 or inter % 64):
        pytest.skip("the tiled GEMM needs K % 64 == 0")
    torch.manual_seed(0)
    num_tokens = 128
    w13 = torch.randint(
        -127, 128, (E, 2 * inter, hidden), dtype=torch.int8, device="cuda"
    )
    w2 = torch.randint(-127, 128, (E, hidden, inter), dtype=torch.int8, device="cuda")
    s13 = torch.rand(E, 2 * inter, 1, device="cuda") * 2e-4 + 1e-5
    s2 = torch.rand(E, hidden, 1, device="cuda") * 2e-4 + 1e-5
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda") * 0.5
    act = torch.randn(num_tokens * TOPK, inter, dtype=x.dtype, device="cuda") * 0.5
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
        getattr(ops, op)(
            out, a, a_s.view(-1), w, w_s, *align, topk_weights, top_k, mul, block_m
        )
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("zp", [False, True])
@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("block_m", [64, 128])
def test_moe_w4a8_gemm_rdna2_matches_requantized_reference(block_m, group, zp):
    """The W4A8 opt-in's routed GEMMs: int8 activation rows against the experts
    re-quantized to int8 per output channel in the kernel, exact in int32, so
    both GEMMs match that reference to fp16 output rounding."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    torch.manual_seed(0)
    num_tokens = 96
    w13, s13, w13_ref, z13 = _experts(2 * INTER, H, group, zp, requant=True)
    w2, s2, w2_ref, z2 = _experts(H, INTER, group, zp, requant=True)
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    act = torch.randn(num_tokens * TOPK, INTER, dtype=x.dtype, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    topk_ids = topk_ids.int()
    align = moe_align_block_size(topk_ids, block_m, E)
    ids = topk_ids.view(-1)
    rows = torch.arange(num_tokens * TOPK, device="cuda")
    for a, w, s, z, ref_w, top_k, mul in (
        (x, w13, s13, z13, w13_ref, TOPK, False),
        (act, w2, s2, z2, w2_ref, 1, True),
    ):
        a_q, a_s, _ = ops.scaled_int8_quant(a, None, None, True)
        out = torch.empty(num_tokens * TOPK, w.size(1), dtype=x.dtype, device="cuda")
        ops.moe_w4a8_gemm_rdna2(
            out, a_q, a_s.view(-1), w, s, z, *align, topk_weights, top_k, mul, block_m
        )
        a_f = (a_q.float() * a_s)[rows // top_k]
        ref = torch.empty(out.shape, device="cuda")
        for e in range(E):
            r = (ids == e).nonzero().view(-1)
            ref[r] = a_f[r] @ ref_w[e].t()
        if mul:
            ref *= topk_weights.view(-1, 1)
        assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


def _fp8_experts(n, k, kind):
    """fp8 weights with per-expert, per-channel or 128x128 block scales, and
    the (block_n, block_k) the quant config reports for them."""
    w = (torch.randn(E, n, k, device="cuda") * 2).to(torch.float8_e4m3fn)
    if kind == "block":
        s = torch.rand(E, n // 128, k // 128, device="cuda") * 2e-3 + 1e-4
        full = s.repeat_interleave(128, 1).repeat_interleave(128, 2)
        return w, s, w.float() * full, [128, 128]
    if kind == "channel":
        s = torch.rand(E, n, 1, device="cuda") * 2e-3 + 1e-4
    else:
        s = torch.rand(E, device="cuda") * 2e-3 + 1e-4
    return w, s, w.float() * s.view(E, -1, 1), None


@pytest.mark.parametrize("kind", ["block", "channel", "expert"])
@pytest.mark.parametrize("num_tokens", [1, 3, 64])
@torch.inference_mode()
def test_rdna2_fp8_experts_match_reference(kind, num_tokens):
    """FP8 experts run weight-only on gfx1030 (fp8 widened exactly to fp16,
    activations unquantized): the decode kernel (<= 4 routed rows per expert)
    and the routed GEMMs both stay at fp16-rounding error."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
        Rdna2Fp8Experts,
    )

    torch.manual_seed(0)
    w13, s13, w13_ref, block_shape = _fp8_experts(2 * INTER, H, kind)
    w2, s2, w2_ref, _ = _fp8_experts(H, INTER, kind)
    experts = Rdna2Fp8Experts(
        make_dummy_moe_config(),
        fp8_w8a16_moe_quant_config(s13, s2, block_shape=block_shape),
    )
    x = torch.randn(num_tokens, H, dtype=torch.float16, device="cuda") * 0.5
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    ws = torch.empty(
        num_tokens * TOPK * max(2 * INTER, H), dtype=x.dtype, device="cuda"
    )
    out = torch.empty(num_tokens, H, dtype=x.dtype, device="cuda")

    experts.apply(
        out,
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        E,
        None,
        None,
        None,
        ws,
        ws.clone(),
        None,
        False,
    )

    ref = _reference(x, topk_weights, topk_ids, w13_ref, w2_ref)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("num_tokens", [3, 96])
@torch.inference_mode()
def test_rdna2_wna16_experts_run_bf16_in_fp16(num_tokens, monkeypatch):
    """bf16 activations (with the fp16 group scales the weight conversion
    stores on gfx1030) take the fp16 RDNA2 kernels instead of the Triton
    fallback, at bf16 output accuracy."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
        Rdna2WNA16Experts,
    )

    torch.manual_seed(0)
    w13, s13, w13_ref, _ = _experts(2 * INTER, H)
    w2, s2, w2_ref, _ = _experts(H, INTER)
    experts = Rdna2WNA16Experts(
        make_dummy_moe_config(),
        int4_w4a16_moe_quant_config(s13, s2, block_shape=[0, GROUP]),
    )
    x = (torch.randn(num_tokens, H, device="cuda") * 0.5).bfloat16()
    topk_weights, topk_ids = torch.topk(
        torch.randn(num_tokens, E, device="cuda").softmax(-1), TOPK, dim=-1
    )
    ws = torch.empty(
        num_tokens * TOPK * max(2 * INTER, H), dtype=x.dtype, device="cuda"
    )
    out = torch.empty(num_tokens, H, dtype=x.dtype, device="cuda")
    calls = []
    for name in (
        "moe_wna16_decode_rdna2",
        "moe_wna16_skinny_rdna2",
        "moe_wna16_gemm_rdna2",
    ):
        op = getattr(ops, name)

        def spy(*args, _op=op, **kwargs):
            calls.append(args)
            _op(*args, **kwargs)

        monkeypatch.setattr(ops, name, spy)

    experts.apply(
        out,
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        E,
        None,
        None,
        None,
        ws,
        ws.clone(),
        None,
        False,
    )

    assert calls
    ref = _reference(x, topk_weights, topk_ids, w13_ref, w2_ref)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


def test_fp8_moe_selects_rdna2_backend():
    """Block-scaled FP8 MoE layers pick the weight-only gfx1030 experts."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        select_fp8_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    config = make_dummy_moe_config(
        num_experts=E, experts_per_token=TOPK, hidden_dim=H, in_dtype=torch.float16
    )
    backend, _ = select_fp8_moe_backend(
        config, kFp8Static128BlockSym, kFp8Dynamic128Sym
    )
    assert backend == Fp8MoeBackend.RDNA2


def test_unquantized_experts_become_rdna2_int8_with_opt_in(
    monkeypatch, default_vllm_config
):
    """With VLLM_ROCM_W8A16_UNQUANTIZED, unquantized MoE layers (e.g. an MTP
    drafter a checkpoint leaves in 16 bits) are quantized to int8 at load and
    run on Rdna2Int8Experts, whose kernels need dynamic per-token activations
    in the quant config."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe import routed_experts
    from vllm.model_executor.layers.fused_moe.experts.rdna2_moe import (
        Rdna2Int8Experts,
    )
    from vllm.model_executor.layers.quantization.online.int8 import (
        Rdna2Int8OnlineMoEMethod,
    )

    monkeypatch.setenv("VLLM_ROCM_W8A16_UNQUANTIZED", "1")
    config = make_dummy_moe_config(
        num_experts=E, experts_per_token=TOPK, hidden_dim=H, in_dtype=torch.float16
    )
    method = routed_experts.RoutedExperts._get_quant_method(
        None, "mtp.layers.0.mlp.experts", None, config
    )
    assert isinstance(method, Rdna2Int8OnlineMoEMethod)
    assert method.experts_cls is Rdna2Int8Experts

    layer = torch.nn.Module()
    layer.w13_scale = torch.ones(E, 2 * INTER, device="cuda")
    layer.w2_scale = torch.ones(E, H, device="cuda")
    quant_config = method.get_fused_moe_quant_config(layer)
    assert quant_config is not None
    assert quant_config.use_int8_w8a8 and quant_config.per_act_token_quant

    monkeypatch.setenv("VLLM_ROCM_W8A16_UNQUANTIZED", "0")
    method = routed_experts.RoutedExperts._get_quant_method(
        None, "mtp.layers.0.mlp.experts", None, config
    )
    assert not isinstance(method, Rdna2Int8OnlineMoEMethod)
