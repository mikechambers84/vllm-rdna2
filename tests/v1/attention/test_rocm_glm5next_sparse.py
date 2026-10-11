# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    _use_rocm_sparse_triton,
    fit_kpool_indices_to_aiter,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_kv_row_offset,
    _validate_dsv4_sparse_dims,
    _validate_sparse_dims,
)


@triton.jit
def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr, stride: tl.constexpr):
    slot = tl.load(slot_ptr)
    tl.store(output_ptr, _sparse_kv_row_offset(slot, stride))


def _fit_kpool_indices_reference(
    token_indices: torch.Tensor, topk_tokens: int
) -> torch.Tensor:
    history = token_indices[:, :topk_tokens]
    tail = token_indices[:, topk_tokens:]
    valid_history = (history >= 0).sum(dim=1)
    valid_tail = (tail >= 0).sum(dim=1)
    keep_history = torch.minimum(valid_history, topk_tokens - valid_tail)

    columns = torch.arange(topk_tokens, device=token_indices.device).unsqueeze(0)
    tail_offsets = columns - keep_history.unsqueeze(1)
    tail_values = torch.gather(
        tail, 1, tail_offsets.clamp(min=0, max=tail.shape[1] - 1)
    )
    output = torch.where(columns < keep_history.unsqueeze(1), history, tail_values)
    valid_output = columns < (keep_history + valid_tail).unsqueeze(1)
    return torch.where(valid_output, output, -1)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_fit_kpool_indices_preserves_tail_and_best_history():
    token_indices = torch.tensor(
        [
            [10, 9, 8, 7, 6, 5, 100, 101],
            [10, 9, 8, -1, -1, -1, 100, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=6)

    assert fitted.tolist() == [
        [10, 9, 8, 7, 100, 101],
        [10, 9, 8, 100, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize(
    ("num_tokens", "topk_tokens", "tail_width"),
    [(1, 4, 1), (7, 6, 2), (13, 128, 128), (3, 2048, 128)],
)
def test_fit_kpool_indices_matches_eager_reference(num_tokens, topk_tokens, tail_width):
    """The Triton fit must reproduce the eager packing rule exactly."""
    set_random_seed(topk_tokens + num_tokens)
    width = topk_tokens + tail_width
    token_indices = torch.randint(0, 4096, (num_tokens, width), dtype=torch.int32)
    valid = torch.rand((num_tokens, width)) < 0.5
    valid[0] = False
    valid[-1] = True
    token_indices = torch.where(valid, token_indices, -1).cuda()

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=topk_tokens)

    torch.testing.assert_close(
        fitted, _fit_kpool_indices_reference(token_indices, topk_tokens)
    )


def test_fit_kpool_indices_exact_width_is_noop():
    token_indices = torch.tensor([[3, 2, 1, -1]], dtype=torch.int32)

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=4)

    assert fitted.data_ptr() == token_indices.data_ptr()


def test_fit_kpool_indices_rejects_narrow_input():
    with pytest.raises(ValueError, match="at least topk_tokens"):
        fit_kpool_indices_to_aiter(
            torch.zeros((1, 3), dtype=torch.int32), topk_tokens=4
        )


@pytest.mark.parametrize(
    (
        "kv_cache_dtype",
        "head_size",
        "num_prefills",
        "num_decodes",
        "num_decode_tokens",
        "max_query_len",
        "expected",
    ),
    [
        ("auto", 512, 1, 0, 0, 32, True),
        ("auto", 512, 1, 2, 2, 32, True),
        ("auto", 512, 0, 2, 2, 1, True),
        ("fp8", 512, 1, 0, 0, 32, False),
        ("auto", 576, 1, 0, 0, 32, False),
        ("auto", 512, 0, 2, 4, 2, True),
        ("auto", 512, 0, 2, 12, 6, True),
        ("auto", 512, 0, 0, 0, 0, False),
    ],
)
def test_rocm_sparse_triton_route(
    kv_cache_dtype,
    head_size,
    num_prefills,
    num_decodes,
    num_decode_tokens,
    max_query_len,
    expected,
):
    """Validate Triton routing for prefill, decode, and MTP verification."""
    assert (
        _use_rocm_sparse_triton(
            kv_cache_dtype=kv_cache_dtype,
            head_size=head_size,
            kv_lora_rank=512,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            max_query_len=max_query_len,
        )
        is expected
    )


@pytest.mark.parametrize("num_heads", [8, 12])
def test_rocm_sparse_triton_route_preserves_padded_sinks(monkeypatch, num_heads):
    captured = {}

    def fake_rocm_sparse_attn_prefill(**kwargs):
        output = kwargs["output"]
        captured["attn_sink"] = kwargs["attn_sink"]
        output.copy_(
            captured["attn_sink"].to(output.dtype).view(1, -1, 1).expand_as(output)
        )

    monkeypatch.setattr(
        sparse_mod, "rocm_sparse_attn_prefill", fake_rocm_sparse_attn_prefill
    )

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "auto"
    impl.scale = 512**-0.5
    impl.sinks = torch.arange(num_heads, dtype=torch.float32)

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        max_query_len=2,
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
    )

    output, lse = impl._forward_mla(SimpleNamespace(), q, kv, metadata)

    if num_heads == 8:
        expected_sinks = impl.sinks.repeat_interleave(2)
    else:
        expected_sinks = torch.cat((impl.sinks, impl.sinks[:4]))
    torch.testing.assert_close(captured["attn_sink"], expected_sinks)
    assert output.shape == (2, num_heads, 512)
    torch.testing.assert_close(
        output[:, :, 0].float(),
        impl.sinks.expand(2, -1),
    )
    assert lse is None


def test_rocm_sparse_attention_accepts_glm_nope_dimensions():
    _validate_sparse_dims(512, 512, 0, "test")


def test_rocm_sparse_attention_rejects_inconsistent_dimensions():
    with pytest.raises(AssertionError, match="expected head_dim"):
        _validate_sparse_dims(511, 512, 0, "test")


def test_dsv4_sparse_attention_keeps_layout_constraint():
    _validate_dsv4_sparse_dims(512, 448, 64, "test")
    with pytest.raises(AssertionError, match="expects 448 NoPE dims"):
        _validate_dsv4_sparse_dims(512, 512, 0, "test")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_kv_row_offset_does_not_overflow_int32():
    # GLM's 640-token pages cross the signed-int32 address boundary at block
    # 6554 for a 512-element KV row. The production kernel must promote the
    # slot before multiplying by the row stride.
    slot = torch.tensor([6554 * 640], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_kv_row_offset_kernel[(1,)](slot, output, stride=512)

    assert output.item() == 6554 * 640 * 512


def _on_gfx1030() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030()


def _ragged_lists(num_tokens, num_slots, max_keys, generator):
    """Per-token slot lists with invalid entries, one empty list, and lengths
    crossing the 128-key tile boundary."""
    lists = []
    for t in range(num_tokens):
        n = (
            0
            if t == 1
            else int(torch.randint(1, max_keys + 1, (1,), generator=generator))
        )
        sl = torch.randperm(num_slots, generator=generator)[:n].to(torch.int32)
        if n > 4:
            sl[::7] = -1
            sl[3] = num_slots + 5
        lists.append(sl)
    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor([len(x) for x in lists]), 0)
    return torch.cat(lists).cuda(), indptr.cuda()


def _sparse_mla_reference(q, kv, indices, indptr, scale):
    out = torch.zeros_like(q)
    for t in range(q.shape[0]):
        sl = indices[indptr[t] : indptr[t + 1]].long()
        sl = sl[(sl >= 0) & (sl < kv.shape[0])]
        if sl.numel():
            k = kv[sl].float()
            p = torch.softmax(q[t].float() @ k.T * scale, dim=-1)
            out[t] = (p @ k).to(q.dtype)
    return out


@pytest.mark.skipif(not _on_gfx1030(), reason="gfx1030 HIP kernel")
@pytest.mark.parametrize("num_tokens", [1, 3, 40, 130])
@pytest.mark.parametrize("max_keys", [5, 300])
def test_rdna2_sparse_mla_matches_reference(num_tokens, max_keys):
    """Covers the split-KV (few tokens) and direct paths, invalid slots and an
    empty key list."""
    from vllm import _custom_ops as ops

    g = torch.Generator().manual_seed(num_tokens * 1000 + max_keys)
    num_heads, num_slots = 64, 700
    q = torch.randn(num_tokens, num_heads, 512, generator=g).half().cuda()
    kv = torch.randn(num_slots, 512, generator=g).half().cuda()
    indices, indptr = _ragged_lists(num_tokens, num_slots, max_keys, g)
    scale = 512**-0.5
    out = torch.full_like(q, float("nan"))
    ops.sparse_mla_attn_rdna2(q, kv, indices, indptr, out, scale)
    ref = _sparse_mla_reference(q, kv, indices, indptr, scale)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _on_gfx1030(), reason="gfx1030 HIP kernel")
def test_rocm_sparse_prefill_routes_glm_shapes_to_rdna2(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill

    calls = []
    kernel = ops.sparse_mla_attn_rdna2

    def spy(*args):
        calls.append(1)
        kernel(*args)

    monkeypatch.setattr(ops, "sparse_mla_attn_rdna2", spy)
    g = torch.Generator().manual_seed(7)
    q = torch.randn(9, 64, 512, generator=g).half().cuda()
    kv = torch.randn(300, 512, generator=g).half().cuda()
    indices, indptr = _ragged_lists(9, 300, 200, g)
    out = torch.empty_like(q)
    rocm_sparse_attn_prefill(
        q=q,
        kv=kv.view(-1, 1, 512),
        indices=None,
        topk_length=None,
        scale=0.05,
        head_dim=512,
        nope_head_dim=512,
        rope_head_dim=0,
        attn_sink=None,
        output=out,
        ragged_indices=indices,
        ragged_indptr=indptr,
    )
    assert calls
    ref = _sparse_mla_reference(q, kv, indices, indptr, 0.05)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def _paged_fp8_cache(num_blocks, block_size, generator):
    """Packed per-block [block_size x 128 fp8 | block_size fp32 scales]."""
    vals = torch.randn(num_blocks, block_size, 128, generator=generator) * 2
    vals = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(num_blocks, block_size, generator=generator) + 0.5
    packed = torch.cat(
        [
            vals.reshape(num_blocks, -1),
            scales.view(torch.uint8).reshape(num_blocks, -1),
        ],
        dim=1,
    )
    return packed.reshape(num_blocks, block_size, 1, 132).cuda()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("ctx_per_row", [True, False])
def test_paged_mqa_logits_reference_mtp_rows_match_single_rows(ctx_per_row):
    """MTP rows of the reference read the block-flat cache like single-token
    rows (a past version read per-key [values | scale] rows)."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    g = torch.Generator().manual_seed(11)
    batch, next_n, heads, block_size, max_len = 2, 3, 32, 64, 300
    pages = (max_len + block_size - 1) // block_size
    cache = _paged_fp8_cache(batch * pages, block_size, g)
    q = (torch.randn(batch, next_n, heads, 128, generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(batch * next_n, heads, generator=g)
    lens = torch.tensor([70, 299])
    row_lens = (lens[:, None] - next_n + 1 + torch.arange(next_n)).int()
    context_lens = row_lens if ctx_per_row else lens.int()
    block_tables = torch.arange(batch * pages).reshape(batch, pages).int()
    q, weights, context_lens, block_tables, row_lens = (
        t.cuda() for t in (q, weights, context_lens, block_tables, row_lens)
    )
    mtp = fp8_paged_mqa_logits_torch(
        q, cache, weights, context_lens, block_tables, max_len
    )
    single = fp8_paged_mqa_logits_torch(
        q.reshape(batch * next_n, 1, heads, 128),
        cache,
        weights,
        row_lens.reshape(-1, 1),
        block_tables.repeat_interleave(next_n, dim=0),
        max_len,
    )
    torch.testing.assert_close(mtp, single)


def _shuffle_fp8_cache(cache, block_size):
    """The indexer cache writers' layout for blocks of more than one entry:
    each block's values in 16-token tiles of eight 16 x 16-byte groups."""
    flat = cache.reshape(cache.shape[0], -1)
    vals = flat[:, : block_size * 128].reshape(-1, block_size // 16, 16, 8, 16)
    vals = vals.permute(0, 1, 3, 2, 4).reshape(cache.shape[0], -1)
    return torch.cat([vals, flat[:, block_size * 128 :]], dim=1).reshape(cache.shape)


@pytest.mark.skipif(not _on_gfx1030(), reason="gfx1030 HIP kernel")
@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("block_size", [64, 160])
@pytest.mark.parametrize("next_n,ctx_per_row", [(1, True), (2, True), (2, False)])
def test_rdna2_paged_mqa_logits_matches_torch(
    block_size, next_n, ctx_per_row, shuffled
):
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
    )

    g = torch.Generator().manual_seed(block_size + next_n)
    batch, heads, max_len = 3, 32, 1000
    pages = (max_len + block_size - 1) // block_size
    num_blocks = batch * pages + 5
    cache = _paged_fp8_cache(num_blocks, block_size, g)
    q = (torch.randn(batch, next_n, heads, 128, generator=g) * 2).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(batch * next_n, heads, generator=g)
    lens = torch.tensor([1, 333, 999])[:batch]
    if ctx_per_row:
        context_lens = (lens[:, None] - next_n + 1 + torch.arange(next_n)).int()
    else:
        context_lens = lens.int()
    block_tables = torch.randperm(num_blocks, generator=g)[: batch * pages]
    block_tables = block_tables.reshape(batch, pages).int()
    q, weights, context_lens, block_tables = (
        t.cuda() for t in (q, weights, context_lens, block_tables)
    )
    out = ops.paged_mqa_logits_rdna2(
        q,
        _shuffle_fp8_cache(cache, block_size) if shuffled else cache,
        weights,
        context_lens,
        block_tables,
        max_len,
        shuffled=shuffled,
    )
    ref = fp8_paged_mqa_logits_torch(
        q, cache, weights, context_lens, block_tables, max_len
    )
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out))
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not _on_gfx1030(), reason="gfx1030 HIP kernel")
@pytest.mark.parametrize("rows,keys,heads", [(5, 700, 32), (40, 300, 64)])
def test_rdna2_mqa_logits_matches_fp32_reference(rows, keys, heads):
    """The prefill indexer logits (rocm_fp8_mqa_logits on gfx1030) equal the
    fp32 sum over heads, with -inf outside each row's key range (empty and
    partial ranges included)."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_fp8_mqa_logits

    g = torch.Generator().manual_seed(rows + keys)
    q = (torch.randn(rows, heads, 128, generator=g) * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(keys, 128, generator=g) * 2).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-6, 2, (keys,), generator=g).float())
    weights = torch.randn(rows, heads, generator=g)
    ke = torch.randint(0, keys + 1, (rows,), generator=g).int()
    ks = (ke * torch.rand(rows, generator=g)).int()
    score = torch.einsum("mhd,nd->mhn", q.float(), k.float()) * scale
    ref = (score.relu() * weights[:, :, None]).sum(1)
    cols = torch.arange(keys)
    ref[(cols[None] < ks[:, None]) | (cols[None] >= ke[:, None])] = float("-inf")
    out = rocm_fp8_mqa_logits(
        q.cuda(),
        (k.cuda(), scale.view(-1, 1).cuda()),
        weights.cuda(),
        ks.cuda(),
        ke.cuda(),
    ).cpu()
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out))
    torch.testing.assert_close(out[finite], ref[finite], atol=1e-3, rtol=1e-4)


@pytest.mark.skipif(not _on_gfx1030(), reason="gfx1030 HIP kernel")
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("k,c", [(256, 512), (512, 256)])
def test_rdna2_bmm_skinny_matches_bmm(rows, k, c):
    """MLA's decode absorption products (q_nope @ W_UK^T, o @ W_UV) on the
    batched GEMV, with the token-major views the MLA layer passes."""
    from vllm import _custom_ops as ops

    g = torch.Generator().manual_seed(rows * 7 + k)
    heads = 64
    x = (torch.randn(rows, heads, k, generator=g) * 0.5).half().cuda()
    w = (torch.randn(heads, k, c, generator=g) * 0.05).half().cuda()
    out = torch.full((rows, heads, c), float("nan"), dtype=torch.half, device="cuda")
    x_nbk = x.transpose(0, 1)
    assert ops.bmm_skinny_rdna2_supported(x_nbk, w)
    ops.bmm_skinny_rdna2(x_nbk, w, out.transpose(0, 1))
    ref = torch.bmm(x_nbk.float(), w.float()).transpose(0, 1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)
