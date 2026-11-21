# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode


def _build(
    monkeypatch,
    builder_cls,
    capability,
    num_kv_heads,
    query_lens,
    max_num_seqs=16,
    num_speculative_tokens=None,
    parallel_drafting=False,
    kv_quant_mode=KVQuantMode.NONE,
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap: cap == capability
    )
    monkeypatch.setattr(current_platform, "num_compute_units", lambda: 188)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: num_kv_heads,
            get_head_size=lambda: 128,
            rswa_window=None,
        ),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(
                num_speculative_tokens=num_speculative_tokens,
                speculator_config=SimpleNamespace(parallel_drafting=parallel_drafting),
            )
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = AttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.bfloat16,
        kv_quant_mode=kv_quant_mode,
    )
    builder = builder_cls(spec, ["layer.0"], config, "cpu")
    batch = BatchSpec(seq_lens=[128] * len(query_lens), query_lens=query_lens)
    metadata = builder.build(0, create_common_attn_metadata(batch, 16, "cpu"))
    assert metadata.softmax_segm_output.shape[2] == metadata.num_par_softmax_segments
    assert metadata.softmax_segm_max.shape[0] >= sum(query_lens)
    # DiffKV re-allocates the output; it must stay sized like the base buffers.
    assert builder.softmax_segm_output.shape[0] == builder.softmax_segm_max.shape[0]
    assert (
        metadata.softmax_segm_output.data_ptr()
        == builder.softmax_segm_output.data_ptr()
    )
    return builder, metadata


BUILDERS = [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder]


# Verify batches: the 64-segment scratch holds min(11, max_num_seqs) * (1 + K)
# rows, and its 16-segment view covers every 3D-eligible sequence. 64 segments
# follow the request count, not Q blocks: on SM12.0 they stay faster for verify
# batches up to 11 requests (22 Q blocks at K=3).
@pytest.mark.parametrize("builder_cls", BUILDERS)
@pytest.mark.parametrize(
    "max_num_seqs,num_speculative_tokens,num_seqs,rows,segments",
    [
        (16, 3, 10, 44, 64),
        (16, 3, 11, 44, 64),
        (16, 3, 12, 44, 16),
        (16, 3, 16, 44, 16),
        (1, 5, 1, 6, 64),
    ],
)
def test_split_k_segment_rows_cover_verify_queries(
    monkeypatch,
    builder_cls,
    max_num_seqs,
    num_speculative_tokens,
    num_seqs,
    rows,
    segments,
):
    builder, metadata = _build(
        monkeypatch,
        builder_cls,
        (12, 0),
        1,
        [1 + num_speculative_tokens] * num_seqs,
        max_num_seqs,
        num_speculative_tokens,
    )
    assert builder.softmax_segm_max.shape[:2] == (rows, 8)
    assert metadata.num_par_softmax_segments == segments


# Without SM12.0 64-segment reuse, the scratch holds exactly
# min(seq_threshold_3D, max_num_seqs) * query_len rows for split-K query_lens.
@pytest.mark.parametrize(
    "num_speculative_tokens,parallel_drafting,max_num_seqs,rows",
    [
        (None, False, 8, 8),
        (4, False, 8, 40),
        (7, False, 8, 64),
        (8, False, 8, 64),
        (3, True, 8, 56),
        (4, True, 8, 64),
        (5, False, 1, 6),
    ],
)
@pytest.mark.parametrize(
    "kv_quant_mode", [KVQuantMode.NONE, KVQuantMode.INT4_PER_TOKEN_HEAD]
)
def test_split_k_scratch_rows_follow_speculative_config(
    monkeypatch,
    num_speculative_tokens,
    parallel_drafting,
    max_num_seqs,
    rows,
    kv_quant_mode,
):
    # The INT4 launcher never runs multi-query split-K.
    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        rows = max_num_seqs
    builder, _ = _build(
        monkeypatch,
        TritonAttentionMetadataBuilder,
        (9, 0),
        1,
        [1],
        max_num_seqs,
        num_speculative_tokens,
        parallel_drafting,
        kv_quant_mode,
    )
    assert builder.softmax_segm_output.shape == (rows, 8, 16, 128)
    assert builder.softmax_segm_max.shape == (rows, 8, 16)
    assert builder.softmax_segm_expsum.shape == (rows, 8, 16)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm gfx1030 only")
@pytest.mark.parametrize(
    "num_seqs,query_len,threshold,segments",
    [
        (16, 1, 16, 32),
        (64, 1, 64, 8),
        (128, 1, 128, 4),
        (200, 1, 16, 32),
        (64, 4, 16, 32),
    ],
)
def test_gfx1030_large_decode_batches_keep_3d_path(
    monkeypatch, num_seqs, query_len, threshold, segments
):
    """On gfx1030 decode batches above the 3D threshold (HIP decode kernel)
    keep the 3D path with proportionally fewer segments in the same scratch,
    down to 4; larger or multi-token batches keep the 2D threshold."""
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx10", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx1030", lambda: True)
    monkeypatch.setattr(
        torch.ops._rocm_C, "decode_attention_rdna2", object(), raising=False
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 32,
            get_num_kv_heads=lambda _: 8,
            get_head_size=lambda: 128,
            rswa_window=None,
            dtype=torch.float16,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=256),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = AttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    builder = TritonAttentionMetadataBuilder(spec, ["layer.0"], config, "cpu")
    batch = BatchSpec(seq_lens=[128] * num_seqs, query_lens=[query_len] * num_seqs)
    metadata = builder.build(0, create_common_attn_metadata(batch, 16, "cpu"))

    assert metadata.seq_threshold_3D == threshold
    assert metadata.num_par_softmax_segments == segments
    assert metadata.softmax_segm_output.shape[2] == segments
    if num_seqs <= threshold and query_len == 1:  # 3D path: scratch holds all
        assert metadata.softmax_segm_max.shape[0] >= num_seqs
    assert (
        metadata.softmax_segm_output.data_ptr()
        == builder.softmax_segm_output.data_ptr()
    )
