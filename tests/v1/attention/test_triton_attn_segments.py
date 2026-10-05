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
from vllm.v1.kv_cache_interface import FullAttentionSpec


# 64 segments only while num_seqs * num_kv_heads * 16 < 188 SMs on SM12.0 and
# the 64-segment scratch holds every query token (e.g. multi-query verify).
@pytest.mark.parametrize(
    "builder_cls",
    [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder],
)
@pytest.mark.parametrize(
    "capability,num_kv_heads,num_seqs,query_len,segments",
    [
        ((12, 0), 1, 11, 1, 64),
        ((12, 0), 1, 12, 1, 16),
        ((12, 0), 8, 1, 1, 64),
        ((12, 0), 8, 2, 1, 16),
        ((9, 0), 1, 1, 1, 16),
        ((12, 0), 1, 2, 4, 64),
        ((12, 0), 1, 10, 4, 16),
    ],
)
def test_split_k_segments_follow_sm_occupancy(
    monkeypatch, builder_cls, capability, num_kv_heads, num_seqs, query_len, segments
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
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
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=num_kv_heads, head_size=128, dtype=torch.bfloat16
    )
    builder = builder_cls(spec, ["layer.0"], config, "cpu")
    batch = BatchSpec(seq_lens=[128] * num_seqs, query_lens=[query_len] * num_seqs)
    metadata = builder.build(0, create_common_attn_metadata(batch, 16, "cpu"))
    assert metadata.num_par_softmax_segments == segments
    assert metadata.softmax_segm_output.shape[2] == segments
    assert metadata.softmax_segm_max.shape[0] >= num_seqs * query_len
    # DiffKV re-allocates the output; it must stay sized like the base buffers.
    assert builder.softmax_segm_output.shape[0] == builder.softmax_segm_max.shape[0]
    assert (
        metadata.softmax_segm_output.data_ptr()
        == builder.softmax_segm_output.data_ptr()
    )


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
    spec = FullAttentionSpec(
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
