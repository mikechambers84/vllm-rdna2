# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.offloader.uva import offload_input_embeddings
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="needs UVA")
def test_offload_input_embeddings_keeps_lookups_and_skips_tied_heads(dist_init):
    """Untied input embeddings move to pinned host memory and still return the
    same rows; one tied to an LM head stays on device, since the head's GEMM
    reads the whole table every step."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = VocabParallelEmbedding(1024, 64)
            self.tied_embed = VocabParallelEmbedding(1024, 64)
            self.lm_head = ParallelLMHead(1024, 64).tie_weights(self.tied_embed)

    with torch.device(current_platform.device_type):
        model = Model()
    model.embed.weight.data.normal_()
    ids = torch.randint(0, 1024, (16,), device=current_platform.device_type)
    expected = model.embed(ids)

    moved = offload_input_embeddings(model)

    assert moved == model.embed.weight.numel() * model.embed.weight.element_size()
    assert model.embed.weight._vllm_is_uva_offloaded
    assert not hasattr(model.tied_embed.weight, "_vllm_is_uva_offloaded")
    torch.testing.assert_close(model.embed(ids), expected, rtol=0, atol=0)
