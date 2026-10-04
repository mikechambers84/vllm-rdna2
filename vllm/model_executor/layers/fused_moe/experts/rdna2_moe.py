# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 fused-MoE experts for RDNA2 (gfx1030).

Same weights as ``TritonWNA16Experts`` (``[E, N, K/2]`` uint8, N-first group
scales). Decode batches of symmetric int4 g32 experts with fp16 activations and
SiLU run on ``moe_wna16_decode_rdna2`` (gate/up with SiLU fused, then down with
the top-k sum in registers): 12-15x the Triton path on a V620 for 1-16 tokens.
Everything else takes the Triton path.
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonWNA16Experts
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform


def rdna2_moe_kernel_available() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, "moe_wna16_decode_rdna2")


class Rdna2WNA16Experts(TritonWNA16Experts):
    MAX_DECODE_TOKENS = 16

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna2_moe_kernel_available()

    def _use_decode_kernel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> bool:
        intermediate = w2.size(2) * 2
        return (
            topk_ids.size(0) <= self.MAX_DECODE_TOKENS
            and hidden_states.dtype == torch.float16
            and self.quant_config.use_int4_w4a16
            and self.block_shape == [0, 32]
            and self.quant_config.w1_zp is None
            and activation == MoEActivation.SILU
            and expert_map is None
            and not apply_router_weight_on_input
            and w1.dtype == torch.uint8
            and w1.size(1) == 2 * intermediate
            and hidden_states.size(1) % 256 == 0
            and intermediate % 256 == 0
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if not self._use_decode_kernel(
            hidden_states,
            w1,
            w2,
            topk_ids,
            activation,
            expert_map,
            apply_router_weight_on_input,
        ):
            return super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
        num_tokens, top_k = topk_ids.shape
        act = _resize_cache(workspace13, (num_tokens * top_k, w2.size(2) * 2))
        ops.moe_wna16_decode_rdna2(
            output,
            hidden_states,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            w1,
            self.w1_scale,
            w2,
            self.w2_scale,
            act,
        )
