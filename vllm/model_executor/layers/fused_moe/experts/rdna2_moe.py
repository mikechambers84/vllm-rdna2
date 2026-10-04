# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused-MoE experts for RDNA2 (gfx1030): decode kernels on the Triton paths'
weights.

W4A16:
Same weights as ``TritonWNA16Experts`` (``[E, N, K/2]`` uint8, N-first group
scales). Batches of symmetric int4 g32 experts with fp16 activations and SiLU
run on ``moe_wna16_decode_rdna2`` (gate/up with SiLU fused, then down with the
top-k sum in registers) up to MAX_DECODE_TOKENS: on a V620 (E=256, top-8) it beats
tuned Triton 4.5x at 1-32 tokens, 3x at 64, 1.8x at 128 and breaks even near 256,
since each token re-reads its experts. Larger batches of symmetric int4 experts
(group size a multiple of 32) run the two routed GEMMs on
``moe_wna16_gemm_rdna2``: 1.4x Triton at 512 tokens, 1.8x at 2048, 2.1x at 8192.
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform


def rdna2_moe_kernel_available(op: str = "moe_wna16_decode_rdna2") -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, op)


class Rdna2WNA16Experts(TritonWNA16Experts):
    MAX_DECODE_TOKENS = 192

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

    def _use_prefill_kernel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> bool:
        block_shape = self.block_shape
        return (
            rdna2_moe_kernel_available("moe_wna16_gemm_rdna2")
            and hidden_states.dtype == torch.float16
            and self.quant_config.use_int4_w4a16
            and block_shape is not None
            and block_shape[0] == 0
            and block_shape[1] % 32 == 0
            and self.quant_config.w1_zp is None
            and self.w1_bias is None
            and self.w2_bias is None
            and expert_map is None
            and self._lora_context is None
            and w1.dtype == torch.uint8
            and hidden_states.size(1) % 32 == 0
            and w2.size(2) % 16 == 0
            and w1.size(1) % 8 == 0
            and w2.size(1) % 8 == 0
        )

    @staticmethod
    def _prefill_block_m(rows_per_expert: float) -> int:
        # Tile height by routed rows per expert, from a sweep on E=256 top-8.
        if rows_per_expert <= 8:
            return 16
        if rows_per_expert <= 16:
            return 32
        if rows_per_expert <= 64:
            return 64
        return 128

    def _apply_prefill(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        num_tokens, top_k = topk_ids.shape
        num_experts, n = w1.size(0), w1.size(1)
        block_m = self._prefill_block_m(num_tokens * top_k / num_experts)
        sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        weights = topk_weights.to(torch.float32).contiguous()
        cache1 = _resize_cache(workspace2, (num_tokens, top_k, n))
        ops.moe_wna16_gemm_rdna2(
            cache1,
            hidden_states,
            w1,
            self.w1_scale,
            sorted_ids,
            expert_ids,
            num_post_padded,
            weights,
            top_k,
            False,
            block_m,
        )
        cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k, self.adjust_N_for_activation(n, activation)),
        )
        self.activation(activation, cache2, cache1.view(-1, n))
        cache3 = _resize_cache(workspace2, (num_tokens, top_k, hidden_states.size(1)))
        ops.moe_wna16_gemm_rdna2(
            cache3,
            cache2,
            w2,
            self.w2_scale,
            sorted_ids,
            expert_ids,
            num_post_padded,
            weights,
            1,
            not apply_router_weight_on_input,
            block_m,
        )
        self.moe_sum(cache3, output)

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
            if self._use_prefill_kernel(hidden_states, w1, w2, expert_map):
                return self._apply_prefill(
                    output,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    activation,
                    workspace13,
                    workspace2,
                    apply_router_weight_on_input,
                )
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


class Rdna2Int8Experts(TritonExperts):
    """int8 W8A8 experts (per-channel weight scales) on the Triton weights
    (``[E, N, K]`` int8, ``[E, N, 1]`` fp32 scales). Decode batches run on
    ``moe_int8_decode_rdna2``, which dots the exactly converted int8 weights
    with the fp16 activations (no activation quantization, so more accurate
    than the Triton W8A8 path): 7x tuned Triton at 1 token, 2.2x at 8, 1.5x at
    32 on a V620 (E=256, top-8), even near 64. Larger batches take the Triton
    path."""

    MAX_DECODE_TOKENS = 32

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The decode kernel takes the fp16 activations; on the Triton fallback
        # TritonExperts.apply quantizes them itself.
        return self.quant_dtype is not None

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna2_moe_kernel_available("moe_int8_decode_rdna2")

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
        w1_scale = self.w1_scale
        return (
            topk_ids.size(0) <= self.MAX_DECODE_TOKENS
            and hidden_states.dtype == torch.float16
            and self.quant_config.use_int8_w8a8
            and self.block_shape is None
            and w1_scale is not None
            and w1_scale.numel() == w1.size(0) * w1.size(1)
            and activation == MoEActivation.SILU
            and expert_map is None
            and not apply_router_weight_on_input
            and w1.dtype == torch.int8
            and w1.size(1) == 2 * w2.size(2)
            and hidden_states.size(1) % 256 == 0
            and w2.size(2) % 256 == 0
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
        act = _resize_cache(workspace13, (num_tokens * top_k, w2.size(2)))
        ops.moe_int8_decode_rdna2(
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
