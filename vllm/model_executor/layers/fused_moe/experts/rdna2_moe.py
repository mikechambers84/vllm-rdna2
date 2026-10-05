# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused-MoE experts for RDNA2 (gfx1030): decode kernels on the Triton paths'
weights.

W4A16:
Same weights as ``TritonWNA16Experts`` (``[E, N, K/2]`` uint8, N-first group
scales, optional N-packed zero points). Batches of int4 experts (group size a
multiple of 32, symmetric or with zero points) with fp16 activations and SiLU
run on ``moe_wna16_decode_rdna2`` (gate/up with SiLU fused, then down with the
top-k sum in registers) while each expert gets at most MAX_DECODE_ROWS_PER_EXPERT
routed rows (each token re-reads its experts): on a V620 it beats tuned Triton
5-7x at 1 token and 1.5-2.3x at 2-4 rows per expert (E=8-256, group size 32 or
128). Larger batches run the two routed GEMMs on ``moe_wna16_gemm_rdna2``,
which beats Triton at every batch size: 1.2-1.4x up to ~16 rows per expert,
1.8-2.2x at 1024+ tokens.

FP8: ``Rdna2Fp8Experts`` (weight-only, fp16 activations) on the same two kinds
of kernels.
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
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform


def rdna2_moe_kernel_available(op: str = "moe_wna16_decode_rdna2") -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, op)


class Rdna2WNA16Experts(TritonWNA16Experts):
    # The decode kernel beats the routed GEMMs up to ~4-5 rows per expert and
    # loses 20-60% at 6-8 (qwen3.6-35b-a3b, qwen3-30b-a3b, Mixtral shapes).
    MAX_DECODE_ROWS_PER_EXPERT = 4

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna2_moe_kernel_available()

    def _zeros_ok(self, w1: torch.Tensor, w2: torch.Tensor, group: int) -> bool:
        """No zero points, or both in the contiguous [E, N/2, K/G] uint8 layout
        the kernels read (two columns per byte)."""
        z1, z2 = self.w1_zp, self.w2_zp
        if z1 is None and z2 is None:
            return True
        if z1 is None or z2 is None or group <= 0:
            return False
        e, n1, n2 = w1.size(0), w1.size(1), w2.size(1)
        k1, k2 = w1.size(2) * 2, w2.size(2) * 2
        return (
            z1.dtype == z2.dtype == torch.uint8
            and z1.is_contiguous()
            and z2.is_contiguous()
            and tuple(z1.shape) == (e, n1 // 2, k1 // group)
            and tuple(z2.shape) == (e, n2 // 2, k2 // group)
        )

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
        group = self.block_shape[1] if self.block_shape else 0
        return (
            topk_ids.numel() <= self.MAX_DECODE_ROWS_PER_EXPERT * w1.size(0)
            and hidden_states.dtype == torch.float16
            and self.quant_config.use_int4_w4a16
            and self.block_shape is not None
            and self.block_shape[0] == 0
            and group > 0
            and group % 32 == 0
            and hidden_states.size(1) % group == 0
            and intermediate % group == 0
            and self._zeros_ok(w1, w2, group)
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
            and block_shape[1] > 0
            and block_shape[1] % 32 == 0
            and self._zeros_ok(w1, w2, block_shape[1])
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
            self.w1_zp,
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
            self.w2_zp,
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
            self.w1_zp,
            self.w2_zp,
        )


class Rdna2Int8Experts(TritonExperts):
    """int8 W8A8 experts (per-channel weight scales) on the Triton weights
    (``[E, N, K]`` int8, ``[E, N, 1]`` fp32 scales). Decode batches run on
    ``moe_int8_decode_rdna2``, which dots the exactly converted int8 weights
    with the fp16 activations (no activation quantization, so more accurate
    than the Triton W8A8 path): 7x tuned Triton at 1 token, 2.2x at 8, 1.5x at
    32 on a V620 (E=256, top-8), even near 64. With dynamic per-token
    activation scales, batches with at least 32 routed rows per expert run
    the two routed GEMMs on ``moe_int8_gemm_rdna2`` (same integer math as the
    Triton W8A8 kernel; 1.24x at 1024 tokens, 1.3x at 2048, 1.47x at 8192). Other
    batches take the Triton path."""

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

    def _use_prefill_kernel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> bool:
        w1_scale, w2_scale = self.w1_scale, self.w2_scale
        num_tokens, top_k = topk_ids.shape
        return (
            num_tokens * top_k >= 32 * w1.size(0)
            and rdna2_moe_kernel_available("moe_int8_gemm_rdna2")
            and hidden_states.dtype == torch.float16
            and self.quant_config.use_int8_w8a8
            and self.block_shape is None
            and self.per_act_token_quant
            and self.a1_scale is None
            and self.a2_scale is None
            and w1_scale is not None
            and w2_scale is not None
            and w1_scale.numel() == w1.size(0) * w1.size(1)
            and w2_scale.numel() == w2.size(0) * w2.size(1)
            and self.w1_bias is None
            and self.w2_bias is None
            and expert_map is None
            and self._lora_context is None
            and w1.dtype == torch.int8
            and hidden_states.size(1) % 64 == 0
            and w2.size(2) % 64 == 0
            and w1.size(1) % 8 == 0
            and w2.size(1) % 8 == 0
        )

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
        block_m = 64 if num_tokens * top_k <= 96 * num_experts else 128
        sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        weights = topk_weights.to(torch.float32).contiguous()
        x_q, x_s, _ = ops.scaled_int8_quant(hidden_states, None, None, True)
        cache1 = _resize_cache(workspace2, (num_tokens, top_k, n))
        ops.moe_int8_gemm_rdna2(
            cache1,
            x_q,
            x_s.view(-1),
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
        a2_q, a2_s, _ = ops.scaled_int8_quant(cache2, None, None, True)
        cache3 = _resize_cache(workspace2, (num_tokens, top_k, hidden_states.size(1)))
        ops.moe_int8_gemm_rdna2(
            cache3,
            a2_q,
            a2_s.view(-1),
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
            if self._use_prefill_kernel(hidden_states, w1, w2, topk_ids, expert_map):
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


class Rdna2Fp8Experts(TritonExperts):
    """FP8 (e4m3fn) experts with fp16 activations on gfx1030, which has no FP8
    instructions (the Triton FP8 path is unavailable there). Weights stay
    ``[E, N, K]`` fp8 with per-expert ``[E]``, per-channel ``[E, N, 1]`` or 2D
    block ``[E, N / bn, K / bk]`` fp32 scales and are widened exactly to fp16
    inside the kernels; activations are never quantized. Batches with at most
    MAX_DECODE_ROWS_PER_EXPERT routed rows per expert (SiLU) run
    ``moe_fp8_decode_rdna2``, the rest the two routed GEMMs on
    ``moe_fp8_gemm_rdna2``."""

    MAX_DECODE_ROWS_PER_EXPERT = 4

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna2_moe_kernel_available("moe_fp8_gemm_rdna2")

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # The checkpoint's activation scheme is ignored (weight-only FP8).
        if weight_key is None or weight_key.dtype != torch.float8_e4m3fn:
            return False
        scale = weight_key.scale
        if not (weight_key.symmetric and scale.static) or scale.dtype != (
            torch.float32
        ):
            return False
        row, col = scale.group_shape
        if col <= 1:  # per tensor or per channel
            return True
        return row >= 1 and col >= 32 and col & (col - 1) == 0

    @staticmethod
    def supports_lora() -> bool:
        return False

    def _block(self) -> tuple[int, int]:
        """(block_n, block_k) for the kernels; block_k 0 without 2D blocks.
        W8A16 configs carry the block shape on the weights only."""
        shape = self.quant_config._w1.shape
        if shape is not None and shape.col > 1:
            return shape.row, shape.col
        return 1, 0

    def _use_decode_kernel(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
    ) -> bool:
        return (
            topk_ids.numel() <= self.MAX_DECODE_ROWS_PER_EXPERT * w1.size(0)
            and activation == MoEActivation.SILU
            and not apply_router_weight_on_input
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
        if (
            expert_map is not None
            or hidden_states.dtype != torch.float16
            or self.w1_bias is not None
            or self.w2_bias is not None
        ):
            raise NotImplementedError(
                "gfx1030 FP8 MoE needs fp16 activations, no expert "
                "parallelism and no expert biases"
            )
        block_n, block_k = self._block()
        num_tokens, top_k = topk_ids.shape
        weights = topk_weights.to(torch.float32).contiguous()
        if self._use_decode_kernel(
            hidden_states, w1, w2, topk_ids, activation, apply_router_weight_on_input
        ):
            act = _resize_cache(workspace13, (num_tokens * top_k, w2.size(2)))
            ops.moe_fp8_decode_rdna2(
                output,
                hidden_states,
                topk_ids.to(torch.int32).contiguous(),
                weights,
                w1,
                self.w1_scale,
                w2,
                self.w2_scale,
                block_n,
                block_k,
                act,
            )
            return
        num_experts, n = w1.size(0), w1.size(1)
        block_m = Rdna2WNA16Experts._prefill_block_m(num_tokens * top_k / num_experts)
        sorted_ids, expert_ids, num_post_padded = moe_align_block_size(
            topk_ids, block_m, num_experts
        )
        cache1 = _resize_cache(workspace2, (num_tokens, top_k, n))
        ops.moe_fp8_gemm_rdna2(
            cache1,
            hidden_states,
            w1,
            self.w1_scale,
            block_n,
            block_k,
            sorted_ids,
            expert_ids,
            num_post_padded,
            weights,
            top_k,
            apply_router_weight_on_input,
            block_m,
        )
        cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k, self.adjust_N_for_activation(n, activation)),
        )
        self.activation(activation, cache2, cache1.view(-1, n))
        cache3 = _resize_cache(workspace2, (num_tokens, top_k, hidden_states.size(1)))
        ops.moe_fp8_gemm_rdna2(
            cache3,
            cache2,
            w2,
            self.w2_scale,
            block_n,
            block_k,
            sorted_ids,
            expert_ids,
            num_post_padded,
            weights,
            1,
            not apply_router_weight_on_input,
            block_m,
        )
        self.moe_sum(cache3, output)
