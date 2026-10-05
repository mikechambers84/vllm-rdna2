# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 for gfx1030 (RDNA2), which has no FP8 instructions.

e4m3fn weights stay one byte each, stored K-major (``kmajor_w8``), and are
widened exactly to fp16 in registers by ``gemm_w8_rdna2`` at every batch
size. The activations keep full precision, so a checkpoint's activation
scheme (static or dynamic FP8) is ignored, as with Marlin on CUDA GPUs
without FP8.
"""

import torch

from vllm.model_executor.kernels.linear.rdna2_w8a16 import (
    kmajor_w8,
    rdna2_w8_linear,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _channel_scales(
    layer: torch.nn.Module, scale: torch.Tensor, n: int
) -> torch.Tensor:
    """Per-tensor, per-shard or per-channel weight scales as [n]."""
    scale = scale.float().reshape(-1)
    widths = getattr(layer, "logical_widths", None) or [n]
    if scale.numel() == len(widths) > 1:
        # A fused checkpoint module loads one scale; the other shards keep
        # their initial (very negative) value.
        if not (scale > torch.finfo(torch.float8_e4m3fn).min).all():
            scale = scale.max().expand(len(widths))
        scale = scale.repeat_interleave(scale.new_tensor(widths, dtype=torch.long))
    elif scale.numel() == 1:
        scale = scale.expand(n)
    assert scale.numel() == n, f"cannot map {scale.numel()} weight scales to {n}"
    return scale.contiguous()


def _split_block_scales(
    scale: torch.Tensor, block_n: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D block scales [ceil(n / block_n), ceil(k / block_k)] as a scale per
    output channel (the largest of its blocks) and fp16 ratios <= 1 per
    (k-block, channel), the form ``gemm_w8_rdna2`` takes."""
    s = scale.float().repeat_interleave(block_n, 0)[:n]
    channel = s.amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
    return channel.contiguous(), (s / channel[:, None]).t().contiguous().half()


class RDNA2FP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """FP8 e4m3fn weights with fp16/bf16 activations on gfx1030.

    Takes the weights the way MarlinFP8ScaledMMLinearKernel does: (K, N) with
    per-tensor, per-shard or per-channel scales, or (N, K) with 2D block
    scales in ``weight_scale_inv`` or ``weight_scale``.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        from vllm.platforms.rocm import on_gfx1030

        if not on_gfx1030():
            return False, "requires gfx1030."
        if not hasattr(torch.ops._rocm_C, "gemm_w8_rdna2"):
            return False, "requires the gfx1030 ROCm kernels."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.input_dtype not in (torch.float16, torch.bfloat16):
            return False, "requires fp16 or bf16 activations."
        if c.weight_quant_key.dtype != torch.float8_e4m3fn:
            return False, "requires float8_e4m3fn weights."
        if c.weight_shape[1] % 16 != 0:
            return False, "requires an input size divisible by 16."
        if c.weight_shape[0] % 4 != 0:
            return False, "requires an output size divisible by 4."
        block_n, block_k = cls._block_shape(c)
        if block_k and (block_n < 1 or block_k < 8 or block_k & (block_k - 1)):
            return False, "requires block_k a power of two >= 8."
        return True, None

    @staticmethod
    def _block_shape(c: FP8ScaledMMLinearLayerConfig) -> tuple[int, int]:
        """(block_n, block_k) of 2D block scales, else (1, 0)."""
        group = c.weight_quant_key.scale.group_shape
        if group.col > 1:
            return group.row, group.col
        return 1, 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        block_n, block_k = self._block_shape(self.config)
        block_scale = None
        if block_k:
            name = "weight_scale"
            if getattr(layer, "weight_scale_inv", None) is not None:
                name = "weight_scale_inv"
            weight = layer.weight
            scale, block_scale = _split_block_scales(
                getattr(layer, name), block_n, weight.shape[0]
            )
        else:
            name = "weight_scale"
            weight = layer.weight.t()
            scale = _channel_scales(layer, layer.weight_scale, weight.shape[0])
        replace_parameter(layer, "weight", kmajor_w8(weight))
        replace_parameter(layer, name, scale)
        layer.rdna2_fp8_scale_name = name
        layer.rdna2_fp8_block_scale = block_scale
        layer.rdna2_fp8_block_k = block_k

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return rdna2_w8_linear(
            x,
            layer.weight,
            getattr(layer, layer.rdna2_fp8_scale_name),
            layer.rdna2_fp8_block_scale,
            layer.rdna2_fp8_block_k,
            bias,
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError
