# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 and W8A8 int8 checkpoints on gfx1030 (RDNA2) through
``gemm_w8_rdna2``, at every batch size, on weights stored K-major
(``kmajor_w8``).

gfx1030 has no FP8 instructions: e4m3fn weights stay one byte each and are
widened exactly to fp16 in registers. The activations keep full precision, so
a checkpoint's activation scheme (static or dynamic FP8) is ignored, as with
Marlin on CUDA GPUs without FP8. Symmetric W8A8 int8 runs v_dot4 on the int8
activations, with the same results as triton_scaled_mm.
"""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.rdna2_w8a16 import (
    kmajor_w8,
    rdna2_w8_linear,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearLayerConfig,
)
from .triton import TritonInt8ScaledMMLinearKernel


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
    output channel (the largest block of its row) and fp16 ratios <= 1 per
    block, transposed to [ceil(k / block_k), ceil(n / block_n)]: the form
    ``gemm_w8_rdna2`` takes."""
    s = scale.float()
    row = s.amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
    channel = row.repeat_interleave(block_n)[:n].contiguous()
    return channel, (s / row[:, None]).t().contiguous().half()


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
        layer.rdna2_fp8_block = (block_n, block_k)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block_n, block_k = layer.rdna2_fp8_block
        return rdna2_w8_linear(
            x,
            layer.weight,
            getattr(layer, layer.rdna2_fp8_scale_name),
            layer.rdna2_fp8_block_scale,
            block_n,
            block_k,
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


class RDNA2Int8ScaledMMLinearKernel(TritonInt8ScaledMMLinearKernel):
    """Symmetric W8A8 int8 on gfx1030. A layer whose shape the kernel cannot
    take (N % 4 or K % 16) keeps the Triton kernel's layout and path."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."
        from vllm.platforms.rocm import on_gfx1030

        if not on_gfx1030():
            return False, "requires gfx1030."
        if not envs.VLLM_ROCM_USE_SKINNY_GEMM:
            return False, "disabled by VLLM_ROCM_USE_SKINNY_GEMM=0."
        if not hasattr(torch.ops._rocm_C, "gemm_w8_rdna2"):
            return False, "requires the gfx1030 ROCm kernels."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "requires symmetric activation quantization."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight = getattr(layer, w_q_name)
        n, k = weight.shape
        layer.rdna2_w8a8 = n % 4 == 0 and k % 16 == 0
        if not layer.rdna2_w8a8:
            super().process_weights_after_loading(layer)
            return
        scale = getattr(layer, w_s_name)
        if len(layer.logical_widths) > 1 and not self.config.is_channelwise:
            scale = convert_to_channelwise(scale, layer.logical_widths)
        scale = scale.float().reshape(-1).expand(n).contiguous()
        replace_parameter(layer, w_q_name, kmajor_w8(weight.data))
        replace_parameter(layer, w_s_name, scale)
        if self.config.is_static_input_scheme:
            replace_parameter(layer, i_s_name, getattr(layer, i_s_name).max())
        else:
            setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not layer.rdna2_w8a8:
            return super().apply_weights(layer, x, bias)
        w_q, w_s, i_s, _, _ = self._get_layer_params(layer)
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), i_s, None, symmetric=True)
        return rdna2_w8_linear(x_q, w_q, w_s, bias=bias, scale_a=x_s, out_dtype=x.dtype)
