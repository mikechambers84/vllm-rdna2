# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 for gfx1030 (RDNA2), which has no FP8 instructions.

e4m3fn weights stay one byte each and are widened exactly to fp16 in
registers: decode runs a GEMV on them (in 8-token chunks up to
_MAX_GEMV_TOKENS tokens), larger batches dequantize into a shared workspace
(in row chunks for big weights) and use the regular fp16/bf16 GEMM. The
activations keep full precision, so a checkpoint's activation scheme (static
or dynamic FP8) is ignored, as with Marlin on CUDA GPUs without FP8.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.rdna2_w8a16 import (
    reserve_dequant_workspace,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_GEMV_TOKENS = 8
_MAX_GEMV_TOKENS = 32
# Bigger weights dequantize and multiply in row chunks of at most this many
# elements, which bounds the shared workspace.
_MAX_WORKSPACE_NUMEL = 64 << 20


def _rdna2_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
    bias: torch.Tensor | None,
    workspace: torch.Tensor,
) -> torch.Tensor:
    n, k = weight.shape
    x_2d = x.reshape(-1, k)
    if x_2d.shape[0] <= _MAX_GEMV_TOKENS:
        x_2d = x_2d.contiguous()
        if x_2d.data_ptr() % 16 == 0:
            out = torch.cat(
                [
                    ops.gemv_fp8_rdna2(chunk, weight, scale, block_n, block_k, bias)
                    for chunk in x_2d.split(_GEMV_TOKENS)
                ]
            )
            return out.reshape(*x.shape[:-1], n)
    rows = workspace.numel() // k // block_n * block_n
    if rows >= n:
        dense = workspace.view(x.dtype)[: n * k].view(n, k)
        ops.dequant_fp8_rdna2(dense, weight, scale, block_n, block_k)
        return torch.nn.functional.linear(x, dense, bias)
    out = x_2d.new_empty(x_2d.shape[0], n)
    for n0 in range(0, n, rows):
        nr = min(rows, n - n0)
        dense = workspace.view(x.dtype)[: nr * k].view(nr, k)
        s0 = n0 // block_n
        ops.dequant_fp8_rdna2(
            dense,
            weight[n0 : n0 + nr],
            scale[s0 : s0 + (nr + block_n - 1) // block_n],
            block_n,
            block_k,
        )
        out[:, n0 : n0 + nr] = x_2d @ dense.t()
    if bias is not None:
        out += bias
    return out.reshape(*x.shape[:-1], n)


def _rdna2_fp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
    bias: torch.Tensor | None,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


# Opaque to Dynamo, so the GEMV/GEMM split follows the runtime token count.
direct_register_custom_op(
    "rdna2_fp8_linear", _rdna2_fp8_linear, fake_impl=_rdna2_fp8_linear_fake
)


def _channel_scales(
    layer: torch.nn.Module, scale: torch.Tensor, n: int
) -> torch.Tensor:
    """Per-tensor, per-shard or per-channel weight scales as [n, 1]."""
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
    return scale.reshape(n, 1).contiguous()


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
        if not hasattr(torch.ops._rocm_C, "gemv_fp8_rdna2"):
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
        block_n, block_k = cls._block_shape(c)
        if block_k and (block_n < 1 or block_k < 16 or block_k & (block_k - 1)):
            return False, "requires block_k a power of two >= 16."
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
        if block_k:
            name = "weight_scale"
            if getattr(layer, "weight_scale_inv", None) is not None:
                name = "weight_scale_inv"
            weight = layer.weight
            scale = getattr(layer, name).float().contiguous()
        else:
            name = "weight_scale"
            weight = layer.weight.t()
            scale = _channel_scales(layer, layer.weight_scale, weight.shape[0])
        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, name, scale)
        layer.rdna2_fp8_scale_name = name
        layer.rdna2_fp8_block = (block_n, block_k)
        reserve_dequant_workspace(
            layer.weight.device,
            min(
                layer.weight.numel(),
                max(_MAX_WORKSPACE_NUMEL, block_n * weight.shape[1]),
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        block_n, block_k = layer.rdna2_fp8_block
        return torch.ops.vllm.rdna2_fp8_linear(
            x,
            layer.weight,
            getattr(layer, layer.rdna2_fp8_scale_name),
            block_n,
            block_k,
            bias,
            reserve_dequant_workspace(layer.weight.device, 0),
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
