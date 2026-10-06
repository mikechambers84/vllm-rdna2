# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1030 dispatch for ExllamaLinearKernel's 4-bit layers, by row count.

Up to RDNA2_MAX_ROWS rows run ``gemm_w4a16_exl_rdna2`` on Exllama's own
tensors (summed over the linear layers of eight 1.5B-70B models, 1.04-1.5x
gptq_gemm at 1 row, 1.5-3x at 8-64 rows, 1.06-1.25x at 512, near parity by
1024); larger batches run gptq_gemm's reconstruct + GEMM. With
VLLM_ROCM_W4A8_PREFILL, batches of more than VLLM_ROCM_W4A8_MIN_ROWS rows run
as W4A8 (per-token int8 activations): up to W4A8_REQUANT_MIN_ROWS rows
``gemm_w4a8_exl_rdna2`` on the same 4-bit tensors (1.25-1.85x
gemm_w4a16_exl_rdna2 at 8-256 rows), above it the int8 GEMM on the weight
re-quantized per call (symmetric weights), which amortizes the
re-quantization there and outruns the group-exact kernel.
"""

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .exllama_w4a8 import w4a8_gemm
from .MPLinearKernel import MPLinearLayerConfig

RDNA2_MAX_ROWS = 512
# Crossover of the two W4A8 paths on the Qwen3.8-27B layers: the group-exact
# kernel is 1.05-1.65x faster up to 384 rows, 1.3-1.55x slower at 2048.
W4A8_REQUANT_MIN_ROWS = 384


def use_rdna2_gemm(c: MPLinearLayerConfig) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return (
        on_gfx1030()
        and hasattr(torch.ops._rocm_C, "gemm_w4a16_exl_rdna2")
        and c.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
        and c.act_type == torch.float16
        and c.group_size % 32 == 0
    )


def use_rdna2_w4a8(c: MPLinearLayerConfig) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return (
        on_gfx1030()
        and hasattr(torch.ops._rocm_C, "gemm_w4a8_exl_rdna2")
        and c.weight_type in (scalar_types.uint4b8, scalar_types.uint4)
        and c.act_type in (torch.float16, torch.bfloat16)
        and c.group_size % 32 == 0
    )


def _exllama_gfx1030_gemm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    channel_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    group_size: int,
    symmetric: bool,
    v2_format: bool,
    rdna2_rows: int,
    w4a8_min_rows: int,
) -> torch.Tensor:
    m = x.shape[0]
    if m > max(w4a8_min_rows, W4A8_REQUANT_MIN_ROWS) and channel_scale is not None:
        return w4a8_gemm(x, w_q, w_s, channel_scale, group_size, workspace)
    if (0 < m <= rdna2_rows or m > w4a8_min_rows) and (
        x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16
    ):
        x = x.clone(memory_format=torch.contiguous_format)
    if m > w4a8_min_rows:
        x_q, x_s, x_sum = ops.quant_int8_exl_rdna2(x)
        return ops.gemm_w4a8_exl_rdna2(
            x_q, x_s, x_sum, w_q, w_zp, w_s, symmetric, v2_format, x.dtype
        )
    if 0 < m <= rdna2_rows:
        return ops.gemm_w4a16_exl_rdna2(x, w_q, w_zp, w_s, symmetric, v2_format)
    return ops.gptq_gemm(x, w_q, w_zp, w_s, True, v2_format, 4, workspace)


def _exllama_gfx1030_gemm_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_zp: torch.Tensor,
    w_s: torch.Tensor,
    channel_scale: torch.Tensor | None,
    workspace: torch.Tensor,
    group_size: int,
    symmetric: bool,
    v2_format: bool,
    rdna2_rows: int,
    w4a8_min_rows: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], w_q.shape[1]), dtype=x.dtype, device=x.device)


# Opaque to Dynamo, so the kernel choice follows the runtime row count.
direct_register_custom_op(
    "exllama_gfx1030_gemm", _exllama_gfx1030_gemm, fake_impl=_exllama_gfx1030_gemm_fake
)
