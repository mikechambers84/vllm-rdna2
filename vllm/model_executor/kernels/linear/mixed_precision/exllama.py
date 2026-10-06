# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# Per-device buffer for the reconstruct path (the dequantized fp16 weight fed to
# the prefill GEMM), sized for the largest Exllama layer. Reserved at load time
# so CUDA graphs never allocate it.
_dq_workspaces: dict[torch.device, torch.Tensor] = {}


def _reserve_dq_workspace(numel: int, dtype: torch.dtype, device: torch.device):
    workspace = _dq_workspaces.get(device)
    if workspace is None or workspace.numel() < numel or workspace.dtype != dtype:
        _dq_workspaces[device] = torch.empty(numel, dtype=dtype, device=device)


def _asymmetric_uint4(c: MPLinearLayerConfig) -> bool:
    """uint4 with stored zeros (AWQ, compressed-tensors asymmetric) on RDNA2:
    gptq_gemm and the gfx1030 GEMM take the zeros as stored (GPTQv2)."""
    if not (current_platform.is_rocm() and c.weight_type == scalar_types.uint4):
        return False
    from vllm.platforms.rocm import on_gfx10

    return c.zero_points and on_gfx10()


class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
    # In theory supports `scalar_types.uint2b2, scalar_types.uint3b4` too but
    # currently untested so not added to the list

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES and not _asymmetric_uint4(c):
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config

        # For Exllama, we need to set a zero-point tensor if there is not one
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            device = getattr(layer, self.w_q_name).device
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group (-1 due to
                # a bug in the original GPTQ checkpoint format leading to
                # exllama kernel adding 1 to the zero points during inference)
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "A 0 zero-point is not supported by Exllama due to "
                    "a bug in the original GPTQ checkpoint format leading to "
                    "exllama kernel adding 1 to the zero points during "
                    "inference"
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            ops.gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if c.zero_points:
            # gptq_gemm takes [K / G, N / 8] zeros (packed along N), the layout
            # of GPTQ checkpoints; compressed-tensors and converted AWQ zeros
            # are [N / 8, K / G].
            assert self.w_zp_name is not None
            zp = getattr(layer, self.w_zp_name)
            if getattr(zp, "output_dim", 1) != 1:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

        k, n = c.partition_weight_shape
        _reserve_dq_workspace(k * n, c.act_type, getattr(layer, self.w_q_name).device)
        from .exllama_rdna2 import RDNA2_MAX_ROWS, use_rdna2_gemm, use_rdna2_w4a8

        self._rdna2_rows = RDNA2_MAX_ROWS if use_rdna2_gemm(c) else 0
        self._w4a8_min_rows = (
            envs.VLLM_ROCM_W4A8_MIN_ROWS
            if envs.VLLM_ROCM_W4A8_PREFILL and use_rdna2_w4a8(c)
            else 2**31 - 1
        )
        self._gfx1030 = self._rdna2_rows > 0 or self._w4a8_min_rows < 2**31 - 1
        if self._w4a8_min_rows < 2**31 - 1 and not c.zero_points:
            from .exllama_w4a8 import channel_scales

            layer.w4a8_channel_scale = channel_scales(getattr(layer, self.w_s_name))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp = self._get_weight_params(layer)
        # gptq_gemm supports GPTQv2 format by passing use_v2_format=True.
        # However, the MPLinearLayerConfig doesn't contain format info, so
        # types with a bias (GPTQ checkpoints) keep GPTQv1's zero + 1; uint4
        # zeros (AWQ, compressed-tensors) are the zeros as stored.
        use_v2_format = not c.weight_type.has_bias()

        assert w_zp is not None, "Zero points are required by Exllama"
        if self._gfx1030:
            output = torch.ops.vllm.exllama_gfx1030_gemm(
                x_2d,
                w_q,
                w_zp,
                w_s,
                getattr(layer, "w4a8_channel_scale", None),
                _dq_workspaces[x_2d.device],
                c.group_size,
                not c.zero_points,
                use_v2_format,
                self._rdna2_rows,
                self._w4a8_min_rows,
            )
            if bias is not None:
                output.add_(bias)
            return output.reshape(out_shape)
        output = ops.gptq_gemm(
            x_2d,
            w_q,
            w_zp,
            w_s,
            True,
            use_v2_format,
            c.weight_type.size_bits,
            _dq_workspaces.get(x_2d.device),
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
