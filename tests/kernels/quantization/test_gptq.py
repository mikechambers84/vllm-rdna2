# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, True, bit))
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, False, bit))


def _gptq_int4_sym(
    k: int, n: int, group_size: int, random_zeros: bool = False, v2: bool = False
):
    """Random int4 GPTQ weights (stored zero 7, or random stored zeros; GPTQv1
    adds 1 to them, v2 does not), shuffled for the exllama kernel, and their
    fp32 dequantized reference."""

    def pack(v, dim):
        # Eight 4-bit values along dim per int32, low nibble first.
        v = v.movedim(dim, -1).to(torch.int64)
        packed = sum(v[..., i::8] << (4 * i) for i in range(8))
        packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
        return packed.movedim(-1, dim).contiguous()

    q = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    w_q = pack(q, 0)
    ops.gptq_shuffle(w_q, 4)
    z = torch.full((k // group_size, n), 7, device="cuda", dtype=torch.int32)
    if random_zeros:
        z = torch.randint(0, 15, z.shape, device="cuda", dtype=torch.int32)
    zeros = pack(z, 1)
    scales = (torch.rand(k // group_size, n, device="cuda") * 0.01 + 1e-3).half()
    w_ref = (q - z.repeat_interleave(group_size, 0) - (0 if v2 else 1)).float()
    w_ref *= scales.float().repeat_interleave(group_size, 0)
    return w_q, zeros, scales, w_ref


@pytest.mark.parametrize(
    "m,k,n",
    [(m, k, 1024) for m in (1, 16, 64, 512) for k in (5120, 17408)]
    # Wide outputs run the reconstruct-path GEMM as two column halves on ROCm.
    + [(512, 5120, 34816)],
)
def test_gptq_gemm_int4_matches_reference(m, k, n):
    """Covers the fused exllama path (m <= 50) and the reconstruct + GEMM path,
    which must accumulate in fp32: fp16 accumulation costs ~1e-2 at this K."""
    torch.manual_seed(0)
    w_q, zeros, scales, w_ref = _gptq_int4_sym(k, n, 32)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = ops.gptq_gemm(a, w_q, zeros, scales, True, False, 4)
    ref = a.float() @ w_ref
    # The fused path accumulates K/128 partial sums with fp16 atomics.
    tol = 1e-3 if m > 50 else 3e-3
    assert ((out.float() - ref).norm() / ref.norm()).item() < tol


def test_gptq_gemm_reconstruct_path_cuda_graph():
    """The reconstruct path (m > 50) calls into BLAS; vLLM captures it in
    piecewise CUDA/HIP graphs, so capture and replay must reproduce eager,
    also when the dequantized weight goes to a caller-owned workspace."""
    torch.manual_seed(0)
    w_q, zeros, scales, _ = _gptq_int4_sym(1024, 1024, 32)
    a = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    args = (a, w_q, zeros, scales, True, False, 4)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.float16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eager = ops.gptq_gemm(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.gptq_gemm(*args, workspace)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(out, eager)


def _gfx1030_with(op: str) -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, op)


gfx1030_only = pytest.mark.skipif(
    not _gfx1030_with("gemm_w4a16_exl_rdna2"),
    reason="requires gfx1030 with _rocm_C.gemm_w4a16_exl_rdna2 built",
)


@gfx1030_only
@pytest.mark.parametrize("random_zeros", [False, True], ids=["sym", "zeros"])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("m", [1, 2, 3, 8, 16, 24, 64, 300])
def test_gemm_w4a16_exl_rdna2_matches_reference(m, group_size, random_zeros):
    """The gfx1030 GEMM on gptq_gemm's tensors, for each default tile config
    (1-512 rows), symmetric (zeros ignored) and with stored zeros; N = 1000
    leaves a partial 128-column tile."""
    torch.manual_seed(0)
    k, n = 2048, 1000
    w_q, zeros, scales, w_ref = _gptq_int4_sym(k, n, group_size, random_zeros)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = ops.gemm_w4a16_exl_rdna2(a, w_q, zeros, scales, not random_zeros)
    ref = a.float() @ w_ref
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@gfx1030_only
@pytest.mark.parametrize("m", [1, 64])
def test_gemm_w4a16_exl_rdna2_v2_zeros(m):
    """GPTQv2 / AWQ zero semantics: the stored zero is the zero."""
    torch.manual_seed(0)
    k, n = 2048, 1000
    w_q, zeros, scales, w_ref = _gptq_int4_sym(k, n, 128, random_zeros=True, v2=True)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = ops.gemm_w4a16_exl_rdna2(a, w_q, zeros, scales, False, True)
    ref = a.float() @ w_ref
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@gfx1030_only
@pytest.mark.parametrize("cfg", range(13))
def test_gemm_w4a16_exl_rdna2_configs(cfg):
    """Every tile config, with a partial token tile."""
    torch.manual_seed(0)
    k, n = 1024, 512
    w_q, zeros, scales, w_ref = _gptq_int4_sym(k, n, 32, random_zeros=True)
    a = torch.randn(37, k, device="cuda", dtype=torch.float16)
    out = ops.gemm_w4a16_exl_rdna2(a, w_q, zeros, scales, False, False, cfg)
    ref = a.float() @ w_ref
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.skipif(
    not _gfx1030_with("w8a8_gemm_rdna2"), reason="requires gfx1030 int8 GEMM"
)
@pytest.mark.parametrize("rdna2_rows", [0, 512], ids=["gptq_gemm", "rdna2"])
@pytest.mark.parametrize("m", [16, 512])
def test_exllama_gfx1030_gemm_runs_int8_only_for_prefill(m, rdna2_rows):
    """Opt-in W4A8 prefill: rows above the int8 threshold run as an int8 GEMM
    on the re-quantized weight (per-token activation quant costs ~1e-2; a wrong
    nibble order would cost ~1), while decode rows stay W4A16."""
    from vllm.model_executor.kernels.linear.mixed_precision import exllama_rdna2
    from vllm.model_executor.kernels.linear.mixed_precision.exllama_w4a8 import (
        channel_scales,
        w4a8_gemm,
    )

    torch.manual_seed(0)
    k, n, group_size = 5120, 1024, 32
    w_q, zeros, scales, w_ref = _gptq_int4_sym(k, n, group_size)
    channel_scale = channel_scales(scales)
    workspace = torch.empty(k * n, device="cuda", dtype=torch.float16)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    min_rows = (
        exllama_rdna2.W4A8_MIN_ROWS_RDNA2 if rdna2_rows else exllama_rdna2.W4A8_MIN_ROWS
    )
    out = torch.ops.vllm.exllama_gfx1030_gemm(
        a,
        w_q,
        zeros,
        scales,
        channel_scale,
        workspace,
        group_size,
        True,
        False,
        rdna2_rows,
        min_rows,
    )
    ref = a.float() @ w_ref
    err = ((out.float() - ref).norm() / ref.norm()).item()
    if m > min_rows:
        expected = w4a8_gemm(a, w_q, scales, channel_scale, group_size, workspace)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert err < 2e-2
    else:
        assert err < 3e-3


@gfx1030_only
@pytest.mark.parametrize("asymmetric", [False, True], ids=["uint4b8", "uint4_zp"])
@pytest.mark.parametrize("m", [1, 40, 600])
def test_exllama_linear_kernel_gfx1030(m, asymmetric, dist_init):
    """ExllamaLinearKernel on a compressed-tensors layout layer (symmetric, or
    uint4 with stored zeros as compressed-tensors and converted AWQ
    checkpoints have them): the gfx1030 GEMM up to 512 rows, gptq_gemm
    above."""
    from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
        ExllamaLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(0)
    k, n, group_size = 1024, 512, 128
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    scales = (torch.rand(n, k // group_size, device="cuda") * 0.01 + 1e-3).half()
    z = torch.full((n, k // group_size), 8, device="cuda", dtype=torch.int32)
    if asymmetric:
        z = torch.randint(0, 16, z.shape, device="cuda", dtype=torch.int32)
    w_ref = (q - z.repeat_interleave(group_size, 1)).float()
    w_ref *= scales.float().repeat_interleave(group_size, 1)
    weight_type = scalar_types.uint4 if asymmetric else scalar_types.uint4b8
    layer = torch.nn.Module()
    layer.w_q = PackedvLLMParameter(
        data=pack_quantized_values_into_int32(q, weight_type, 1),
        weight_loader=None,
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=8,
    )
    layer.w_s = GroupQuantScaleParameter(
        data=scales, weight_loader=None, input_dim=1, output_dim=0
    )
    if asymmetric:
        layer.w_zp = PackedvLLMParameter(
            data=pack_quantized_values_into_int32(z, weight_type, 0),
            weight_loader=None,
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=8,
        )
    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=asymmetric,
    )
    assert ExllamaLinearKernel.can_implement(config)[0]
    kernel = ExllamaLinearKernel(
        config,
        w_q_param_name="w_q",
        w_s_param_name="w_s",
        w_zp_param_name="w_zp" if asymmetric else None,
    )
    kernel.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    out = kernel.apply_weights(layer, x, bias)
    ref = x.float() @ w_ref.t() + bias.float()
    assert ((out.float() - ref).norm() / ref.norm()).item() < 2e-3
