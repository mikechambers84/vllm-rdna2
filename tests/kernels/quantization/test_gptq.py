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


def _gptq_int4_sym(k: int, n: int, group_size: int):
    """Random symmetric int4 GPTQ weights (GPTQv1: stored zero 7), shuffled for
    the exllama kernel, and their fp32 dequantized reference."""
    q = torch.randint(0, 16, (k, n), device="cuda", dtype=torch.int32)
    packed = torch.zeros(k // 8, n, device="cuda", dtype=torch.int64)
    for i in range(8):
        packed |= q[i::8].to(torch.int64) << (4 * i)
    w_q = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    ops.gptq_shuffle(w_q, 4)
    zeros = torch.full(
        (k // group_size, n // 8), 0x77777777, device="cuda", dtype=torch.int32
    )
    scales = (torch.rand(k // group_size, n, device="cuda") * 0.01 + 1e-3).half()
    w_ref = (q - 8).float() * scales.float().repeat_interleave(group_size, 0)
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
