# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1030

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not on_gfx1030(),
    reason="RDNA2 (gfx1030) hyper-connection kernels",
)

HC = 4
HIDDEN_SIZE = 2560
LORA_RANK = 320
DOWN_N = LORA_RANK + HC + 12  # merged down+inject output, 16-row padded


@pytest.mark.parametrize("num_tokens", [1, 3, 4, 9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hc_up_mix_matches_unfused(num_tokens: int, dtype: torch.dtype) -> None:
    """The gate of int8 K-major up weights (fused kernel up to 4 rows, silu +
    GEMM + gate mix above) matches the three ops computed in fp32 with their
    intermediates rounded to the activation dtype; lora is the strided slice
    of the merged down+inject output."""
    from vllm.model_executor.kernels.linear.rdna2_w8a16 import kmajor_w8
    from vllm.models.qwen4_exp.amd.ops.hc import hc_up_mix

    torch.manual_seed(0)
    w = torch.randn(HC * HIDDEN_SIZE, LORA_RANK, device="cuda") * 0.05
    scale = w.abs().amax(1) / 127
    q = torch.round(w / scale[:, None]).to(torch.int8)
    down = torch.randn(num_tokens, DOWN_N, device="cuda", dtype=dtype) * 2
    lora = down[:, :LORA_RANK]
    xn = torch.randn(num_tokens, HC * HIDDEN_SIZE, device="cuda", dtype=dtype)

    out = hc_up_mix(lora, xn, kmajor_w8(q), scale, HC, 0)

    y = torch.nn.functional.silu(lora.float() / HC).to(dtype)
    gate = (y.float() @ (q.float() * scale[:, None]).t()).to(dtype)
    ref = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * xn.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
