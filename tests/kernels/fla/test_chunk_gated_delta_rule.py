# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLA's chunked gated delta rule (varlen prefill with initial states) against
a per-token fp32 recurrence."""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk, chunk_gated_delta_rule

if not current_platform.is_cuda_alike():
    pytest.skip("GPU only", allow_module_level=True)


def recurrent_reference(q, k, v, g, beta, scale, h0, cu_seqlens):
    """o_t = S_t q_t scale, S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T)
    + beta_t v_t k_t^T, with S [V, K] per value head."""
    H, Hg = v.shape[2], k.shape[2]
    o = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    states = h0.float().clone()
    for n in range(len(cu_seqlens) - 1):
        for t in range(int(cu_seqlens[n]), int(cu_seqlens[n + 1])):
            for h in range(H):
                kt = k[0, t, h // (H // Hg)].float()
                qt = q[0, t, h // (H // Hg)].float()
                S = states[n, h] * torch.exp(g[0, t, h])
                vt = (v[0, t, h].float() - S @ kt) * beta[0, t, h]
                S = S + torch.outer(vt, kt)
                states[n, h] = S
                o[0, t, h] = (S @ qt) * scale
    return o, states


@pytest.mark.parametrize("lens", [[100], [1, 64, 130, 7]])
@pytest.mark.parametrize("heads", [(2, 4), (4, 4)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_varlen(lens, heads, state_dtype, monkeypatch):
    """fp16 varlen prefill matches the recurrence; on gfx1030 the pipeline runs
    in the HIP kernels."""
    torch.manual_seed(0)
    Hg, H = heads
    K = V = 128
    T = sum(lens)
    q = F.normalize(torch.randn(1, T, Hg, K, device="cuda"), dim=-1).half()
    k = F.normalize(torch.randn(1, T, Hg, K, device="cuda"), dim=-1).half()
    v = torch.randn(1, T, H, V, device="cuda").half()
    g = -torch.rand(1, T, H, device="cuda") * 0.3
    beta = torch.rand(1, T, H, device="cuda")
    h0 = (torch.randn(len(lens), H, V, K, device="cuda") * 0.1).to(state_dtype)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
    )

    calls = []
    if chunk._rdna2_gdn_available():
        hip_wy = ops.gdn_wy_rdna2

        def spy(*args) -> None:
            calls.append(args)
            hip_wy(*args)

        monkeypatch.setattr(ops, "gdn_wy_rdna2", spy)

    o, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=K**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ref_o, ref_state = recurrent_reference(q, k, v, g, beta, K**-0.5, h0, cu_seqlens)
    assert len(calls) == (1 if chunk._rdna2_gdn_available() else 0)
    torch.testing.assert_close(o.float(), ref_o, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(final_state.float(), ref_state, atol=2e-2, rtol=2e-2)
