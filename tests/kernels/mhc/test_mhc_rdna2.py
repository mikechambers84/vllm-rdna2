# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RDNA2 mHC kernels against the torch reference (post, pre + Sinkhorn,
RMS norm), standalone and through the mHC custom ops."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA2 kernels", allow_module_level=True)
from vllm.platforms.rocm import on_gfx1030  # noqa: E402

pytestmark = pytest.mark.skipif(not on_gfx1030(), reason="gfx1030 only")

HC = 4
EPS = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6)


def _rms_norm(x, w, eps):
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(x.dtype)


def _inputs(t, hidden, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dt=torch.float32):
        return torch.randn(*shape, generator=g, device="cuda", dtype=dt)

    nmix = 2 * HC + HC * HC
    return dict(
        residual=rand(t, HC, hidden, dt=dtype),
        x=rand(t, hidden, dt=dtype),
        post=torch.rand(t, HC, 1, generator=g, device="cuda") * 2,
        comb=torch.softmax(rand(t, HC, HC), dim=-1),
        fn=rand(nmix, HC * hidden) / (HC * hidden) ** 0.5,
        hc_scale=torch.rand(3, generator=g, device="cuda") + 0.5,
        hc_base=rand(nmix) * 0.5,
        norm_w=(1 + 0.1 * rand(hidden)).to(dtype),
    )


def _ref_pre(residual, d, norm):
    post, comb, y = mhc_pre_torch(
        residual,
        d["fn"],
        d["hc_scale"],
        d["hc_base"],
        EPS["rms_eps"],
        EPS["hc_pre_eps"],
        EPS["hc_sinkhorn_eps"],
        2.0,
        20,
    )
    return post, comb, _rms_norm(y, d["norm_w"], 1e-5) if norm else y


def _check_pre(got, ref):
    post, comb, y = got
    post_r, comb_r, y_r = ref
    torch.testing.assert_close(post, post_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb, comb_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y.float(), y_r.float(), atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("t", [0, 1, 7, 130])
@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm", [False, True])
def test_mhc_rdna2_fused_post_pre(t, hidden, dtype, norm):
    d = _inputs(t, hidden, dtype)
    res, post, comb, y = ops.mhc_rdna2(
        d["residual"],
        d["fn"],
        d["hc_scale"],
        d["hc_base"],
        **EPS,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_weight=d["norm_w"] if norm else None,
        norm_eps=1e-5,
        x=d["x"],
        post_layer_mix=d["post"],
        comb_res_mix=d["comb"],
    )
    res_r = mhc_post_torch(d["x"], d["residual"], d["post"], d["comb"])
    torch.testing.assert_close(res, res_r, atol=2e-2, rtol=1e-2)
    # The pre step must see the kernel's own rounded residual.
    _check_pre((post, comb, y), _ref_pre(res, d, norm))


@pytest.mark.parametrize("t", [1, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm", [False, True])
def test_mhc_rdna2_pre(t, dtype, norm):
    d = _inputs(t, 4096, dtype, seed=1)
    res, post, comb, y = ops.mhc_rdna2(
        d["residual"],
        d["fn"],
        d["hc_scale"],
        d["hc_base"],
        **EPS,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_weight=d["norm_w"] if norm else None,
        norm_eps=1e-5,
    )
    assert res is None
    _check_pre((post, comb, y), _ref_pre(d["residual"], d, norm))


@pytest.mark.parametrize("t", [1, 33])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_post_rdna2(t, dtype):
    d = _inputs(t, 4096, dtype, seed=2)
    out = ops.mhc_post_rdna2(d["x"], d["residual"], d["post"], d["comb"])
    ref = mhc_post_torch(d["x"], d["residual"], d["post"], d["comb"])
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


def test_mhc_ops_route_to_rdna2(default_vllm_config, monkeypatch):
    """The mHC custom ops take the RDNA2 kernel on gfx1030 and agree with
    their native decomposition."""
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp

    calls = []
    kernel = ops.mhc_rdna2

    def spy(*args, **kwargs):
        calls.append(1)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(ops, "mhc_rdna2", spy)
    d = _inputs(5, 4096, torch.float16, seed=3)
    op = MHCFusedPostPreOp()
    args = (
        d["x"],
        d["residual"],
        d["post"],
        d["comb"],
        d["fn"],
        d["hc_scale"],
        d["hc_base"],
        EPS["rms_eps"],
        EPS["hc_pre_eps"],
        EPS["hc_sinkhorn_eps"],
        2.0,
        20,
    )
    res, post, comb, y = op.forward_hip(*args)
    assert calls
    res_r, post_r, comb_r, y_r = op.forward_native(*args)
    torch.testing.assert_close(res, res_r, atol=2e-2, rtol=1e-2)
    _check_pre((post, comb, y), (post_r, comb_r, y_r))
