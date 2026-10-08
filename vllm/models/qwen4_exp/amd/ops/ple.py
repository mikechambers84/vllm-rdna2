# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp PLE dilated short convolution (prefill)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ple_conv_prefill_kernel(
    x_ptr,
    w_ptr,
    state_ptr,
    out_ptr,
    req_ptr,
    qstart_ptr,
    sidx_ptr,
    flags_ptr,
    stride_x,
    stride_out,
    stride_slot,
    stride_sh,
    stride_ss,
    H,
    K: tl.constexpr,
    DIL: tl.constexpr,
    S: tl.constexpr,
    BH: tl.constexpr,
) -> None:
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    hm = h < H
    p = tl.load(req_ptr + t)
    q0 = tl.load(qstart_ptr + p)
    col = t - q0
    slot = tl.load(sidx_ptr + p).to(tl.int64)
    flags = tl.load(flags_ptr + p)  # bit 0: valid state, bit 1: initial state
    acc = tl.zeros((BH,), dtype=tl.float32)
    for j in tl.static_range(K):
        # Tap j reads the token DIL * (K - 1 - j) positions back, or the cached
        # state before the request's first token.
        src = col - (K - 1 - j) * DIL
        xv = tl.load(
            x_ptr + (q0 + src).to(tl.int64) * stride_x + h,
            mask=hm & (src >= 0),
            other=0.0,
        ).to(tl.float32)
        sv = tl.load(
            state_ptr + slot * stride_slot + h * stride_sh + (S + src) * stride_ss,
            mask=hm & (src < 0) & ((flags & 2) != 0),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_ptr + h * K + j, mask=hm, other=0.0).to(tl.float32)
        acc += (xv + sv) * w
    out = acc * tl.sigmoid(acc)
    out = tl.where((flags & 1) != 0, out, 0.0)
    tl.store(out_ptr + t.to(tl.int64) * stride_out + h, out, mask=hm)


@triton.jit
def _ple_conv_state_kernel(
    x_ptr,
    state_ptr,
    qstart_ptr,
    sidx_ptr,
    flags_ptr,
    stride_x,
    stride_slot,
    stride_sh,
    stride_ss,
    H,
    S: tl.constexpr,
    BH: tl.constexpr,
) -> None:
    p = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    hm = h < H
    q0 = tl.load(qstart_ptr + p)
    length = tl.load(qstart_ptr + p + 1) - q0
    slot = tl.load(sidx_ptr + p).to(tl.int64)
    flags = tl.load(flags_ptr + p)
    write = hm & ((flags & 1) != 0) & (length > 0)
    base = state_ptr + slot * stride_slot + h * stride_sh
    # The new state is the last S positions of [initial state, tokens]. Slot s
    # reads old-state index length + s > s, so writing in increasing s never
    # clobbers a value still to be read.
    for s in tl.static_range(S):
        pos = length - S + s
        xv = tl.load(
            x_ptr + (q0 + pos).to(tl.int64) * stride_x + h,
            mask=write & (pos >= 0),
            other=0.0,
        )
        sv = tl.load(
            base + (S + pos) * stride_ss,
            mask=write & (pos < 0) & ((flags & 2) != 0),
            other=0.0,
        )
        tl.store(base + s * stride_ss, tl.where(pos >= 0, xv, sv), mask=write)


def ple_short_conv_prefill(
    x_p: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    query_start_loc_p: torch.Tensor,
    state_indices: torch.Tensor,
    valid_state: torch.Tensor,
    has_initial_state: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Causal depthwise conv (taps every ``dilation`` tokens, kernel size
    conv_weights.shape[1]) + SiLU over the prefill tokens ``x_p`` [T, H] of
    requests starting at ``query_start_loc_p``, continuing each request's cached
    state (conv_state [slots, H, >= S], S = (K - 1) * dilation, read where
    ``has_initial_state``) and writing its last S inputs back. Rows of requests
    without a ``valid_state`` are zero and their state is left alone."""
    num_tokens, hidden = x_p.shape
    kernel_size = conv_weights.shape[1]
    state_len = (kernel_size - 1) * dilation
    out = torch.empty_like(x_p)
    if num_tokens == 0:
        return out
    if x_p.stride(1) != 1:
        x_p = x_p.contiguous()
    q_starts = query_start_loc_p.to(torch.int32).contiguous()
    req = torch.searchsorted(
        q_starts[1:], torch.arange(num_tokens, device=x_p.device), right=True
    ).to(torch.int32)
    has_state = conv_state.shape[0] > 0
    flags = valid_state.to(torch.int32)
    if has_state:
        flags = flags | ((valid_state & has_initial_state).to(torch.int32) << 1)
    state_indices = state_indices.to(torch.int32).contiguous()
    weights = conv_weights.contiguous()
    state = conv_state if has_state else x_p.new_empty((1, hidden, state_len))
    block_h = 256
    _ple_conv_prefill_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        x_p,
        weights,
        state,
        out,
        req,
        q_starts,
        state_indices,
        flags,
        x_p.stride(0),
        out.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        hidden,
        K=kernel_size,
        DIL=dilation,
        S=state_len,
        BH=block_h,
        num_warps=4,
    )
    if has_state and state_len > 0:
        _ple_conv_state_kernel[(state_indices.numel(), triton.cdiv(hidden, block_h))](
            x_p,
            conv_state,
            q_starts,
            state_indices,
            flags,
            x_p.stride(0),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            hidden,
            S=state_len,
            BH=block_h,
            num_warps=4,
        )
    return out


__all__ = ["ple_short_conv_prefill"]
