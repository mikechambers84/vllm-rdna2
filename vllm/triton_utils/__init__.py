# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.language.core import _aggregate as aggregate  # noqa: E501
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from vllm.triton_utils.tensor_descriptor import use_tensor_descriptor

if HAS_TRITON:
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx10

        if on_gfx10():
            from vllm.triton_utils.rocm_gfx10 import patch_triton_bf16_for_gfx10

            patch_triton_bf16_for_gfx10()

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "use_tensor_descriptor",
]
