# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 Triton kernels on gfx10 (RDNA1/RDNA2).

Triton's AMD backend lowers bf16 multiplies to
``llvm.amdgcn.fdot2.bf16.bf16(<a, 0>, <b, 0>, 0)``, an instruction gfx11 and
later have but gfx10 does not, so any kernel doing bf16 arithmetic aborts the
process in code generation ("LLVM ERROR: Cannot select: intrinsic
%llvm.amdgcn.fdot2.bf16.bf16"). For gfx10 targets, the LLVM IR handed to code
generation gets each such call rewritten as the same dot product in fp32,
rounded to bf16 once.
"""

import regex as re

_FDOT2_BF16 = re.compile(
    r"^(?P<indent>\s*)(?P<res>%[-\w.$]+) = (?:tail |notail |musttail )?call "
    r"(?:[\w ]+ )?bfloat @llvm\.amdgcn\.fdot2\.bf16\.bf16\("
    r"<2 x bfloat> (?P<a><[^>]*>|[^,]+), <2 x bfloat> (?P<b><[^>]*>|[^,]+), "
    r"bfloat (?P<c>[^,)]+)(?:, i1 [^)]+)?\).*$"
)


def rewrite_bf16_fdot2(llir: str) -> str:
    """LLVM IR with every llvm.amdgcn.fdot2.bf16.bf16 call replaced by fp32
    arithmetic."""

    def expand(m: re.Match) -> str:
        i, res = m["indent"], m["res"]
        t = "%vllm.fdot2." + res[1:]
        lines = []
        for v, src in (("a", m["a"]), ("b", m["b"])):
            for lane in (0, 1):
                lines += [
                    f"{t}.{v}{lane} = extractelement <2 x bfloat> {src}, i32 {lane}",
                    f"{t}.f{v}{lane} = fpext bfloat {t}.{v}{lane} to float",
                ]
        lines += [
            f"{t}.fc = fpext bfloat {m['c']} to float",
            f"{t}.m0 = fmul float {t}.fa0, {t}.fb0",
            f"{t}.m1 = fmul float {t}.fa1, {t}.fb1",
            f"{t}.s = fadd float {t}.m0, {t}.m1",
            f"{t}.t = fadd float {t}.s, {t}.fc",
            f"{res} = fptrunc float {t}.t to bfloat",
        ]
        return "\n".join(i + line for line in lines)

    out = []
    for line in llir.split("\n"):
        m = _FDOT2_BF16.match(line) if "fdot2.bf16.bf16(" in line else None
        out.append(expand(m) if m else line)
    return "\n".join(out)


def patch_triton_bf16_for_gfx10() -> None:
    """Install the rewrite in Triton's HIP backend (idempotent)."""
    try:
        from triton.backends.amd.compiler import HIPBackend
    except ImportError:
        return
    if getattr(HIPBackend, "_vllm_gfx10_bf16", False):
        return
    make_amdgcn = HIPBackend.make_amdgcn

    def make_amdgcn_gfx10(src, metadata, options):
        if options.arch.startswith("gfx10") and "fdot2.bf16.bf16(" in src:
            src = rewrite_bf16_fdot2(src)
        return make_amdgcn(src, metadata, options)

    HIPBackend.make_amdgcn = staticmethod(make_amdgcn_gfx10)
    HIPBackend._vllm_gfx10_bf16 = True
