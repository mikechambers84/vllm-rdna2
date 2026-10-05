# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rocBLAS solution choice for fp16/bf16 GEMMs on gfx1030 (RDNA2).

rocBLAS's default Tensile solution is often 2-4x off the best one on gfx1030
for 64-512 rows (e.g. 6 vs 20 TFLOPS for x [128, 2048] @ W [12288, 2048]^T)
and up to 1.3x at prefill sizes. A table of benchmarked solution indices per
weight shape and row bucket routes those GEMMs through
``gemm_rocblas_rdna2``. Indices belong to one rocBLAS build, so a table records
its version. The shipped table covers the shapes of the models this fork was
tuned on; VLLM_ROCM_RDNA2_GEMM_TUNE=1 tunes the other weight shapes of a model
while it loads into VLLM_ROCM_RDNA2_GEMM_TABLE (also read at startup), and
``python -m vllm.model_executor.kernels.linear.rdna2_gemm N,K ...`` tunes
shapes offline.
"""

import functools
import json
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# A GEMM with m rows uses the solution of the smallest bucket >= m.
BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# Tuning while a model loads keeps its temporary outputs small.
_MAX_TUNE_OUT_BYTES = 1 << 30
_SHIPPED = (
    Path(__file__).parents[2]
    / "layers"
    / "quantization"
    / "utils"
    / "configs"
    / "rdna2_rocblas_gemm.json"
)


@functools.cache
def available() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx1030

    return on_gfx1030() and hasattr(torch.ops._rocm_C, "gemm_rocblas_rdna2")


def _key(n: int, k: int, dtype: torch.dtype, w_kn: bool) -> str:
    return f"{n},{k},{str(dtype).removeprefix('torch.')},{'kn' if w_kn else 'nk'}"


def _user_table_path() -> Path:
    path = envs.VLLM_ROCM_RDNA2_GEMM_TABLE
    return Path(path or Path(envs.VLLM_CACHE_ROOT) / "rdna2_rocblas_gemm.json")


def _read(path: Path, version: str) -> dict[str, list[int]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    if data.get("rocblas_version") != version or tuple(data["buckets"]) != BUCKETS:
        logger.warning_once(
            "Ignoring %s: tuned for rocBLAS %s, running %s",
            path,
            data.get("rocblas_version"),
            version,
        )
        return {}
    return data["solutions"]


@functools.cache
def _table() -> dict[str, list[int]]:
    """{shape key: solution per bucket (0: rocBLAS's choice)}."""
    if not available():
        return {}
    version = torch.ops._rocm_C.rocblas_version_rdna2()
    return _read(_SHIPPED, version) | _read(_user_table_path(), version)


def solution(n: int, k: int, m: int, dtype: torch.dtype, w_kn: bool = False) -> int:
    sols = _table().get(_key(n, k, dtype, w_kn))
    if sols is None:
        return 0
    for bucket, sol in zip(BUCKETS, sols):
        if m <= bucket:
            return sol
    return sols[-1]


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """F.linear(x, weight, bias) with the tuned rocBLAS solution, if any."""
    n, k = weight.shape
    x_2d = x.reshape(-1, k)
    sol = solution(n, k, x_2d.shape[0], x.dtype)
    if (
        sol == 0
        or weight.dtype != x.dtype
        or x_2d.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        return torch.nn.functional.linear(x, weight, bias)
    out = torch.ops._rocm_C.gemm_rocblas_rdna2(x_2d, weight, sol, False)
    if bias is not None:
        out += bias
    return out.reshape(*x.shape[:-1], n)


def register(
    n: int, k: int, dtype: torch.dtype, weight: torch.Tensor | None = None
) -> None:
    """Note a weight [n, k] (``weight`` itself, if it exists in that form) run
    through ``linear``; with VLLM_ROCM_RDNA2_GEMM_TUNE, tune the shape now if
    no table has it."""
    if not (envs.VLLM_ROCM_RDNA2_GEMM_TUNE and available()):
        return
    if dtype not in (torch.float16, torch.bfloat16):
        return
    key = _key(n, k, dtype, False)
    if key in _table():
        return
    logger.info("Tuning rocBLAS solutions for a [%d, %d] weight", n, k)
    _table()[key] = tune(n, k, dtype, weight=weight)
    _save(_user_table_path(), {key: _table()[key]})


def _save(path: Path, solutions: dict[str, list[int]]) -> None:
    version = torch.ops._rocm_C.rocblas_version_rdna2()
    data = {"rocblas_version": version, "buckets": list(BUCKETS)}
    data["solutions"] = _read(path, version) | solutions
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def tune(
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    w_kn: bool = False,
    guard=None,
    weight: torch.Tensor | None = None,
) -> list[int]:
    """Fastest solution per bucket (0 unless >= 3% faster than rocBLAS's, or
    where the output would exceed _MAX_TUNE_OUT_BYTES)."""
    op = torch.ops._rocm_C.gemm_rocblas_rdna2
    w = weight
    if w is None:
        w = torch.randn((k, n) if w_kn else (n, k), dtype=dtype, device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def time_ms(a: torch.Tensor, sol: int, budget_ms: float) -> float:
        if guard is not None:
            guard()
        op(a, w, sol, w_kn)
        start.record()
        op(a, w, sol, w_kn)
        end.record()
        end.synchronize()
        iters = max(1, int(budget_ms / max(start.elapsed_time(end), 1e-3)))
        start.record()
        for _ in range(iters):
            op(a, w, sol, w_kn)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iters

    best = []
    for m in BUCKETS:
        if m * n * w.element_size() > _MAX_TUNE_OUT_BYTES:
            best.append(0)
            continue
        a = torch.randn(m, k, dtype=dtype, device="cuda")
        ref = op(a, w, 0, w_kn).float()
        t_default = time_ms(a, 0, 20.0)
        screened = []
        for sol in torch.ops._rocm_C.gemm_rocblas_solutions_rdna2(a, w, w_kn):
            err = (op(a, w, sol, w_kn).float() - ref).norm()
            if err <= 1e-3 * ref.norm():
                screened.append((time_ms(a, sol, 2.0), sol))
        screened.sort()
        choice, t_best = 0, t_default
        for _, sol in screened[:4]:
            t = time_ms(a, sol, 20.0)
            if t < t_best:
                choice, t_best = sol, t
        if t_best > 0.97 * t_default:
            choice, t_best = 0, t_default
        best.append(choice)
        logger.debug(
            "[%d, %d] x %d rows: rocBLAS %.1f, tuned %.1f TFLOPS",
            n,
            k,
            m,
            2 * m * n * k / t_default / 1e9,
            2 * m * n * k / t_best / 1e9,
        )
    return best


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapes", nargs="+", help="N,K of a weight [N, K]")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--kn", action="store_true", help="weights stored [K, N]")
    parser.add_argument("--out", default=str(_user_table_path()))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    for shape in args.shapes:
        n, k = map(int, shape.split(","))
        sols = tune(n, k, dtype, args.kn)
        _save(Path(args.out), {_key(n, k, dtype, args.kn): sols})
        print(shape, sols, flush=True)


if __name__ == "__main__":
    main()
