#!/usr/bin/env python3
"""Compare one zcutlass shape against cuBLAS and, optionally, CUTLASS profiler."""

import argparse
import json
import pathlib
import shutil
import subprocess
from typing import Any


def find_bench(root: pathlib.Path, build_dir: str) -> pathlib.Path:
    candidates = [
        root / build_dir / "zcutlass_bench",
        root / build_dir / "benchmarks" / "zcutlass_bench",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"zcutlass_bench not found under {root / build_dir}")


def find_cutlass_profiler(cutlass_dir: pathlib.Path) -> pathlib.Path:
    candidates = [
        cutlass_dir / "build" / "tools" / "profiler" / "cutlass_profiler",
        cutlass_dir / "build" / "tools" / "profiler" / "cutlass_profiler.exe",
        cutlass_dir / "tools" / "profiler" / "cutlass_profiler",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which("cutlass_profiler")
    if found:
        return pathlib.Path(found)
    raise SystemExit("cutlass_profiler not found; build CUTLASS profiler first or add it to PATH")


def run_zcutlass(bench: pathlib.Path, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        str(bench),
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_cutlass(profiler: pathlib.Path, args: argparse.Namespace) -> int:
    # CUTLASS profiler argument syntax is kept isolated here so the library never
    # depends on CUTLASS headers or source.
    dtype = "f16" if args.dtype == "f16" else "bf16"
    cmd = [
        str(profiler),
        "--operation=Gemm",
        f"--m={args.m}",
        f"--n={args.n}",
        f"--k={args.k}",
        f"--A={dtype}:row",
        f"--B={dtype}:row",
        f"--C={dtype}:row",
        "--accum=f32",
        "--providers=cutlass",
    ]
    print("CUTLASS profiler command:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--cutlass-dir")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    bench = find_bench(root, args.build_dir)
    result = run_zcutlass(bench, args)
    print(json.dumps(result, indent=2))

    if args.cutlass_dir:
        profiler = find_cutlass_profiler(pathlib.Path(args.cutlass_dir).expanduser().resolve())
        return run_cutlass(profiler, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

