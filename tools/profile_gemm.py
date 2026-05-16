#!/usr/bin/env python3
"""Run Nsight Compute for one zcutlass GEMM benchmark shape."""

import argparse
import pathlib
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--section", default="SpeedOfLight")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    bench = root / args.build_dir / "zcutlass_bench"
    if not bench.exists():
        bench = root / args.build_dir / "benchmarks" / "zcutlass_bench"
    if not bench.exists():
        raise SystemExit(f"benchmark not found: {bench}")

    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--section",
        args.section,
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
        "3",
        "--iterations",
        "5",
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())

