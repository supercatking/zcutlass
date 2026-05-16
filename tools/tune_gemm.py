#!/usr/bin/env python3
"""Collect zcutlass benchmark JSON for the built-in shape suites."""

import argparse
import json
import pathlib
import subprocess
from typing import Any


def run_suite(bench: pathlib.Path, suite: str, dtype: str, warmup: int, iterations: int) -> list[dict[str, Any]]:
    cmd = [
        str(bench),
        "--suite",
        suite,
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--suite", choices=("smoke", "llm"), default="smoke")
    parser.add_argument("--dtype", choices=("f16", "bf16", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="build/tuning_results.json")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    bench = root / args.build_dir / "zcutlass_bench"
    if not bench.exists():
        bench = root / args.build_dir / "benchmarks" / "zcutlass_bench"
    if not bench.exists():
        raise SystemExit(f"benchmark not found: {bench}")

    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    results: list[dict[str, Any]] = []
    for dtype in dtypes:
        results.extend(run_suite(bench, args.suite, dtype, args.warmup, args.iterations))

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out} with {len(results)} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

