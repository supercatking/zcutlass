#!/usr/bin/env python3
"""Summarize zcutlass benchmark JSONL by problem shape."""

import argparse
import collections
import json
import math
import pathlib
from typing import Any


def problem_key(record: dict[str, Any]) -> tuple[Any, ...]:
    problem = record["problem"]
    return (
        problem.get("operation", "gemm"),
        problem["m"],
        problem["n"],
        problem["k"],
        problem["dtype"],
        problem.get("layout", "row,row,row,row"),
        problem.get("alpha", 1.0),
        problem.get("beta", 0.0),
    )


def label(key: tuple[Any, ...]) -> str:
    operation, m, n, k, dtype, layout, alpha, beta = key
    return f"{operation} m={m} n={n} k={k} dtype={dtype} layout={layout} alpha={alpha} beta={beta}"


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=pathlib.Path)
    parser.add_argument("--baseline", default="cublas")
    parser.add_argument("--candidate", default="zcutlass")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for record in load_records(args.jsonl):
        grouped[problem_key(record)][record["provider"]] = record

    rows = []
    for key, providers in grouped.items():
        candidate = providers.get(args.candidate)
        baseline = providers.get(args.baseline)
        if candidate is None:
            continue
        candidate_ms = candidate["performance"]["median_ms"]
        baseline_ms = baseline["performance"]["median_ms"] if baseline else math.nan
        speedup = baseline_ms / candidate_ms if baseline and candidate_ms else math.nan
        rows.append((speedup, key, candidate, baseline))

    rows.sort(key=lambda row: (-math.inf if math.isnan(row[0]) else row[0]), reverse=True)
    if args.top_k > 0:
        rows = rows[: args.top_k]

    print("speedup\tcandidate_ms\tbaseline_ms\tcandidate_kernel\tproblem")
    for speedup, key, candidate, baseline in rows:
        candidate_ms = candidate["performance"]["median_ms"]
        baseline_ms = baseline["performance"]["median_ms"] if baseline else math.nan
        print(
            f"{speedup:.4f}\t{candidate_ms:.4f}\t{baseline_ms:.4f}\t"
            f"{candidate.get('kernel', '')}\t{label(key)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
