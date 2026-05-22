#!/usr/bin/env python3
"""Gate benchmark candidates against a prior JSONL baseline.

The script compares matching schema-v1 benchmark records for one provider,
usually `zcutlass`, and fails when a candidate run is slower than the configured
per-shape or geomean tolerance. It is intentionally dependency-free so kernel
agents can use it before promoting a dispatch or kernel variant.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Key:
    operation: str
    m: int
    n: int
    k: int
    dtype: str
    layout: str
    alpha: float
    beta: float

    def label(self) -> str:
        return (
            f"{self.operation} {self.dtype} {self.m}x{self.n}x{self.k} "
            f"layout={self.layout} alpha={self.alpha:g} beta={self.beta:g}"
        )


@dataclass(frozen=True)
class Record:
    key: Key
    provider: str
    median_ms: float
    kernel: str


def load_jsonl(path: pathlib.Path, provider: str) -> dict[Key, Record]:
    records: dict[Key, Record] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        if raw.get("provider") != provider or raw.get("status", "success") != "success":
            continue
        problem = raw["problem"]
        performance = raw["performance"]
        key = Key(
            operation=str(problem.get("operation", "gemm")),
            m=int(problem["m"]),
            n=int(problem["n"]),
            k=int(problem["k"]),
            dtype=str(problem.get("dtype") or problem.get("a_type") or "unknown"),
            layout=str(problem.get("layout", "row,row,row,row")),
            alpha=float(problem.get("alpha", 1.0)),
            beta=float(problem.get("beta", 0.0)),
        )
        records[key] = Record(
            key=key,
            provider=provider,
            median_ms=float(performance["median_ms"]),
            kernel=str(raw.get("kernel", "")),
        )
    return records


def fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def write_markdown(path: pathlib.Path, rows: list[tuple[Key, Record, Record, float]]) -> None:
    lines = [
        "# Benchmark Regression Check",
        "",
        "| Shape | Baseline ms | Candidate ms | Baseline / candidate | Candidate kernel |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, baseline, candidate, speed in rows:
        lines.append(
            f"| {key.label()} | {baseline.median_ms:.4f} | {candidate.median_ms:.4f} | "
            f"{speed:.4f}x | `{candidate.kernel}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path, help="Baseline schema-v1 benchmark JSONL.")
    parser.add_argument("candidate", type=pathlib.Path, help="Candidate schema-v1 benchmark JSONL.")
    parser.add_argument("--provider", default="zcutlass")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.05,
        help="Fail if candidate_ms / baseline_ms exceeds this value for any matching shape.",
    )
    parser.add_argument(
        "--min-geomean-speedup",
        type=float,
        default=0.98,
        help="Fail if geomean(baseline_ms / candidate_ms) is below this value.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--markdown", type=pathlib.Path, help="Optional Markdown table output.")
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline, args.provider)
    candidate = load_jsonl(args.candidate, args.provider)
    if not baseline:
        raise SystemExit(f"No baseline records found for provider '{args.provider}' in {args.baseline}")
    if not candidate:
        raise SystemExit(f"No candidate records found for provider '{args.provider}' in {args.candidate}")

    rows: list[tuple[Key, Record, Record, float]] = []
    failures: list[str] = []
    missing = sorted(set(baseline) - set(candidate), key=lambda key: key.label())
    if missing and not args.allow_missing:
        failures.append(f"missing candidate records: {len(missing)}")

    for key in sorted(set(baseline) & set(candidate), key=lambda item: item.label()):
        base = baseline[key]
        cand = candidate[key]
        speed = base.median_ms / cand.median_ms if cand.median_ms > 0 else math.nan
        rows.append((key, base, cand, speed))
        slowdown = cand.median_ms / base.median_ms if base.median_ms > 0 else math.inf
        if slowdown > args.max_slowdown:
            failures.append(
                f"{key.label()} slowed {slowdown:.4f}x "
                f"({base.median_ms:.4f} ms -> {cand.median_ms:.4f} ms)"
            )

    speeds = [speed for _key, _base, _cand, speed in rows if speed > 0 and not math.isnan(speed)]
    geomean = math.exp(sum(math.log(value) for value in speeds) / len(speeds)) if speeds else math.nan
    if math.isnan(geomean) or geomean < args.min_geomean_speedup:
        failures.append(
            f"geomean speedup {fmt(geomean)}x below required {args.min_geomean_speedup:.4f}x"
        )

    print("shape\tbaseline_ms\tcandidate_ms\tbaseline/candidate\tcandidate_kernel")
    for key, base, cand, speed in rows:
        print(f"{key.label()}\t{base.median_ms:.4f}\t{cand.median_ms:.4f}\t{speed:.4f}\t{cand.kernel}")
    print(f"geomean_speedup\t{fmt(geomean)}x")

    if args.markdown:
        write_markdown(args.markdown, rows)
        print(f"wrote {args.markdown}")

    if failures:
        print("benchmark regression gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
