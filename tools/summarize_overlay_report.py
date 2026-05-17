#!/usr/bin/env python3
"""Summarize PyTorch overlay JSONL reports.

The benchmark harnesses emit one stock record and one zcutlass_overlay record
for each callsite. This helper pairs them by problem and tags, then prints the
latency speedup plus routing telemetry that matters for v1.5 promotion gates.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Any


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    problem = record["problem"]
    tags = record["tags"]
    return (
        problem.get("operation"),
        problem.get("dtype"),
        problem.get("m"),
        problem.get("n"),
        problem.get("k"),
        tags.get("suite"),
        tags.get("layer"),
        tags.get("module"),
        tags.get("callsite"),
    )


def label_for(record: dict[str, Any]) -> str:
    problem = record["problem"]
    tags = record["tags"]
    layer = tags.get("layer") or "-"
    module = tags.get("module") or problem.get("operation")
    dtype = problem.get("dtype")
    if problem.get("n") is not None and problem.get("k") is not None:
        return f"{layer}/{module} {dtype} {problem['m']}x{problem['n']}x{problem['k']}"
    if problem.get("hidden") is not None and problem.get("intermediate") is not None:
        return f"{layer}/{module} {dtype} m={problem['m']} h={problem['hidden']} i={problem['intermediate']}"
    if problem.get("m") is None:
        return f"{layer}/{module} {dtype}"
    return f"{layer}/{module} {dtype} m={problem['m']}"


def summarize(path: pathlib.Path) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in load_records(path):
        groups[record_key(record)][record["provider"]] = record

    rows: list[dict[str, Any]] = []
    for providers in groups.values():
        stock = providers.get("pytorch_stock")
        overlay = providers.get("zcutlass_overlay")
        if stock is None or overlay is None:
            continue
        stock_ms = float(stock["performance"]["median_ms"])
        overlay_ms = float(overlay["performance"]["median_ms"])
        tags = overlay["tags"]
        rows.append(
            {
                "label": label_for(overlay),
                "stock_ms": stock_ms,
                "overlay_ms": overlay_ms,
                "speedup": stock_ms / overlay_ms if overlay_ms > 0 else 0.0,
                "hit_rate": float(tags.get("hit_rate", 0.0)),
                "kernel_path": tags.get("kernel_path", "aggregate"),
                "fallback_reason": tags.get("fallback_reason"),
                "fallback_reasons": tags.get("fallback_reasons", {}),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()

    rows = summarize(args.jsonl)
    if not rows:
        raise SystemExit(f"No stock/overlay pairs found in {args.jsonl}")

    lines = [
        "| Callsite | Stock ms | Overlay ms | Speedup | Hit rate | Path | Fallback |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        fallback = row["fallback_reason"] or json.dumps(row["fallback_reasons"], sort_keys=True)
        if fallback == "{}":
            fallback = "-"
        lines.append(
            f"| {row['label']} | {row['stock_ms']:.4f} | {row['overlay_ms']:.4f} | "
            f"{row['speedup']:.3f}x | {row['hit_rate']:.2f} | {row['kernel_path']} | {fallback} |"
        )

    text = "\n".join(lines)
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n")
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
