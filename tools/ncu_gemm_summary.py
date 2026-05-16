#!/usr/bin/env python3
"""Summarize Nsight Compute CSV for one GEMM profile."""

import argparse
import csv
import json
import pathlib
import re
from typing import Any


THROUGHPUT_METRICS = {
    "sm": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_active",
    ),
    "tensor": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum.pct_of_peak_sustained_active",
    ),
    "dram": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_active",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
}

OCCUPANCY_METRICS = {
    "achieved": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ),
    "theoretical": (
        "sm__maximum_warps_avg_per_active_cycle_pct",
        "launch__occupancy_limit_registers",
    ),
}

RESOURCE_METRICS = {
    "registers_per_thread": (
        "launch__registers_per_thread",
        "launch__registers_per_thread_allocated",
    ),
    "shared_memory_per_block": (
        "launch__shared_mem_per_block_allocated",
        "launch__shared_mem_per_block",
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_mem_per_block_static",
    ),
    "block_size": ("launch__block_size", "launch__threads_per_block"),
    "grid_size": ("launch__grid_size", "launch__blocks_per_grid"),
}

STALL_RE = re.compile(r"(?:smsp__)?(?:average_)?warps?_issue_stalled_(.+?)(?:_per_warp_active)?(?:\.|$)")


def parse_number(value: str) -> float | int | str | None:
    value = value.strip().replace(",", "")
    if not value or value.upper() == "N/A":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def normalize_unit(unit: str) -> str:
    return unit.strip().replace("byte", "B").replace("cycle", "cycle")


def metric_value(row: dict[str, str]) -> dict[str, Any]:
    unit = row.get("Metric Unit", "") or row.get("Unit", "")
    value = row.get("Metric Value", "") or row.get("Value", "")
    return {"value": parse_number(value), "unit": normalize_unit(unit)}


def row_metric_name(row: dict[str, str]) -> str:
    return (row.get("Metric Name") or row.get("Name") or row.get("Metric") or "").strip()


def row_kernel_name(row: dict[str, str]) -> str:
    return (row.get("Kernel Name") or row.get("Kernel") or "").strip()


def read_ncu_csv(path: pathlib.Path) -> list[dict[str, str]]:
    text = path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("==")]
    if not lines:
        return []

    header_index = 0
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            header_index = index
            break
    reader = csv.DictReader(lines[header_index:])
    return [dict(row) for row in reader if row]


def read_ncu_wide_csv(path: pathlib.Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    text = path.read_text(errors="replace")
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("==")]
    if len(lines) < 3:
        return {}, []

    rows = list(csv.reader(lines))
    header = rows[0]
    if "Metric Name" in header and "Metric Value" in header:
        return {}, []

    units = rows[1]
    metrics: dict[str, dict[str, Any]] = {}
    kernels: list[str] = []
    kernel_index = header.index("Kernel Name") if "Kernel Name" in header else -1
    for values in rows[2:]:
        if not any(cell.strip() for cell in values):
            continue
        if kernel_index >= 0 and kernel_index < len(values):
            kernel = values[kernel_index].strip()
            if kernel and kernel not in kernels:
                kernels.append(kernel)
        for index, name in enumerate(header):
            name = name.strip()
            if not name or index >= len(values):
                continue
            value = values[index]
            unit = units[index] if index < len(units) else ""
            metrics[name] = {"value": parse_number(value), "unit": normalize_unit(unit)}
    return metrics, kernels


def first_metric(metrics: dict[str, dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        if name in metrics:
            return {"metric": name, **metrics[name]}
    return None


def summarize_csv(path: pathlib.Path, shape: dict[str, Any] | None = None, top_stalls: int = 8) -> dict[str, Any]:
    metrics, kernels = read_ncu_wide_csv(path)
    if not metrics:
        rows = read_ncu_csv(path)
        for row in rows:
            name = row_metric_name(row)
            if not name:
                continue
            metrics[name] = metric_value(row)
            kernel = row_kernel_name(row)
            if kernel and kernel not in kernels:
                kernels.append(kernel)

    throughput = {
        key: first_metric(metrics, metric_names)
        for key, metric_names in THROUGHPUT_METRICS.items()
    }
    occupancy = {
        key: first_metric(metrics, metric_names)
        for key, metric_names in OCCUPANCY_METRICS.items()
    }
    resources = {
        key: first_metric(metrics, metric_names)
        for key, metric_names in RESOURCE_METRICS.items()
    }

    stalls = []
    for name, value in metrics.items():
        match = STALL_RE.search(name)
        if not match or not isinstance(value.get("value"), (int, float)):
            continue
        reason = match.group(1)
        reason = re.sub(r"(_per_issue_active|_per_warp_active|\.ratio)$", "", reason)
        stalls.append({"reason": reason.replace("_", " "), "metric": name, **value})
    stalls.sort(key=lambda item: float(item["value"]), reverse=True)

    return {
        "source": str(path),
        "shape": shape or {},
        "kernels": kernels,
        "throughput_pct_of_peak": throughput,
        "occupancy": occupancy,
        "resources": resources,
        "top_stalls": stalls[:top_stalls],
        "metric_count": len(metrics),
    }


def format_metric(item: dict[str, Any] | None) -> str:
    if not item:
        return "n/a"
    value = item.get("value")
    unit = item.get("unit", "")
    if value is None:
        return "n/a"
    if isinstance(value, float):
        rendered = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        rendered = str(value)
    return f"{rendered} {unit}".strip()


def summary_markdown(summary: dict[str, Any]) -> str:
    shape = summary.get("shape") or {}
    title_shape = " ".join(f"{key}={value}" for key, value in shape.items()) or "unknown shape"
    lines = [f"# Nsight Compute GEMM Summary: {title_shape}", ""]
    if summary.get("kernels"):
        lines.extend(["## Kernels", ""])
        lines.extend(f"- `{kernel}`" for kernel in summary["kernels"])
        lines.append("")

    lines.extend(["## Throughput", "", "| Signal | Value | Metric |", "| --- | ---: | --- |"])
    for name, item in summary.get("throughput_pct_of_peak", {}).items():
        metric = item.get("metric") if item else ""
        lines.append(f"| {name.upper()} | {format_metric(item)} | `{metric}` |")

    lines.extend(["", "## Occupancy And Resources", "", "| Signal | Value | Metric |", "| --- | ---: | --- |"])
    combined = {}
    combined.update(summary.get("occupancy", {}))
    combined.update(summary.get("resources", {}))
    for name, item in combined.items():
        metric = item.get("metric") if item else ""
        lines.append(f"| {name.replace('_', ' ')} | {format_metric(item)} | `{metric}` |")

    lines.extend(["", "## Top Stalls", "", "| Reason | Value | Metric |", "| --- | ---: | --- |"])
    for stall in summary.get("top_stalls", []):
        lines.append(f"| {stall['reason']} | {format_metric(stall)} | `{stall['metric']}` |")
    if not summary.get("top_stalls"):
        lines.append("| n/a | n/a |  |")
    lines.append("")
    return "\n".join(lines)


def write_summary(
    summary: dict[str, Any],
    summary_json: pathlib.Path | None = None,
    summary_md: pathlib.Path | None = None,
) -> None:
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    if summary_md:
        summary_md.parent.mkdir(parents=True, exist_ok=True)
        summary_md.write_text(summary_markdown(summary))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--top-stalls", type=int, default=8)
    parser.add_argument("--summary-json", type=pathlib.Path)
    parser.add_argument("--summary-md", type=pathlib.Path)
    args = parser.parse_args()

    shape = {
        key: value
        for key, value in {"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype}.items()
        if value is not None
    }
    summary = summarize_csv(args.csv, shape=shape, top_stalls=args.top_stalls)
    write_summary(summary, summary_json=args.summary_json, summary_md=args.summary_md)
    if not args.summary_json and not args.summary_md:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
