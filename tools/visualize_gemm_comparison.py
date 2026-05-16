#!/usr/bin/env python3
"""Generate an HTML report comparing zcutlass GEMM against CUTLASS.

The preferred flow is:

  1. Run zcutlass_bench to collect zcutlass measurements.
  2. Provide CUTLASS baseline measurements as zcutlass-schema JSONL or profiler CSV.
  3. Generate a self-contained HTML report with latency, TFLOP/s, and speedup charts.

The script intentionally keeps dependencies to Python's standard library so it
works in a fresh WSL checkout.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    dtype: str
    alpha: float = 1.0
    beta: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.dtype} {self.m}x{self.n}x{self.k}"


@dataclass
class Measurement:
    provider: str
    shape: Shape
    median_ms: float
    tflops: float
    kernel: str = ""
    status: str = "success"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_bench_path(root: pathlib.Path) -> pathlib.Path:
    candidates = [
        root / "build" / "zcutlass_bench",
        root / "build" / "benchmarks" / "zcutlass_bench",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_shape_token(token: str, dtype: str) -> Shape:
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", token.strip().lower())
    if not match:
        raise SystemExit(f"Invalid shape '{token}', expected MxNxK such as 256x4096x4096")
    return Shape(int(match.group(1)), int(match.group(2)), int(match.group(3)), dtype)


def builtin_shapes(suite: str, dtypes: Iterable[str]) -> list[Shape]:
    base: list[tuple[int, int, int]]
    if suite == "smoke":
        base = [(1, 1024, 1024), (8, 2048, 2048), (64, 4096, 4096), (256, 2048, 8192)]
    elif suite == "correctness":
        base = [(15, 17, 19), (16, 16, 16), (65, 129, 31), (67, 127, 29)]
    elif suite == "square":
        base = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    elif suite == "ragged":
        base = [(63, 127, 65), (65, 129, 127), (127, 255, 129), (257, 511, 255)]
    elif suite in {"llm-v1.5", "llm-canonical"}:
        base = [
            (8, 4096, 4096),
            (128, 4096, 4096),
            (128, 16384, 4096),
            (128, 4096, 16384),
            (4096, 4096, 4096),
        ]
    elif suite in {"llm-decode", "llm-prefill", "llm"}:
        hs = [1024, 2048, 4096, 8192]
        if suite == "llm-decode":
            ms = [1, 2, 4, 8, 16]
        elif suite == "llm-prefill":
            ms = [64, 128, 256, 512, 1024]
        else:
            ms = [1, 8, 16, 64, 256, 1024]
        base = []
        for h in hs:
            for m in ms:
                base.extend([(m, h, h), (m, 4 * h, h), (m, h, 4 * h)])
    else:
        raise SystemExit(f"Unknown suite '{suite}'")

    return [Shape(m, n, k, dtype) for dtype in dtypes for (m, n, k) in base]


def load_jsonl(path: pathlib.Path) -> list[Measurement]:
    measurements: list[Measurement] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
        problem = record["problem"]
        perf = record["performance"]
        measurements.append(
            Measurement(
                provider=record.get("provider", "unknown"),
                shape=Shape(
                    int(problem["m"]),
                    int(problem["n"]),
                    int(problem["k"]),
                    str(problem.get("dtype") or problem.get("a_type") or "unknown"),
                    float(problem.get("alpha", 1.0)),
                    float(problem.get("beta", 0.0)),
                ),
                median_ms=float(perf.get("median_ms", perf.get("runtime_ms"))),
                tflops=float(perf.get("tflops", 0.0)),
                kernel=str(record.get("kernel", "")),
                status=str(record.get("status", "success")),
            )
        )
    return measurements


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def find_first(row: dict[str, Any], names: Iterable[str]) -> Any:
    lower = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def load_cutlass_csv(path: pathlib.Path, dtype: str) -> list[Measurement]:
    measurements: list[Measurement] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            m = parse_float(find_first(row, ["m", "M"]))
            n = parse_float(find_first(row, ["n", "N"]))
            k = parse_float(find_first(row, ["k", "K"]))
            if m is None or n is None or k is None:
                continue
            runtime = parse_float(
                find_first(row, ["runtime", "runtime_ms", "time", "time_ms", "median_ms", "Runtime"])
            )
            gflops = parse_float(find_first(row, ["GFLOPs", "gflops", "gflop/s", "flops"]))
            tflops = parse_float(find_first(row, ["TFLOPs", "tflops", "tflop/s"]))
            if runtime is None:
                continue
            # CUTLASS profiler commonly reports runtime in milliseconds and GFLOP/s.
            if tflops is None and gflops is not None:
                tflops = gflops / 1000.0
            if tflops is None:
                flops = 2.0 * m * n * k
                tflops = flops / (runtime * 1.0e-3) / 1.0e12
            kernel = str(find_first(row, ["Operation", "operation", "Name", "name", "Kernel", "kernel"]) or "")
            measurements.append(
                Measurement("cutlass", Shape(int(m), int(n), int(k), dtype), runtime, tflops, kernel)
            )
    return measurements


def run_zcutlass(bench: pathlib.Path, shapes: list[Shape], warmup: int, iterations: int) -> list[Measurement]:
    measurements: list[Measurement] = []
    for shape in shapes:
        cmd = [
            str(bench),
            "--m",
            str(shape.m),
            "--n",
            str(shape.n),
            "--k",
            str(shape.k),
            "--dtype",
            shape.dtype,
            "--providers",
            "zcutlass",
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--json",
        ]
        proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for line in proc.stdout.splitlines():
            if not line.strip().startswith("{"):
                continue
            record = json.loads(line)
            measurements.append(
                Measurement(
                    "zcutlass",
                    shape,
                    float(record["zcutlass_ms"]),
                    float(record["zcutlass_tflops"]),
                    str(record.get("kernel", "")),
                )
            )
    return measurements


def run_cutlass_profiler(profiler: pathlib.Path, shapes: list[Shape], warmup: int, iterations: int) -> list[Measurement]:
    measurements: list[Measurement] = []
    for shape in shapes:
        dtype = "f16" if shape.dtype == "f16" else "bf16"
        c_dtype = dtype
        tmp_base = repo_root() / "build" / f".cutlass_{shape.dtype}_{shape.m}_{shape.n}_{shape.k}"
        tmp = tmp_base.with_suffix(".csv")
        tmp_gemm = tmp_base.with_suffix(".gemm.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        kernel_filter = "*s1688gemm_f16*tt*align8" if shape.dtype == "f16" else "*s16816gemm_bf16*tt*align8"
        cmd = [
            str(profiler),
            "--operation=Gemm",
            f"--m={shape.m}",
            f"--n={shape.n}",
            f"--k={shape.k}",
            f"--A={dtype}:row",
            f"--B={dtype}:row",
            f"--C={c_dtype}:column",
            f"--D={c_dtype}:column",
            "--accum=f32",
            "--providers=cutlass",
            f"--kernels={kernel_filter}",
            "--verification-enabled=false",
            "--verbose=false",
            f"--warmup-iterations={warmup}",
            f"--profiling-iterations={iterations}",
            f"--output={tmp_base}",
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print(f"[warn] CUTLASS profiler failed for {shape.label}: {proc.stderr.strip()}", file=sys.stderr)
            continue
        if tmp_gemm.exists():
            tmp = tmp_gemm
        elif not tmp.exists() and proc.stdout.strip():
            tmp.write_text(proc.stdout)
        parsed = load_cutlass_csv(tmp, shape.dtype)
        if parsed:
            best = max(parsed, key=lambda item: item.tflops)
            best.shape = shape
            measurements.append(best)
    return measurements


def best_by_shape(measurements: list[Measurement], provider: str) -> dict[Shape, Measurement]:
    best: dict[Shape, Measurement] = {}
    for measurement in measurements:
        if measurement.provider != provider or measurement.status != "success":
            continue
        current = best.get(measurement.shape)
        if current is None or measurement.tflops > current.tflops:
            best[measurement.shape] = measurement
    return best


def bar(width: float, max_width: int = 180) -> str:
    if math.isnan(width) or width <= 0:
        return ""
    return f'<div class="bar" style="width:{min(width * max_width, max_width):.1f}px"></div>'


def make_report(zcutlass: list[Measurement], cutlass: list[Measurement], output: pathlib.Path, title: str) -> None:
    zbest = best_by_shape(zcutlass, "zcutlass")
    cbest = best_by_shape(cutlass, "cutlass")
    shapes = sorted(set(zbest) | set(cbest), key=lambda s: (s.dtype, s.m, s.n, s.k))
    rows = []
    speedups = []
    max_tflops = max([m.tflops for m in zbest.values()] + [m.tflops for m in cbest.values()] + [1.0])
    for shape in shapes:
        z = zbest.get(shape)
        c = cbest.get(shape)
        speedup = (c.median_ms / z.median_ms) if z and c and z.median_ms > 0 else math.nan
        if not math.isnan(speedup):
            speedups.append(speedup)
        rows.append((shape, z, c, speedup))

    geomean = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else math.nan
    wins = sum(1 for x in speedups if x > 1.0)
    losses = sum(1 for x in speedups if x <= 1.0)

    table_rows = []
    for shape, z, c, speedup in rows:
        z_ms = f"{z.median_ms:.4f}" if z else "missing"
        c_ms = f"{c.median_ms:.4f}" if c else "missing"
        z_tf = f"{z.tflops:.2f}" if z else "missing"
        c_tf = f"{c.tflops:.2f}" if c else "missing"
        speed = f"{speedup:.3f}x" if not math.isnan(speedup) else "n/a"
        z_bar = bar((z.tflops / max_tflops) if z else 0)
        c_bar = bar((c.tflops / max_tflops) if c else 0)
        verdict = "win" if not math.isnan(speedup) and speedup > 1 else "behind"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(shape.label)}</td>"
            f"<td>{z_ms}</td><td>{c_ms}</td><td class='{verdict}'>{speed}</td>"
            f"<td>{z_tf} {z_bar}</td><td>{c_tf} {c_bar}</td>"
            f"<td>{html.escape(z.kernel if z else '')}</td>"
            f"<td>{html.escape(c.kernel if c else '')}</td>"
            "</tr>"
        )

    speed_points = []
    for idx, (shape, _z, _c, speedup) in enumerate(rows):
        if math.isnan(speedup):
            continue
        x = 70 + idx * 34
        y = 220 - min(speedup, 2.0) * 90
        color = "#11805a" if speedup >= 1.0 else "#bb3e03"
        speed_points.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{color}"><title>{html.escape(shape.label)} {speedup:.3f}x</title></circle>')
    svg_width = max(720, 120 + len(rows) * 34)

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 20px 0; }}
    .card {{ border: 1px solid #d8dee4; border-radius: 8px; padding: 12px 16px; min-width: 150px; }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: middle; }}
    th {{ background: #f6f8fa; position: sticky; top: 0; }}
    .bar {{ display: inline-block; height: 8px; background: #2f80ed; border-radius: 4px; margin-left: 8px; }}
    .win {{ color: #11805a; font-weight: 700; }}
    .behind {{ color: #bb3e03; font-weight: 700; }}
    .note {{ color: #52606d; }}
    svg {{ width: 100%; max-width: {svg_width}px; height: 250px; border: 1px solid #e5e7eb; border-radius: 8px; background: white; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="note">Speedup is CUTLASS latency / zcutlass latency. Values above 1.0 mean zcutlass is faster.</div>
  <div class="cards">
    <div class="card"><div>Compared Shapes</div><div class="metric">{len(speedups)}</div></div>
    <div class="card"><div>Geomean Speedup</div><div class="metric">{geomean:.3f}x</div></div>
    <div class="card"><div>zcutlass Wins</div><div class="metric">{wins}</div></div>
    <div class="card"><div>zcutlass Behind</div><div class="metric">{losses}</div></div>
  </div>
  <h2>Speedup Scatter</h2>
  <svg viewBox="0 0 {svg_width} 250" role="img">
    <line x1="50" y1="220" x2="{svg_width - 20}" y2="220" stroke="#cbd5e1"/>
    <line x1="50" y1="130" x2="{svg_width - 20}" y2="130" stroke="#11805a" stroke-dasharray="4 4"/>
    <text x="8" y="134" fill="#11805a">1.0x</text>
    <text x="8" y="44" fill="#52606d">2.0x</text>
    <line x1="50" y1="40" x2="{svg_width - 20}" y2="40" stroke="#e5e7eb"/>
    {''.join(speed_points)}
  </svg>
  <h2>Shape Results</h2>
  <table>
    <thead>
      <tr><th>Shape</th><th>zcutlass ms</th><th>CUTLASS ms</th><th>Speedup</th><th>zcutlass TFLOP/s</th><th>CUTLASS TFLOP/s</th><th>zcutlass kernel</th><th>CUTLASS kernel</th></tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)


def write_jsonl(path: pathlib.Path, measurements: list[Measurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for m in measurements:
            record = {
                "schema_version": 1,
                "problem": {
                    "operation": "gemm",
                    "m": m.shape.m,
                    "n": m.shape.n,
                    "k": m.shape.k,
                    "dtype": m.shape.dtype,
                    "layout": "row,row,row,row",
                    "alpha": m.shape.alpha,
                    "beta": m.shape.beta,
                },
                "provider": m.provider,
                "status": m.status,
                "kernel": m.kernel,
                "performance": {"median_ms": m.median_ms, "tflops": m.tflops},
            }
            f.write(json.dumps(record) + "\n")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--shape", action="append", help="Explicit shape MxNxK; may be repeated.")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16", "both"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bench", type=pathlib.Path, default=default_bench_path(root))
    parser.add_argument("--zcutlass-jsonl", type=pathlib.Path)
    parser.add_argument("--cutlass-jsonl", type=pathlib.Path)
    parser.add_argument("--cutlass-csv", type=pathlib.Path)
    parser.add_argument("--cutlass-profiler", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=root / "build" / "gemm_comparison.html")
    parser.add_argument("--save-jsonl", type=pathlib.Path, default=root / "build" / "gemm_comparison.jsonl")
    parser.add_argument("--title", default="zcutlass GEMM vs NVIDIA CUTLASS")
    args = parser.parse_args()

    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    shapes = (
        [parse_shape_token(token, dtype) for dtype in dtypes for token in args.shape]
        if args.shape
        else builtin_shapes(args.suite, dtypes)
    )

    zcutlass = load_jsonl(args.zcutlass_jsonl) if args.zcutlass_jsonl else run_zcutlass(args.bench, shapes, args.warmup, args.iterations)

    cutlass: list[Measurement] = []
    if args.cutlass_jsonl:
        cutlass.extend(load_jsonl(args.cutlass_jsonl))
        for m in cutlass:
            if m.provider != "cutlass":
                m.provider = "cutlass"
    if args.cutlass_csv:
        # CSV does not always encode dtype in a stable way, so use the selected dtype
        # unless the user provides JSONL.
        csv_dtype = dtypes[0] if len(dtypes) == 1 else "f16"
        cutlass.extend(load_cutlass_csv(args.cutlass_csv, csv_dtype))
    if args.cutlass_profiler:
        cutlass.extend(run_cutlass_profiler(args.cutlass_profiler, shapes, args.warmup, args.iterations))
    if not cutlass:
        raise SystemExit("Provide --cutlass-jsonl, --cutlass-csv, or --cutlass-profiler for the CUTLASS baseline")

    write_jsonl(args.save_jsonl, zcutlass + cutlass)
    make_report(zcutlass, cutlass, args.output, args.title)
    print(f"wrote {args.output}")
    print(f"wrote {args.save_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
