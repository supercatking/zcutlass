#!/usr/bin/env python3
"""Collect zcutlass and official CUTLASS profiler GEMM baselines as JSONL.

The output is schema_version=1 JSONL accepted by visualize_gemm_comparison.py.
This script shells out to an external CUTLASS profiler binary and never vendors
or imports CUTLASS source.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_CUTLASS_PROFILER = pathlib.Path(
    "/home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler"
)
DEFAULT_ROW_MAJOR_CAVEAT = (
    "zcutlass measures row-major A/B/C/D. The CUTLASS profiler baseline uses "
    "row-major A/B and column-major C/D tensor-op instances because the current "
    "official profiler build may not enumerate matching f16/bf16 row-row-row-row "
    "GEMM instances for these shapes."
)


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    dtype: str

    @property
    def label(self) -> str:
        return f"{self.dtype} {self.m}x{self.n}x{self.k}"


@dataclass
class Measurement:
    provider: str
    shape: Shape
    median_ms: float
    tflops: float
    kernel: str
    command: list[str]
    environment: dict[str, Any]
    tags: dict[str, Any]
    status: str = "success"
    stdout: str = ""
    stderr: str = ""


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def run_text(cmd: list[str], cwd: pathlib.Path | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return ""
    return proc.stdout.strip()


def git_value(path: pathlib.Path, *args: str) -> str:
    if not path.exists():
        return ""
    proc = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def find_bench(root: pathlib.Path, build_dir: str) -> pathlib.Path:
    candidates = [
        root / build_dir / "zcutlass_bench",
        root / build_dir / "benchmarks" / "zcutlass_bench",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"zcutlass_bench not found under {root / build_dir}")


def find_cutlass_profiler(path_or_dir: pathlib.Path | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if path_or_dir:
        path_or_dir = path_or_dir.expanduser()
        candidates.append(path_or_dir)
        candidates.extend(
            [
                path_or_dir / "build-profiler" / "tools" / "profiler" / "cutlass_profiler",
                path_or_dir / "build" / "tools" / "profiler" / "cutlass_profiler",
                path_or_dir / "tools" / "profiler" / "cutlass_profiler",
            ]
        )
    candidates.append(DEFAULT_CUTLASS_PROFILER)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    found = shutil.which("cutlass_profiler")
    if found:
        return pathlib.Path(found).resolve()
    raise SystemExit(
        "cutlass_profiler not found; pass --cutlass-profiler or build "
        "/home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler"
    )


def infer_cutlass_source_dir(profiler: pathlib.Path, explicit: pathlib.Path | None) -> pathlib.Path | None:
    if explicit:
        explicit = explicit.expanduser().resolve()
        if (explicit / ".git").exists():
            return explicit
        for parent in explicit.parents:
            if (parent / ".git").exists():
                return parent
    for parent in profiler.parents:
        if parent.name in {"build-profiler", "build"} and (parent.parent / ".git").exists():
            return parent.parent
        if (parent / ".git").exists():
            return parent
    official = pathlib.Path("/home/zyz/cutlass-official")
    return official if (official / ".git").exists() else None


def cutlass_environment(profiler: pathlib.Path, source_dir: pathlib.Path | None) -> dict[str, Any]:
    version = run_text([str(profiler), "--version"])
    env: dict[str, Any] = {
        "cutlass_profiler": str(profiler),
        "cutlass_profiler_version": version,
        "cutlass_build_dir": str(profiler.parents[2]) if len(profiler.parents) >= 3 else "",
    }
    if source_dir:
        env.update(
            {
                "cutlass_source_dir": str(source_dir),
                "cutlass_commit": git_value(source_dir, "rev-parse", "HEAD"),
                "cutlass_describe": git_value(source_dir, "describe", "--always", "--dirty", "--tags"),
            }
        )
    match = re.search(r"built on (.*?) with commit", version)
    if match:
        env["cutlass_profiler_build_time"] = match.group(1)
    return {key: value for key, value in env.items() if value}


def parse_shape_token(token: str, dtype: str) -> Shape:
    match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", token.strip().lower())
    if not match:
        raise SystemExit(f"Invalid shape '{token}', expected MxNxK such as 256x4096x4096")
    return Shape(int(match.group(1)), int(match.group(2)), int(match.group(3)), dtype)


def builtin_shapes(suite: str, dtypes: Iterable[str]) -> list[Shape]:
    if suite == "single":
        raise SystemExit("--suite single requires --shape or --m/--n/--k")
    if suite == "smoke":
        base = [(1, 1024, 1024), (8, 2048, 2048), (64, 4096, 4096), (256, 2048, 8192)]
    elif suite == "correctness":
        base = [(15, 17, 19), (16, 16, 16), (65, 129, 31), (67, 127, 29)]
    elif suite == "square":
        base = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    elif suite == "ragged":
        base = [(63, 127, 65), (65, 129, 127), (127, 255, 129), (257, 511, 255)]
    elif suite in {"llm-decode", "llm-prefill", "llm"}:
        hidden = [1024, 2048, 4096, 8192]
        ms = {
            "llm-decode": [1, 2, 4, 8, 16],
            "llm-prefill": [64, 128, 256, 512, 1024],
            "llm": [1, 8, 16, 64, 256, 1024],
        }[suite]
        base = []
        for h in hidden:
            for m in ms:
                base.extend([(m, h, h), (m, 4 * h, h), (m, h, 4 * h)])
        base.extend([(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)])
    else:
        raise SystemExit(f"Unknown suite '{suite}'")
    return [Shape(m, n, k, dtype) for dtype in dtypes for (m, n, k) in base]


def selected_shapes(args: argparse.Namespace) -> list[Shape]:
    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    if args.shape:
        return [parse_shape_token(token, dtype) for dtype in dtypes for token in args.shape]
    if args.suite == "single":
        return [Shape(args.m, args.n, args.k, dtype) for dtype in dtypes]
    return builtin_shapes(args.suite, dtypes)


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


def load_cutlass_csv(path: pathlib.Path, shape: Shape) -> list[Measurement]:
    measurements: list[Measurement] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            m = parse_float(find_first(row, ["m", "problem_size::m", "problem::m"]))
            n = parse_float(find_first(row, ["n", "problem_size::n", "problem::n"]))
            k = parse_float(find_first(row, ["k", "problem_size::k", "problem::k"]))
            if (m, n, k) != (float(shape.m), float(shape.n), float(shape.k)):
                continue
            runtime = parse_float(
                find_first(row, ["runtime", "runtime_ms", "time", "time_ms", "median_ms", "Runtime"])
            )
            gflops = parse_float(find_first(row, ["GFLOPs", "gflops", "gflop/s", "flops"]))
            tflops = parse_float(find_first(row, ["TFLOPs", "tflops", "tflop/s"]))
            if runtime is None:
                continue
            if tflops is None and gflops is not None:
                tflops = gflops / 1000.0
            if tflops is None:
                flops = 2.0 * shape.m * shape.n * shape.k
                tflops = flops / (runtime * 1.0e-3) / 1.0e12
            kernel = str(find_first(row, ["operation", "Operation", "name", "Name", "kernel", "Kernel"]) or "")
            status = str(find_first(row, ["status", "Status"]) or "success").lower()
            if status in {"passed", "verified", "not_verified"}:
                status = "success"
            measurements.append(
                Measurement("cutlass", shape, runtime, tflops, kernel, [], {}, {}, status=status)
            )
    return measurements


def run_zcutlass(bench: pathlib.Path, shapes: list[Shape], args: argparse.Namespace) -> list[Measurement]:
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
            str(args.warmup),
            "--iterations",
            str(args.iterations),
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
                    cmd,
                    {"zcutlass_repo": str(repo_root())},
                    {"suite": args.suite},
                )
            )
    return measurements


def cutlass_kernel_filter(dtype: str) -> str:
    return "*s1688gemm_f16*tt*align8" if dtype == "f16" else "*s16816gemm_bf16*tt*align8"


def run_cutlass_profiler(
    profiler: pathlib.Path,
    shapes: list[Shape],
    args: argparse.Namespace,
    environment: dict[str, Any],
) -> list[Measurement]:
    measurements: list[Measurement] = []
    for shape in shapes:
        dtype = "f16" if shape.dtype == "f16" else "bf16"
        tmp_base = repo_root() / "build" / f".cutlass_compare_{shape.dtype}_{shape.m}_{shape.n}_{shape.k}"
        tmp = tmp_base.with_suffix(".gemm.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        for old in (tmp, tmp_base.with_suffix(".csv")):
            old.unlink(missing_ok=True)
        layout = args.cutlass_layout
        cmd = [
            str(profiler),
            "--operation=Gemm",
            f"--m={shape.m}",
            f"--n={shape.n}",
            f"--k={shape.k}",
            f"--A={dtype}:row",
            f"--B={dtype}:row",
            f"--C={dtype}:{layout}",
            f"--D={dtype}:{layout}",
            "--accum=f32",
            "--providers=cutlass",
            f"--kernels={args.kernels or cutlass_kernel_filter(shape.dtype)}",
            "--verification-enabled=false",
            "--verbose=false",
            f"--warmup-iterations={args.warmup}",
            f"--profiling-iterations={args.iterations}",
            f"--output={tmp_base}",
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            measurements.append(
                Measurement(
                    "cutlass",
                    shape,
                    0.0,
                    0.0,
                    "",
                    cmd,
                    environment,
                    {"suite": args.suite, "row_major_caveat": DEFAULT_ROW_MAJOR_CAVEAT},
                    status="failed",
                    stdout=proc.stdout.strip(),
                    stderr=proc.stderr.strip(),
                )
            )
            print(f"[warn] CUTLASS profiler failed for {shape.label}: {proc.stderr.strip()}", file=sys.stderr)
            continue
        csv_path = tmp if tmp.exists() else tmp_base.with_suffix(".csv")
        if not csv_path.exists() and proc.stdout.strip():
            csv_path = tmp_base.with_suffix(".csv")
            csv_path.write_text(proc.stdout)
        parsed = load_cutlass_csv(csv_path, shape) if csv_path.exists() else []
        if not parsed:
            print(f"[warn] no CUTLASS CSV rows parsed for {shape.label}", file=sys.stderr)
            continue
        best = max((m for m in parsed if m.status == "success"), key=lambda item: item.tflops, default=None)
        if best is None:
            best = max(parsed, key=lambda item: item.tflops)
        best.command = cmd
        best.environment = environment
        best.tags = {"suite": args.suite, "row_major_caveat": DEFAULT_ROW_MAJOR_CAVEAT}
        measurements.append(best)
    return measurements


def record_for(measurement: Measurement, args: argparse.Namespace) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "problem": {
            "operation": "gemm",
            "m": measurement.shape.m,
            "n": measurement.shape.n,
            "k": measurement.shape.k,
            "dtype": measurement.shape.dtype,
            "layout": "row,row,row,row",
            "alpha": args.alpha,
            "beta": args.beta,
        },
        "provider": measurement.provider,
        "status": measurement.status,
        "kernel": measurement.kernel,
        "performance": {
            "warmup_iterations": args.warmup,
            "profiling_iterations": args.iterations,
            "median_ms": measurement.median_ms,
            "tflops": measurement.tflops,
        },
        "environment": measurement.environment,
        "tags": measurement.tags,
        "command": measurement.command,
    }
    if measurement.provider == "cutlass":
        record["problem"]["cutlass_profiler_layout"] = f"row,row,{args.cutlass_layout},{args.cutlass_layout}"
    if measurement.stdout:
        record["stdout"] = measurement.stdout[-2000:]
    if measurement.stderr:
        record["stderr"] = measurement.stderr[-2000:]
    return record


def write_jsonl(path: pathlib.Path, measurements: list[Measurement], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for measurement in measurements:
            f.write(json.dumps(record_for(measurement, args), sort_keys=True) + "\n")


def print_summary(measurements: list[Measurement]) -> None:
    for m in measurements:
        if m.status != "success":
            print(f"{m.provider:8s} {m.shape.label:22s} {m.status}")
            continue
        print(
            f"{m.provider:8s} {m.shape.label:22s} {m.median_ms:9.4f} ms "
            f"{m.tflops:8.2f} TFLOP/s {m.kernel}"
        )


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--bench", type=pathlib.Path)
    parser.add_argument("--cutlass-dir", type=pathlib.Path, help="CUTLASS source or build directory.")
    parser.add_argument("--cutlass-source-dir", type=pathlib.Path)
    parser.add_argument("--cutlass-profiler", type=pathlib.Path)
    parser.add_argument("--suite", default="single")
    parser.add_argument("--shape", action="append", help="Explicit shape MxNxK; may be repeated.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=("f16", "bf16", "both"), default="f16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--providers", default="zcutlass,cutlass", help="Comma list: zcutlass,cutlass")
    parser.add_argument("--kernels", help="CUTLASS profiler --kernels filter.")
    parser.add_argument("--cutlass-layout", choices=("column", "row"), default="column")
    parser.add_argument("--output", type=pathlib.Path, default=root / "build" / "cutlass_baseline.jsonl")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be >= 0 and --iterations must be > 0")

    providers = {item.strip() for item in args.providers.split(",") if item.strip()}
    shapes = selected_shapes(args)
    measurements: list[Measurement] = []

    if "zcutlass" in providers:
        bench = args.bench or find_bench(root, args.build_dir)
        measurements.extend(run_zcutlass(bench, shapes, args))

    if "cutlass" in providers:
        profiler_hint = args.cutlass_profiler or args.cutlass_dir
        profiler = find_cutlass_profiler(profiler_hint)
        source_dir = infer_cutlass_source_dir(profiler, args.cutlass_source_dir or args.cutlass_dir)
        environment = cutlass_environment(profiler, source_dir)
        measurements.extend(run_cutlass_profiler(profiler, shapes, args, environment))

    if not measurements:
        raise SystemExit("No measurements collected. Check --providers.")
    write_jsonl(args.output, measurements, args)
    if args.summary:
        print_summary(measurements)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
