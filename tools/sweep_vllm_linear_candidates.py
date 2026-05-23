#!/usr/bin/env python3
"""Extract unique vLLM Linear callsite candidates from route JSONL logs.

The default mode is intentionally dry-run: it emits one JSON object per unique
callsite candidate with the command that would probe the shape through
check_vllm_linear_method.py. Pass --run to execute the generated commands and
include the measured stock/overlay timing and hit/fallback details.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable


DEFAULT_CHECK_SCRIPT = "/home/zyz/zcutlass/tools/check_vllm_linear_method.py"
KNOWN_FAMILIES = {"decode", "prefill", "large", "fallback"}


@dataclass(frozen=True)
class CandidateKey:
    m: int
    n: int
    k: int
    dtype: str
    bias: bool
    layer: str
    family: str


@dataclass
class CandidateStats:
    files: set[str] = field(default_factory=set)
    rows: int = 0
    routes: collections.Counter[str] = field(default_factory=collections.Counter)
    fallback_reasons: collections.Counter[str] = field(default_factory=collections.Counter)
    latency_us: list[float] = field(default_factory=list)
    kernels: collections.Counter[str] = field(default_factory=collections.Counter)
    selected_configs: collections.Counter[str] = field(default_factory=collections.Counter)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def flatten_paths(groups: list[list[pathlib.Path]]) -> list[pathlib.Path]:
    return [path for group in groups for path in group]


def nested_get(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    problem = row.get("problem")
    if isinstance(problem, dict):
        for key in keys:
            if key in problem and problem[key] not in (None, ""):
                return problem[key]
    tags = row.get("tags")
    if isinstance(tags, dict):
        for key in keys:
            if key in tags and tags[key] not in (None, ""):
                return tags[key]
    return None


def normalize_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"bf16", "bfloat16", "torch.bfloat16"}:
        return "bf16"
    if text in {"f16", "fp16", "float16", "half", "torch.float16", "torch.half"}:
        return "f16"
    return text or "unknown"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_mnk(row: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    m = parse_int(nested_get(row, ("m",)))
    n = parse_int(nested_get(row, ("n",)))
    k = parse_int(nested_get(row, ("k",)))

    input_shape = row.get("input_shape")
    if isinstance(input_shape, list) and len(input_shape) >= 2:
        m = m if m is not None else parse_int(input_shape[0])
        k = k if k is not None else parse_int(input_shape[-1])

    weight_shape = row.get("weight_shape")
    if isinstance(weight_shape, list) and len(weight_shape) >= 2:
        n = n if n is not None else parse_int(weight_shape[0])
        k = k if k is not None else parse_int(weight_shape[-1])

    output_shape = row.get("output_shape")
    if isinstance(output_shape, list) and len(output_shape) >= 2:
        m = m if m is not None else parse_int(output_shape[0])
        n = n if n is not None else parse_int(output_shape[-1])

    return m, n, k


def shape_family(m: int, n: int, k: int) -> str:
    if m <= 16 and n >= 1024 and k >= 1024:
        return "decode"
    if 32 <= m <= 256 and n >= 1024 and k >= 1024:
        return "prefill"
    if m >= 512 and n >= 1024 and k >= 1024:
        return "large"
    return "fallback"


def candidate_from_row(row: dict[str, Any]) -> CandidateKey | None:
    m, n, k = infer_mnk(row)
    if m is None or n is None or k is None:
        return None

    dtype = normalize_dtype(nested_get(row, ("dtype",)))
    bias = parse_bool(nested_get(row, ("bias",)))
    layer = str(
        nested_get(
            row,
            (
                "layer",
                "layer_prefix",
                "callsite",
                "name",
                "layer_class",
            ),
        )
        or "unknown"
    )
    family = str(nested_get(row, ("family", "shape_family", "route_family")) or "").strip()
    if not family:
        family = shape_family(m, n, k)

    return CandidateKey(m=m, n=n, k=k, dtype=dtype, bias=bias, layer=layer, family=family)


def update_stats(stats: CandidateStats, row: dict[str, Any], source: pathlib.Path) -> None:
    stats.files.add(str(source))
    stats.rows += 1

    route = str(row.get("route") or "unknown")
    stats.routes[route] += 1

    fallback_reasons = row.get("fallback_reasons")
    used_fallback_reasons = False
    if isinstance(fallback_reasons, dict):
        for reason, count in fallback_reasons.items():
            stats.fallback_reasons[str(reason)] += int(count or 0)
            used_fallback_reasons = True

    fallback_reason = row.get("fallback_reason")
    if fallback_reason and not used_fallback_reasons:
        stats.fallback_reasons[str(fallback_reason)] += 1

    latency_us = row.get("latency_us")
    if isinstance(latency_us, (int, float)):
        stats.latency_us.append(float(latency_us))

    kernel_name = str(row.get("kernel_name") or row.get("kernel_path") or "")
    if kernel_name:
        stats.kernels[kernel_name] += 1

    selected_config = row.get("selected_config")
    if selected_config:
        stats.selected_configs[json.dumps(selected_config, sort_keys=True, default=str)] += 1


def load_candidates(paths: list[pathlib.Path]) -> tuple[dict[CandidateKey, CandidateStats], dict[str, int]]:
    candidates: dict[CandidateKey, CandidateStats] = {}
    diagnostics: collections.Counter[str] = collections.Counter()
    for path in paths:
        diagnostics["files"] += 1
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            print(f"warning: cannot open route log {path}: {exc}", file=sys.stderr)
            diagnostics["missing_files"] += 1
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                diagnostics["rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warning: invalid JSON in {path}:{line_number}: {exc}", file=sys.stderr)
                    diagnostics["invalid_json"] += 1
                    continue
                if not isinstance(row, dict):
                    diagnostics["non_object_rows"] += 1
                    continue
                key = candidate_from_row(row)
                if key is None:
                    diagnostics["missing_shape_rows"] += 1
                    continue
                update_stats(candidates.setdefault(key, CandidateStats()), row, path)
    diagnostics["candidates"] = len(candidates)
    return candidates, dict(diagnostics)


def priority_key(item: tuple[CandidateKey, CandidateStats]) -> tuple[int, int, int, int, int, int, str]:
    key, stats = item
    family_rank = 0 if key.family == "prefill" else 1
    dtype_rank = 0 if key.dtype == "bf16" else 1
    bias_rank = 0 if key.bias else 1
    known_family_rank = 0 if key.family in KNOWN_FAMILIES else 1
    return (
        family_rank,
        dtype_rank,
        bias_rank,
        known_family_rank,
        -stats.rows,
        key.m * key.n * key.k,
        key.layer,
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * pct)
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min_us": min(values) if values else None,
        "median_us": percentile(values, 0.5),
        "max_us": max(values) if values else None,
    }


def build_command(
    key: CandidateKey,
    *,
    python_executable: str,
    check_script: str,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    require_hit: bool,
    require_fallback: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        python_executable,
        str(check_script),
        "--m",
        str(key.m),
        "--n",
        str(key.n),
        "--k",
        str(key.k),
        "--dtype",
        key.dtype,
        "--allow-family",
        key.family,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--rtol",
        str(rtol),
        "--atol",
        str(atol),
    ]
    if key.bias:
        command.append("--bias")
    if require_hit:
        command.append("--require-hit")
    if require_fallback:
        command.append("--require-fallback")
    command.extend(extra_args)
    return command


def shell_command(command: list[str]) -> str:
    return shlex.join(command)


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def tail_text(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_command(command: list[str], timeout_seconds: float | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "stdout_tail": tail_text(exc.stdout or ""),
            "stderr_tail": tail_text(exc.stderr or ""),
            "result": None,
        }
    except OSError as exc:
        return {
            "status": "error",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "result": None,
        }

    result = extract_json_object(completed.stdout)
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_tail": "" if completed.returncode == 0 else tail_text(completed.stdout),
        "stderr_tail": "" if completed.returncode == 0 else tail_text(completed.stderr),
        "result": result,
    }


def measurement_fields(run: dict[str, Any] | None) -> dict[str, Any]:
    if not run or not isinstance(run.get("result"), dict):
        return {
            "stock_ms": None,
            "overlay_ms": None,
            "speedup": None,
            "hit_rate": None,
            "hits": None,
            "misses": None,
            "fallback_reasons": {},
            "last_trace": {},
        }
    result = run["result"]
    performance = result.get("performance") if isinstance(result.get("performance"), dict) else {}
    routing = result.get("routing") if isinstance(result.get("routing"), dict) else {}
    return {
        "stock_ms": performance.get("stock_ms"),
        "overlay_ms": performance.get("overlay_ms"),
        "speedup": performance.get("speedup_vs_stock"),
        "hit_rate": routing.get("hit_rate"),
        "hits": routing.get("hits"),
        "misses": routing.get("misses"),
        "fallback_reasons": routing.get("fallback_reasons") or {},
        "last_trace": routing.get("last_trace") or {},
    }


def make_summary_row(
    key: CandidateKey,
    stats: CandidateStats,
    command: list[str],
    run: dict[str, Any] | None,
) -> dict[str, Any]:
    measured = measurement_fields(run)
    row = {
        "schema_version": 1,
        "operation": "vllm_linear_candidate_sweep",
        "candidate": {
            "m": key.m,
            "n": key.n,
            "k": key.k,
            "dtype": key.dtype,
            "bias": key.bias,
            "layer": key.layer,
            "family": key.family,
        },
        "route_log": {
            "rows": stats.rows,
            "files": sorted(stats.files),
            "routes": dict(sorted(stats.routes.items())),
            "fallback_reasons": dict(sorted(stats.fallback_reasons.items())),
            "latency": latency_summary(stats.latency_us),
            "kernels": dict(stats.kernels.most_common()),
        },
        "command": command,
        "dry_run_command": shell_command(command),
        "run": {
            "enabled": run is not None,
            "status": run.get("status") if run else "dry_run",
            "returncode": run.get("returncode") if run else None,
        },
        "stock_ms": measured["stock_ms"],
        "overlay_ms": measured["overlay_ms"],
        "speedup": measured["speedup"],
        "routing": {
            "hit_rate": measured["hit_rate"],
            "hits": measured["hits"],
            "misses": measured["misses"],
            "fallback_reasons": measured["fallback_reasons"],
            "last_trace": measured["last_trace"],
        },
    }
    if run and run.get("status") != "success":
        row["run"]["stdout_tail"] = run.get("stdout_tail", "")
        row["run"]["stderr_tail"] = run.get("stderr_tail", "")
    return row


def open_output(path: pathlib.Path | None, force: bool):
    if path is None:
        return None
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing output: {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unique GEMM callsite candidates from vLLM route JSONL logs "
            "and emit dry-run or executed check_vllm_linear_method.py summaries."
        )
    )
    parser.add_argument("--route-log", type=pathlib.Path, nargs="+", action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, help="Optional JSONL summary path. Defaults to stdout.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting --output.")
    parser.add_argument("--limit", type=int, help="Maximum number of sorted candidates to emit.")
    parser.add_argument("--run", action="store_true", help="Execute each generated check command.")
    parser.add_argument("--python", default="python3", help="Python executable for generated commands.")
    parser.add_argument(
        "--check-script",
        default=DEFAULT_CHECK_SCRIPT,
        help="Path to check_vllm_linear_method.py used in generated commands.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--require-hit", action="store_true", help="Pass --require-hit to the check script.")
    parser.add_argument(
        "--require-fallback",
        action="store_true",
        help="Pass --require-fallback to the check script.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per candidate when --run is set.",
    )
    parser.add_argument(
        "--extra-check-arg",
        action="append",
        default=[],
        help="Extra argument appended verbatim to every check command. Repeat as needed.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first executed candidate that fails or times out.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_logs = flatten_paths(args.route_log)
    candidates, diagnostics = load_candidates(route_logs)
    ordered = sorted(candidates.items(), key=priority_key)
    if args.limit is not None:
        ordered = ordered[: args.limit]

    print(
        "loaded "
        f"{diagnostics.get('rows', 0)} rows from {diagnostics.get('files', 0)} files; "
        f"{diagnostics.get('candidates', 0)} unique candidates",
        file=sys.stderr,
    )

    output_handle = open_output(args.output, args.force)
    exit_code = 0
    try:
        for key, stats in ordered:
            command = build_command(
                key,
                python_executable=args.python,
                check_script=args.check_script,
                warmup=args.warmup,
                iterations=args.iterations,
                rtol=args.rtol,
                atol=args.atol,
                require_hit=args.require_hit,
                require_fallback=args.require_fallback,
                extra_args=args.extra_check_arg,
            )
            run = run_command(command, args.timeout_seconds) if args.run else None
            row = make_summary_row(key, stats, command, run)
            text = json.dumps(row, sort_keys=True) + "\n"
            if output_handle is None:
                sys.stdout.write(text)
            else:
                output_handle.write(text)
            if run and run.get("status") != "success":
                exit_code = 1
                if args.stop_on_error:
                    break
    finally:
        if output_handle is not None:
            output_handle.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
