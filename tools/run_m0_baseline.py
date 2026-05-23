#!/usr/bin/env python3
"""Run the zcutlass v1.5 M0 baseline evidence chain."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Step:
    name: str
    command: list[str]
    required: bool = True


@dataclass
class Result:
    name: str
    command: list[str]
    returncode: int
    stdout_path: pathlib.Path
    stderr_path: pathlib.Path


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_step(step: Step, cwd: pathlib.Path, output_dir: pathlib.Path) -> Result:
    safe_name = step.name.replace(" ", "_").replace("/", "_")
    stdout_path = output_dir / f"{safe_name}.stdout.txt"
    stderr_path = output_dir / f"{safe_name}.stderr.txt"
    print(f"[m0] {step.name}: {quote(step.command)}", flush=True)
    proc = subprocess.run(
        step.command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    print(f"[m0] {step.name}: exit {proc.returncode}", flush=True)
    return Result(step.name, step.command, proc.returncode, stdout_path, stderr_path)


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def kernel_summary_lines(path: pathlib.Path) -> list[str]:
    records = [record for record in load_jsonl(path) if record.get("provider") == "zcutlass"]
    if not records:
        return []
    lines = [
        "## Kernel Baseline Summary",
        "",
        "| DType | Shape | Kernel | Path | ms | TFLOP/s |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in records:
        problem = record["problem"]
        perf = record["performance"]
        tags = record.get("tags", {})
        shape = f"{problem['m']}x{problem['n']}x{problem['k']}"
        lines.append(
            f"| {problem.get('dtype', 'unknown')} | {shape} | `{record.get('kernel', '')}` | "
            f"{tags.get('kernel_path', '')} | {float(perf['median_ms']):.4f} | "
            f"{float(perf['tflops']):.4f} |"
        )
    return lines


def vllm_summary_lines(path: pathlib.Path) -> list[str]:
    records = [
        record for record in load_jsonl(path) if record.get("provider") == "zcutlass_vllm_overlay"
    ]
    if not records:
        return []
    lines = [
        "## vLLM LinearMethod Summary",
        "",
        "| DType | Case | Shape | Hit rate | Kernel | Speedup vs stock |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for record in records:
        problem = record["problem"]
        tags = record.get("tags", {})
        trace = tags.get("last_trace", {})
        shape = f"{problem['m']}x{problem['n']}x{problem['k']}"
        lines.append(
            f"| {problem.get('dtype', 'unknown')} | {tags.get('case', '')} | {shape} | "
            f"{float(tags.get('hit_rate', 0.0)):.2f} | `{trace.get('kernel_name', '-')}` | "
            f"{float(tags.get('speedup_vs_stock', 0.0)):.3f}x |"
        )
    return lines


def command_output(command: list[str], cwd: pathlib.Path) -> str:
    proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def write_report(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    results: list[Result],
    required_failures: list[Result],
    metadata: dict[str, str],
    kernel_jsonl: pathlib.Path,
    vllm_jsonl: pathlib.Path,
) -> None:
    lines = [
        "# zcutlass M0 Baseline Report",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Step Results", ""])
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        rel_stdout = result.stdout_path.relative_to(root)
        rel_stderr = result.stderr_path.relative_to(root)
        lines.append(
            f"- {status} `{result.name}` exit={result.returncode} "
            f"stdout=`{rel_stdout}` stderr=`{rel_stderr}`"
        )
    lines.extend(["", "## Required Gate", ""])
    if required_failures:
        lines.append("Required M0 checks failed:")
        for failure in required_failures:
            lines.append(f"- `{failure.name}` exit={failure.returncode}")
    else:
        lines.append("All required M0 checks passed.")
    for summary in (kernel_summary_lines(kernel_jsonl), vllm_summary_lines(vllm_jsonl)):
        if summary:
            lines.extend(["", *summary])
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report freezes the current baseline before explicit-MMA work.",
            "- vLLM LinearMethod results prove routing and telemetry only; they are not serving-level value claims.",
            "- Existing unrelated dirty/manual report files are intentionally left untouched.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--build-jobs", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--vllm-venv", type=pathlib.Path, default=pathlib.Path("/home/zyz/vllm/.venv"))
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-kernel-bench", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    today = dt.date.today().isoformat()
    output_dir = args.output_dir or root / "reports" / f"{today}-m0-baseline"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_jsonl = output_dir / "llm-v15-zcutlass-cublas.jsonl"
    vllm_jsonl = output_dir / "vllm-linear-method-smoke.jsonl"

    steps = [
        Step("cmake_build", ["cmake", "--build", "build", "-j", str(args.build_jobs)]),
        Step("ctest", ["ctest", "--test-dir", "build", "--output-on-failure"]),
    ]
    if not args.skip_kernel_bench:
        steps.append(
            Step(
                "kernel_llm_v15",
                [
                    "./build/zcutlass_bench",
                    "--suite",
                    "llm-v1.5",
                    "--dtype",
                    "both",
                    "--providers",
                    "zcutlass,cublas",
                    "--json",
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--output",
                    str(kernel_jsonl),
                ],
            )
        )

    if not args.skip_vllm:
        activate = args.vllm_venv / "bin" / "activate"
        if activate.exists():
            vllm_prefix = f"source {shlex.quote(str(activate))} && "
            steps.extend(
                [
                    Step(
                        "torch_overlay_check",
                        ["bash", "-lc", vllm_prefix + "python tools/check_torch_overlay.py --require-extension"],
                    ),
                    Step(
                        "vllm_plugin_check",
                        [
                            "bash",
                            "-lc",
                            vllm_prefix
                            + "python tools/check_vllm_plugin.py --require-entry-point --require-vllm",
                        ],
                    ),
                    Step(
                        "vllm_linear_method_smoke",
                        [
                            "bash",
                            "-lc",
                            vllm_prefix
                            + "python tools/benchmark_vllm_linear_method.py "
                            + "--suite smoke --dtype both --allow-family prefill "
                            + "--materialize-inputs "
                            + f"--warmup {args.warmup} --iterations {args.iterations} "
                            + f"--output {shlex.quote(str(vllm_jsonl))} --summary",
                        ],
                    ),
                ]
            )
        else:
            steps.append(Step("vllm_venv_missing", ["bash", "-lc", f"test -f {shlex.quote(str(activate))}"], required=False))

    metadata = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": command_output(["git", "rev-parse", "HEAD"], root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "dirty_status": command_output(["git", "status", "--short"], root).replace("\n", "; ") or "clean",
        "baseline_note": "M0 may be run from an integration workspace; inspect dirty_status before comparing reports.",
        "cuda_nvcc": command_output(["bash", "-lc", "nvcc --version | tail -1"], root),
        "nvidia_smi": command_output(["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1"], root),
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    results: list[Result] = []
    required_failures: list[Result] = []
    for step in steps:
        result = run_step(step, root, output_dir)
        results.append(result)
        if result.returncode != 0 and step.required:
            required_failures.append(result)
            if not args.continue_on_error:
                break

    report = output_dir / "m0-baseline.md"
    write_report(
        report,
        root=root,
        results=results,
        required_failures=required_failures,
        metadata=metadata,
        kernel_jsonl=kernel_jsonl,
        vllm_jsonl=vllm_jsonl,
    )
    print(f"[m0] wrote {report}")
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
