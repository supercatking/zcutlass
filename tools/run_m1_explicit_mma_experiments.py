#!/usr/bin/env python3
"""Run the zcutlass v1.5 M1 explicit-MMA reproducibility chain."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelExperiment:
    name: str
    dtype: str
    kernel_filter: str
    output_name: str


@dataclass
class Step:
    name: str
    command: list[str]
    required: bool = True
    skip_reason: str | None = None


@dataclass
class Result:
    name: str
    command: list[str]
    returncode: int
    stdout_path: pathlib.Path
    stderr_path: pathlib.Path
    required: bool = True
    skipped: bool = False
    note: str = ""


KERNEL_EXPERIMENTS = [
    KernelExperiment(
        "direct_f16",
        "f16",
        "sm120_mma_f16_64x128x64_prefill_reg_epilogue",
        "explicit-mma-direct-f16.jsonl",
    ),
    KernelExperiment(
        "direct_bf16",
        "bf16",
        "sm120_mma_bf16_64x128x64_prefill_reg_epilogue",
        "explicit-mma-direct-bf16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x16_f16",
        "f16",
        "sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue",
        "explicit-mma-shared-smem-warp16x16-f16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x16_bf16",
        "bf16",
        "sm120_mma_bf16_64x128x64_prefill_smem_reg_epilogue",
        "explicit-mma-shared-smem-warp16x16-bf16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x32_f16",
        "f16",
        "sm120_mma_f16_64x128x64_prefill_smem_warp16x32_reg_epilogue",
        "explicit-mma-shared-smem-warp16x32-f16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x32_bf16",
        "bf16",
        "sm120_mma_bf16_64x128x64_prefill_smem_warp16x32_reg_epilogue",
        "explicit-mma-shared-smem-warp16x32-bf16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_ldm_warp16x32_f16",
        "f16",
        "sm120_mma_f16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue",
        "explicit-mma-shared-smem-ldm-warp16x32-f16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_ldm_warp16x32_bf16",
        "bf16",
        "sm120_mma_bf16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue",
        "explicit-mma-shared-smem-ldm-warp16x32-bf16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x64_f16",
        "f16",
        "sm120_mma_f16_64x128x64_prefill_smem_warp16x64_reg_epilogue",
        "explicit-mma-shared-smem-warp16x64-f16.jsonl",
    ),
    KernelExperiment(
        "shared_smem_warp16x64_bf16",
        "bf16",
        "sm120_mma_bf16_64x128x64_prefill_smem_warp16x64_reg_epilogue",
        "explicit-mma-shared-smem-warp16x64-bf16.jsonl",
    ),
]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command) if command else "-"


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def run_step(step: Step, cwd: pathlib.Path, output_dir: pathlib.Path) -> Result:
    stdout_path = output_dir / f"{safe_name(step.name)}.stdout.txt"
    stderr_path = output_dir / f"{safe_name(step.name)}.stderr.txt"
    if step.skip_reason is not None:
        stdout_path.write_text(step.skip_reason + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        print(f"[m1] {step.name}: skipped ({step.skip_reason})", flush=True)
        return Result(
            step.name,
            step.command,
            0,
            stdout_path,
            stderr_path,
            required=step.required,
            skipped=True,
            note=step.skip_reason,
        )

    print(f"[m1] {step.name}: {quote(step.command)}", flush=True)
    try:
        proc = subprocess.run(
            step.command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except FileNotFoundError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    print(f"[m1] {step.name}: exit {returncode}", flush=True)
    return Result(
        step.name,
        step.command,
        returncode,
        stdout_path,
        stderr_path,
        required=step.required,
    )


def relpath(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def record_key(record: dict) -> tuple:
    problem = record.get("problem", {})
    return (
        problem.get("operation"),
        problem.get("m"),
        problem.get("n"),
        problem.get("k"),
        problem.get("dtype"),
    )


def speedups(records: list[dict]) -> dict[tuple, float]:
    zcutlass = {
        record_key(record): record
        for record in records
        if record.get("provider") == "zcutlass"
    }
    cublas = {
        record_key(record): record
        for record in records
        if record.get("provider") == "cublas"
    }
    ratios: dict[tuple, float] = {}
    for key, z_record in zcutlass.items():
        c_record = cublas.get(key)
        if not c_record:
            continue
        z_ms = float(z_record.get("performance", {}).get("median_ms", 0.0))
        c_ms = float(c_record.get("performance", {}).get("median_ms", 0.0))
        if z_ms:
            ratios[key] = c_ms / z_ms
    return ratios


def kernel_summary_lines(kernel_outputs: list[tuple[str, pathlib.Path]]) -> list[str]:
    rows: list[str] = []
    for experiment, path in kernel_outputs:
        records = load_jsonl(path)
        ratios = speedups(records)
        for record in records:
            if record.get("provider") != "zcutlass":
                continue
            problem = record.get("problem", {})
            perf = record.get("performance", {})
            tags = record.get("tags", {})
            shape = f"{problem.get('m')}x{problem.get('n')}x{problem.get('k')}"
            speedup = ratios.get(record_key(record))
            rows.append(
                "| "
                + " | ".join(
                    [
                        experiment,
                        str(problem.get("dtype", "unknown")),
                        shape,
                        f"`{record.get('kernel', '')}`",
                        str(tags.get("kernel_path", "")),
                        str(tags.get("epilogue_kind", "")),
                        f"{float(perf.get('median_ms', 0.0)):.4f}",
                        f"{float(perf.get('tflops', 0.0)):.4f}",
                        f"{speedup:.3f}x" if speedup is not None else "-",
                    ]
                )
                + " |"
            )
    if not rows:
        return []
    return [
        "## Explicit-MMA Kernel Summary",
        "",
        "| Experiment | DType | Shape | Kernel | Path | Epilogue | ms | TFLOP/s | Speedup vs cuBLAS |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        *rows,
    ]


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
        problem = record.get("problem", {})
        tags = record.get("tags", {})
        trace = tags.get("last_trace", {})
        shape = f"{problem.get('m')}x{problem.get('n')}x{problem.get('k')}"
        lines.append(
            f"| {problem.get('dtype', 'unknown')} | {tags.get('case', '')} | {shape} | "
            f"{float(tags.get('hit_rate', 0.0)):.2f} | `{trace.get('kernel_name', '-')}` | "
            f"{float(tags.get('speedup_vs_stock', 0.0)):.3f}x |"
        )
    return lines


def command_output(command: list[str], cwd: pathlib.Path) -> str:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        return f"unavailable: {exc}"
    return proc.stdout.strip()


def write_report(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    results: list[Result],
    required_failures: list[Result],
    metadata: dict[str, str],
    kernel_outputs: list[tuple[str, pathlib.Path]],
    vllm_jsonl: pathlib.Path,
) -> None:
    lines = [
        "# zcutlass M1 Explicit-MMA Experiment Report",
        "",
        "## Metadata",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Step Results", ""])
    for result in results:
        status = "SKIP" if result.skipped else ("PASS" if result.returncode == 0 else "FAIL")
        rel_stdout = relpath(result.stdout_path, root)
        rel_stderr = relpath(result.stderr_path, root)
        required = "required" if result.required else "optional"
        line = (
            f"- {status} `{result.name}` ({required}) exit={result.returncode} "
            f"stdout=`{rel_stdout}` stderr=`{rel_stderr}`"
        )
        if result.note:
            line += f" note=`{result.note}`"
        lines.append(line)

    lines.extend(["", "## Required Gate", ""])
    if required_failures:
        lines.append("Required M1 checks failed:")
        for failure in required_failures:
            lines.append(f"- `{failure.name}` exit={failure.returncode}")
    else:
        lines.append("All required M1 checks passed.")

    for summary in (kernel_summary_lines(kernel_outputs), vllm_summary_lines(vllm_jsonl)):
        if summary:
            lines.extend(["", *summary])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Explicit-MMA kernels are selected with the existing experimental kernel filter.",
            "- Decode, large, or off-bucket shapes may report fallback kernels by design.",
            "- vLLM LinearMethod results are a smoke check for routing and telemetry, not a serving claim.",
            "- Existing unrelated dirty/manual report files are intentionally left untouched.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_kernel_step(
    experiment: KernelExperiment,
    *,
    bench: pathlib.Path,
    suite: str,
    warmup: int,
    iterations: int,
    output_dir: pathlib.Path,
) -> tuple[Step, pathlib.Path]:
    output = output_dir / experiment.output_name
    return (
        Step(
            f"kernel_{experiment.name}",
            [
                str(bench),
                "--suite",
                suite,
                "--dtype",
                experiment.dtype,
                "--providers",
                "zcutlass,cublas",
                "--json",
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--experimental-kernels",
                "--experimental-kernel",
                experiment.kernel_filter,
                "--output",
                str(output),
            ],
        ),
        output,
    )


def build_sanitizer_step(
    experiment: KernelExperiment,
    *,
    bench: pathlib.Path,
    output_dir: pathlib.Path,
) -> tuple[Step, pathlib.Path]:
    output = output_dir / f"compute-sanitizer-{experiment.output_name}"
    return (
        Step(
            f"compute_sanitizer_{experiment.name}",
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                str(bench),
                "--suite",
                "single",
                "--m",
                "64",
                "--n",
                "1024",
                "--k",
                "1024",
                "--dtype",
                experiment.dtype,
                "--providers",
                "zcutlass",
                "--json",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--experimental-kernels",
                "--experimental-kernel",
                experiment.kernel_filter,
                "--output",
                str(output),
            ],
        ),
        output,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--build-dir", type=pathlib.Path, default=pathlib.Path("build"))
    parser.add_argument("--build-jobs", type=int, default=24)
    parser.add_argument("--suite", default="llm-v1.5")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--compute-sanitizer", action="store_true")
    parser.add_argument("--vllm-venv", type=pathlib.Path, default=pathlib.Path("/home/zyz/vllm/.venv"))
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--skip-kernel-bench", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.skip_vllm and args.require_vllm:
        parser.error("--skip-vllm and --require-vllm cannot be used together")

    root = repo_root()
    today = dt.date.today().isoformat()
    output_dir = args.output_dir or root / "reports" / f"{today}-m1-explicit-mma"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    build_dir = args.build_dir if args.build_dir.is_absolute() else root / args.build_dir
    bench = build_dir / "zcutlass_bench"
    vllm_jsonl = output_dir / "vllm-linear-method-smoke.jsonl"

    steps = [
        Step("cmake_build", ["cmake", "--build", str(build_dir), "-j", str(args.build_jobs)]),
        Step("ctest", ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]),
    ]
    kernel_outputs: list[tuple[str, pathlib.Path]] = []

    if args.compute_sanitizer:
        for experiment in KERNEL_EXPERIMENTS:
            step, output = build_sanitizer_step(experiment, bench=bench, output_dir=output_dir)
            steps.append(step)
            kernel_outputs.append((f"sanitizer_{experiment.name}", output))

    if not args.skip_kernel_bench:
        for experiment in KERNEL_EXPERIMENTS:
            step, output = build_kernel_step(
                experiment,
                bench=bench,
                suite=args.suite,
                warmup=args.warmup,
                iterations=args.iterations,
                output_dir=output_dir,
            )
            steps.append(step)
            kernel_outputs.append((experiment.name, output))

    if args.skip_vllm:
        steps.append(Step("vllm_linear_method_smoke", [], required=False, skip_reason="disabled by --skip-vllm"))
    else:
        activate = args.vllm_venv / "bin" / "activate"
        if activate.exists():
            vllm_prefix = f"source {shlex.quote(str(activate))} && "
            steps.append(
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
                )
            )
        elif args.require_vllm:
            steps.append(
                Step(
                    "vllm_venv_missing",
                    ["bash", "-lc", f"test -f {shlex.quote(str(activate))}"],
                    required=True,
                )
            )
        else:
            steps.append(
                Step(
                    "vllm_linear_method_smoke",
                    [],
                    required=False,
                    skip_reason=f"missing vLLM venv at {args.vllm_venv}",
                )
            )

    metadata = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": command_output(["git", "rev-parse", "HEAD"], root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "dirty_status": command_output(["git", "status", "--short"], root).replace("\n", "; ") or "clean",
        "benchmark_suite": args.suite,
        "warmup": str(args.warmup),
        "iterations": str(args.iterations),
        "compute_sanitizer": str(args.compute_sanitizer),
        "vllm_venv": str(args.vllm_venv),
        "cuda_nvcc": command_output(["bash", "-lc", "nvcc --version | tail -1"], root),
        "nvidia_smi": command_output(
            ["bash", "-lc", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1"],
            root,
        ),
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

    report = output_dir / "m1-explicit-mma.md"
    write_report(
        report,
        root=root,
        results=results,
        required_failures=required_failures,
        metadata=metadata,
        kernel_outputs=kernel_outputs,
        vllm_jsonl=vllm_jsonl,
    )
    print(f"[m1] wrote {report}")
    return 1 if required_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
