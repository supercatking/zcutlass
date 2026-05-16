#!/usr/bin/env python3
"""Run Nsight Compute for one zcutlass GEMM benchmark shape."""

import argparse
import pathlib
import shlex
import shutil
import subprocess
import sys

from ncu_gemm_summary import summarize_csv, write_summary


DEFAULT_SECTIONS = (
    "SpeedOfLight",
    "Occupancy",
    "LaunchStats",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
)

ERR_NVGPUCTRPERM_HELP = """\
Nsight Compute could not access GPU performance counters (ERR_NVGPUCTRPERM).

Enable NVIDIA performance counter access on the host/driver and rerun this
command. The zcutlass benchmark can still report latency without these counters,
but Nsight throughput, occupancy, and stall metrics require permission.
"""


def find_bench(root: pathlib.Path, build_dir: str) -> pathlib.Path:
    candidates = [
        root / build_dir / "zcutlass_bench",
        root / build_dir / "benchmarks" / "zcutlass_bench",
        root / build_dir / "zcutlass_bench.exe",
        root / build_dir / "benchmarks" / "zcutlass_bench.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"zcutlass_bench not found under {root / build_dir}")


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def shape_stem(args: argparse.Namespace) -> str:
    return f"gemm_m{args.m}_n{args.n}_k{args.k}_{args.dtype}"


def run(cmd: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    print(quote_cmd(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def handle_ncu_failure(proc: subprocess.CompletedProcess[str]) -> int:
    output = f"{proc.stdout}\n{proc.stderr}"
    if "ERR_NVGPUCTRPERM" in output:
        print(ERR_NVGPUCTRPERM_HELP, file=sys.stderr)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def export_csv_from_report(
    ncu: str, report: pathlib.Path, csv_output: pathlib.Path, root: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    cmd = [ncu, "--import", str(report), "--page", "raw", "--csv"]
    proc = run(cmd, root)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.write_text(proc.stdout)
    return proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--provider", default="zcutlass")
    parser.add_argument(
        "--section",
        action="append",
        dest="sections",
        help="Nsight Compute section to collect; repeatable. Defaults cover SOL, occupancy, memory, and stalls.",
    )
    parser.add_argument("--set", dest="section_set", help="Nsight Compute section set, e.g. full.")
    parser.add_argument("--kernel-name", help="Optional ncu --kernel-name filter.")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--ncu", default=shutil.which("ncu") or "ncu")
    parser.add_argument("--output-dir", default="build/profiles")
    parser.add_argument("--report", type=pathlib.Path, help="Report path without or with .ncu-rep suffix.")
    parser.add_argument("--csv-output", type=pathlib.Path, help="Write imported ncu raw CSV here.")
    parser.add_argument("--summary-json", type=pathlib.Path, help="Write parsed summary JSON here.")
    parser.add_argument("--summary-md", type=pathlib.Path, help="Write parsed summary Markdown here.")
    parser.add_argument("--no-import", action="store_true", help="Only collect the .ncu-rep report.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    bench = find_bench(root, args.build_dir)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = shape_stem(args)
    report = args.report or output_dir / f"{stem}.ncu-rep"
    if report.suffix != ".ncu-rep":
        report = report.with_suffix(".ncu-rep")
    if not report.is_absolute():
        report = root / report
    csv_output = args.csv_output or report.with_suffix(".csv")
    summary_json = args.summary_json or report.with_suffix(".summary.json")
    summary_md = args.summary_md or report.with_suffix(".summary.md")

    cmd = [
        args.ncu,
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(report),
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
    ]
    if args.section_set:
        cmd.extend(["--set", args.section_set])
    else:
        for section in args.sections or DEFAULT_SECTIONS:
            cmd.extend(["--section", section])
    if args.kernel_name:
        cmd.extend(["--kernel-name", args.kernel_name])

    cmd.extend(
        [
            str(bench),
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--dtype",
            args.dtype,
            "--providers",
            args.provider,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
    )
    if args.benchmark_args and args.benchmark_args[0] == "--":
        cmd.extend(args.benchmark_args[1:])
    else:
        cmd.extend(args.benchmark_args)

    if args.dry_run:
        print(quote_cmd(cmd))
        return 0

    proc = run(cmd, root)
    if proc.returncode != 0:
        return handle_ncu_failure(proc)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"wrote report: {report}")

    if args.no_import:
        return 0

    import_proc = export_csv_from_report(args.ncu, report, csv_output, root)
    if import_proc.returncode != 0:
        return handle_ncu_failure(import_proc)
    summary = summarize_csv(csv_output, shape={"m": args.m, "n": args.n, "k": args.k, "dtype": args.dtype})
    write_summary(summary, summary_json=summary_json, summary_md=summary_md)
    print(f"wrote csv: {csv_output}")
    print(f"wrote summary json: {summary_json}")
    print(f"wrote summary markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
