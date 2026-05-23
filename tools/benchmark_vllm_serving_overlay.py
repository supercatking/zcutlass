#!/usr/bin/env python3
"""Bounded stock-vs-overlay vLLM serving benchmark harness.

The default mode is dry-run command generation. Passing ``--run`` starts one
bounded server/client pair at a time and parses the saved vLLM bench result when
available. Model downloads are disabled by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LAYER_FILTER = "qkv_proj,gate_up_proj,down_proj,o_proj"
DEFAULT_ALLOW_FAMILIES = "prefill"
OVERLAY_ENV_KEYS = [
    "VLLM_PLUGINS",
    "ZCUTLASS_VLLM_ENABLE",
    "ZCUTLASS_VLLM_ALLOW_FAMILIES",
    "ZCUTLASS_VLLM_LAYER_FILTER",
    "ZCUTLASS_VLLM_LOG_ROUTES",
    "ZCUTLASS_VLLM_ROUTE_LOG",
    "ZCUTLASS_VLLM_MODEL_ID",
]


@dataclass(frozen=True)
class ServingPreset:
    name: str
    random_input_len: int
    random_output_len: int
    num_prompts: int
    request_rate: float
    max_concurrency: int
    description: str


@dataclass(frozen=True)
class CommandPlan:
    variant: str
    provider: str
    served_model_name: str
    port: int
    base_url: str
    result_file: pathlib.Path
    server_log: pathlib.Path
    client_log: pathlib.Path
    route_log: pathlib.Path | None
    set_env: dict[str, str]
    unset_env: list[str]
    server_argv: list[str]
    client_argv: list[str]
    server_command: str
    client_command: str


PRESETS = {
    "decode-heavy": ServingPreset(
        name="decode-heavy",
        random_input_len=128,
        random_output_len=256,
        num_prompts=64,
        request_rate=4.0,
        max_concurrency=4,
        description="Short prompts with longer generation to emphasize decode/TPOT.",
    ),
    "prefill-heavy": ServingPreset(
        name="prefill-heavy",
        random_input_len=1024,
        random_output_len=64,
        num_prompts=64,
        request_rate=4.0,
        max_concurrency=4,
        description="Long prompts with short generation to emphasize prefill/TTFT.",
    ),
    "mixed": ServingPreset(
        name="mixed",
        random_input_len=512,
        random_output_len=128,
        num_prompts=64,
        request_rate=4.0,
        max_concurrency=4,
        description="Balanced prompt and generation lengths for end-to-end serving.",
    ),
}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def default_vllm_root() -> pathlib.Path:
    candidate = pathlib.Path("/home/zyz/vllm")
    return candidate if candidate.exists() else repo_root()


def default_hf_home() -> pathlib.Path:
    if os.environ.get("HF_HOME"):
        return pathlib.Path(os.environ["HF_HOME"])
    return pathlib.Path("/home/zyz/vllm/hf-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or run bounded vLLM stock-vs-zcutlass-overlay serving "
            "benchmarks. Dry-run command generation is the default."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local model path.")
    parser.add_argument("--tokenizer", help="Optional tokenizer id/path.")
    parser.add_argument("--served-name-prefix", default="qwen15")
    parser.add_argument("--preset", default="prefill-heavy", choices=tuple(PRESETS))
    parser.add_argument("--variant", default="both", choices=("stock", "overlay", "both"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--stock-port", type=int, default=8000)
    parser.add_argument("--overlay-port", type=int, default=8001)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vllm-command", default="vllm", help="vLLM CLI executable for --run.")
    parser.add_argument("--vllm-root", type=pathlib.Path, default=default_vllm_root())
    parser.add_argument(
        "--venv-activate",
        type=pathlib.Path,
        default=pathlib.Path("/home/zyz/vllm/.venv/bin/activate"),
        help="Activation script shown in generated shell commands.",
    )
    parser.add_argument("--hf-home", type=pathlib.Path, default=default_hf_home())
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Do not force HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE in generated or run env.",
    )
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--allow-families", default=DEFAULT_ALLOW_FAMILIES)
    parser.add_argument("--layer-filter", default=DEFAULT_LAYER_FILTER)
    parser.add_argument("--output-dir", type=pathlib.Path, default=repo_root() / "reports" / "vllm-serving-overlay")
    parser.add_argument("--plan-output", type=pathlib.Path)
    parser.add_argument("--summary-output", type=pathlib.Path)
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--percentile-metrics", default="ttft,tpot,itl")
    parser.add_argument("--metric-percentiles", default="50,95,99")
    parser.add_argument("--random-input-len", type=int)
    parser.add_argument("--random-output-len", type=int)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--extra-serve-arg", action="append", default=[])
    parser.add_argument("--extra-bench-arg", action="append", default=[])
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--client-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--model-check-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--print-commands", action=argparse.BooleanOptionalAction, default=True)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run", action="store_true", help="Start server/client benchmarks.")
    run_group.add_argument("--no-run", dest="run", action="store_false", help="Only generate commands.")
    parser.set_defaults(run=False)
    return parser.parse_args()


def selected_variants(name: str) -> list[str]:
    if name == "both":
        return ["stock", "overlay"]
    return [name]


def resolved_preset(args: argparse.Namespace) -> ServingPreset:
    base = PRESETS[args.preset]
    return ServingPreset(
        name=base.name,
        random_input_len=args.random_input_len or base.random_input_len,
        random_output_len=args.random_output_len or base.random_output_len,
        num_prompts=args.num_prompts or base.num_prompts,
        request_rate=args.request_rate if args.request_rate is not None else base.request_rate,
        max_concurrency=args.max_concurrency or base.max_concurrency,
        description=base.description,
    )


def quote_command(argv: list[str]) -> str:
    return shlex.join([str(arg) for arg in argv])


def shell_lines(
    *,
    cwd: pathlib.Path,
    activate: pathlib.Path,
    set_env: dict[str, str],
    unset_env: list[str],
    argv: list[str],
) -> str:
    lines = [f"cd {shlex.quote(str(cwd))}"]
    if activate:
        lines.append(f"source {shlex.quote(str(activate))}")
    if unset_env:
        lines.append("unset " + " ".join(shlex.quote(name) for name in unset_env))
    for key, value in sorted(set_env.items()):
        if key == "PYTHONPATH" and value.endswith("${PYTHONPATH:+:$PYTHONPATH}"):
            lines.append(f"export {key}={value}")
        else:
            lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(quote_command(argv))
    return "\n".join(lines)


def common_env(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {
        "HF_HOME": str(args.hf_home.expanduser()),
    }
    if not args.allow_downloads:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def overlay_env(args: argparse.Namespace, route_log: pathlib.Path) -> dict[str, str]:
    python_path = f"{repo_root() / 'python'}:${{PYTHONPATH:+:$PYTHONPATH}}"
    return {
        **common_env(args),
        "PYTHONPATH": python_path,
        "VLLM_PLUGINS": "zcutlass_overlay",
        "ZCUTLASS_VLLM_ENABLE": "1",
        "ZCUTLASS_VLLM_ALLOW_FAMILIES": args.allow_families,
        "ZCUTLASS_VLLM_LAYER_FILTER": args.layer_filter,
        "ZCUTLASS_VLLM_LOG_ROUTES": "1",
        "ZCUTLASS_VLLM_ROUTE_LOG": str(route_log),
        "ZCUTLASS_VLLM_MODEL_ID": args.model,
    }


def provider_for(variant: str) -> str:
    return "vllm_stock" if variant == "stock" else "zcutlass_vllm_overlay"


def served_model_name(args: argparse.Namespace, variant: str) -> str:
    return f"{args.served_name_prefix}-{variant}"


def port_for(args: argparse.Namespace, variant: str) -> int:
    return args.stock_port if variant == "stock" else args.overlay_port


def append_extra(argv: list[str], extras: list[str]) -> list[str]:
    out = list(argv)
    for extra in extras:
        out.extend(shlex.split(extra))
    return out


def build_plan(args: argparse.Namespace, preset: ServingPreset, variant: str) -> CommandPlan:
    output_dir = args.output_dir.expanduser().resolve()
    result_dir = output_dir / variant
    result_file = result_dir / f"{variant}-{preset.name}.json"
    route_log = output_dir / f"routes-{variant}-{preset.name}.jsonl" if variant == "overlay" else None
    set_env = overlay_env(args, route_log) if route_log else common_env(args)
    unset_env = [] if variant == "overlay" else OVERLAY_ENV_KEYS
    port = port_for(args, variant)
    name = served_model_name(args, variant)
    base_url = f"http://{args.host}:{port}"

    server_argv = [
        args.vllm_command,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--served-model-name",
        name,
    ]
    if args.enforce_eager:
        server_argv.append("--enforce-eager")
    if args.trust_remote_code:
        server_argv.append("--trust-remote-code")
    if args.tokenizer:
        server_argv.extend(["--tokenizer", args.tokenizer])
    server_argv = append_extra(server_argv, args.extra_serve_arg)

    client_argv = [
        args.vllm_command,
        "bench",
        "serve",
        "--backend",
        args.backend,
        "--base-url",
        base_url,
        "--endpoint",
        args.endpoint,
        "--model",
        name,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(preset.random_input_len),
        "--random-output-len",
        str(preset.random_output_len),
        "--num-prompts",
        str(preset.num_prompts),
        "--request-rate",
        str(preset.request_rate),
        "--max-concurrency",
        str(preset.max_concurrency),
        "--ignore-eos",
        "--percentile-metrics",
        args.percentile_metrics,
        "--metric-percentiles",
        args.metric_percentiles,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_file.name,
    ]
    client_argv = append_extra(client_argv, args.extra_bench_arg)

    server_command = shell_lines(
        cwd=args.vllm_root.expanduser(),
        activate=args.venv_activate.expanduser(),
        set_env=set_env,
        unset_env=unset_env,
        argv=server_argv,
    )
    client_command = shell_lines(
        cwd=args.vllm_root.expanduser(),
        activate=args.venv_activate.expanduser(),
        set_env=set_env,
        unset_env=unset_env,
        argv=client_argv,
    )
    return CommandPlan(
        variant=variant,
        provider=provider_for(variant),
        served_model_name=name,
        port=port,
        base_url=base_url,
        result_file=result_file,
        server_log=output_dir / f"server-{variant}-{preset.name}.log",
        client_log=output_dir / f"client-{variant}-{preset.name}.log",
        route_log=route_log,
        set_env=set_env,
        unset_env=unset_env,
        server_argv=server_argv,
        client_argv=client_argv,
        server_command=server_command,
        client_command=client_command,
    )


def plan_to_json(plan: CommandPlan) -> dict[str, Any]:
    raw = asdict(plan)
    for key in ("result_file", "server_log", "client_log", "route_log"):
        if raw[key] is not None:
            raw[key] = str(raw[key])
    return raw


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_env(plan: CommandPlan) -> dict[str, str]:
    env = dict(os.environ)
    for key in plan.unset_env:
        env.pop(key, None)
    for key, value in plan.set_env.items():
        if key == "PYTHONPATH" and value.endswith("${PYTHONPATH:+:$PYTHONPATH}"):
            prefix = value.removesuffix(":${PYTHONPATH:+:$PYTHONPATH}")
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"
        else:
            env[key] = value
    return env


def skip_or_fail(message: str, *, required: bool) -> tuple[str, int]:
    print(("FAIL: " if required else "SKIP: ") + message)
    return ("failed" if required else "skipped", 1 if required else 0)


def cli_available(command: str) -> bool:
    executable = shlex.split(command)[0] if command else ""
    return bool(executable and shutil.which(executable))


def model_is_local_path(model: str) -> bool:
    return pathlib.Path(model).expanduser().exists()


def check_model_available(args: argparse.Namespace, env: dict[str, str]) -> tuple[bool, str]:
    if args.skip_model_preflight or model_is_local_path(args.model):
        return True, ""
    python = shutil.which("python3") or shutil.which("python") or sys.executable
    code = (
        "from transformers import AutoConfig\n"
        "AutoConfig.from_pretrained("
        f"{args.model!r}, local_files_only=True, trust_remote_code={bool(args.trust_remote_code)!r})\n"
    )
    try:
        completed = subprocess.run(
            [python, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.model_check_timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"model cache preflight could not run: {exc}"
    if completed.returncode == 0:
        return True, ""
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    tail = detail[-1] if detail else "unknown local-files-only failure"
    return False, tail


def wait_for_server(plan: CommandPlan, process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    urls = [f"{plan.base_url}/health", f"{plan.base_url}/v1/models"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        for url in urls:
            try:
                with urlopen(url, timeout=2.0) as response:
                    if 200 <= int(response.status) < 500:
                        return True
            except URLError:
                pass
            except OSError:
                pass
        time.sleep(2.0)
    return False


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=20.0)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)


def run_plan(args: argparse.Namespace, preset: ServingPreset, plan: CommandPlan) -> tuple[str, int, dict[str, Any]]:
    if not cli_available(args.vllm_command):
        status, code = skip_or_fail(f"vLLM CLI unavailable: {args.vllm_command!r}", required=args.require_vllm)
        return status, code, {"skip_reason": "vllm_cli_unavailable"}

    env = prepare_env(plan)
    model_ok, model_reason = check_model_available(args, env)
    if not model_ok:
        status, code = skip_or_fail(
            f"model unavailable in local/offline cache for {args.model!r}: {model_reason}",
            required=args.require_model,
        )
        return status, code, {"skip_reason": "model_unavailable", "detail": model_reason}

    plan.result_file.parent.mkdir(parents=True, exist_ok=True)
    plan.server_log.parent.mkdir(parents=True, exist_ok=True)
    if plan.route_log:
        plan.route_log.parent.mkdir(parents=True, exist_ok=True)
        plan.route_log.write_text("", encoding="utf-8")

    print(f"RUN: starting {plan.variant} server on {plan.base_url}")
    with plan.server_log.open("w", encoding="utf-8") as server_handle:
        try:
            process = subprocess.Popen(
                plan.server_argv,
                cwd=str(args.vllm_root.expanduser()),
                env=env,
                stdout=server_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            status = "failed" if args.require_vllm else "skipped"
            code = 1 if args.require_vllm else 0
            print(f"{'FAIL' if code else 'SKIP'}: could not start vLLM server: {exc}")
            return status, code, {"skip_reason": "server_launch_failed", "detail": str(exc)}
        try:
            if not wait_for_server(plan, process, args.startup_timeout_seconds):
                status = "failed" if args.require_model else "skipped"
                code = 1 if args.require_model else 0
                print(
                    f"{'FAIL' if code else 'SKIP'}: server did not become ready; "
                    f"see {plan.server_log}"
                )
                return status, code, {"skip_reason": "server_not_ready"}

            print(f"RUN: starting {plan.variant} benchmark client")
            with plan.client_log.open("w", encoding="utf-8") as client_handle:
                try:
                    completed = subprocess.run(
                        plan.client_argv,
                        cwd=str(args.vllm_root.expanduser()),
                        env=env,
                        stdout=client_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=args.client_timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    print(f"FAIL: benchmark client timed out; see {plan.client_log}")
                    return "failed", 1, {"client_timeout_seconds": args.client_timeout_seconds}
                except OSError as exc:
                    print(f"FAIL: benchmark client could not start: {exc}")
                    return "failed", 1, {"client_launch_error": str(exc)}
            if completed.returncode != 0:
                print(f"FAIL: benchmark client exited {completed.returncode}; see {plan.client_log}")
                return "failed", completed.returncode, {"client_returncode": completed.returncode}
        finally:
            terminate_process(process)

    metrics = parse_vllm_result(plan.result_file)
    return "success", 0, {"metrics": metrics}


def load_jsonish(path: pathlib.Path) -> Any:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-1] if rows else None


def flatten_json(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(value, child))
        return out
    if isinstance(data, list):
        out = {}
        for index, value in enumerate(data):
            child = f"{prefix}.{index}" if prefix else str(index)
            out.update(flatten_json(value, child))
        return out
    return {prefix: data}


def canonical_key(key: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def find_value(flat: dict[str, Any], patterns: list[str]) -> float | None:
    canonical = [(canonical_key(key), value) for key, value in flat.items()]
    for pattern in patterns:
        pattern_key = canonical_key(pattern)
        for key, value in canonical:
            if key == pattern_key:
                out = numeric(value)
                if out is not None:
                    return out
    for pattern in patterns:
        parts = [part for part in canonical_key(pattern).split("_") if part]
        for key, value in canonical:
            if all(part in key for part in parts):
                out = numeric(value)
                if out is not None:
                    return out
    return None


def percentile_patterns(metric: str, percentile: int) -> list[str]:
    labels = [f"p{percentile}", f"percentile_{percentile}", f"{percentile}th", f"{percentile}_percentile"]
    if percentile == 50:
        labels.append("median")
    patterns: list[str] = []
    for label in labels:
        patterns.extend(
            [
                f"{label}_{metric}_ms",
                f"{metric}_{label}_ms",
                f"{label}_{metric}",
                f"{metric}_{label}",
            ]
        )
    return patterns


def parse_vllm_result(path: pathlib.Path) -> dict[str, Any]:
    raw = load_jsonish(path)
    if raw is None:
        return {"result_path": str(path), "available": False}

    flat = flatten_json(raw)
    metrics: dict[str, Any] = {
        "result_path": str(path),
        "available": True,
        "tokens_per_second": find_value(
            flat,
            [
                "total_token_throughput",
                "total_token_throughput_per_s",
                "token_throughput",
                "tokens_per_second",
                "tokens_per_sec",
                "output_throughput",
                "output_tokens_per_second",
            ],
        ),
        "request_throughput": find_value(flat, ["request_throughput", "requests_per_second"]),
    }
    for metric in ("ttft", "tpot", "itl"):
        entry: dict[str, float] = {}
        mean = find_value(flat, [f"mean_{metric}_ms", f"{metric}_mean_ms", f"avg_{metric}_ms"])
        if mean is not None:
            entry["mean"] = mean
        for percentile in (50, 95, 99):
            value = find_value(flat, percentile_patterns(metric, percentile))
            if value is not None:
                entry[f"p{percentile}"] = value
        if entry:
            metrics[f"{metric}_ms"] = entry
    return metrics


def build_record(
    *,
    args: argparse.Namespace,
    preset: ServingPreset,
    plan: CommandPlan,
    status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    metrics = details.get("metrics") or parse_vllm_result(plan.result_file)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": plan.provider,
        "status": status,
        "model": args.model,
        "preset": asdict(preset),
        "serving": {
            "host": args.host,
            "port": plan.port,
            "base_url": plan.base_url,
            "served_model_name": plan.served_model_name,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "env": {
            "set": plan.set_env,
            "unset": plan.unset_env,
        },
        "route_log": str(plan.route_log) if plan.route_log else None,
        "commands": {
            "server": plan.server_command,
            "client": plan.client_command,
        },
        "artifacts": {
            "result_file": str(plan.result_file),
            "server_log": str(plan.server_log),
            "client_log": str(plan.client_log),
        },
        "metrics": metrics,
        "details": details,
    }


def print_commands(plans: list[CommandPlan]) -> None:
    for plan in plans:
        print(f"\n# {plan.variant} server")
        print(plan.server_command)
        print(f"\n# {plan.variant} client")
        print(plan.client_command)


def main() -> int:
    args = parse_args()
    preset = resolved_preset(args)
    output_dir = args.output_dir.expanduser().resolve()
    plan_output = (args.plan_output or output_dir / f"plan-{preset.name}.json").expanduser().resolve()
    summary_output = (args.summary_output or output_dir / f"summary-{preset.name}.jsonl").expanduser().resolve()

    plans = [build_plan(args, preset, variant) for variant in selected_variants(args.variant)]
    plan_doc = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "run" if args.run else "dry-run",
        "model": args.model,
        "preset": asdict(preset),
        "downloads_allowed": bool(args.allow_downloads),
        "plans": [plan_to_json(plan) for plan in plans],
    }
    write_json(plan_output, plan_doc)

    if args.print_commands:
        print_commands(plans)

    records: list[dict[str, Any]] = []
    exit_code = 0
    for plan in plans:
        if args.run:
            status, code, details = run_plan(args, preset, plan)
            exit_code = exit_code or code
        else:
            status, details = "planned", {"dry_run": True}
        records.append(build_record(args=args, preset=preset, plan=plan, status=status, details=details))

    append_jsonl(summary_output, records)
    print(f"\nwrote plan: {plan_output}")
    print(f"wrote summary: {summary_output}")
    if not args.run:
        print("dry-run only; pass --run to start bounded serving benchmarks")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
