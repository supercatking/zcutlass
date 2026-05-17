#!/usr/bin/env python3
"""Print a compact vLLM/zcutlass environment report."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


def import_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def entry_points(group: str) -> list[dict[str, str]]:
    eps = importlib.metadata.entry_points()
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    return [{"name": ep.name, "value": ep.value} for ep in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-zcutlass", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": sys.executable,
        "packages": {
            name: import_status(name)
            for name in ("torch", "vllm", "zcutlass_torch", "zcutlass_vllm")
        },
        "vllm_general_plugins": entry_points("vllm.general_plugins"),
        "nvcc": shutil.which("nvcc"),
    }

    try:
        import torch

        cuda_info: dict[str, object] = {
            "torch_cuda": torch.version.cuda,
            "available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["device_capability"] = torch.cuda.get_device_capability(0)
        report["cuda"] = cuda_info
    except Exception as exc:
        report["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}

    runtime_lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cuda_runtime"
        / "lib"
    )
    report["cuda_runtime_lib"] = {
        "path": str(runtime_lib),
        "libcudart_so": (runtime_lib / "libcudart.so").exists(),
        "libcudart_so_12": (runtime_lib / "libcudart.so.12").exists(),
    }

    print(json.dumps(report, indent=2, sort_keys=True, default=str))

    packages = report["packages"]
    assert isinstance(packages, dict)
    if args.require_vllm and not packages["vllm"]["ok"]:
        return 2
    if args.require_zcutlass and (
        not packages["zcutlass_torch"]["ok"] or not packages["zcutlass_vllm"]["ok"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
