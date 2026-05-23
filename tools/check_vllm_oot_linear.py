#!/usr/bin/env python3
"""Check zcutlass vLLM OOT Linear registration without loading a real model."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-vllm", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))
    os.environ.setdefault("ZCUTLASS_VLLM_ENABLE", "1")
    os.environ.setdefault("ZCUTLASS_VLLM_ALLOW_FAMILIES", "prefill")
    os.environ.setdefault("ZCUTLASS_VLLM_LAYER_FILTER", "qkv_proj,gate_up_proj,down_proj,o_proj")

    try:
        from vllm.model_executor.custom_op import op_registry_oot
    except Exception as exc:
        if args.require_vllm:
            raise SystemExit(f"FAIL: vLLM is required but unavailable: {exc}") from exc
        print(f"SKIP: vLLM unavailable: {exc}")
        return 0

    import zcutlass_vllm

    zcutlass_vllm.register()
    expected = {
        "QKVParallelLinear",
        "MergedColumnParallelLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
    }
    registered = expected.intersection(op_registry_oot)
    missing = sorted(expected - registered)
    if missing:
        raise SystemExit(f"FAIL: missing OOT Linear registrations: {missing}")
    print(
        "PASS: zcutlass vLLM OOT Linear registrations installed: "
        + ", ".join(sorted(registered))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
