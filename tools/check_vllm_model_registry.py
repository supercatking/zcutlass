#!/usr/bin/env python3
"""Check zcutlass vLLM custom model registration and synthetic forward."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from types import SimpleNamespace


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--require-hit", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))

    import torch
    import vllm.plugins
    import zcutlass_vllm
    from vllm.model_executor.models import ModelRegistry

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")

    before_supported = "ZCutlassToyForCausalLM" in ModelRegistry.get_supported_archs()
    vllm.plugins.load_general_plugins()
    after_supported = "ZCutlassToyForCausalLM" in ModelRegistry.get_supported_archs()
    if not after_supported:
        raise SystemExit("FAIL: zcutlass toy model was not registered in vLLM ModelRegistry")

    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    fake_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=args.hidden_size,
                vocab_size=args.vocab_size,
            )
        )
    )
    model = zcutlass_vllm.ZCutlassToyForCausalLM(fake_config).to(device="cuda", dtype=dtype).eval()
    input_ids = torch.randint(0, args.vocab_size, (args.m,), device="cuda", dtype=torch.long)
    positions = torch.arange(args.m, device="cuda", dtype=torch.long)
    with torch.inference_mode():
        hidden_states = model(input_ids=input_ids, positions=positions)
        logits = model.compute_logits(hidden_states)
    torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "operation": "vllm_custom_model_registry_probe",
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "plugin_registered": zcutlass_vllm.is_registered(),
            "model_supported_before_plugin_load": before_supported,
            "model_supported_after_plugin_load": after_supported,
        },
        "problem": {
            "m": args.m,
            "hidden_size": args.hidden_size,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
        },
        "routing": {
            "hit_rate": model.adapter.stats.hit_rate,
            "hits": model.adapter.stats.hits,
            "misses": model.adapter.stats.misses,
            "fallback_reasons": dict(model.adapter.stats.fallback_reasons),
            "last_trace": model.adapter.last_trace,
        },
        "outputs": {
            "hidden_shape": list(hidden_states.shape),
            "logits_shape": list(logits.shape),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if args.require_hit and model.adapter.stats.hits == 0:
        raise SystemExit("FAIL: model forward did not route to zcutlass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
