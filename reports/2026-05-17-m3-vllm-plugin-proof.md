# M3 vLLM Plugin Proof

Date: 2026-05-17

## Goal

M3 proves that a vLLM process can discover zcutlass as an opt-in overlay plugin
before we wire a concrete vLLM model or worker path. This is not yet a
performance claim.

## Implemented

- `vllm.general_plugins` entry point:
  - name: `zcutlass_overlay`
  - target: `zcutlass_vllm.plugin:register`
- `zcutlass_vllm.register()` is idempotent and side-effect-light.
- `zcutlass_vllm.ZCutlassVllmLinearAdapter` exposes an explicit Linear adapter
  for vLLM custom model/worker experiments.
- `tools/check_vllm_plugin.py` validates import, registration, packaging entry
  point discovery, and optionally vLLM availability.

## Validation

Command:

```bash
cd /home/zyz/zcutlass
source build/torch-cu130-venv/bin/activate
python3 -m pip install -e ./python --no-build-isolation --force-reinstall
python3 tools/check_vllm_plugin.py --require-entry-point
```

Result:

```text
PASS: zcutlass_vllm import/register ok; entry_point_installed=True; vllm=not_installed (No module named 'vllm')
```

This proves that the package entry point exists and the adapter package can be
loaded. It does not prove vLLM end-to-end execution because vLLM is not installed
in the current validation environment.

## Commercial Value Status

Not established.

The current zcutlass WMMA path is still slower than stock PyTorch on promoted
module-level prefill smoke measurements. vLLM performance value requires all of
the following:

- vLLM installed in a compatible environment.
- A concrete vLLM custom model/worker integration that routes selected Linear
  callsites through `ZCutlassVllmLinearAdapter`.
- Stock vLLM vs vLLM plus zcutlass overlay benchmark under identical model,
  prompt/output, batch, sampling, CUDA, and driver settings.
- TTFT or TPOT improvement with no material p95/p99 regression.
- Fallback reasons and hit rate recorded for every zcutlass-eligible callsite.

## Next Steps

1. Install vLLM in a dedicated environment compatible with the RTX 5080/CUDA
   stack.
2. Add a vLLM custom model or worker-side experiment that uses
   `ZCutlassVllmLinearAdapter` for selected Linear callsites.
3. Reuse the PyTorch module harness JSONL schema for vLLM stock-vs-overlay
   measurements.
4. Do not claim superiority over CUTLASS until the vLLM end-to-end benchmark
   beats the stock engine and the kernel-level evidence explains the win.
