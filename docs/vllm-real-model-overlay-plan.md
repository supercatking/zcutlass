# vLLM Real Model Overlay Plan

## Purpose

Move from dummy LinearMethod probes to a real vLLM model execution path without
editing vLLM source and without global cuBLAS interception.

The first product proof target is `Qwen/Qwen2.5-1.5B-Instruct` in BF16 on the
local RTX 5080. Avoid AWQ/quantized models for the first pass because the
current zcutlass wrapper only handles vLLM `UnquantizedLinearMethod`.

## Integration Path

Use vLLM out-of-tree linear layer registration and keep it env-gated.

Implemented zcutlass files:

- `python/zcutlass_vllm/oot_linear.py`: registers OOT replacements for selected
  vLLM Linear layer classes.
- `python/zcutlass_vllm/route_logger.py`: JSONL route logger with per-layer
  hit/miss counters.
- `tools/check_vllm_oot_linear.py`: verifies plugin-driven OOT registration
  without loading a full model.

Planned next zcutlass file:

- `tools/check_vllm_real_model_overlay.py`: imports vLLM, loads the plugin, and
  verifies that a small model can install the overlay and generate text.

First layer classes to target:

- `QKVParallelLinear`
- `MergedColumnParallelLinear`
- `ColumnParallelLinear`
- `RowParallelLinear`

Later candidates:

- `ReplicatedLinear`
- `ParallelLMHead`

## Environment Switches

- `ZCUTLASS_VLLM_ENABLE=1`
- `ZCUTLASS_VLLM_ALLOW_FAMILIES=prefill`
- `ZCUTLASS_VLLM_LAYER_FILTER=qkv_proj,gate_up_proj,down_proj,o_proj`
- `ZCUTLASS_VLLM_LOG_ROUTES=1`
- `ZCUTLASS_VLLM_ROUTE_LOG=/home/zyz/zcutlass/reports/.../routes.jsonl`

The overlay must remain disabled by default.

## Route Log Fields

Each JSONL row should include:

- model id and architecture.
- layer prefix and layer class.
- input, weight, and output shapes.
- dtype and bias.
- tensor-parallel rank/size when available.
- shape family.
- hit or fallback.
- fallback reason.
- selected kernel name/path/tile.
- weight-cache hit/miss.
- materialize flag.
- latency in microseconds.
- pid, worker id, and rank when available.
- cumulative layer hit/miss counters.

## Baseline Commands

Stock server:

```bash
cd /home/zyz/vllm
source .venv/bin/activate
export HF_HOME=/home/zyz/vllm/hf-cache
MODEL=Qwen/Qwen2.5-1.5B-Instruct

vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --served-model-name qwen15-stock \
  --enforce-eager
```

Benchmark:

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model qwen15-stock \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 64 \
  --num-prompts 64 \
  --request-rate 4 \
  --max-concurrency 4 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir /home/zyz/zcutlass/reports/vllm-serving-baseline \
  --result-filename stock-prefill.json
```

Overlay server after OOT linear registration exists:

```bash
export VLLM_PLUGINS=zcutlass_overlay
export ZCUTLASS_VLLM_ENABLE=1
export ZCUTLASS_VLLM_ALLOW_FAMILIES=prefill
export ZCUTLASS_VLLM_LAYER_FILTER=qkv_proj,gate_up_proj,down_proj,o_proj
export ZCUTLASS_VLLM_LOG_ROUTES=1
export ZCUTLASS_VLLM_ROUTE_LOG=/home/zyz/zcutlass/reports/vllm-serving-baseline/routes-overlay-prefill.jsonl

vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --served-model-name qwen15-overlay \
  --enforce-eager
```

Run the same benchmark with `--model qwen15-overlay` and
`--result-filename overlay-prefill.json`.

Registration smoke check:

```bash
cd /home/zyz/zcutlass
source /home/zyz/vllm/.venv/bin/activate
python3 tools/check_vllm_oot_linear.py --require-vllm
```

## Acceptance

- Stock serving baseline completes.
- Overlay server starts only when explicitly enabled.
- Route log shows real model Linear callsites with hit/fallback rows.
- Unsupported quantized/non-unquantized methods stay on stock vLLM.
- No vLLM source checkout edits are required.
