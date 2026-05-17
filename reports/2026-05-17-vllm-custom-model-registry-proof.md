# vLLM Custom Model Registry Proof - 2026-05-17

## Result

zcutlass now has a non-invasive vLLM custom model probe:
`ZCutlassToyForCausalLM`.

This model is not a production LLM and does not load HuggingFace checkpoints.
It exists to verify the next integration seam:

- vLLM loads the zcutlass general plugin.
- The plugin registers `ZCutlassToyForCausalLM` via
  `vllm.model_executor.models.ModelRegistry.register_model`.
- The registered model can run a synthetic forward pass in the vLLM Python
  environment.
- That forward pass routes one prefill-shaped Linear call through
  `ZCutlassVllmLinearAdapter`.

## Commands

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_model_registry.py \
  --m 128 \
  --hidden-size 1024 \
  --vocab-size 4096 \
  --dtype f16 \
  --require-hit \
  --output reports/2026-05-17-vllm-custom-model-registry-f16.jsonl

python tools/check_vllm_model_registry.py \
  --m 128 \
  --hidden-size 1024 \
  --vocab-size 4096 \
  --dtype bf16 \
  --require-hit \
  --output reports/2026-05-17-vllm-custom-model-registry-bf16.jsonl
```

## Observations

| Case | Registry before plugin | Registry after plugin | zcutlass hit rate | Output |
| --- | ---: | ---: | ---: | --- |
| FP16 | false | true | 1.00 | hidden `[128,1024]`, logits `[128,4096]` |
| BF16 | false | true | 1.00 | hidden `[128,1024]`, logits `[128,4096]` |

This proves that zcutlass can participate in vLLM's external model registration
path without editing `/home/zyz/vllm`.

## Remaining Gap

This still is not a stock model serving benchmark. The next step is to choose a
small real model and add a controlled adapter experiment around real vLLM Linear
layers or an out-of-tree model variant, then measure:

- request correctness,
- zcutlass hit/fallback rate,
- TTFT,
- TPOT,
- tokens/s,
- p50/p95/p99 latency.
