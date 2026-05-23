# 2026-05-23 vLLM Prefill Serving Smoke

## Scope

This run validates the bounded stock-vs-overlay vLLM serving harness on the
local Qwen2.5-1.5B BF16 model. It is a data-path proof and gap measurement, not
a product-value win.

Model:

```text
/home/zyz/vLLM_deploy/localModels/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

Workload:

```text
random_input_len=384
random_output_len=8
num_prompts=4
request_rate=1
max_concurrency=1
dtype=bfloat16
max_model_len=512
enforce_eager=true
```

## Harness Fix

The first attempt failed because `vllm bench serve` received
`--model qwen15-stock` or `--model qwen15-overlay` and then tried to load the
tokenizer from that served-model alias. The harness now passes
`--tokenizer <local model path>` to the benchmark client by default.

## Result

| Provider | TTFT mean ms | TTFT p50 ms | TPOT mean ms | TPOT p50 ms | Tokens/s | Request/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock vLLM | `115.99` | `27.46` | `12.13` | `10.31` | `382.44` | `0.976` |
| vLLM + zcutlass overlay | `138.94` | `43.14` | `14.29` | `14.27` | `378.49` | `0.966` |

Overlay is slower on this bounded run:

- TTFT mean: `0.835x` of stock.
- TPOT mean: `0.849x` of stock.
- Tokens/s: `0.990x` of stock.

## Route Summary

The overlay run recorded `3920` route rows:

| Route | Count |
| --- | ---: |
| zcutlass | `112` |
| fallback | `3808` |

Shape families:

| Family | Count |
| --- | ---: |
| decode | `3136` |
| fallback | `448` |
| large | `224` |
| prefill | `112` |

Fallback reasons:

| Reason | Count |
| --- | ---: |
| family_not_promoted | `3360` |
| shape_not_target_bucket | `448` |

zcutlass hit kernels:

| Kernel | Count |
| --- | ---: |
| `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_ldm_cpasync_warp32x32_reg_epilogue` | `84` |
| `zcutlass_sm120_tensorop_bf16_64x128x16` | `28` |

Hit shapes:

| Shape | Count |
| --- | ---: |
| `256x2048x1536` | `28` |
| `256x1536x1536` | `28` |
| `256x17920x1536` | `28` |
| `256x1536x8960` | `28` |

## Decision

Do not claim serving value yet. The current overlay stack is functional and
observable, but the zcutlass path is not fast enough and the hit rate is too
low for this serving workload.

Next work should focus on:

1. Promoting a faster prefill kernel for the Qwen2.5-1.5B hit shapes.
2. Separating decode and large-family optimization because most route rows are
   outside the current prefill bucket.
3. Adding route-summary extraction to the serving harness so future reports do
   not require ad hoc log parsing.
