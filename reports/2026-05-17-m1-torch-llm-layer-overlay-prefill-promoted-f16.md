| Callsite | Stock ms | Overlay ms | Speedup | Hit rate | Path | Fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| smoke_decode/qkv f16 8x1536x512 | 0.0118 | 0.0235 | 0.503x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/o_proj f16 8x512x512 | 0.0218 | 0.0107 | 2.030x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/mlp_up_gate f16 8x4096x512 | 0.0084 | 0.0113 | 0.740x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/mlp_down f16 8x512x2048 | 0.0200 | 0.0175 | 1.147x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_prefill/qkv f16 128x3072x1024 | 0.0207 | 0.0818 | 0.254x | 1.00 | zcutlass | - |
| smoke_prefill/o_proj f16 128x1024x1024 | 0.0135 | 0.0808 | 0.167x | 1.00 | zcutlass | - |
| smoke_prefill/mlp_up_gate f16 128x8192x1024 | 0.0331 | 0.1195 | 0.277x | 1.00 | zcutlass | - |
| smoke_prefill/mlp_down f16 128x1024x4096 | 0.0312 | 0.3227 | 0.097x | 1.00 | zcutlass | - |
| smoke_decode/synthetic_llm_layer f16 | 0.0620 | 0.0630 | 0.984x | 0.00 | aggregate | {"shape_not_target_bucket": 20} |
| smoke_prefill/synthetic_llm_layer f16 | 0.0984 | 0.6048 | 0.163x | 1.00 | aggregate | - |
