| Callsite | Stock ms | Overlay ms | Speedup | Hit rate | Path | Fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| smoke_decode/qkv f16 8x1536x512 | 0.0266 | 0.0177 | 1.498x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/o_proj f16 8x512x512 | 0.0083 | 0.0088 | 0.945x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/mlp_up_gate f16 8x4096x512 | 0.0250 | 0.0116 | 2.163x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_decode/mlp_down f16 8x512x2048 | 0.0147 | 0.0223 | 0.658x | 0.00 | pytorch_fallback | shape_not_target_bucket |
| smoke_prefill/qkv f16 128x3072x1024 | 0.0210 | 0.0216 | 0.973x | 0.00 | pytorch_fallback | family_not_promoted |
| smoke_prefill/o_proj f16 128x1024x1024 | 0.0279 | 0.0161 | 1.730x | 0.00 | pytorch_fallback | family_not_promoted |
| smoke_prefill/mlp_up_gate f16 128x8192x1024 | 0.0344 | 0.0340 | 1.013x | 0.00 | pytorch_fallback | family_not_promoted |
| smoke_prefill/mlp_down f16 128x1024x4096 | 0.0239 | 0.0234 | 1.020x | 0.00 | pytorch_fallback | family_not_promoted |
| smoke_decode/synthetic_llm_layer f16 | 0.0745 | 0.0603 | 1.235x | 0.00 | aggregate | {"shape_not_target_bucket": 20} |
| smoke_prefill/synthetic_llm_layer f16 | 0.1072 | 0.0951 | 1.128x | 0.00 | aggregate | {"family_not_promoted": 20} |
