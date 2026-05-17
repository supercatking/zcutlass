| Callsite | Stock ms | Overlay ms | Speedup | Hit rate | Path | Fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| smoke_decode/mini_decoder_block f16 m=8 h=512 i=2048 | 0.1066 | 0.0702 | 1.518x | 0.00 | aggregate | {"non_contiguous_ab": 5, "shape_not_target_bucket": 15} |
| smoke_prefill/mini_decoder_block f16 m=128 h=1024 i=4096 | 0.0970 | 0.0906 | 1.071x | 0.00 | aggregate | {"family_not_promoted": 15, "non_contiguous_ab": 5} |
