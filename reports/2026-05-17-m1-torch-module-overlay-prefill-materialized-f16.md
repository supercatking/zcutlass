| Callsite | Stock ms | Overlay ms | Speedup | Hit rate | Path | Fallback |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| smoke_decode/mini_decoder_block f16 m=8 h=512 i=2048 | 0.1341 | 0.0935 | 1.434x | 0.00 | aggregate | {"shape_not_target_bucket": 20} |
| smoke_prefill/mini_decoder_block f16 m=128 h=1024 i=4096 | 0.1030 | 0.5897 | 0.175x | 1.00 | aggregate | - |
