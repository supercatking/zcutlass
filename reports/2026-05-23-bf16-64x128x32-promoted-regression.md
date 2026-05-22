# Benchmark Regression Check

| Shape | Baseline ms | Candidate ms | Baseline / candidate | Candidate kernel |
| --- | ---: | ---: | ---: | --- |
| gemm bf16 128x16384x4096 layout=row,row,row,row alpha=1 beta=0 | 0.8826 | 0.8294 | 1.0641x | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` |
| gemm bf16 128x4096x16384 layout=row,row,row,row alpha=1 beta=0 | 1.6936 | 1.5626 | 1.0838x | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` |
| gemm bf16 128x4096x4096 layout=row,row,row,row alpha=1 beta=0 | 0.2838 | 0.2548 | 1.1138x | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` |
