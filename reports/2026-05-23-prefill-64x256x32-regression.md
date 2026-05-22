# Benchmark Regression Check

| Shape | Baseline ms | Candidate ms | Baseline / candidate | Candidate kernel |
| --- | ---: | ---: | ---: | --- |
| gemm f16 128x16384x4096 layout=row,row,row,row alpha=1 beta=0 | 0.9834 | 0.9390 | 1.0473x | `zcutlass_sm120_tensorop_f16_64x256x32_aligned_prefill_n_gt_k_experimental` |
| gemm f16 128x4096x16384 layout=row,row,row,row alpha=1 beta=0 | 1.5851 | 1.5979 | 0.9920x | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` |
| gemm f16 128x4096x4096 layout=row,row,row,row alpha=1 beta=0 | 0.2518 | 0.2512 | 1.0024x | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` |
