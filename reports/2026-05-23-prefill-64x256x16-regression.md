# Benchmark Regression Check

| Shape | Baseline ms | Candidate ms | Baseline / candidate | Candidate kernel |
| --- | ---: | ---: | ---: | --- |
| gemm f16 128x16384x4096 layout=row,row,row,row alpha=1 beta=0 | 0.9834 | 1.1719 | 0.8392x | `zcutlass_sm120_tensorop_f16_64x256x16_aligned_prefill_n_gt_k_experimental` |
| gemm f16 128x4096x16384 layout=row,row,row,row alpha=1 beta=0 | 1.5851 | 1.5951 | 0.9937x | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` |
| gemm f16 128x4096x4096 layout=row,row,row,row alpha=1 beta=0 | 0.2518 | 0.2515 | 1.0012x | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` |
