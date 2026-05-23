## Explicit-MMA Comparison Summary

_JSONL source: `reports/2026-05-23-m1-explicit-mma`_

| Rank | File | DType | Shape | Kernel | Tile | Stages | Epilogue | zcutlass ms | cuBLAS ms | Speedup vs cuBLAS |
| ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_vec_k128_warp32x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x128_prefill_smem_ldm_vec_warp32x32_reg_epilogue` | 64x128x128 | 2 | register_linear | 0.1022 | 0.0573 | 0.561x |
| 2 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_vec_k128_warp32x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x128_prefill_smem_ldm_vec_warp32x32_reg_epilogue` | 64x128x128 | 2 | register_linear | 0.1048 | 0.0545 | 0.520x |
| 3 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_vec_k128_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x128_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x128x128 | 2 | register_linear | 0.1079 | 0.0585 | 0.542x |
| 4 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_vec_k128_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x128_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x128x128 | 2 | register_linear | 0.1096 | 0.0567 | 0.517x |
| 5 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_vec_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1164 | 0.0588 | 0.505x |
| 6 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_vec_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1181 | 0.0553 | 0.468x |
| 7 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_vec_n64_k128_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x64x128_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x64x128 | 2 | register_linear | 0.1249 | 0.0547 | 0.438x |
| 8 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_vec_n64_k128_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x64x128_prefill_smem_ldm_vec_warp16x32_reg_epilogue` | 64x64x128 | 2 | register_linear | 0.1262 | 0.0554 | 0.439x |
| 9 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_vec_lb2_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_ldm_vec_lb2_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1434 | 0.0555 | 0.387x |
| 10 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_vec_lb2_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_vec_lb2_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1442 | 0.0574 | 0.398x |
| 11 | `reports/2026-05-23-m1-explicit-mma/bf16_ldm_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1627 | 0.0549 | 0.337x |
| 12 | `reports/2026-05-23-m1-explicit-mma/f16_ldm_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.1633 | 0.0549 | 0.336x |
| 13 | `reports/2026-05-23-m1-explicit-mma/f16_warp16x32_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2455 | 0.0575 | 0.234x |
| 14 | `reports/2026-05-23-m1-explicit-mma/bf16_warp16x32_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x32_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2459 | 0.0559 | 0.227x |
| 15 | `reports/2026-05-23-m1-explicit-mma/bf16_warp16x64_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x64_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2678 | 0.0555 | 0.207x |
| 16 | `reports/2026-05-23-m1-explicit-mma/f16_warp16x64_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x64_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2690 | 0.0563 | 0.209x |
| 17 | `reports/2026-05-23-m1-explicit-mma/f16_smem_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2712 | 0.0544 | 0.201x |
| 18 | `reports/2026-05-23-m1-explicit-mma/bf16_smem_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2713 | 0.0555 | 0.205x |
| 19 | `reports/2026-05-23-m1-explicit-mma/f16_128x4096x4096.jsonl` | f16 | 128x4096x4096 | `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2922 | 0.0574 | 0.196x |
| 20 | `reports/2026-05-23-m1-explicit-mma/bf16_128x4096x4096.jsonl` | bf16 | 128x4096x4096 | `zcutlass_sm120_mma_bf16_64x128x64_prefill_reg_epilogue` | 64x128x64 | 2 | register_linear | 0.2929 | 0.0575 | 0.196x |
