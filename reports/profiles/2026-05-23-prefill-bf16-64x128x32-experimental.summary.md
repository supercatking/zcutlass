# Nsight Compute GEMM Summary: m=128 n=4096 k=4096 dtype=bf16

## Kernels

- `void unnamed>::wmma_gemm_kernel<__nv_bfloat16, 64, 128, 1, 2>(GemmDesc, const T1 *, const T1 *, const T1 *, T1 *)`

## Throughput

| Signal | Value | Metric |
| --- | ---: | --- |
| SM | 14.43 % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| TENSOR | 9.53 % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` |
| DRAM | 9.05 % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |

## Occupancy And Resources

| Signal | Value | Metric |
| --- | ---: | --- |
| achieved | 16.66 % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| theoretical | 3 block | `launch__occupancy_limit_registers` |
| registers per thread | 71 register/thread | `launch__registers_per_thread` |
| shared memory per block | 46.08 KB/block | `launch__shared_mem_per_block_allocated` |
| block size | 256 | `launch__block_size` |
| grid size | 64 | `launch__grid_size` |

## Top Stalls

| Reason | Value | Metric |
| --- | ---: | --- |
| long scoreboard | 9.51 inst | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` |
| wait | 1.36 inst | `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` |
| lg throttle | 1.04 inst | `smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio` |
| selected | 1 inst | `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` |
| short scoreboard | 0.98 inst | `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` |
| math pipe throttle | 0.88 inst | `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` |
| mio throttle | 0.54 inst | `smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio` |
| barrier | 0.25 inst | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` |
