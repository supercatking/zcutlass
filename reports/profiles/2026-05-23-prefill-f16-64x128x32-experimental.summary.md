# Nsight Compute GEMM Summary: m=128 n=4096 k=4096 dtype=f16

## Kernels

- `void unnamed>::wmma_gemm_kernel<__half, 64, 128, 1, 2>(GemmDesc, const T1 *, const T1 *, const T1 *, T1 *)`

## Throughput

| Signal | Value | Metric |
| --- | ---: | --- |
| SM | 8.18 % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| TENSOR | 8.18 % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` |
| DRAM | 7.76 % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |

## Occupancy And Resources

| Signal | Value | Metric |
| --- | ---: | --- |
| achieved | 16.67 % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| theoretical | 4 block | `launch__occupancy_limit_registers` |
| registers per thread | 64 register/thread | `launch__registers_per_thread` |
| shared memory per block | 46.08 KB/block | `launch__shared_mem_per_block_allocated` |
| block size | 256 | `launch__block_size` |
| grid size | 64 | `launch__grid_size` |

## Top Stalls

| Reason | Value | Metric |
| --- | ---: | --- |
| long scoreboard | 13.16 inst | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` |
| short scoreboard | 2.51 inst | `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio` |
| wait | 1.15 inst | `smsp__average_warps_issue_stalled_wait_per_issue_active.ratio` |
| selected | 1 inst | `smsp__average_warps_issue_stalled_selected_per_issue_active.ratio` |
| math pipe throttle | 0.75 inst | `smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio` |
| barrier | 0.73 inst | `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio` |
| dispatch stall | 0.17 inst | `smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio` |
| not selected | 0.15 inst | `smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio` |
