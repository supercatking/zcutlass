# Kernel Optimization Notes

The current implementation is a correct WMMA Tensor Core baseline. It is a
foundation for measurement and dispatch work, not yet a peak SM120 kernel.

## Registered Kernels

The built-in manifest currently registers these row-major operations:

- `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill`
- `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k`
- `zcutlass_sm120_tensorop_f16_64x256x16_aligned_prefill_n_gt_k_experimental`
- `zcutlass_sm120_tensorop_f16_64x256x32_aligned_prefill_n_gt_k_experimental`
- `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_experimental`
- `zcutlass_sm120_tensorop_f16_128x128x16_aligned_prefill_experimental`
- `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill`
- `zcutlass_sm120_tensorop_f16_64x128x16`
- `zcutlass_sm120_tensorop_f16_64x64x16`
- `zcutlass_sm120_tensorop_bf16_64x128x16`
- `zcutlass_sm120_tensorop_bf16_64x64x16`
- `zcutlass_sm120_tensorop_f16_64x128x16_aligned`
- `zcutlass_sm120_tensorop_bf16_64x128x16_aligned`
- `zcutlass_sm120_tensorop_f16_32x128x16`
- `zcutlass_sm120_tensorop_f16_32x64x16`
- `zcutlass_sm120_tensorop_bf16_32x128x16`
- `zcutlass_sm120_tensorop_bf16_32x64x16`

The benchmark JSONL includes the selected kernel name and the registered
operation count so measurements can be traced back to a concrete implementation.
The `32x*` entries are registered after the `64x*` fallback kernels because
initial spot checks showed the reused WMMA body is slower for decode shapes; they
are placeholders for a future dedicated small-M mainloop, not the default path.

The f16 prefill `64x128x32` and `64x128x64_n_le_k` entries are promoted
mainloop variants. `64x128x32` groups two WMMA K slices per synchronization
point and handles `N > K` prefill shapes. `64x128x64_n_le_k` groups four K
slices and handles prefill shapes where `N <= K`. The unrestricted
`64x128x64`, `128x128x16`, and the f16 `64x256x*` N>K entries remain
experimental and require `--experimental-kernels` or
`ZCUTLASS_EXPERIMENTAL_KERNELS=1`.

The bf16 prefill `64x128x32` entry is promoted for all aligned prefill buckets.
It groups two WMMA K slices per synchronization point and replaces the previous
bf16 `64x128x16_aligned` prefill default. The old bf16 KGroup=1 aligned kernel
remains available for large/throughput and non-prefill aligned shapes.

## Known Gap

The baseline uses synchronous WMMA, scalar global-to-shared copies, and a generic
fallback epilogue. The aligned operation variants compile a `FastPath=true`
kernel for dense `alpha=1,beta=0,bias=null` problems. That path removes boundary
predicates and beta/bias work while preserving the fallback kernel for ragged
shapes.

The WMMA fast path still spills FP32 accumulator fragments to shared memory
before converting to FP16/BF16 output. That is not just an implementation
oversight: `wmma::store_matrix_sync` stores accumulator fragments as their
accumulator element type, so the current FP32-accumulate WMMA path cannot
directly store to FP16/BF16 D. Removing this shared-memory accumulator spill
requires a lower-level MMA/register-epilogue implementation.

## Next Optimization Order

1. Continue the K-loop pipeline after the f16 aspect-bucket promotion: reduce
   long-scoreboard stalls, test whether grouped loading can extend to BF16, and
   investigate a better `N > K` policy for MLP up-projection.
2. Replace WMMA with explicit MMA/register epilogue once the profiler makes the
   current bottleneck clear.
3. Investigate SM120-native Blackwell tensor-core paths only after the WMMA
   baseline has a stable measurement story.

See [LLM Kernel Family Plan](llm-kernel-family-plan.md) for the first
profile-driven experiments by Decode, Prefill, Large/Throughput, and Fallback
families. The plan intentionally avoids full-shape-specific kernels; experiments
should improve tile families, mainloop behavior, and epilogue policy so the
manifest remains small and explainable.

## Profiling Command

```bash
ncu --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section Occupancy \
  --section SourceCounters \
  ./build/zcutlass_bench --m 256 --n 4096 --k 4096 --dtype f16 --warmup 3 --iterations 5
```
