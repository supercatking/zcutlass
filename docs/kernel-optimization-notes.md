# Kernel Optimization Notes

The current implementation is a correct WMMA Tensor Core baseline. It is a
foundation for measurement and dispatch work, not yet a peak SM120 kernel.

## Registered Kernels

The built-in manifest currently registers four row-major operations:

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

## Known Gap

The baseline uses synchronous WMMA, scalar global-to-shared copies, and a generic
fallback epilogue. The aligned operation variants compile a `FastPath=true`
kernel for dense `alpha=1,beta=0,bias=null` problems. That path removes boundary
predicates and beta/bias work while preserving the fallback kernel for ragged
shapes.

## Next Optimization Order

1. Double-buffer the K loop to reduce barrier and memory dependency stalls.
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
