# Kernel Optimization Notes

The current implementation is a correct WMMA Tensor Core baseline. It is a
foundation for measurement and dispatch work, not yet a peak SM120 kernel.

## Registered Kernels

The built-in manifest currently registers four row-major operations:

- `zcutlass_sm120_tensorop_f16_64x128x16`
- `zcutlass_sm120_tensorop_f16_64x64x16`
- `zcutlass_sm120_tensorop_bf16_64x128x16`
- `zcutlass_sm120_tensorop_bf16_64x64x16`

The benchmark JSONL includes the selected kernel name and the registered
operation count so measurements can be traced back to a concrete implementation.

## Known Gap

The baseline uses synchronous WMMA, scalar global-to-shared copies, and a generic
epilogue. It is expected to trail cuBLAS on large aligned GEMMs until the
mainloop and epilogue are specialized.

## Next Optimization Order

1. Add small-M kernels for LLM decode shapes (`M <= 16`) to avoid wasting the
   `64x*` tile.
2. Add aligned no-bias fast paths for dense multiples of tile sizes.
3. Double-buffer the K loop to reduce barrier and memory dependency stalls.
4. Replace WMMA with explicit MMA/register epilogue once the profiler makes the
   current bottleneck clear.
5. Investigate SM120-native Blackwell tensor-core paths only after the WMMA
   baseline has a stable measurement story.

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

