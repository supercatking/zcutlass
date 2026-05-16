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

The baseline uses synchronous WMMA, scalar global-to-shared copies, and mostly a
generic epilogue. The aligned operation variants currently provide a separate
dispatch point for dense `alpha=1,beta=0,bias=null` problems; their implementation
still shares the WMMA kernel body, so they are a staging point for future fast
path work rather than a peak kernel.

## Next Optimization Order

1. Add aligned no-bias fast paths for dense multiples of tile sizes.
2. Double-buffer the K loop to reduce barrier and memory dependency stalls.
3. Replace WMMA with explicit MMA/register epilogue once the profiler makes the
   current bottleneck clear.
4. Investigate SM120-native Blackwell tensor-core paths only after the WMMA
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
