# GPU performance optimization notes

This folder contains [`gpu_performance.py`](./gpu_performance.py), the CUDA
benchmark used to compare throughput and simulation results, and its
[`gpu_performance_reference.json`](./gpu_performance_reference.json) reference.
Run the benchmark from this folder with:

```bash
python gpu_performance.py
```

It runs 10,000 timed steps for increasing square grids and reports iterations
per second, Nusselt number, and mean-square velocity. The physics metrics must
remain within the project's 2% tolerance when evaluating an optimization.
The current size sequence starts at N=1500; Rayleigh number retains the
established scaling Ra = 1e7 * (N / 300)^3.
Do not update the reference just to record a faster or slower timing from one
run; update it only when a validated implementation is deliberately retained.

## Recent experiment: CUDA Graph step batching

Commit `ecfebb8` captured pairs of LBM steps in CUDA Graphs to reduce host-side
kernel launch overhead. It was reverted in `b328bd5`: the large-grid results
from the user's sequential A/B runs were effectively similar, rather than
showing a repeatable gain. At sizes 1435 and larger, the graph run's throughput
relative to the non-graph run was:

| Grid | Change in iterations/second |
| ---: | ---: |
| 1435 | +2.1% |
| 1794 | +6.1% |
| 2243 | -0.1% |
| 2804 | +2.2% |

Across all 11 sizes, the reported benchmark time was 36.25 s with graphs and
36.86 s without them (about 1.6% less time with graphs in these single runs).
Nu and mean-square velocity were unchanged. This small aggregate difference,
with mixed per-size results and only one run per version, is not enough to
establish a reliable improvement.

Other experiments that did not show a consistent benefit and were discarded:

- Batching 16 step-pairs per graph reduced graph replay submissions in a
  profile, but did not consistently improve end-to-end benchmark time.
- Simplifying per-cell index arithmetic.
- Switching to a 2D tiled launch.
- Fixed 128-thread and 512-thread launch blocks.

Avoid repeating these exact variants without new profiling evidence or a
meaningfully different implementation.

## Potential optimization work

Prioritize changes using measurements from the current non-graph baseline:

1. **Make A/B measurements robust.** Alternate benchmark versions (for example,
   A/B/B/A), use multiple runs and compare medians per grid size. Keep the GPU
   in a comparable thermal and power state. The current benchmark covers a
   broad range of sizes, so retain per-size results rather than relying only on
   one aggregate number.
2. **Profile representative large grids.** Use Nsight Systems to identify host
   launch and synchronization costs, then Nsight Compute on the dominant
   kernels to measure achieved memory bandwidth, cache behavior, occupancy,
   register use, and instruction mix. A previous short Nsight Systems profile
   attributed about 52% of its CUDA API time to kernel launches, but the
   graph experiment did not translate that observation into a clear large-grid
   throughput gain.
3. **Reduce population data movement.** If profiling shows the dominant
   kernels are bandwidth-bound, investigate memory access coalescing, data
   layout, redundant loads/stores, and whether adjacent stages can safely be
   fused. Check register pressure and occupancy as well as bytes transferred;
   fusion or layout changes can trade one bottleneck for another.
4. **Tune launch geometry based on occupancy data.** Explore size- or
   architecture-specific block shapes only when profiler results indicate
   under-occupancy or poor memory access. The previously tried fixed block
   sizes and 2D launch were not consistently faster.
5. **Revisit launch overhead only with end-to-end evidence.** The graph test
   shows that fewer host submissions alone are not a sufficient success
   metric. Any new batching or device-side scheduling approach should be
   tested on the large grids and benchmark duration that matter in practice.

For every retained solver change, check the exact benchmark physics outputs
against the reference (within 2%), and run the relevant CPU/CUDA and boundary
condition comparisons. In particular, preserve odd-grid behavior and both
wall modes when changing streaming, boundary handling, data layout, or kernel
ordering. The existing float-literal and grid-index optimizations are part of
the current baseline and should not be reverted as part of these experiments.
