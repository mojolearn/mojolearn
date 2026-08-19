# Where CatBoost calls a vendor library, and what we should call

**Rule (PORTING_RULES 0, sharpened 2026-08-19):** when the incumbent calls a
VENDOR LIBRARY, the faithful port calls OUR vendor library. Hand-writing a
replacement is the same category error as inventing an algorithm. CatBoost
did not hand-write a segmented radix sort; they called CUB. Reproducing the
CALL is the port. Reproducing the IMPLEMENTATION is inventing, and it loses
to a kernel somebody tuned.

## What Modular actually ships, probed 2026-08-19

Confirmed to import and compile in this toolchain, on Metal:

| ours | resolves | CUDA analog |
|---|---|---|
| `linalg.matmul.matmul` | yes | cuBLAS GEMM |
| `nn.cumsum.cumsum` | yes | `cub::DeviceScan` |
| `nn.argsort.argsort` | yes | `cub::DeviceRadixSort::SortPairs` |
| `nn.gather_scatter.gather` | yes | Thrust gather |
| `nn.topk` (module) | yes, symbol name differs | `cub::DeviceSelect` |

Probed and NOT found under the obvious names: `algorithm.reduce`,
`algorithm.sum`, `algorithm.scan`, `nn.reductions`, `nn.sort`. A reduction
primitive may exist under a name not guessed here; that search is not
finished and nothing should be hand-written on the strength of it.

## What CatBoost calls, by file

Grepped from their tree, `cub::|thrust::|cublas|cusolver`:

| their file | vendor call | ours today |
|---|---|---|
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceRadixSort::SortPairs` | **hand-written** segmented stable partition |
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceScan::ExclusiveSum` | **hand-written** chunk offsets |
| `greedy_subsets_searcher/kernel/compute_scores.cu` | `cub::BlockReduce` | **hand-written** tree reduction |
| `greedy_subsets_searcher/kernel/histogram_utils.cu` | `cub::WarpScan`, `cub::ShuffleIndex` | **hand-written** serial scan |
| `cuda_util/kernel/reduce.cu` | `cub::DeviceSegmentedReduce`, `thrust::maximum/minimum` | **hand-written** `partitions_reduce` |
| `cuda_util/kernel/scan.cu` | `cub::DeviceScan::{Inclusive,Exclusive}{Sum,Scan}` | not ported |
| `cuda_util/kernel/segmented_sort.cu` | `cub::DeviceSegmentedRadixSort::*` | not ported |
| `methods/kernel/linear_cusolver.cuh` | cuSOLVER | not ported (multiclass Newton) |

**Five hand-written replacements for CUB primitives**, and every one of them
is marked `replaced` or `partial+replaced` in `PORTED_MAP.tsv`, which is how
they were found.

## `SortPairs` -> `nn.argsort`: DO NOT. Measured and refuted, 2026-08-19

This looked like the obvious first swap and it is the wrong one, for two
independent reasons.

**1. They sort ONE BIT.** `split_points.cu:676-685`:

    cub::DeviceRadixSort::SortPairs<bool, ui32>(..., part.Size,
                                                0,   // begin_bit
                                                1,   // end_bit
                                                stream);

`begin_bit=0, end_bit=1` is a SINGLE radix pass, which is a stable partition
by a boolean, O(n). Our three kernels -- count chunks, scan chunks, scatter
-- are precisely that. `nn.argsort` is a full argsort over a 32-bit key and
is not segmented, so using it here means several radix passes and a composite
key to fake the segments. It would be slower AND less faithful: the primitive
they call and the primitive we would call are not the same primitive.

**2. It is under a tenth of the time.** Phase timings at 800k rows:

    histogram alone           2.199 ms   (at only 32 features)
    stable partition alone    0.376 ms
    the other six phases      1.381 ms

The partition is about 9.5% of a level at 32 features, and its share FALLS as
features grow because it does not scale with them while the histogram does.
At the 100-feature benchmark it is a rounding error.

**The histogram is the target.** Within it, the barrier count: their
`AddPoint` syncs a `tiled_partition<32>` and ours uses a threadgroup
`barrier()`, which at 8-bit features is 4 features x 8 passes = 32
threadgroup barriers per point. No vendor call fixes that; it needs a
warp-level sync that Mojo does not currently expose.

## The order to fix the rest in
2. `partitions_reduce` -> whatever ships for segmented reduce, once the
   search above is finished.
3. The two scans -> `nn.cumsum.cumsum`.
4. `compute_scores`'s block reduce: LAST, and possibly never. It is a
   32-thread reduction inside one kernel, not a device-wide primitive, so
   there may be nothing to call.

## The rule for a genuine gap

If nothing ships, hand-write it AND record the search that came up empty in a
`DEVIATION BLOCK`, so the next person does not repeat it and does not assume
it was never looked for.
