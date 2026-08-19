# Vendor calls: what CatBoost calls, and what we should call

**Rule (PORTING_RULES 0, sharpened 2026-08-19 by Andrew):** when the
incumbent calls a VENDOR LIBRARY, the faithful port calls OUR vendor library.
Hand-writing a replacement is the same category error as inventing an
algorithm. CatBoost did not hand-write a segmented radix sort; they called
CUB. **Reproducing the CALL is the port. Reproducing the IMPLEMENTATION is
inventing**, and it loses to a kernel somebody tuned.

Modular ships tuned kernels that compile in this toolchain and run on Metal.
They are the counterpart to cuBLAS, cuSOLVER, CUB and Thrust, and they are
what the incumbents' vendor calls should map onto.

---

## 1. What Modular ships, probed 2026-08-19

Every row below was confirmed to import and compile here.

| Modular | CUDA counterpart |
|---|---|
| `linalg.matmul.matmul` | cuBLAS GEMM |
| `nn.cumsum.cumsum` | `cub::DeviceScan::{Inclusive,Exclusive}Sum` |
| `nn.argsort.argsort` | `cub::DeviceRadixSort::SortKeys` (full width) |
| `nn.gather_scatter.gather` | `thrust::gather` |
| `nn.topk.top_k` | `cub::DeviceSelect` / partial sort |
| `nn.softmax.softmax` | fused reduce + exp, no single CUB analog |
| `nn.concat.concat` | `thrust::copy` into a joined buffer |

**Probed and NOT found** under any name tried: a device-wide `reduce`, a
`scan` taking a custom operator, and any SEGMENTED form of sort or reduce.
`algorithm.{reduce,sum,scan}`, `nn.{reductions,reduce,sort}` and
`algorithm.reduction` all fail to resolve. That search is not finished; do not
hand-write on the strength of it without extending the search first.

---

## 2. Every vendor primitive CatBoost calls

Grepped from their whole `catboost/cuda` tree.

### Device-wide, i.e. things with a Modular counterpart

| their file | vendor call | ours today | swap |
|---|---|---|---|
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceRadixSort::SortPairs` | hand-written stable partition | **NO, see section 3** |
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceScan::ExclusiveSum` | hand-written chunk offsets | `nn.cumsum` -- OPEN |
| `cuda_util/kernel/scan.cu` | `cub::DeviceScan::{Inclusive,Exclusive}{Sum,Scan}` | not ported | `nn.cumsum` for the Sum forms |
| `cuda_util/kernel/reduce.cu` | `cub::DeviceReduce::Reduce`, `ReduceByKey` | not ported | nothing found yet |
| `cuda_util/kernel/reduce.cu` | `cub::DeviceSegmentedReduce::Reduce` | hand-written `partitions_reduce` | nothing found yet |
| `cuda_util/kernel/segmented_sort.cu` | `cub::DeviceSegmentedRadixSort::*` | not ported | nothing segmented found |
| `cuda_util/kernel/reduce.cu` | `thrust::maximum`, `minimum`, `plus` | not ported | plain comparisons |
| `cuda_util/kernel/mvs.cu` | `thrust::transform` | not ported | elementwise |
| `methods/kernel/linear_cusolver.cuh` | cuSOLVER dense (`cusolverDn*`) | not ported | `linalg` -- OPEN, multiclass Newton |

### Block and warp scope, i.e. things INSIDE one kernel

These have no device-wide analog and are not swap candidates. They are
ordinary kernel code and get transliterated.

| their file | vendor call | ours today |
|---|---|---|
| `greedy_subsets_searcher/kernel/compute_scores.cu` | `cub::BlockReduce` | hand-written tree reduction |
| `greedy_subsets_searcher/kernel/histogram_utils.cu` | `cub::WarpScan` | hand-written serial scan |
| `greedy_subsets_searcher/kernel/histogram_utils.cu` | `cub::ShuffleIndex` | **BLOCKED**: no warp shuffles in Mojo 1.0 |
| various | `cub::BlockRadixSort`, `cub::BlockScan` | not ported |
| various | `cub::ThreadLoad/ThreadStore`, `cub::CacheModified*Iterator` | plain loads; these are cache-modifier hints |
| various | `cub::LoadDirectWarpStriped` | our striping is written out longhand |

---

## 3. `SortPairs` -> `nn.argsort`: REFUTED, do not do it

This looked like the obvious first swap. It is wrong twice over.

**They sort ONE BIT.** `split_points.cu:676-685`:

    cub::DeviceRadixSort::SortPairs<bool, ui32>(..., part.Size,
                                                0,   // begin_bit
                                                1,   // end_bit
                                                stream);

`begin_bit=0, end_bit=1` is a SINGLE radix pass, which is a stable partition
by a boolean, O(n). Our count-chunks / scan-chunks / scatter is exactly that.
`nn.argsort` is a full argsort over a 32-bit key and is not segmented, so
using it means several passes plus a composite key to fake the segments.
Slower, and LESS faithful: the primitive they call and the one we would call
are not the same primitive.

**And it is under a tenth of the time.** At 800k rows: histogram 2.199 ms at
only 32 features, stable partition 0.376 ms, the other six phases 1.381 ms.
The partition's share FALLS as features grow, because it does not scale with
them and the histogram does.

---

## 4. What actually paid, and it was not a vendor call

The 4.3x gap was in the histogram, and reading their kernel found two things
that were mis-ported rather than un-ported:

- `ELoadSize::FourElements` (`hist_one_byte.cu:47`). They load four elements
  per thread; we loaded one. Ported with `AlignMemoryAccess`. ~4%.
- `tiled_partition<32>(...).sync()`. They sync a WARP; we widened it to a
  threadgroup `barrier()`, which at 8 bits is 32 threadgroup barriers per
  point. `syncwarp` imports from `max.gpu.sync` and had been wrongly believed
  absent. **21%: 800k went 122.9 -> 97.3 ms/tree.**

`hist_binary` and `hist_half_byte` still widen `tiled_partition<8>` the same
way. Same swap, not yet made.

**The lesson for this file: check for a MIS-port before reaching for a vendor
swap.** Both wins were things they already do that we did differently, and
neither needed a library.

---

## 5. The rule for a genuine gap

If nothing ships, hand-write it AND record the search that came up empty in a
`DEVIATION BLOCK`, so the next person does not repeat it and does not assume
it was never looked for. See `metal-hardware-gaps` in memory for what the
hardware genuinely lacks: no float `atomicAdd`, no warp shuffles, no Metal
streams, no device-to-device copy.
