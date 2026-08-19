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
| `nn.cumsum.cumsum` | `cub::DeviceScan::{Inclusive,Exclusive}Sum` -- **CPU ONLY**, see 3b |
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

## 3b. `ExclusiveSum` -> `nn.cumsum`: BLOCKED, it is CPU only

`nn.cumsum` looked like the cleanest swap on the list. Its signature even
carries `exclusive: Bool` and an `axis`, so a `[leaf][chunk]` tensor scanned
on axis 1 would be a SEGMENTED exclusive scan, which is exactly the shape our
chunk offsets have.

**It has no GPU form.** Probed 2026-08-19:

    nn.argsort.argsort   ctx: DeviceContext, target: StringSpan = "cpu"
    nn.cumsum.cumsum     neither

`argsort` carries both a `DeviceContext` and a `target` parameter, so it
dispatches to a device kernel. `cumsum` has a single overload with neither,
so it runs on the host. Calling it per level would mean a device-to-host copy,
a host scan, and a copy back, once per level per tree. That undoes the
control-plane work outright.

So the scan stays hand-written, and the DEVIATION BLOCK reason is now "the
shipped primitive is CPU only", not "nothing ships". Re-probe when Modular
adds a target to it.

**Where this leaves the vendor pass:** of everything CatBoost calls
device-wide, only `argsort` and `matmul` have GPU forms here, and `argsort`
is the wrong primitive for a 1-bit partition (section 3). The honest result
of the whole audit is that the vendor swaps available to us today are
`matmul`, once multiclass needs cuSOLVER's job, and nothing else.

---

## 3c. Their partition has TWO paths, and the DEFAULT is not CUB

Reading `SortByFlagsInLeaf` (`split_points.cu:741`) rather than only the CUB
call it contains:

    if (part.Size > FastSortSize()) {          // FastSortSize() == 500000
        cub::DeviceRadixSort::SortPairs(..., 0, 1, stream);
    } else {
        SortWithoutCub(leafId, ...);
    }

**Below 500,000 rows the CUB sort is not used at all.** After the first split
every leaf of an 800k dataset is under that, so the path CatBoost actually
runs almost everywhere is `SortWithoutCub`, which is:

1. `cub::DeviceScan::ExclusiveSum` over the flag bits, per leaf
2. `ReorderOneBitImpl<bool, ui32, N=1, blockSize=512>`
   (`cuda_util/kernel/reorder_one_bit_impl.cuh:127`), which is ORDINARY
   PORTABLE CUDA, not a vendor call

`ReorderOneBitImpl`, transliterated from their source:

    totalOnes  = offsets[size-1] + ((tempKeys[size-1] >> bit) & 1)
    totalZeros = size - totalOnes
    onesBefore   = offsets[idx]
    zeroesBefore = idx - onesBefore
    offset = isZero ? zeroesBefore : (totalZeros + onesBefore)
    keys[offset] = key;  values[offset] = value

So the scan is over the ELEMENTS, not over chunks, and the scatter reads a
per-element rank. Ours counts zeros per 256-row chunk, scans the chunk
counts, then scatters. Functionally the same partition; structurally a
different decomposition, and ours is the one nobody has diffed against a
reference.

**Per rule 0b, ours is the suspect.** The port to make is `ReorderOneBitImpl`
plus a device-wide exclusive scan, with their 500,000-row switch to a 1-bit
sort above it. The scan is the piece with no shipped GPU primitive, which is
what section 3b established.

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

## 5. Wrong DEFAULTS found alongside the vendor audit

Not vendor calls, but the same category of error: a value we chose where
CatBoost had already chosen one. Each is a live divergence from stock
CatBoost behaviour, not a scope gap.

| ours | theirs | citation |
|---|---|---|
| score function is L2, and not configurable | **Cosine** is the shipped default | `oblivious_tree_options.cpp:22` |
| `max_leaf_value` clamp | **no such parameter exists in CatBoost**; they have `RegularizeImpl`, which zeroes leaves under `MinLeafWeight` | `oracle_interface.h:43-53` |
| `l2_leaf_reg` fed the leaf kernel but the score kernel got a literal `3.0` | one field feeds both | `greedy_search_helper.cpp:466`, `:646` -- FIXED |
| tie-break kept the lower thread id | tie-break keeps the lower BIN index | `compute_scores.cu:30` -- FIXED |
| leaf guard `w <= 0`, no epsilon in the denominator | `w > 1e-20`, and `+1e-20` | `greedy_search_helper.cpp:646`, `descent_helpers.cpp:87` -- FIXED |
| `replicas_for`, an invented width heuristic | occupancy: `blocksPerSm * SMCount()` | `hist_binary.cu:95` -- FIXED |
| threadgroup `barrier()` | `tiled_partition<32>().sync()` | `hist_one_byte.cu` -- FIXED |
| `LOAD_SIZE = 1` | `ELoadSize::FourElements` | `hist_one_byte.cu:47` -- FIXED for one-byte |
| tree grows to `max_depth` | stop when a split repeats | `oblivious_tree_doc_parallel_structure_searcher.cpp:134` -- FIXED |
| no `Score < 0` gate | a leaf is only put forward if its best score is negative | `greedy_search_helper.cpp:362` |
| no feature weights on the gain | `gain *= binFeaturesWeights[featureId]` | `compute_scores.cu:136` |
| `partStats` are Float32 | **double**, and the right-child subtraction is done in double before narrowing | `compute_scores.cu:60`, `:99` |

## 6. Structural substitutions still open

Places where theirs has a shape ours does not, found by the same audits.
None of these is a vendor call; all are portable CUDA we simply have not
ported.

| theirs | ours | why it matters |
|---|---|---|
| `GatherInplaceLeqSize<SIZE>` shared-memory fast path below 1024 rows, ONE launch (`split_points.cu:52-112`) | always the global path, three launches | every leaf at depth 4+ on normal data; directly the fixed per-tree overhead |
| Copy then Gather, both restricted to leaf ranges (`split_points.cpp:71-88`) | Gather then a FULL-ARRAY copy | correctness landmine plus full-`n_rows` traffic per level |
| stat columns chunked 8 at a time (`split_points.cpp:66`) | all stats in one shot | temp storage sized to the chunk, not the whole thing |
| five single-leaf kernel variants, dispatched at count `<= 2` | always the plural id-list form | a host id-buffer write and copy at depths 0 and 1 |
| `partsCpu` is PINNED host memory written by the kernel (`split_points.cpp:177`) | an ordinary device buffer nothing reads | a full D2H stall every level |
| `NonZeroLeaves` / `ZeroLeaves` disjoint sets (`split_properties_helper.cpp:762-784`) | one undifferentiated list fed to both | a wasted histogram launch per empty leaf per level |
| grid x from `SMCount()`, halved above 4 leaves (`split_points.cu:220`) | `min(256, ceil(n_rows/256))`, device-blind | never adapts across GPUs |
| `WriteThrough` = `cub::ThreadStore<STORE_WT>` on every streaming store | plain stores | keeps write-once data out of L1; may not port to Apple, but the drop is unrecorded |

## 7. The rule for a genuine gap

If nothing ships, hand-write it AND record the search that came up empty in a
`DEVIATION BLOCK`, so the next person does not repeat it and does not assume
it was never looked for. See `metal-hardware-gaps` in memory for what the
hardware genuinely lacks: no float `atomicAdd`, no warp shuffles, no Metal
streams, no device-to-device copy.
