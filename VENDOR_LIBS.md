# Vendor calls: what CatBoost calls, and what we should call

> **RULE CHANGE, Andrew, 2026-08-19 (evening). READ THIS BEFORE THE REST OF
> THIS FILE.** The rule this document was written to serve — "where the
> incumbent calls a vendor primitive, call OURS" — is **DELETED**. It is
> replaced by one rule: **FOLLOW THEIR DISPATCH.** Port the path their library
> actually takes for the parameters in question.
>
> All that survives is a narrow exception: where the path their dispatch
> actually takes calls a **CLOSED** library we cannot read or port (cuBLAS,
> cuSOLVER), call the MAX equivalent, because there is nothing to port. **CUB
> and Thrust are OPEN.** Their kernels are readable, so they are PORT
> candidates, not substitution candidates.
>
> **What killed it: a device-wide vendor call cannot be fused.** It reads its
> input from memory and writes its output to memory, by construction.
> Substituting one for a step that belongs INSIDE a kernel freezes the unfused
> structure permanently. k-NN substituted `linalg.matmul` for its distance step
> and `nn.topk` for its selection under the old rule, so the distance matrix had
> to be materialized, so a 13 ms job takes 306 ms — while cuVS's default path
> for those same parameters (`knn_brute_force.cuh:443` -> `fusedL2Knn`) calls no
> vendor primitive at all. It is their own tile kernel with a register-resident
> `faiss_select::WarpSelect` queue. **When they fuse, they hand-write.**
>
> The probe results below are still VALID and still useful — what MAX ships and
> what it does not is a fact. What is no longer valid is treating an available
> primitive as a reason to substitute. Before citing any row of this file,
> answer in writing: does their dispatch take this path for these parameters,
> and is this step standalone in THEIR code too? If either answer is no, port
> their kernel.

**What this file is now.** It is an inventory of what MAX ships and what it
does not, and a ledger of which substitutions were made, which were rejected
and why. It is no longer an argument that a shipped primitive should be
called. The rule that made that argument is deleted by the banner above and
the sentences asserting it are deleted with it rather than annotated.

The narrow case where a substitution is still right: the path their dispatch
actually takes calls a CLOSED library. cuBLAS and cuSOLVER have no source, so
there is nothing to port and `linalg.matmul` / `linalg.gemv.gemv_gpu` are the
faithful mirror. CUB and Thrust are open and are port candidates.

**Block and warp collectives are a THIRD case and the banner does not reach
them.** `block.sum` and `warp.prefix_sum` run INSIDE one kernel. They cannot
materialize an intermediate, they cannot break a fusion, and they are what
CUB's own `BlockReduce` is: a collective, not an algorithm. Substituting one
changes which instructions a kernel issues and nothing about its structure.
Every swap this file's section 8 records as MADE is either that or a closed
library.

---

## 1. What Modular ships, probed 2026-08-19

Every row below was confirmed to import and compile here, and the whole table
was re-checked against the published API on 2026-08-19.

| Modular | CUDA counterpart |
|---|---|
| `linalg.matmul.matmul` | cuBLAS GEMM (`transpose_b` only; `transpose_a` is refused) |
| `linalg.gemv.gemv_gpu` | cuBLAS GEMV. `linalg.gemv.gemv` is the CPU one; taking the obvious name is the wrong call |
| `nn.cumsum.cumsum` | `cub::DeviceScan::{Inclusive,Exclusive}Sum` -- **CPU ONLY**, see 3b |
| `nn.argsort.argsort` | `cub::DeviceRadixSort::SortKeys` (full width) |
| `nn.gather_scatter.gather` | `thrust::gather` |
| `nn.topk.top_k` | `cuvs::selection::select_k`, CUB partial sort. Takes `ctx`, `target`, `largest`, `axis`, `sorted` |
| `nn.softmax.softmax` | fused reduce + exp, no single CUB analog |
| `nn.concat.concat` | `thrust::copy` into a joined buffer |
| `algorithm.reductions.reduce_{sum,max,min,mean,product,argmax,argmin}` | `cub::DeviceReduce::Reduce`. Takes an `InputFn`, an `OutputFn`, a `reduce_dim`, a `target` and an optional `DeviceContext` |
| `max.gpu.primitives.block.{sum,max,min,broadcast,prefix_sum}` | `cub::Block{Reduce,Scan}` |
| `std.gpu.primitives.warp.{sum,max,min,prefix_sum,shuffle_up,shuffle_down,shuffle_xor,shuffle_idx,broadcast}` | `cub::Warp{Reduce,Scan}`, `__shfl_*` |

**TWO ROWS OF THIS TABLE WERE FALSE NEGATIVES AND BOTH WERE IMPORT PATHS.**

- The warp primitives are under `std.gpu.primitives.warp`. Searches under
  `max.gpu.primitives.warp` fail with "unable to locate module 'warp'", which
  reads exactly like the primitive being absent. The block primitives ARE
  under `max`, so the two namespaces are split and neither one implies the
  other. Probed on this M4: `shuffle_xor(lane_id, 1)` returns the partner's
  lane id, 0 wrong of 32; warp `prefix_sum(1)` 0 wrong of 32.
- The reduction module is `algorithm.reductions`, PLURAL. This file previously
  recorded `algorithm.reduce`, `algorithm.reduction` and `nn.reductions` as
  failing to resolve, which they do, and concluded no device-wide reduce
  ships, which is wrong.

**Still NOT found**, and these are the honest gaps: a device-wide SCAN with a
GPU form (see 3b), and any SEGMENTED form of sort or reduce. Segmented is the
one that matters for boosting, because leaf partitions are ragged and every
shipped reduction reduces a dense axis of a rectangular shape.

The lesson the two false negatives buy: **a NOT FOUND is only worth writing
down if it names the exact paths tried, and it expires.** Re-probe before
hand-writing anything on the strength of one.

---

## 2. Every vendor primitive CatBoost calls

Grepped from their whole `catboost/cuda` tree.

### Device-wide, i.e. things with a Modular counterpart

| their file | vendor call | ours today | swap |
|---|---|---|---|
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceRadixSort::SortPairs` | hand-written stable partition | **NO, see section 3** |
| `greedy_subsets_searcher/kernel/split_points.cu` | `cub::DeviceScan::ExclusiveSum` | three-pass decoupled scan; its BLOCK scan is now `block.prefix_sum` | `nn.cumsum` is CPU-only, so the device-wide decoupling stays ours -- see 3b |
| `cuda_util/kernel/scan.cu` | `cub::DeviceScan::{Inclusive,Exclusive}{Sum,Scan}` | not ported | same: no GPU form ships |
| `cuda_util/kernel/reduce.cu` | `cub::DeviceReduce::Reduce`, `ReduceByKey` | not ported | `algorithm.reductions.reduce_*`, which DOES ship with a GPU target. Nothing calls it yet |
| `cuda_util/kernel/reduce.cu` | `cub::DeviceSegmentedReduce::Reduce` | hand-written `partitions_reduce`, per-block reduce now `block.sum` | nothing segmented ships; our segments are ragged leaf ranges and no dense-axis reduction expresses them |
| `cuda_util/kernel/segmented_sort.cu` | `cub::DeviceSegmentedRadixSort::*` | not ported | nothing segmented found |
| `cuda_util/kernel/reduce.cu` | `thrust::maximum`, `minimum`, `plus` | not ported | plain comparisons |
| `cuda_util/kernel/mvs.cu` | `thrust::transform` | not ported | elementwise |
| `methods/kernel/linear_cusolver.cuh` | cuSOLVER dense (`cusolverDn*`) | not ported | `linalg` -- OPEN, multiclass Newton |

### Block and warp scope, i.e. things INSIDE one kernel

**This section used to say these are "not swap candidates". That was wrong,
and it was wrong for the whole repository, not just for one row.** Modular
ships the block collectives under `max.gpu.primitives.block` and the warp
collectives and shuffles under `std.gpu.primitives.warp`. A `cub::BlockReduce`
in their kernel has a one-call counterpart in ours. Hand-writing a tree
reduction next to it is the same category error as hand-writing a GEMM.

| their file | vendor call | Mojo counterpart | ours today |
|---|---|---|---|
| `greedy_subsets_searcher/kernel/compute_scores.cu` | `cub::BlockReduce` | `block.sum` / `block.max` / `block.min` | hand-written tree reduction -- **SWAP OPEN**, boosting lane |
| `greedy_subsets_searcher/kernel/histogram_utils.cu` | `cub::WarpScan` | `warp.prefix_sum` | hand-written serial scan -- **SWAP OPEN**, boosting lane |
| `greedy_subsets_searcher/kernel/histogram_utils.cu` | `cub::ShuffleIndex` | `warp.shuffle_idx`, or `shuffle_xor` where the source lane is a fixed XOR partner | was recorded **BLOCKED, "no warp shuffles in Mojo 1.0"**. False; the import path was wrong. **SWAP OPEN**, boosting lane |
| `cuda_util/kernel/partitions_reduce` call path | `cub::BlockReduce` | `block.sum` | **SUBSTITUTED** in `ported/gpu_util/partitions_reduce.mojo`. It had been faking a reduce with `prefix_sum` plus a `broadcast` of the last lane, on a note claiming Mojo exposed a scan and not a reduce |
| `reorder_one_bit_impl.cuh` scan | `cub::BlockScan::ExclusiveSum` | `block.prefix_sum[exclusive=True]` | **SUBSTITUTED** in `ported/gpu_util/kernel/reorder_one_bit.mojo`, replacing a Hillis-Steele loop and its shared page |
| various | `cub::BlockRadixSort` | **NOT FOUND** | not ported |
| various | `cub::ThreadLoad/ThreadStore`, `cub::CacheModified*Iterator` | **NOT FOUND** | plain loads; these are cache-modifier hints |
| various | `cub::LoadDirectWarpStriped` | **NOT FOUND** | our striping is written out longhand |

The three rows marked SWAP OPEN are in `ported/methods/`, which this pass does
not own. The exact anchors are in the vendor lane's cross-file note.

---

## 3. `SortPairs` -> `nn.argsort`: REFUTED, do not do it

RE-CHECKED 2026-08-19 against the published signature, and it still holds.
`argsort` is `argsort[ascending, target](output, input, ctx)`: a full-width
argsort of one flat tensor, with no bit range and no segments. Nothing in it
has changed.

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

**It has no GPU form.** Probed 2026-08-19, and re-checked against the
published signature the same day:

    nn.argsort.argsort   ctx: DeviceContext, target: StringSpan = "cpu"
    nn.cumsum.cumsum     neither

    cumsum[dtype, exclusive, reverse, *, axis](output, input)

`argsort` carries both a `DeviceContext` and a `target` parameter, so it
dispatches to a device kernel. `cumsum` has a single overload with neither,
so it runs on the host. Calling it per level would mean a device-to-host copy,
a host scan, and a copy back, once per level per tree. That undoes the
control-plane work outright.

So the scan stays hand-written, and the DEVIATION BLOCK reason is now "the
shipped primitive is CPU only", not "nothing ships". Re-probe when Modular
adds a target to it.

**Where this leaves the vendor pass, corrected 2026-08-19.** The earlier
version of this paragraph said the only swap available to us was `matmul`.
That followed from two false negatives in section 1 and is wrong. Of what
CatBoost calls device-wide, `argsort`, `matmul`, `gemv_gpu`, `top_k` and
`algorithm.reductions.reduce_*` all have GPU forms; `cumsum` does not; nothing
SEGMENTED does. `argsort` is still the wrong primitive for a 1-bit partition
(section 3), which is a different objection from "nothing ships".

At BLOCK and WARP scope, which section 2 used to exclude from the audit
outright, the counterparts are complete: reduce, scan, shuffle, broadcast.
That is where the swaps this pass actually made are.

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

That port exists: `ported/gpu_util/kernel/reorder_one_bit.mojo`. Its scan is
three passes, and as of this round the WITHIN-BLOCK pass is
`block.prefix_sum[exclusive=True]` rather than a hand-written Hillis-Steele
loop over a shared page. What is still ours is the decoupling across blocks,
which is exactly the piece `cub::DeviceScan` exists for and `nn.cumsum`
cannot do on a device.

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
| five single-leaf kernel variants, dispatched at count `<= 2` | always the plural id-list form | a host id-buffer write and copy at depths 0 and 1 |
| `partsCpu` is PINNED host memory written by the kernel (`split_points.cpp:177`) | an ordinary device buffer nothing reads | a full D2H stall every level |
| `NonZeroLeaves` / `ZeroLeaves` disjoint sets (`split_properties_helper.cpp:762-784`) | one undifferentiated list fed to both | a wasted histogram launch per empty leaf per level |
| grid x from `SMCount()`, halved above 4 leaves (`split_points.cu:220`) | `min(256, ceil(n_rows/256))`, device-blind | never adapts across GPUs |
| `Ldg` / `__ldg` = `cub::ThreadLoad<LOAD_LDG>` on every streaming read (`kernel_helpers.cuh:180`) | plain loads, in every kernel in the tree | `std.gpu.intrinsics.ldg` and `max.gpu.memory.load[read_only=True]` BOTH ship and neither is called anywhere; verified 2026-08-19 |

`GatherInplaceLeqSize`, the leaf-range-restricted Copy-then-Gather pair and the
eight-column chunk were all three on this list. They are ported and reached as
of 2026-08-19; `launch_reorder_in_leaves` is their `TSplitPointsKernel::Run`
(`split_points.cpp:64-136`) including the fused `1 + statCount` fast launch, so
the fast path is ONE launch per level, as theirs is.

`WriteThrough` = `cub::ThreadStore<cub::STORE_WT>` (`kernel_helpers.cuh:190`)
was also on this list as "the drop is unrecorded". It is recorded now, in the
DEVIATION BLOCK on `gather_inplace_kernel`: `max.gpu.memory.memory` ships
`load` with `read_only`, `cache_policy` and `eviction_policy` and ships no
store at all; `std.gpu.intrinsics` ships `ldg` and `threadfence` and no store;
`std.sys.intrinsics` ships `masked_store`, `compressed_store`, `strided_store`
and `scatter`, all addressing patterns with no cache or temporality hint. The
load half of their pair ships, the store half does not.

## 7. The rule for a genuine gap

If nothing ships, hand-write it AND record the search that came up empty in a
`DEVIATION BLOCK`, so the next person does not repeat it and does not assume
it was never looked for. A `DEVIATION BLOCK` that names no searched path is
not evidence of anything.

### What the hardware and the toolchain GENUINELY lack

This list was wrong in two places and both were load-bearing. Corrected
2026-08-19, each line against a probe on this M4:

| claim | status |
|---|---|
| no float `atomicAdd` | **FALSE.** 1024 threads adding 1.0 return exactly 1024.0. Every design in this tree that cited this has to stand on another argument or be undone. `mojo_only/fixed_point.mojo` stands, on DETERMINISM: an integer accumulator is order-independent, a float atomic is not, and reproducibility across devices is the claim this repository makes |
| no warp shuffles | **FALSE.** `std.gpu.primitives.warp` ships `shuffle_{up,down,xor,idx}`, `broadcast`, `sum`, `max`, `min`, `prefix_sum`. `shuffle_xor(lane_id, 1)` returns the partner lane, 0 wrong of 32. The earlier searches used `max.gpu.primitives.warp`, which does not resolve; the BLOCK primitives are the ones under `max` |
| no Metal streams | true, unretested. `ctx.stream()` raises; one queue |
| no device-to-device copy | true, unretested |
| no float64 on Apple | true. `decomposition/mojo_only/jacobi_eigh_device.mojo` documents what it costs |
| `linalg.matmul` refuses `transpose_a` | true, still. Not a blocker: transpose and use the N-T shape |
| `linalg.matmul` at `n = 1` returns zeros | true. Call `gemv_gpu` there, which is what RAFT calls too |
| `linalg.transpose` signals on device buffers | true. It dispatches to a host strided copy |

### DEVIATION BLOCKS this pass added or corrected

- `ported/gpu_util/partitions_reduce.mojo`, segmented reduce: the search is
  now written into the banner. `algorithm.reductions` reduces a dense axis;
  our segments are ragged leaf ranges; nothing shipped expresses that.
- `ported/gpu_util/kernel/reorder_one_bit.mojo`, device-wide scan: `nn.cumsum`
  has no `ctx` and no `target`, re-checked against the published signature.
  Only the cross-block decoupling is hand-written now.
- `cluster/mojo_only/reduce_by_key.mojo`: the fixed-point accumulator was
  justified by "Metal has no float atomic add". That sentence is deleted, not
  annotated, and the determinism argument replaces it.
- `ported/methods/greedy_subsets_searcher/kernel/split_points.mojo`,
  `gather_inplace_kernel`: `WriteThrough`. The old text said "No Mojo
  spelling; a plain store" and named no search, which section 7 says is not
  evidence of anything. The three module paths searched are now written into
  the block, along with the finding that the LOAD half of the same CUB pair
  (`ldg`) does ship and this tree calls it nowhere.
- The same file's launch-count deviation is DELETED rather than annotated: it
  claimed no call site could hold both payload pointers, which was true only
  because the driver had been split in two. `launch_reorder_in_leaves` is
  their single `Run` and the fast path is their single launch.

---

## 8. The substitution ledger, 2026-08-19 vendor pass

Every swap MADE, every swap REJECTED, and the reason. A rejection with a
reason is worth as much as a swap, because it is what stops the next person
repeating the search.

**Written before the rule change at the top of this file and re-checked
against it afterwards.** Each MADE row is now labelled with which of the two
surviving cases it falls under:

- **CLOSED**: their dispatch calls cuBLAS or cuSOLVER, which have no source.
  Nothing to port, so the MAX equivalent is the faithful mirror.
- **COLLECTIVE**: a block or warp collective inside one kernel. It cannot
  materialize an intermediate and cannot break a fusion.

One row was neither, and it is the one the rule change was written about. It
is reverted.

### MADE

| case | file | was | now calls |
|---|---|---|---|
| COLLECTIVE | `ported/gpu_util/kernel/reorder_one_bit.mojo` | Hillis-Steele block scan over a shared page, 9 iterations and 18 barriers | `max.gpu.primitives.block.prefix_sum[block_size=512, exclusive=True]` |
| COLLECTIVE | `ported/gpu_util/partitions_reduce.mojo`, both kernels | `block.prefix_sum` followed by `block.broadcast` of the last lane, to fake a reduce | `max.gpu.primitives.block.sum[block_size=256]` |
| CLOSED | `core/column_stats.mojo` | `covariance_kernel` (RAFT's column-major contraction plus our split-K) and `covariance_reduce_kernel`, ~250 lines with no caller | nothing. Deleted. `raft::stats::cov` and `tsvd_fit` both call cuBLAS for this shape and materialize the result; `gemm_tn` now DISPATCHES — the hand-written split-K Gram kernel (`core/gram_splitk.mojo`, added 2026-08-19 after `linalg.matmul` measured ~25 GFLOP/s on the tile-starved 32x32x4M shape) on small outputs, transpose + `linalg.matmul` above that |
| CLOSED | `glm/ported/linalg/detail/lstsq.mojo` step 6 | `use_vendor_gemv` flag with a ported contraction on the false arm | `linalg.gemv.gemv_gpu` only. `raft::linalg::gemv` is cuBLAS and is standalone in their code too |
| CLOSED | `core/gemm.mojo` | `gemm_nt_kernel`, RAFT's register-tiled row-major contraction, kept as "the reference" and reached by nothing once the flag above went | nothing. Deleted. **The one to look at first if the rule change should undo more than the k-NN row**: RAFT's contraction is open source and portable, and the argument for deleting it is that the FUSED instantiation of the same policy already lives in `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo`, which is the copy the new rule cares about. The standalone one was 16x slower than `linalg.matmul` at 1M x 128 and had no caller |
| **neither** | `neighbors/` selection | `use_vendor_topk` flag, ported radix select as default | **PROPOSED AND REVERTED**, see the REJECTED table |

### REJECTED, with the reason

| candidate | why not |
|---|---|
| `cub::DeviceRadixSort::SortPairs` -> `nn.argsort` | Section 3. They sort ONE BIT, which is a stable partition; `argsort` is a full 32-bit argsort with no segments. Different primitive, more passes, and under a tenth of the time |
| `cub::DeviceScan::ExclusiveSum` -> `nn.cumsum` | Section 3b. `cumsum` has no `ctx` and no `target`. Calling it per level means a device-to-host copy, a host scan and a copy back, once per level per tree |
| `cub::DeviceSegmentedReduce` -> `algorithm.reductions` | Ragged segments. Every shipped reduction reduces a dense axis of a rectangular shape; leaf partitions are `[offset, offset + size)` and differ every level |
| `raft::linalg::map_offset` (the L2 epilogue) -> anything | It is RAFT's OWN portable elementwise map, not a vendor call. Rule 1: port their kernel. `core/expand_distances.mojo` is that port |
| **the k-NN selection -> `nn.topk.top_k` as the ONLY arm** | **MADE, THEN REVERTED THE SAME HOUR.** I deleted `select_radix.mojo` (390 lines) and `select_warpsort.mojo` (833) and made `top_k` the only selection. The rule changed underneath it: cuVS's DISPATCH for these parameters does not call `select_k` at all, it calls `fusedL2Knn`, whose selection is a register-resident `faiss_select::WarpSelect` queue INSIDE the distance kernel. `select_warpsort.cuh` is the port of exactly that family, so it is the file the new rule wants kept, not deleted. Restored to HEAD; `neighbors/` is another lane's now |
| `fusedDistanceNN`'s contraction -> `linalg.matmul` | RAFT fuses the argmin epilogue into the contraction and never materializes the distance tile. No BLAS call can do that, and the fusion is the entire point of the kernel |
| `row_norm_kernel`, `column_mean_kernel`, `xty_kernel` -> `algorithm.reductions.reduce_*` | Looked like the best of the newly available swaps and is not one. Their upstreams are `raft::linalg::norm`, `raft::stats::mean` and `raft::linalg::gemv(trans)`, all RAFT's OWN open-source kernels, so the rule makes them PORT candidates. All three already call `block.sum` inside, so the arithmetic is already the collective's; only the launch shape differs. Any swap also moves the last bits of a float summation that is in the `IDENTICAL` column's scope, so it needs a run this lane cannot do. Table in `VENDOR_LIBRARIES.md` |
| `unfused_distance_nn::reduce_min_kernel` -> `algorithm.reductions.reduce_argmin` | Worse than neutral. cuVS's dispatch fuses that argmin INTO the distance kernel (`fusedDistanceNN`), and a device-wide reduction is the one thing that cannot be fused into anything |
| `linalg.transpose` -> our `transpose_kernel` | The shipped one signals at runtime on device buffers; it dispatches into a host strided-copy path. Twenty lines of ours is the workaround, and it is recorded rather than assumed |
