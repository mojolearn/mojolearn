"""`TOptimizationSubsets`: the state CatBoost's two OBLIVIOUS tree searchers
carry down a level, and the three-step move that takes it from depth `d` to
depth `d+1`.

PORT OF `catboost/cuda/methods/pointwise_optimization_subsets.{h,cpp}` at
CatBoost `54a8143a`, the `TStripeMapping` specialization. Transliterated. Do
not improve.

WHICH FAMILY THIS IS, because this tree already has a different one
--------------------------------------------------------------------
`PORTING.md` 91 B: CatBoost has THREE GPU searchers, not two.
`TFeatureParallelObliviousTreeSearcher` and
`TDocParallelObliviousTreeSearcher` share this entire stack --
`pointwise_optimization_subsets.h`, `pointwise_scores_calcer.h`,
`histograms_helper.{h,cpp}`, `pointwise_kernels.{h,cpp}` and the
`pointwise_hist2*` kernel family. `TGreedySubsetsSearcher`, which is what
`gbdt/methods/greedy_subsets_searcher/` already ports, shares NONE of it and
has its own histogram family.

So this file is not a second spelling of `TPointsSubsets`. The two structs
answer different questions:

    TPointsSubsets   (greedy)     an explicit TVector<TLeaf>, leaves that
                                  come and go, `Partitions` indexed by a
                                  leaf id the host tracks
    TOptimizationSubsets (this)   NO leaf list at all. The live leaves at
                                  depth d are exactly slots
                                  [0, 1 << (CurrentDepth + FoldBits)) and a
                                  document's leaf IS the low bits of its
                                  entry in `Bins`

That is why the whole of `Split` is: write one more bit into every
document's bin, stable-sort by that bit, recount the partitions. There is no
leaf bookkeeping because an oblivious level does not have any.

THE THREE STEPS, and where each one already lives in this repository
---------------------------------------------------------------------
`TSubsetsHelper<TStripeMapping>::Split` (`pointwise_optimization_subsets.cpp
:26-52`) is four calls, and three of them are primitives this tree already
has. NOTHING BELOW REIMPLEMENTS THEM.

    UpdateBinFromCompressedIndex(cindex, feature, bin, docsForBins,
                                 CurrentDepth + FoldBits, Bins)
        -> `update_bins_from_compressed_index_kernel` below. NEW: the port
           of `gpu_data/kernel/split.cu:179-201`. The greedy family's
           `split_points.mojo` computes the same predicate but writes a
           per-row FLAG for a segmented partition, not a bit into a bin.

    ReorderBins(Bins, Indices, CurrentDepth + FoldBits, 1)
        -> `gbdt/gpu_util/kernel/radix_sort.launch_radix_sort_bins`,
           UNCHANGED. It is already the port of `cuda_util/sort.cpp:544`,
           already one-bit-per-pass, and `bits == 1` here means exactly ONE
           pass -- an odd count, so their `if (doubleBufferKeys.Current() !=
           keys)` copy-back (`sort_templ.cuh:53`) fires on every level.

    UpdatePartitionDimensions(Bins, currentParts)
        -> `launch_update_partition_dimensions` below. ALL THREE of their
           kernels are ported here -- `UpdatePartitionOffsets` in its
           `TPartitionOffsetWriter` instantiation, `UpdatePartitionSizes` and
           `ZeroPartitions` (`cuda_util/kernel/partitions.cu:81-107`, `:14-38`
           and `:109-117`). An earlier draft of this file REUSED
           `gbdt/gpu_util/kernel/partitions.update_partition_offsets_kernel`
           for the offsets half, on the argument that in a parallel-array
           partition layout their `TPartitionOffsetWriter::Write(bin, offset)`
           and `TVecOffsetWriter::Write(bin, offset)` collapse to the same
           store. **THE INTERLEAVED LAYOUT MAKES THEM DIFFERENT AGAIN** --
           `parts[2 * bin] = offset` is not `offsets[bin] = offset` -- so the
           arm is ported rather than the reuse forced. That is also what
           CatBoost does: two instantiations of one template, because the two
           writers write to different places.

    GatherTarget(WeightedTarget, Weights, source, Indices)
        -> `gbdt/gpu_util/kernel/transform.launch_gather_planes_with_mask_f32`
           with `mask = 0xFFFFFFFF`, UNCHANGED. Their `GatherTarget`
           (`weak_target_helpers.h:17-30`) is two plain `Gather` calls; the
           masked form with an all-ones mask is the same arithmetic, and the
           planes form does both columns in one launch.

    UpdatePartitionStats(PartitionStats, currentParts, WeightedTarget,
                         Weights)
        -> `gbdt/gpu_util/partitions_reduce.compute_partition_stats`,
           UNCHANGED. See DEVIATION 98, which is about WHICH of their two
           partition reducers that is.

    MakeSequence(Indices) / FillBuffer(Bins, 0u)
        -> `gbdt/gpu_util/kernel/fill.launch_make_sequence` and
           `ctx.enqueue_memset`, both UNCHANGED.

FOLDS: TRANSCRIBED, ZERO, AND LOAD-BEARING LATER
-------------------------------------------------
`FoldCount` and `FoldBits` are carried below and are both 0 on this path --
`pointwise_optimization_subsets.cpp:12-14` sets them so, and the whole
difference between the two `TSubsetsHelper` specializations is those two
lines plus where `Bins` is seeded:

    Stripe (this file):  FoldCount = 0;
                         FoldBits  = 0;
                         FillBuffer(Bins, 0u)
    Mirror (NOT ported): FoldCount = initParts.size();
                         FoldBits  = IntLog2(FoldCount);
                         WriteFoldBasedInitialBins(Bins)
                         (`oblivious_tree_structure_searcher.cpp:36-37`)

**THE FOLD ID OCCUPIES THE LOW BITS OF A DOCUMENT'S BIN AND THE DEPTH BITS
SIT ABOVE IT.** That is ordered boosting at the subsets level, in full. It is
why every index downstream reads `CurrentDepth + FoldBits` and never
`CurrentDepth`: the bit a level is about to write is at position
`CurrentDepth + FoldBits`, the sort window is that same single bit, and the
live partition count is `1 << (CurrentDepth + FoldBits)` rather than
`1 << CurrentDepth`. Writing `CurrentDepth` anywhere below would be correct
today, at `FoldBits == 0`, and would silently interleave folds with depth
levels the day the Mirror specialization lands. Every such site is spelled
`current_depth + fold_bits` here for that reason and for no other.

The Mirror `Split` also takes a different first step -- `UpdateBins(Bins,
nextLevelDocBins, docMap, CurrentDepth, FoldBits)`
(`pointwise_optimization_subsets.h:78`), a read out of a precomputed
compressed-bits buffer rather than out of the compressed index. That is
`gpu_data/kernel/split.cu:124-163` and is NOT ported here; this file's
`Split` is the Stripe one, which is the `.cpp`.

`TStripeMapping` VERSUS `TMirrorMapping`, at one device
--------------------------------------------------------
`PORTING.md` 91 A proves line by line that at `GetDeviceCount() == 1` a
stripe buffer and a mirror buffer are the same single slice `[{0, n}]`
(`cuda_lib/mapping.h:256-280`). So the distinction costs nothing to honour
and nothing to ignore -- today. It is honoured in the naming below anyway,
because doc-parallel splits ROWS and feature-parallel splits FEATURES, and on
two devices those are different programs. `DeviceView(dev)` and
`ParallelStripeView` (`pointwise_optimization_subsets.h:24-42`, `:110-120`)
are the multi-device half and have no body here: at one device
`DeviceView(0)` is a const alias of the whole struct and
`ParallelStripeView(Partitions, TSlice(0, k))` is `Partitions[0:k]`.
`current_part_count()` is that slice's length and is the only form of it this
port needs.

============================== DEVIATION 97 ==============================
THE STATE'S LAYOUT: FLOAT32 STATS, AND A `Count` PLANE NOTHING READS.

**What theirs is.** Six device buffers, two of them arrays of structs:

    TBuffer<ui32>                 Bins;
    TBuffer<ui32>                 Indices;
    TBuffer<TDataPartition>       Partitions;      // {ui32 Offset, Size}
    TBuffer<TPartitionStatistics> PartitionStats;  // {double Weight, Sum,
                                                   //  Count}
    TBuffer<float>                WeightedTarget;
    TBuffer<float>                Weights;

**What ours is, and why each change.**

1. `Partitions` is ONE `UInt32` buffer read as `partitions[2 * p]` = Offset
   and `partitions[2 * p + 1]` = Size -- `TDataPartition[]` reinterpreted,
   field for field and in their field order. An earlier draft of this file
   used two parallel arrays and argued from the greedy searcher's convention;
   that was OVERRULED and the reversal is recorded here rather than quietly
   dropped. Two reasons, and the second is the binding one:

   - it mirrors `TDataPartition` exactly, which is the tie-breaker under
     COPY-DO-NOT-IMPROVE;
   - it is already the GATED CONTRACT of two layers of the pointwise stack
     that landed before this file --
     `gbdt/methods/kernel/split_properties_helpers.mojo:193` and `:196` read
     `2 * p + 1` to pick the smaller sibling, and
     `gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo:291-292` reads
     both halves. `mojo_only/pointwise_dispatch_check.mojo` gates that family
     end to end at 3,686 and 12,360 cells on this layout, so a parallel-array
     `TOptimizationSubsets` would have invalidated a green gate to satisfy a
     convention.

   What it costs is named in the reuse table above: the offsets kernel is
   ported rather than reused, because the two writers stop coinciding.

2. `WeightedTarget` and `Weights` are TWO COLUMNS OF ONE BUFFER
   (`TL2Target.stats`, `line_size` apart) rather than two allocations.
   CatBoost's own greedy searcher stores its stats exactly this way --
   `TOptimizationTarget::StatsToAggregate` is one multi-column buffer and
   `GetStatCount()` is its column count (`greedy_subsets_searcher/
   split_properties_helper.h:41`, `:60-62`) -- so the shape is theirs, from
   their other file. It is what makes both device reuses possible:
   `launch_gather_planes_with_mask_f32` moves both columns in one launch and
   `compute_partition_stats` reduces both planes in one launch. Two separate
   allocations would need two of each and could not be adjacent.

3. **SOURCE PLANE 0 IS THE WEIGHT AND PLANE 1 IS THE WEIGHTED TARGET**, which
   is the reverse of `TL2Target`'s field declaration order
   (`weak_target_helpers.h:11-14`). It matches `TPartitionStatistics`'s field
   order `{Weight, Sum, Count}` and this repository's existing stat
   convention, `stat_count = 2  # [weight, gradient], their layout`
   (`greedy_search_helper.mojo:244`). Declaration order in a C++ struct whose
   two members are separate allocations means nothing; plane order in one
   buffer means everything, so it follows the consumer.

4. `PartitionStats` is `Float32`, not `double`. Metal has no float64 -- the
   same wall `gbdt/gpu_util/partitions_reduce.mojo`'s second deviation block
   and `jacobi_eigh_device.mojo` both record. **THE COST IS UNPRICED HERE
   TOO.** Their `PartitionUpdateImpl` accumulates in `double` from end to
   end; ours sums Float32 in a two-level tree.
   `mojo_only/pointwise_subsets_check.mojo` sidesteps the question by
   planting integer stats small enough that every intermediate sum is exact
   in Float32, which is what lets it compare with equality and no tolerance
   -- it does NOT establish that the Float32 reduction is adequate on real
   data. No measurement against a Float64 host reduction of a real fixture
   has been taken.

5. `PartitionStats` is STRIDE 3 -- `[3 * p + 0] = Weight`, `+ 1 = Sum`,
   `+ 2 = Count` -- and **plane 2 is dead weight that is stored anyway.**
   This is worth stating precisely because it is the opposite of what the
   arithmetic wants:

   - Their `Count` is never REDUCED on this path. `PartitionUpdateImpl`
     (`methods/kernel/pointwise_scores.cu:668-676`) reads

         if (counts != 0) { ...reduce the counts column... }
         else            { tmp = size; }
         if (tid == 0) partStats->Count = tmp;

     and the pointwise family's only caller passes `nullptr` for `counts`
     (`TUpdatePartitionPropsKernel::Run`, `pointwise_kernels.h:238-244`). So
     `Count` is identically `parts[i].Size`.
   - **And no scorer ever reads it.** Theirs reads `.Weight` and `.Sum` at
     `methods/kernel/pointwise_scores.cu:89`, `:92`, `:260`, `:263`, `:359`
     and `:362` and reads `.Count` nowhere; the ported scorer in this tree
     does the same, `parts[3 * off + 0]` and `parts[3 * off + 1]` at
     `gbdt/methods/kernel/pointwise_scores.mojo:700-703`, `:884-885` and
     `:1011-1014`, and never `+ 2`.

   So the plane carries no information into any consumer. It is stored
   because `TPartitionStatistics` is three doubles wide and the ported
   scorer's reader is compiled against a stride of 3; a stride-2 record would
   be arithmetically complete and would silently misalign every read in that
   file. `pack_partition_stats_kernel` fills it with the partition's own
   size, which is their `else { tmp = size; }` arm exactly. When the PAIRWISE
   family lands -- it is the one that passes a real `counts` column -- the
   plane starts carrying a reduction and this note stops applying.

============================== DEVIATION 98 ==============================
WHICH PARTITION REDUCER, AND WHY IT IS NOT THE ONE THEIR HEADER NAMES.

**This is the one thing in this port that a reader will get wrong from the
name alone, so it is written out.** CatBoost has TWO different functions
called something like "update partition stats" and they are not the same
kernel:

| symbol | file | shape |
|---|---|---|
| `UpdatePartitionProps` | `methods/kernel/pointwise_scores.cu:681` | `<<<partsCount, 1024>>>`, ONE BLOCK PER PARTITION, three sequential `double` reductions per block |
| `UpdatePartitionsProps` | `cuda_util/kernel/update_part_props.cu:197` | `<<<(CeilDivide(2*SMCount, statCount), partCount, statCount), 512>>>`, grid-strided, plus a `SaveResultsImpl` second phase |

`UpdateSubsetsStats` (`pointwise_optimization_subsets.h:66`) calls
`UpdatePartitionStats` (`pointwise_kernels.h:488`), which dispatches
`TUpdatePartitionPropsKernel`, which calls the FIRST one -- the singular
`UpdatePartitionProps`, one block per partition. The greedy searcher this
repository already ports calls the SECOND one, and
`gbdt/gpu_util/partitions_reduce.mojo` is the port of that second one.

**We call the second one here anyway.** What differs and what does not:

- ARITHMETIC: both sum the same rows of the same columns over the same
  `[Offset, Offset + Size)` spans. `Weight` and `Sum` are the same two
  reductions. `Count` is not a reduction on either side (their pointwise
  caller passes `counts == nullptr`; see DEVIATION 97.5).
- SUMMATION ORDER: different, so the last bits can differ from theirs and
  from each other across machines. That is already true of their own code --
  `UpdatePartitionsProps` sizes its grid from `TArchProps::SMCount()` -- and
  `partition_stats_chunks` is pinned under `NUMERIC_IDENTICAL` for exactly
  this reason (`gbdt/gpu_util/partitions_reduce.mojo:214`).
- LAUNCH SHAPE: theirs is one launch, ours is two (phase 1 + phase 2).
- OCCUPANCY: **theirs is one block per partition, which at depth 0 puts the
  whole dataset through a single threadgroup.** This repository has measured
  that shape and lost to it twice (`partitions_reduce.mojo`, "A
  one-block-per-leaf version would be correct and would serialize the whole
  dataset through one threadgroup at depth 0, which is the mistake this
  repository has now made twice and measured twice"). Writing a faithful
  `PartitionUpdateImpl` here would reintroduce it AND duplicate a reduction
  this tree already has.

**No measurement is offered for the occupancy claim on THIS path**, because
nothing calls this file yet; the two prior measurements are on the greedy
path and are cited, not re-taken. If the pointwise searcher lands and the
one-block form is wanted for bit-parity with CatBoost's GPU, this is the
place it goes, and the parity would be against `double` arithmetic we cannot
do anyway (DEVIATION 97.4).

**THE OTHER PORT EXISTS AND IS NOT CALLED, DELIBERATELY.** A concurrent lane
has landed `PartitionUpdateImpl` and `UpdatePartitionProps` in
`gbdt/methods/kernel/pointwise_scores.mojo` (`partition_update_kernel` at
`:1290`, `update_partition_props` at `:1800`) -- the faithful
one-block-per-partition reducer, with the three null-pointer arms and the
`tmp = size` Count arm intact. It is a correct port and it is the function
their dispatch names. **This file still calls the grid-strided one**, and
that is a decision rather than an accident:

- their form puts the WHOLE DATASET through ONE THREADGROUP at depth 0,
  because the grid is `<<<partsCount, 1024>>>` and `partsCount` is 1 there;
- this repository has built that shape twice and measured against it twice
  (`gbdt/gpu_util/partitions_reduce.mojo`: "A one-block-per-leaf version
  would be correct and would serialize the whole dataset through one
  threadgroup at depth 0, which is the mistake this repository has now made
  twice and measured twice");
- the arithmetic is the same either way, and the `double`-vs-Float32 gap
  (DEVIATION 97.4) means bit-parity with their kernel is unreachable on this
  target regardless of which shape we pick, so there is nothing to buy.

**PRICED, AND THE PRICE IS NOT MEASURED ON THIS PATH.** The two measurements
cited are on the greedy path and are cited, not re-taken; nothing calls this
file yet, so no timing here would mean anything. What IS counted is launches.
Their `Split` is 10 (update bins 1, reorder 4, dimensions 2, gather 2,
partition update 1); ours is 9 (update bins 1, reorder 4, dimensions 2,
de-interleave 1, gather 1, reduce 2, pack 1 -- less the dimensions overlap),
so the two adapter launches below are not bought at the cost of a launch
budget. Both adapters run at most `max_part_count` threads: 64 at depth 6.
If a future measurement says the one-block form wins here, the swap is one
call in `update_subsets_stats` and `update_partition_props` is already
written.

STATUS: UNWIRED. Nothing in this tree calls `create_subsets` or
`split_subsets` -- the searcher above them
(`oblivious_tree_doc_parallel_structure_searcher.{h,cpp}`, 295 lines) and the
`pointwise_hist2*` histogram family below them are both unported, and they
are rung 1 of `PORTING.md` 91 E. `PORTING_RULES.md` rule 3 is why this
sentence is here: a ported file no caller reaches is not done, and the gate
in `mojo_only/pointwise_subsets_check.mojo` is what stands in for a caller
until one exists.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK
from gbdt.gpu_util.kernel.transform import (
    launch_gather_planes_with_mask_f32,
)
from gbdt.gpu_util.partitions_reduce import (
    compute_partition_stats,
    partition_stats_chunks,
)


comptime SPLIT_BLOCK_SIZE = 256
"""`constexpr int blockSize = 256` (`gpu_data/kernel/split.cu:211`)."""

comptime SPLIT_MAX_BLOCKS = 65535
"""Stand-in for `TArchProps::MaxBlockCount()`; the kernel grid-strides, so
any large constant gives the same answer. `fill.mojo` and `transform.mojo`
say the same about their own copies of this constant."""

comptime POINTWISE_STAT_COUNT = 2
"""How many columns `TL2Target` carries and how many planes the partition
reduce actually sums. `TPartitionStatistics`'s third member is not one of
them; see DEVIATION 97.5."""

comptime L2_PLANE_WEIGHT = 0
"""`TL2Target`'s `Weights` column, plane 0 of `stats` and of `gathered`.
DEVIATION 97.3 is why the weight is first."""

comptime L2_PLANE_TARGET = 1
"""`TL2Target`'s `WeightedTarget` column, plane 1."""

comptime PART_OFFSET = 0
comptime PART_SIZE = 1
"""`TDataPartition`'s two `ui32`, in their declaration order
(`cuda_util/gpu_data/partitions.h`). A partition record is
`partitions[2 * p + PART_OFFSET]` and `partitions[2 * p + PART_SIZE]`.
DEVIATION 97.1, and the layout
`gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo:291-292` already
reads."""

comptime PARTITION_RECORD = 2
"""`sizeof(TDataPartition) / sizeof(ui32)`."""

comptime PART_STAT_WEIGHT = 0
comptime PART_STAT_SUM = 1
comptime PART_STAT_COUNT = 2
"""`TPartitionStatistics`'s three members, in their declaration order
(`gpu_data/gpu_structures.h:113-116`). `PART_STAT_COUNT` is stored and never
read by any scorer -- see DEVIATION 97.5, which is the whole note."""

comptime PARTITION_STAT_STRIDE = 3
"""`TPartitionStatistics` is three wide, so the record is three wide, even
though only two of the three planes are reduced and only two are ever read.
`gbdt/methods/kernel/pointwise_scores.mojo:700-703` indexes at this stride."""

comptime GATHER_NO_MASK = UInt32(0xFFFFFFFF)
"""`GatherTarget` calls plain `Gather` (`weak_target_helpers.h:28-29`), not
`GatherWithMask`. An all-ones mask makes the masked kernel the plain one:
`map[i] & 0xFFFFFFFF == map[i]`. The mask exists for the CTR block, whose
indices carry a segment flag in the high bits (`gbdt/ctrs/index_wrapper.mojo`);
`Indices` here carries none."""


struct TL2Target(Movable):
    """`TL2Target<TStripeMapping>` (`methods/weak_target_helpers.h:11-14`),
    as two columns of one buffer. DEVIATION 97.2 and 97.3.

        TCudaBuffer<float, TMapping> WeightedTarget;
        TCudaBuffer<float, TMapping> Weights;

    `stats` holds `2 * line_size` floats: `[L2_PLANE_WEIGHT * line_size + i]` is
    document `i`'s weight and `[L2_PLANE_TARGET * line_size + i]` is its weighted
    target. Both are in ORIGINAL document order; `subsets.gathered` holds the
    same two columns in the CURRENT partition order, which is what
    `GatherTarget` produces and what the partition reduce reads.
    """

    var stats: DeviceBuffer[DType.float32]
    var line_size: Int

    def __init__(out self, var stats: DeviceBuffer[DType.float32], line_size: Int):
        self.stats = stats^
        self.line_size = line_size


struct TOptimizationSubsets(Movable):
    """`TOptimizationSubsets<TStripeMapping, false>`
    (`pointwise_optimization_subsets.h:14-50`).

    Their six buffers and three counters, plus the scratch the reuses need.
    `IsConst` is a compile-time const-ness tag on the buffer views and has no
    counterpart: Mojo's `DeviceBuffer` is not const-parameterized, and the
    only thing `IsConst=true` selects upstream is `DeviceView`'s return type.
    """

    var bins: DeviceBuffer[DType.uint32]
    """`Bins`. One `ui32` per document, IN CURRENT PARTITION ORDER. Bit
    `CurrentDepth + FoldBits` is the one the next `Split` writes; bits below
    it are the path so far, and the lowest `FoldBits` of them are the fold id
    (0 of them on this path)."""

    var indices: DeviceBuffer[DType.uint32]
    """`Indices`. Position -> original document id. Seeded with
    `MakeSequence` and permuted by every `ReorderBins`."""

    var partitions: DeviceBuffer[DType.uint32]
    """`Partitions`: `TDataPartition[]` reinterpreted, two `UInt32` per
    record (DEVIATION 97.1). `1 << (FoldBits + maxDepth)` records, allocated
    once at `CreateSubsets` and never resized -- theirs is the same
    allocation with `SliceView`/`ParallelStripeView` taking the live
    prefix."""

    var partition_stats: DeviceBuffer[DType.float32]
    """`PartitionStats`, `[part * PARTITION_STAT_STRIDE + stat]`, three wide.
    DEVIATION 97.4 and 97.5."""

    var gathered: DeviceBuffer[DType.float32]
    """`WeightedTarget` and `Weights` AFTER `GatherTarget`, two columns
    `doc_count` apart, in current partition order."""

    var doc_count: Int
    var max_part_count: Int
    """`1 << (FoldBits + maxDepth)` (`pointwise_optimization_subsets.cpp:12`).
    Their `maxPartCount`."""

    var fold_count: UInt32
    """`FoldCount`. 0 on this path. `initParts.size()` under the Mirror
    specialization."""

    var current_depth: UInt32
    """`CurrentDepth`. Incremented by `Split` AFTER the reorder and BEFORE
    the stats update (`pointwise_optimization_subsets.cpp:49-51`), which is
    what makes `UpdateSubsetsStats` see the new, wider partition count."""

    var fold_bits: UInt32
    """`FoldBits`. 0 on this path, `IntLog2(FoldCount)` under Mirror. The fold
    id lives in the LOW `FoldBits` bits of a document's bin; see the module
    docstring."""

    var sm_count: Int
    """`TArchProps::SMCount()`, cached at construction. Theirs is a static
    read at init; `partitions_reduce.mojo` records that querying it per call
    costs 1.26 ms on this Metal device."""

    var tmp_bins: DeviceBuffer[DType.uint32]
    var tmp_indices: DeviceBuffer[DType.uint32]
    var scan_offsets: DeviceBuffer[DType.int32]
    var block_sums: DeviceBuffer[DType.int32]
    """`ReorderBins`'s double buffer and scan scratch. Theirs comes out of a
    `TRadixSortContext` the manager owns; ours is held here so `Split` does
    not allocate per level."""

    var stat_partials: DeviceBuffer[DType.float32]
    """Phase 1's per-(chunk, part, stat) partials, their `tempVars`."""

    var reduce_offsets: DeviceBuffer[DType.uint32]
    var reduce_sizes: DeviceBuffer[DType.uint32]
    var reduce_pairs: DeviceBuffer[DType.float32]
    """THE ADAPTER, and it is ours, not theirs.

    `compute_partition_stats` was written for the greedy searcher, which
    keeps its partitions as two parallel arrays and its stats at
    `stat_count` stride. This file's records are interleaved and three wide.
    Rather than fork the reducer -- it is on the greedy searcher's live path
    and its signature is that path's contract -- the partition record is
    de-interleaved into `reduce_offsets`/`reduce_sizes` before the reduce and
    the stride-2 answer is packed out of `reduce_pairs` after it. Two
    launches of at most `max_part_count` threads: 64 at depth 6. See
    DEVIATION 98 for the launch budget this sits inside."""

    var part_ids: DeviceBuffer[DType.uint32]
    """`compute_partition_stats` takes a `partIds` list, their argument of the
    same name. `PartitionUpdateImpl` has none -- it indexes partitions by
    `blockIdx.x` -- so the identity sequence over `[0, max_part_count)` is
    what makes the two agree. Filled once with `MakeSequence`."""

    def __init__(
        out self,
        var bins: DeviceBuffer[DType.uint32],
        var indices: DeviceBuffer[DType.uint32],
        var partitions: DeviceBuffer[DType.uint32],
        var partition_stats: DeviceBuffer[DType.float32],
        var gathered: DeviceBuffer[DType.float32],
        var tmp_bins: DeviceBuffer[DType.uint32],
        var tmp_indices: DeviceBuffer[DType.uint32],
        var scan_offsets: DeviceBuffer[DType.int32],
        var block_sums: DeviceBuffer[DType.int32],
        var stat_partials: DeviceBuffer[DType.float32],
        var part_ids: DeviceBuffer[DType.uint32],
        var reduce_offsets: DeviceBuffer[DType.uint32],
        var reduce_sizes: DeviceBuffer[DType.uint32],
        var reduce_pairs: DeviceBuffer[DType.float32],
        doc_count: Int,
        max_part_count: Int,
        sm_count: Int,
    ):
        self.bins = bins^
        self.indices = indices^
        self.partitions = partitions^
        self.partition_stats = partition_stats^
        self.gathered = gathered^
        self.tmp_bins = tmp_bins^
        self.tmp_indices = tmp_indices^
        self.scan_offsets = scan_offsets^
        self.block_sums = block_sums^
        self.stat_partials = stat_partials^
        self.part_ids = part_ids^
        self.reduce_offsets = reduce_offsets^
        self.reduce_sizes = reduce_sizes^
        self.reduce_pairs = reduce_pairs^
        self.doc_count = doc_count
        self.max_part_count = max_part_count
        # `subsets.CurrentDepth = 0; subsets.FoldCount = 0;
        #  subsets.FoldBits = 0;` (`pointwise_optimization_subsets.cpp:11-13`)
        self.fold_count = 0
        self.current_depth = 0
        self.fold_bits = 0
        self.sm_count = sm_count

    def current_part_count(self) -> Int:
        """`CurrentPartsView`'s slice length: `1ULL << (CurrentDepth +
        FoldBits)` (`pointwise_optimization_subsets.h:99`, `:112`, `:118`).

        `ParallelStripeView(Partitions, TSlice(0, that))` is the view itself,
        and at one device a stripe view of `[0, k)` is `Partitions[0:k]`, so
        the length is all a single-device port needs. The Mirror
        specialization's `SliceView` (`:100`) is the same slice of the same
        buffer.

        **`+ fold_bits` is not decoration.** See the module docstring.
        """
        return 1 << Int(self.current_depth + self.fold_bits)


def update_partition_offsets_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
    sorted_bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdatePartitionOffsets<TPartitionOffsetWriter, false>`
    (`cuda_util/kernel/partitions.cu:81-107`), the instantiation
    `UpdatePartitionDimensions` dispatches (`:128`).

        ui32 lastBin = DONT_WRITE_EMPTY_SUFFIX ? Ldg(sortedBins + size - 1, 0)
                                               : UINT32_MAX;
        while (i < size) {
            ui32 bin0 = __ldg(sortedBins + i);
            ui32 bin1 = i ? __ldg(sortedBins + i - 1) : UINT32_MAX;
            if (bin0 != bin1) {
                ui32 b = bin0;
                while (b != bin1) { writer.Write(b, i); b--; }
            }
            if (i + 1 == size) {
                ui32 b = bin0 + 1;
                while (b < min(lastBin, partCount)) { writer.Write(b, size); b++; }
            }
            i += blockDim.x * gridDim.x;
        }

    `TPartitionOffsetWriter::Write(bin, offset)` is `Parts[bin].Offset =
    offset` (`:49-58`), which in the interleaved record is
    `partitions[2 * bin + PART_OFFSET]`.

    **WHY THIS IS NOT `gbdt/gpu_util/kernel/partitions.update_partition_offsets_kernel`.**
    That file ports the SAME template through its OTHER writer,
    `TVecOffsetWriter::Write(bin, offset)` = `BinOffsets[bin] = offset`
    (`:60-70`). The two writers differ by exactly the stride, so in a
    parallel-array partition layout they are the same store and the reuse is
    free -- which is what an earlier draft of this file did. Under
    DEVIATION 97.1's interleaved record `parts[2 * bin]` is not
    `offsets[bin]`, so they are two different stores again, and CatBoost
    instantiates the template twice for precisely that reason. Ported, not
    forced.

    `DONT_WRITE_EMPTY_SUFFIX` is FALSE on this path, so `lastBin` is
    `UINT32_MAX` and the suffix walk runs to `min(UINT32_MAX, partCount)`,
    i.e. `partCount`. The template parameter is not carried as an argument
    here because `UpdatePartitionDimensions` has no arm that sets it; their
    `ui32` entry point does, and that is the OTHER file's business.

    The `i == 0` sentinel is `UINT32_MAX` and the walk terminates by unsigned
    WRAPAROUND after bin 0 is written. That is theirs, and it is the opposite
    of the sentinel twenty lines below it in their own file -- see
    `update_partition_sizes_kernel`.
    """
    var size = Int(size_in)
    var part_count = UInt32(Int(part_count_in))
    var last_bin = UInt32(0xFFFFFFFF)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var bin0 = sorted_bins.unsafe_load(i)
        var bin1: UInt32
        if i > 0:
            bin1 = sorted_bins.unsafe_load(i - 1)
        else:
            bin1 = UInt32(0xFFFFFFFF)
        if bin0 != bin1:
            var b = bin0
            while b != bin1:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_OFFSET, UInt32(i)
                )
                b -= 1  # wraps past 0 to 0xFFFFFFFF, ending the i==0 walk
        if i + 1 == size:
            var limit = last_bin
            if part_count < limit:
                limit = part_count
            var b = bin0 + 1
            while b < limit:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_OFFSET, UInt32(size)
                )
                b += 1
        i += stride


def update_partition_sizes_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
    sorted_bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdatePartitionSizes` (`cuda_util/kernel/partitions.cu:14-38`),
    transcribed.

        ui32 bin0 = __ldg(sortedBins + i);
        ui32 bin1 = i ? __ldg(sortedBins + i - 1) : 0;
        if (bin0 != bin1) {
            ui32 b = bin1;
            while (b < bin0) { parts[b].Size = i - parts[b].Offset; b++; }
        }
        if ((i + 1) == size) {
            parts[bin0].Size = size - parts[bin0].Offset;
            ui32 b = bin0 + 1;
            while (b < partCount) { parts[b].Size = 0; b++; }
        }

    TWO THINGS HERE ARE NOT WHAT THE OFFSETS KERNEL ABOVE DOES AND BOTH ARE
    THEIRS.

    **The `i == 0` sentinel is `0`, not `0xFFFFFFFF`.**
    `UpdatePartitionOffsets` sixty lines below it in their file uses
    `UINT32_MAX` and relies on unsigned wraparound to terminate; this one
    uses `0` and terminates on `b < bin0`. Swapping them looks harmless and
    is not: with `UINT32_MAX` here the `while (b < bin0)` at `i == 0` would
    not run at all, and every empty partition before the first occupied one
    would keep whatever size it had from the previous level.

    **It READS `parts[b].Offset`, so the offsets kernel must have run first.**
    `Size` is computed as a difference against the offset, not counted. Their
    dispatcher launches offsets then sizes on the same stream
    (`partitions.cu:128-129`) and `launch_update_partition_dimensions` below
    keeps that order.

    The trailing walk zeroes every partition ABOVE the last occupied bin, so
    a level that leaves high slots empty does not inherit stale sizes. The
    walk down from `bin1` covers the empty ones in between.
    """
    var size = Int(size_in)
    var part_count = UInt32(Int(part_count_in))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var bin0 = sorted_bins.unsafe_load(i)
        var bin1: UInt32
        if i > 0:
            bin1 = sorted_bins.unsafe_load(i - 1)
        else:
            bin1 = UInt32(0)
        if bin0 != bin1:
            var b = bin1
            while b < bin0:
                var off = partitions.unsafe_load(
                    Int(b) * PARTITION_RECORD + PART_OFFSET
                )
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_SIZE, UInt32(i) - off
                )
                b += 1
        if (i + 1) == size:
            var off0 = partitions.unsafe_load(
                Int(bin0) * PARTITION_RECORD + PART_OFFSET
            )
            partitions.unsafe_store(
                Int(bin0) * PARTITION_RECORD + PART_SIZE, UInt32(size) - off0
            )
            var b = bin0 + 1
            while b < part_count:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_SIZE, UInt32(0)
                )
                b += 1
        i += stride


def zero_partitions_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
):
    """`ZeroPartitions` (`cuda_util/kernel/partitions.cu:109-117`).

        while (i < partCount) { parts[i].Size = 0; parts[i].Offset = 0;
                                i += blockDim.x * gridDim.x; }

    The `numBlocks == 0` arm of `UpdatePartitionDimensions`, i.e. an EMPTY
    document set. Ported because their dispatcher has it, and because an
    empty level otherwise leaves the previous level's partitions in place --
    a well-formed, entirely wrong answer.
    """
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        partitions.unsafe_store(i * PARTITION_RECORD + PART_SIZE, UInt32(0))
        partitions.unsafe_store(i * PARTITION_RECORD + PART_OFFSET, UInt32(0))
        i += stride


def deinterleave_partitions_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
):
    """ADAPTER, NOT A PORT. CatBoost has no counterpart and wants none.

    `compute_partition_stats` takes the greedy searcher's two parallel
    partition arrays; this file's record is `TDataPartition[]` interleaved.
    Splitting the record for the duration of the reduce is cheaper than
    forking a reducer that is on another searcher's live path, and it is
    visibly not theirs so a reviewer diffing against `partitions.cu` finds
    nothing missing. See the `reduce_offsets` field and DEVIATION 98.
    """
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        part_offset.unsafe_store(
            i, partitions.unsafe_load(i * PARTITION_RECORD + PART_OFFSET)
        )
        part_size.unsafe_store(
            i, partitions.unsafe_load(i * PARTITION_RECORD + PART_SIZE)
        )
        i += stride


def pack_partition_stats_kernel(
    pairs: MutPointer[Float32, MutAnyOrigin],
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    part_count_in: Int32,
):
    """ADAPTER, plus ONE line that IS theirs.

    The adapter half widens the reducer's stride-2 answer to
    `TPartitionStatistics`'s stride 3.

    The line that is theirs is `Count`:

        } else {
           tmp = size;
        }
        if (tid == 0) { partStats->Count = tmp; }

    (`methods/kernel/pointwise_scores.cu:668-676`.) The pointwise family
    always passes `counts == nullptr` (`pointwise_kernels.h:240`), so their
    `Count` is the partition's own size and is not a reduction at all. It is
    written here for the same reason they write it -- the record is three
    wide -- and NOTHING READS IT: their scorer touches `.Weight` and `.Sum`
    at `pointwise_scores.cu:89`, `:92`, `:260`, `:263`, `:359`, `:362` and
    `.Count` nowhere, and the ported scorer indexes `3 * off + 0` and
    `3 * off + 1` only (`gbdt/methods/kernel/pointwise_scores.mojo:700-703`,
    `:884-885`, `:1011-1014`). DEVIATION 97.5.
    """
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_WEIGHT,
            pairs.unsafe_load(i * POINTWISE_STAT_COUNT + L2_PLANE_WEIGHT),
        )
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_SUM,
            pairs.unsafe_load(i * POINTWISE_STAT_COUNT + L2_PLANE_TARGET),
        )
        # their `else { tmp = size; }`
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_COUNT,
            Float32(
                Int(partitions.unsafe_load(i * PARTITION_RECORD + PART_SIZE))
            ),
        )
        i += stride


def launch_update_partition_dimensions(
    ctx: DeviceContext,
    mut partitions: DeviceBuffer[DType.uint32],
    part_count: Int,
    mut sorted_bins: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """`UpdatePartitionDimensions` (`cuda_util/kernel/partitions.cu:121-135`).

        const ui32 numBlocks = min(CeilDivide(size, blockSize),
                                   MaxBlockCount());
        if (numBlocks) {
            UpdatePartitionOffsets<TPartitionOffsetWriter, false>
                <<<numBlocks, blockSize>>>(parts, partCount, sortedBins, size);
            UpdatePartitionSizes
                <<<numBlocks, blockSize>>>(parts, partCount, sortedBins, size);
        } else {
            ZeroPartitions<<<CeilDivide(partCount, blockSize), blockSize>>>(
                parts, partCount);
        }

    Both kernels take the SAME `parts` buffer and the order between them is
    not cosmetic: sizes are a DIFFERENCE against the offsets the first one
    wrote. One queue, their order.
    """
    var num_blocks = (size + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if num_blocks > SPLIT_MAX_BLOCKS:
        num_blocks = SPLIT_MAX_BLOCKS

    if num_blocks > 0:
        ctx.enqueue_function[update_partition_offsets_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            sorted_bins.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[update_partition_sizes_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            sorted_bins.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
    else:
        var clear_blocks = (part_count + SPLIT_BLOCK_SIZE - 1) // (
            SPLIT_BLOCK_SIZE
        )
        if clear_blocks == 0:
            return
        ctx.enqueue_function[zero_partitions_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            grid_dim=(clear_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )


def update_bins_from_compressed_index_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Int32,
    bin_idx: UInt32,
    depth: UInt32,
    bins: MutPointer[UInt32, MutAnyOrigin],
):
    """`UpdateBinsFromCompressedIndexImpl`
    (`gpu_data/kernel/split.cu:179-201`), transcribed.

        compressedIndex += feature.Offset;
        const ui32 value = binIdx << feature.Shift;
        const ui32 mask  = feature.Mask << feature.Shift;
        while (i < size) {
            const ui32 idx = indices ? __ldg(indices + i) : i;
            const ui32 featureVal = __ldg(compressedIndex + idx) & mask;
            const ui32 split = (feature.OneHotFeature ? (featureVal == value)
                                                      : featureVal > value);
            bins[i] |= split << depth;
            i += blockDim.x * gridDim.x;
        }

    FOUR THINGS WORTH SAYING.

    **`|=`, NOT `=`.** A document's bin accumulates one bit per level and the
    lower bits are the path it has already taken. Assigning would flatten the
    tree to its last split, and every level would still produce a valid
    partition array -- 2 non-empty partitions out of `2^depth`, which is the
    exact failure shape `partitions_reduce.mojo` records from a different
    cause.

    **`depth` is `CurrentDepth + FoldBits`** at the call site
    (`pointwise_optimization_subsets.cpp:41`), never `CurrentDepth`. Module
    docstring.

    **`indices` is `docsForBins`, not `subsets->Indices`.** It maps a
    POSITION in the current partition order to an ORIGINAL document id,
    because `bins` is permuted and the compressed index never is. Their
    caller builds it as `Gather(groupedByBinObservations, observations,
    subsets.Indices)`
    (`oblivious_tree_doc_parallel_structure_searcher.cpp:65`). Their kernel
    accepts `nullptr` for the identity case; every call in this family passes
    a real buffer, so the null arm is not ported.

    **The comparison is `>` for a float feature and `==` for a one-hot**,
    which is the same predicate `split_points.mojo:141-147` already carries
    for the greedy family. The `TCFeature` fields arrive as four scalars
    rather than as one struct because a whole-struct load kills the Metal
    compiler -- the deviation `split_points.mojo:102-117` measured and
    recorded; passing the fields individually is the same workaround said at
    the call boundary instead of inside the kernel.
    """
    var size = Int(size_in)
    var value = bin_idx << feature_shift
    var mask = feature_mask << feature_shift
    var f_offset = Int(feature_offset)
    var one_hot = feature_one_hot != Int32(0)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var idx = Int(indices.unsafe_load(i))
        var feature_val = compressed_index.unsafe_load(f_offset + idx) & mask
        var goes_right: Bool
        if one_hot:
            goes_right = feature_val == value
        else:
            goes_right = feature_val > value
        if goes_right:
            bins.unsafe_store(i, bins.unsafe_load(i) | (UInt32(1) << depth))
        i += stride


def launch_update_bin_from_compressed_index(
    ctx: DeviceContext,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut docs_for_bins: DeviceBuffer[DType.uint32],
    size: Int,
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Bool,
    bin_idx: UInt32,
    depth: UInt32,
    mut bins: DeviceBuffer[DType.uint32],
) raises:
    """`UpdateBinsFromCompressedIndex` (`gpu_data/kernel/split.cu:203-218`),
    which is what `UpdateBinFromCompressedIndex` (`gpu_data/splitter.h:174-182`)
    dispatches through `TUpdateBinsFromCompressedIndexKernel`.

        const int numBlocks = min(CeilDivide(size, blockSize),
                                  TArchProps::MaxBlockCount());
        if (numBlocks) { ...launch... }
    """
    var num_blocks = (size + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if num_blocks > SPLIT_MAX_BLOCKS:
        num_blocks = SPLIT_MAX_BLOCKS
    if num_blocks == 0:
        return
    ctx.enqueue_function[update_bins_from_compressed_index_kernel](
        compressed_index.unsafe_ptr(),
        docs_for_bins.unsafe_ptr(),
        Int32(size),
        feature_offset,
        feature_mask,
        feature_shift,
        Int32(1) if feature_one_hot else Int32(0),
        bin_idx,
        depth,
        bins.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )


def update_subsets_stats(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut subsets: TOptimizationSubsets,
) raises:
    """`UpdateSubsetsStats` (`pointwise_optimization_subsets.h:53-70`),
    call for call.

        auto currentParts = TSubsetsHelper<TMapping>::CurrentPartsView(subsets);
        subsets.PartitionStats.Reset(currentParts.GetMapping());
        UpdatePartitionDimensions(subsets.Bins, currentParts);
        GatherTarget(subsets.WeightedTarget, subsets.Weights, source,
                     subsets.Indices);
        UpdatePartitionStats(subsets.PartitionStats, currentParts,
                             subsets.WeightedTarget, subsets.Weights);

    `PartitionStats.Reset(mapping)` is a RESIZE to the live partition count.
    Ours is allocated once at `max_part_count` and the live prefix is
    `current_part_count()`, so the resize has no counterpart -- the same way
    `Partitions` is one allocation on both sides and only the VIEW moves.
    What it does mean is that slots at or above `current_part_count()` hold
    the PREVIOUS level's values rather than being freed, and no consumer may
    read them. `mojo_only/pointwise_subsets_check.mojo` poisons that tail and
    checks the poison survives.

    THE ORDER IS NOT INTERCHANGEABLE. Dimensions before gather before stats:
    the reduce reads spans out of `Partitions`, so the dimensions have to be
    current, and it reads the stat columns THROUGH those spans, so the gather
    has to have happened. Enqueued on one queue in that order, which is the
    stream ordering their calls get.
    """
    var part_count = subsets.current_part_count()
    var adapt_blocks = (part_count + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if adapt_blocks < 1:
        adapt_blocks = 1

    # `UpdatePartitionDimensions(subsets.Bins, currentParts)`
    launch_update_partition_dimensions(
        ctx,
        subsets.partitions,
        part_count,
        subsets.bins,
        subsets.doc_count,
    )

    # ADAPTER, not theirs: split `TDataPartition[]` into the parallel pair
    # `compute_partition_stats` takes. See DEVIATION 98.
    ctx.enqueue_function[deinterleave_partitions_kernel](
        subsets.partitions.unsafe_ptr(),
        subsets.reduce_offsets.unsafe_ptr(),
        subsets.reduce_sizes.unsafe_ptr(),
        Int32(part_count),
        grid_dim=(adapt_blocks, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )

    # `GatherTarget(WeightedTarget, Weights, source, Indices)`
    # (`weak_target_helpers.h:17-30`), both columns in one launch.
    launch_gather_planes_with_mask_f32(
        ctx,
        subsets.gathered,
        source.stats,
        subsets.indices,
        subsets.doc_count,
        GATHER_NO_MASK,
        POINTWISE_STAT_COUNT,
        source.line_size,
    )

    # `UpdatePartitionStats(PartitionStats, currentParts, WeightedTarget,
    #  Weights)`. DEVIATION 98 is which of their two reducers this is and
    #  why. Two planes only -- `Count` is not a reduction on this path.
    compute_partition_stats(
        ctx,
        part_count,
        subsets.doc_count,
        POINTWISE_STAT_COUNT,
        subsets.doc_count,
        subsets.part_ids,
        subsets.reduce_offsets,
        subsets.reduce_sizes,
        subsets.gathered,
        subsets.stat_partials,
        subsets.reduce_pairs,
        subsets.sm_count,
    )

    # ADAPTER + their `else { tmp = size; }` Count arm: widen the stride-2
    # answer to `TPartitionStatistics`'s stride 3.
    ctx.enqueue_function[pack_partition_stats_kernel](
        subsets.reduce_pairs.unsafe_ptr(),
        subsets.partitions.unsafe_ptr(),
        subsets.partition_stats.unsafe_ptr(),
        Int32(part_count),
        grid_dim=(adapt_blocks, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )


def create_subsets(
    ctx: DeviceContext, max_depth: Int, mut source: TL2Target
) raises -> TOptimizationSubsets:
    """`TSubsetsHelper<TStripeMapping>::CreateSubsets`
    (`pointwise_optimization_subsets.cpp:4-24`), line for line.

        subsets.Bins.Reset(src.WeightedTarget.GetMapping());
        subsets.Indices.Reset(src.WeightedTarget.GetMapping());
        subsets.CurrentDepth = 0;
        subsets.FoldCount = 0;
        subsets.FoldBits = 0;
        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions.Reset(TStripeMapping::RepeatOnAllDevices(maxPartCount));
        subsets.PartitionStats.Reset(TStripeMapping::RepeatOnAllDevices(maxPartCount));
        FillBuffer(subsets.Bins, 0u);
        MakeSequence(subsets.Indices);
        UpdateSubsetsStats(src, &subsets);

    `RepeatOnAllDevices(maxPartCount)` gives EVERY device the full partition
    array, where the documents are split between them; at one device it is
    `[{0, maxPartCount}]` and identical to a mirror
    (`PORTING.md` 91 A). The distinction is kept in the name of the type this
    port would use, not in the allocation, because there is nothing to
    distinguish at one device.

    `maxPartCount` uses `FoldBits + maxDepth`, not `maxDepth`: under the
    Mirror specialization a depth-6 tree over 4 folds needs 256 partitions,
    not 64.

    **`FillBuffer(Bins, 0u)` IS THE STRIPE SPECIALIZATION'S WHOLE SEED.** The
    Mirror one calls `WriteFoldBasedInitialBins` instead
    (`oblivious_tree_structure_searcher.cpp:37`), putting each document's fold
    id in the low bits. Zeroing is what "one root partition, no folds" means.
    """
    if max_depth < 0:
        raise Error(
            String("maxDepth must be non-negative, got ") + String(max_depth)
        )
    var doc_count = source.line_size

    # `ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth)`, with
    # FoldBits == 0 on this path.
    var max_part_count = 1 << max_depth

    var bins = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var indices = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var partitions = ctx.enqueue_create_buffer[DType.uint32](
        max_part_count * PARTITION_RECORD
    )
    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_part_count * PARTITION_STAT_STRIDE
    )
    var gathered = ctx.enqueue_create_buffer[DType.float32](
        doc_count * POINTWISE_STAT_COUNT
    )

    # `ReorderBins`'s double buffer and scan scratch, sized the way
    # `radix_sort.mojo` sizes them.
    var tmp_bins = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var tmp_indices = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var scan_offsets = ctx.enqueue_create_buffer[DType.int32](doc_count)
    var n_scan_blocks = (doc_count + REORDER_BLOCK - 1) // REORDER_BLOCK
    if n_scan_blocks < 1:
        n_scan_blocks = 1
    var block_sums = ctx.enqueue_create_buffer[DType.int32](n_scan_blocks)

    # Their `tempVars`, sized from the SAME formula the launch uses --
    # `partition_stats_chunks` is exported for exactly this
    # (`partitions_reduce.mojo:214`), and sizing it any other way is how the
    # kernel walks off the end of its own row.
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var chunks = partition_stats_chunks(sm_count, POINTWISE_STAT_COUNT)
    var stat_partials = ctx.enqueue_create_buffer[DType.float32](
        max_part_count * POINTWISE_STAT_COUNT * chunks
    )

    var part_ids = ctx.enqueue_create_buffer[DType.uint32](max_part_count)

    # The adapter's scratch. See the `reduce_offsets` field docstring.
    var reduce_offsets = ctx.enqueue_create_buffer[DType.uint32](max_part_count)
    var reduce_sizes = ctx.enqueue_create_buffer[DType.uint32](max_part_count)
    var reduce_pairs = ctx.enqueue_create_buffer[DType.float32](
        max_part_count * POINTWISE_STAT_COUNT
    )

    var subsets = TOptimizationSubsets(
        bins^,
        indices^,
        partitions^,
        part_stats^,
        gathered^,
        tmp_bins^,
        tmp_indices^,
        scan_offsets^,
        block_sums^,
        stat_partials^,
        part_ids^,
        reduce_offsets^,
        reduce_sizes^,
        reduce_pairs^,
        doc_count,
        max_part_count,
        sm_count,
    )

    # `FillBuffer(subsets.Bins, 0u)` (`cuda_util/fill.cu`). One root
    # partition, no folds.
    ctx.enqueue_memset(subsets.bins, UInt32(0))
    # `MakeSequence(subsets.Indices)` (`fill.cu:47-55`).
    launch_make_sequence(ctx, UInt32(0), subsets.indices, doc_count)
    # `partIds`: the identity, standing in for `PartitionUpdateImpl`'s
    # `blockIdx.x`. See the field docstring.
    launch_make_sequence(ctx, UInt32(0), subsets.part_ids, max_part_count)

    # `UpdateSubsetsStats(src, &subsets)`
    update_subsets_stats(ctx, source, subsets)
    return subsets^


def split_subsets(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut docs_for_bins: DeviceBuffer[DType.uint32],
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Bool,
    bin_idx: UInt32,
    mut subsets: TOptimizationSubsets,
) raises:
    """`TSubsetsHelper<TStripeMapping>::Split`
    (`pointwise_optimization_subsets.cpp:26-52`), call for call.

        UpdateBinFromCompressedIndex(cindex, feature, bin, docsForBins,
                                     subsets->CurrentDepth + subsets->FoldBits,
                                     subsets->Bins);
        ReorderBins(subsets->Bins, subsets->Indices,
                    subsets->CurrentDepth + subsets->FoldBits, 1);
        ++subsets->CurrentDepth;
        UpdateSubsetsStats(sourceTarget, subsets);

    Their two `profiler.Profile("Update bins")` / `("Reorder bins")` guards
    are the only other statements in the function and have no counterpart:
    this tree has no profiler and inventing one would be a third category of
    file (`PORTING_RULES.md` 0b-ii).

    **`ReorderBins(..., offset, 1)` IS ONE BIT AND IT MUST BE STABLE.** The
    documents were already grouped by their first `d` bits; sorting on bit
    `d` alone splits every existing group in two IN PLACE only because equal
    keys keep their relative order. An unstable sort with the same key
    produces the same multiset in every partition and a different permutation
    inside it, which changes `Indices`, changes every gathered stat's
    position, and changes which document a downstream tie resolves to --
    while leaving partition offsets, partition sizes and every per-partition
    total bit-identical. `mojo_only/pointwise_subsets_check.mojo` gates
    stability separately for that reason.

    `bits == 1` also means an ODD number of radix passes, so
    `launch_radix_sort_bins`'s copy-back arm (their `sort_templ.cuh:53`) runs
    on every level rather than on none.

    **`++CurrentDepth` HAPPENS BEFORE `UpdateSubsetsStats`.** The stats
    update recomputes the partition array at `1 << (CurrentDepth + FoldBits)`
    slots, so incrementing after it would recount the OLD, half-as-wide
    level and leave the new one uncounted. Their order, kept.
    """
    var depth = subsets.current_depth + subsets.fold_bits
    if Int(depth) >= 32:
        raise Error(
            String("Split at depth ") + String(depth)
            + " would write bit " + String(depth)
            + " of a ui32 bin; CatBoost's ReorderBins asserts"
            " (offset + bits) <= 32 (cuda_util/sort.cpp:557)"
        )

    # `UpdateBinFromCompressedIndex(...)`
    launch_update_bin_from_compressed_index(
        ctx,
        compressed_index,
        docs_for_bins,
        subsets.doc_count,
        feature_offset,
        feature_mask,
        feature_shift,
        feature_one_hot,
        bin_idx,
        depth,
        subsets.bins,
    )

    # `ReorderBins(subsets->Bins, subsets->Indices, depth, 1)`
    # (`cuda_util/sort.cpp:544`): keys are the bins, values the indices,
    # window is the single bit just written.
    launch_radix_sort_bins(
        ctx,
        subsets.doc_count,
        Int(depth),
        Int(depth) + 1,
        subsets.bins,
        subsets.indices,
        subsets.tmp_bins,
        subsets.tmp_indices,
        subsets.scan_offsets,
        subsets.block_sums,
    )

    # `++subsets->CurrentDepth;`
    subsets.current_depth += 1

    # `UpdateSubsetsStats(sourceTarget, subsets);`
    update_subsets_stats(ctx, source, subsets)
