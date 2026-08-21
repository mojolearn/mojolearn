"""The kernels, and the one step of theirs this formulation replaces.

A PORT of cuML `cpp/src/decisiontree/batched-levelalgo/kernels/
builder_kernels_impl.cuh`, pinned at `00094f7` in
`~/CascadeProjects/upstream/cuml`.

WHAT IS HERE NOW
----------------
`partition_samples`, their `partitionSamples` (`builder_kernels_impl.cuh:43-88`)
transcribed, as a HOST function with the block size and the thread loop made
explicit. It is written this way on purpose and stays this way permanently:

- it is the oracle the device kernel is checked against, and
  `PORTING_RULES.md` 0b-ii is explicit that a host reference used to CHECK a
  device answer is not a "CPU path" -- it is an oracle, and it stays;
- their algorithm is DETERMINISTIC given the block size (nothing in it depends
  on warp scheduling), so the host form produces the device form's exact
  `row_ids` order, not merely an equivalent partition.

WHAT IS NOT HERE YET, AND WHY THE GAP IS NAMED
----------------------------------------------
`computeSplitKernel` (`builder_kernels_impl.cuh:216-340`) and
`nodeSplitKernel` (`:89-107`) are not ported yet. `nodeSplitKernel` is four
lines around the partition below. `computeSplitKernel` is the ONE place this
directory departs from cuML, and the departure is stated here rather than in a
new file, per `PORTING_RULES.md` rule 4 -- a replacement for a step of their
file lives IN THAT FILE, under a marked deviation block.

    ==================================================================
    DEVIATION BLOCK 137 -- computeSplitKernel has no histogram

    THEIRS (`builder_kernels_impl.cuh:216-340`), per (node, feature) block:
      1. load this feature's quantile borders into shared memory
         (`:265-266`, from `quantiles.quantiles_array`);
      2. one pass over the node's rows, `lower_bound` each value into a bin
         and increment a shared histogram (`:281-291`);
      3. large nodes atomically merge their partial histograms in global
         memory and the last block continues (`:295-313`);
      4. PDF -> CDF in place, one scan per class (`:316-322`);
      5. `objective.Gain(...)` scores EVERY bin border as a candidate
         (`:328`);
      6. `sp.evalBestSplit(...)` reduces the per-thread bests into the
         node's split record (`:340`).

    OURS, per (node, feature) block:
      1. RANGE pass: min and max of the feature over the node's rows.
         sklearn's `find_min_max`, `_partitioner.pyx:129-165`.
      2. CONSTANT test: `max <= min + FEATURE_THRESHOLD` (1e-7,
         `_partitioner.pxd:13`) means skip this feature entirely,
         `_splitter.pyx:614-621`.
      3. DRAW: ONE threshold, `Uniform(min, max)`, `_splitter.pyx:633`,
         from the counter-based key `(seed, tree_id, node_id, feature_id)`
         (DEVIATION 130), not from a sequential stream.
      4. SCORE pass: one pass over the node's rows accumulating left/right
         statistics against that single threshold. Class counts for Gini;
         (count, sum) for the MSE proxy, `_criterion.pyx:944-975`.
      5. `evalBestSplit` UNCHANGED -- step 6 above is theirs, kept.

    WHY: there is no upstream GPU implementation of this formulation to be
    faithful to (verified 2026-08-21: LightGBM's `USE_RAND` kernel draws a
    random BIN in histogram space, a different algorithm; cuML's FEA #8133
    is an unlanded design). The upstream is the paper, Geurts, Ernst &
    Wehenkel 2006, with sklearn's `RandomSplitter` as its reference
    implementation. Steps 1-4 above cite it line by line.

    THE COST THIS TRADE PAYS, stated because it is not free: theirs reads
    the feature column ONCE per node, ours reads it TWICE, because the
    threshold cannot be drawn until the range is known. Drawing from the
    PARENT's range instead would fuse the passes and is what a histogram
    builder effectively does -- it is NOT what the paper says
    (`Pick_a_random_split` draws inside the range of the node's OWN local
    subset) and it is not what sklearn does, so it is not taken. No timing
    number is attached to this and none will be until the perf round; the
    trade is recorded as arithmetic, not as a measurement.

    WHAT IT BUYS: the per-node state is a handful of accumulators per
    candidate feature rather than bins x classes, and there is no global
    quantile array, no `lower_bound` per row, no CDF scan, and no
    cross-block histogram merge -- steps 1, 3 and 4 of theirs disappear
    rather than being made cheaper.
    ==================================================================
"""

from extratrees.mojo_only.pcg_rng import SplitKey, key_for, uniform_float
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    NodeWorkItem,
)


comptime TPB_DEFAULT = 128
"""`builder_kernels_impl.cuh:33`, `static constexpr int TPB_DEFAULT = 128`."""


def partition_samples(
    dataset: Dataset,
    split: Split,
    work_item: NodeWorkItem,
    tpb: Int = TPB_DEFAULT,
):
    """Partition `row_ids` into left/right by the best split, in place.

    `builder_kernels_impl.cuh:43-88`, transcribed. Their header note says this
    is called by exactly ONE block, so `tpb` here is that block's width and the
    `for tid in range(tpb)` loops below are its threads. `cub::BlockScan`'s
    `ExclusiveSum` becomes the running prefix it is defined to be.

    Their two-pointer scheme, in their words: walk a cursor down the left
    region and another down the right, find the "misfits" on each side (a left
    slot holding a row that belongs right, and vice versa), compact them, and
    SWAP them in pairs. It terminates when either cursor runs out, which is
    exactly when no misfits remain, because the two counts of misfits are equal
    by construction once `n_left` is correct.

    NOTE THE COMPARISON DIRECTION, `:65-66`: a left-side misfit is
    `col[row_ids[loff]] > quesval`, so a row EQUAL to the threshold stays LEFT.
    sklearn's `partition_samples` agrees (`_partitioner.pyx:233-236`).
    """
    var row_ids = dataset.row_ids
    var range_start = Int(work_item.instances.begin)
    var range_len = Int(work_item.instances.count)
    var col_base = Int(split.colid) * Int(dataset.m)
    var quesval = split.quesval

    var loffset = range_start
    var part = loffset + Int(split.n_left)
    var roffset = part
    var end = range_start + range_len

    # Per-thread state. Theirs lives in registers; ours in lists of width tpb.
    var lflag = List[Int](length=tpb, fill=0)
    var rflag = List[Int](length=tpb, fill=0)
    var lidx = List[Int](length=tpb, fill=0)
    var ridx = List[Int](length=tpb, fill=0)
    var lcomp = List[Int](length=tpb, fill=0)
    var rcomp = List[Int](length=tpb, fill=0)

    # Block-uniform state.
    var llen = 0
    var rlen = 0
    var minlen = 0

    while loffset < part and roffset < end:
        # `:63-67` -- find the misfits. The `llen == minlen` guard is theirs:
        # a side whose cursor did NOT advance keeps the flags it already has,
        # which is what makes the leftovers carry into the next iteration.
        for tid in range(tpb):
            var loff = loffset + tid
            var roff = roffset + tid
            if llen == minlen:
                if loff < part:
                    lflag[tid] = 1 if dataset.data[
                        unsafe_offset = col_base + Int(row_ids[unsafe_offset=loff])
                    ] > quesval else 0
                else:
                    lflag[tid] = 0
            if rlen == minlen:
                if roff < end:
                    rflag[tid] = 1 if dataset.data[
                        unsafe_offset = col_base + Int(row_ids[unsafe_offset=roff])
                    ] <= quesval else 0
                else:
                    rflag[tid] = 0

        # `:69-71` -- two exclusive sums, each returning its block aggregate.
        var lrun = 0
        var rrun = 0
        for tid in range(tpb):
            lidx[tid] = lrun
            lrun += lflag[tid]
            ridx[tid] = rrun
            rrun += rflag[tid]
        llen = lrun
        rlen = rrun

        # `:73` -- pair up only as many misfits as both sides can supply.
        minlen = llen if llen < rlen else rlen

        # `:75-77` -- compaction.
        for tid in range(tpb):
            if lflag[tid] != 0:
                lcomp[lidx[tid]] = loffset + tid
            if rflag[tid] != 0:
                rcomp[ridx[tid]] = roffset + tid

        # `:79-83` -- clear the flags that are about to be consumed, and
        # advance only the side that was fully consumed.
        for tid in range(tpb):
            if lidx[tid] < minlen:
                lflag[tid] = 0
            if ridx[tid] < minlen:
                rflag[tid] = 0
        if llen == minlen:
            loffset += tpb
        if rlen == minlen:
            roffset += tpb

        # `:84-89` -- swap the paired misfits.
        for tid in range(minlen):
            var a = row_ids[unsafe_offset= lcomp[tid]]
            var b = row_ids[unsafe_offset= rcomp[tid]]
            row_ids[unsafe_offset= lcomp[tid]] = b
            row_ids[unsafe_offset= rcomp[tid]] = a


comptime FEATURE_THRESHOLD: Float32 = 1e-7
"""`sklearn/tree/_partitioner.pxd:13`, `cdef const float32_t FEATURE_THRESHOLD
= 1e-7`. A FLOAT32, and the arithmetic it takes part in below is float32 too.
That is not incidental -- see `node_feature_is_constant`."""


@fieldwise_init
struct FeatureRange(ImplicitlyCopyable, Movable):
    """The output of the range pass: one feature's extent over one node."""

    var min_value: Float32
    var max_value: Float32
    var n_missing: Int32
    """NaN count. sklearn tracks it (`_partitioner.pyx:146`) and it changes the
    constant test; DEVIATION 136 refuses NaN input, so this is expected to be
    zero and is returned so a caller can ENFORCE that rather than assume it."""


def node_feature_min_max(
    dataset: Dataset, work_item: NodeWorkItem, col: Int32
) -> FeatureRange:
    """Min and max of one feature over one node's rows.

    `sklearn/tree/_partitioner.pyx:129-165` (`DensePartitioner.find_min_max`),
    transcribed, with two differences that are consequences of where it runs
    and neither of which changes an answer:

    1. Theirs walks `samples[start:end]` and also CACHES each value into
       `feature_values[p]` as it goes (`:152`), because the best-splitter reuses
       that buffer. This formulation has no such buffer -- it reads the column
       twice (DEVIATION 137) -- so the cache write is dropped.
    2. Theirs reads `self.X[samples[p], current_feature]` from a row-major
       numpy array; ours reads cuML's column-major `Dataset` (`dataset.h:24`).

    THEIR INITIALIZATION IS THE PART TO COPY EXACTLY (`:145-163`): `min` and
    `max` are both set from the FIRST NON-MISSING value, and the subsequent
    tests are `if v < min: elif v > max:` -- an `elif`, not two `if`s. With a
    single seeded value the two forms agree; a reduction that seeded from
    +/-INFINITY instead would differ on an all-NaN column, which is exactly the
    column `n_missing` exists to catch.

    A node with no rows at all returns an empty range with `min > max`, which
    `node_feature_is_constant` reports constant. Theirs cannot reach that state
    (`min_samples_split >= 2`), and neither can ours; it is defined here so the
    function is total.
    """
    var begin = Int(work_item.instances.begin)
    var end = begin + Int(work_item.instances.count)
    var col_base = Int(col) * Int(dataset.m)

    var min_value = Float32(0.0)
    var max_value = Float32(0.0)
    var n_missing = 0
    var seen_non_missing = False

    for p in range(begin, end):
        var row = Int(dataset.row_ids[unsafe_offset=p])
        var v = dataset.data[unsafe_offset = col_base + row]
        if v != v:  # isnan, without importing one
            n_missing += 1
        elif not seen_non_missing:
            min_value = v
            max_value = v
            seen_non_missing = True
        elif v < min_value:
            min_value = v
        elif v > max_value:
            max_value = v

    if not seen_non_missing:
        # Their `min = INFINITY, max = -INFINITY` initial state
        # (`_partitioner.pyx:143-144`) survives untouched when every value is
        # missing. Reproduced with the same ordering property (min > max)
        # without needing an infinity on device.
        return FeatureRange(1.0, -1.0, Int32(n_missing))
    return FeatureRange(min_value, max_value, Int32(n_missing))


def node_feature_is_constant(extent: FeatureRange, n_rows: Int32) -> Bool:
    """Whether this feature is constant over this node, sklearn's test.

    `_splitter.pyx:611-618`, all THREE arms::

        if (end - start == n_missing or
            (max_feature_value <= min_feature_value + FEATURE_THRESHOLD
             and n_missing == 0)):

    So: every value missing is ALSO constant, and a column that would
    otherwise be constant is NOT constant when it contains any NaN.

    THE ADDITION IS FLOAT32 AND THAT WIDENS THE BAND. Both operands are
    `float32_t` in their code, so at `min == 0.5` the sum
    `0.5f + 1e-7f` rounds up to `0.5 + 1.192e-7` and a column whose spread is
    `1.19e-7` -- comfortably ABOVE the nominal 1e-7 -- is still reported
    constant. A port that promoted this to float64 would disagree with sklearn
    on real data. Kept in float32 deliberately.
    """
    if n_rows == extent.n_missing:
        return True
    return (
        extent.max_value <= extent.min_value + FEATURE_THRESHOLD
        and extent.n_missing == 0
    )


def draw_threshold(key: SplitKey, extent: FeatureRange) -> Float32:
    """One threshold for one (node, feature), keyed rather than streamed.

    `_splitter.pyx:632-637` draws `rand_uniform(min, max, random_state)`, and
    `:653-654` then applies a guard that is easy to miss and changes an
    answer::

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

    Their draw can land exactly on `max` because `rand_uniform`
    (`_utils.pyx:57-61`) divides by `RAND_R_MAX` and `our_rand_r` can return
    exactly `RAND_R_MAX`. A threshold equal to `max` would send EVERY row left
    (the test is `<=`), i.e. not split at all; their guard turns that into
    `min`, which sends only the min-valued rows left.

    OURS CARRIES THE SAME GUARD, and whether it can fire here is a question
    about float32 rounding rather than about `RAND_R_MAX`: RAFT's `next_float`
    is `[0, 1 - 2^-24]`, so the mathematical product is strictly below the
    span, but `min + res * span` is rounded in float32 and CAN round up to
    `max`. `mojo_only/range_draw_check.mojo` counts how often -- it is not
    assumed either way.
    """
    var gen = key.generator()
    var threshold = uniform_float(gen, extent.min_value, extent.max_value)
    if threshold == extent.max_value:
        return extent.min_value
    return threshold


def draw_threshold_raw(key: SplitKey, extent: FeatureRange) -> Float32:
    """`draw_threshold` WITHOUT sklearn's `== max` guard.

    Exists so a check can count how often the guard fires. Never call it from
    the builder.
    """
    var gen = key.generator()
    return uniform_float(gen, extent.min_value, extent.max_value)


# ============================================================================
# STEP 1 OF DEVIATION 137, ON THE DEVICE: the per-(node, feature) RANGE pass.
#
#     Verified by:  extratrees/mojo_only/range_kernel_check.mojo
#
# `node_feature_min_max` above stays exactly where it is and is not replaced:
# it is the ORACLE this kernel is checked against, per `PORTING_RULES.md`
# 0b-ii, and the check below compares it per (node, feature) cell on FLOAT BIT
# PATTERNS with no tolerance.
#
#     ==================================================================
#     DEVIATION BLOCK 160 -- the range pass is a device kernel whose
#     answer is EXACTLY the host transcription's, not merely close to it
#
#     THEIRS. `computeSplitKernel` (`builder_kernels_impl.cuh:216-340`)
#     makes one pass over the node's rows per (node, feature) block and
#     accumulates a HISTOGRAM. Its accumulator is a SUM -- an exact
#     integer count for classification, but for regression a running sum
#     of float LABELS (`objectives.cuh`'s `AggregateBin`), which rounds,
#     so which bits that bin ends up holding depends on the order the
#     rows and the partial histograms arrive in.
#
#     OURS. `node_feature_range_kernel` below makes the same pass over the
#     same rows with the same `WorkloadInfo` flattening and the same
#     strided loop, and accumulates MIN, MAX and a NaN COUNT instead.
#
#     WHY THIS PARTICULAR KERNEL CAN BE CHECKED FOR BIT-EQUALITY, AND THE
#     HISTOGRAM ONE CANNOT. `min` and `max` on IEEE-754 floats are
#     associative AND commutative EXACTLY -- they select one of their
#     inputs and return it unchanged, so no rounding happens at any step
#     and no regrouping can change which input survives. (`fmin`/`fmax`'s
#     NaN rule is the one place that is not true; no NaN ever reaches the
#     reduction here, see DEVIATION 163.) A COUNT is an exact integer sum
#     and is associative for the same reason. So the device's
#     block-strided, warp-shuffled, cross-block-merged reduction and the
#     host's sequential `for p in range(begin, end)` loop are obliged to
#     produce the IDENTICAL bits, and a tolerance in the check would be
#     hiding a defect rather than absorbing float noise. That is why the
#     check asserts on `.to_bits()` and never on a difference.
#
#     WHAT IS NOT HERE. Steps 2-5 of DEVIATION 137 -- the constant test,
#     the keyed draw, the score pass and `evalBestSplit`. This kernel
#     computes step 1 and stops; its output is the input to the draw.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 161 -- the cross-block combine is a MUTEX MERGE,
#     not their `atomicAdd` + `signalDone` last-block-continues scheme
#
#     THEIRS, `builder_kernels_impl.cuh:295-317`. A large node's blocks
#     each `BinT::AtomicAdd` their partial histogram into a global slot
#     keyed by `large_nodeid`, `__threadfence()`, then call
#     `MLCommon::signalDone` (`src_prims/common/grid_sync.cuh:238-247`),
#     which `atomicAdd`s a per-(node, feature) counter; the block that
#     drives it to zero is "last" and alone continues to the scoring.
#
#     OURS. Each block publishes its partial range into the SAME output
#     cell under a per-(node, feature) spin mutex, merging with
#     `min`/`max`/`+`. No block needs to know that it is last.
#
#     WHY, and it is a platform wall and not a preference. Their merge
#     step is an ATOMIC ADD, and a histogram bin has one. A RANGE does
#     not: there is no portable device `atomicMin`/`atomicMax` on
#     `float32` -- the CUDA idiom for it is a signed-magnitude bit twiddle
#     into an integer atomic, which is a different instruction sequence on
#     every vendor and is precisely the inline `if apple` this tree
#     forbids. And their `__threadfence()` is not expressible: Mojo 1.0
#     comptime-asserts that `threadfence` "is only implemented on NVIDIA
#     GPUs" (measured by the RF lane, DEVIATION 106).
#
#     WHAT IT IS INSTEAD, and it is not invented here either. The spin is
#     the translation this repository has already established and enqueued
#     twice -- for cuVS's cross-block mutex and for cuML's own `Split`
#     publish (`ensemble/.../split.mojo:548-566`): spin on an ACQUIRE load
#     until the mutex reads free, claim it with a WEAK RELAXED
#     compare-exchange, hand it back with a RELEASE store. Only thread 0
#     of a block ever takes the lock, so no two threads of one warp can
#     contend for it and the spin cannot livelock a warp.
#
#     WHY IT CANNOT CHANGE AN ANSWER. The merge is `min`, `max` and
#     integer `+`, all three associative and commutative exactly
#     (DEVIATION 160), so the order the mutex happens to grant the lock in
#     is not observable in the result. This is the same property that lets
#     the check compare against a host loop, and the multi-block sabotage
#     in `range_kernel_check.mojo` is what proves the merge runs at all.
#
#     PRICE. One `Int32` mutex per (node, feature) that the caller must
#     zero, and a serialized publish per block instead of a wait-free
#     atomic. No timing number is attached and none will be until the perf
#     round.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 162 -- the output is a STRUCT OF ARRAYS, and
#     `Dataset` is passed as its component pointers
#
#     THEIRS. `computeSplitKernel` takes `const Dataset<DataT, LabelT,
#     IdxT> dataset` BY VALUE (`:221`) and reads `work_items[nid]` and
#     `workload_info[blockIdx.x]` as whole-struct loads (`:238-241`).
#
#     OURS. The kernel takes `data`, `row_ids`, `m` and `n` as separate
#     arguments, and its output is FOUR parallel arrays rather than an
#     array of `FeatureRange`.
#
#     WHY. `PORTING_RULES.md` 4: a whole-struct load in a kernel kills the
#     Metal compiler ("Metal Compiler failed to compile metallib. Please
#     submit a bug report"), reproduced in a 25-line probe on 2026-08-19,
#     with per-field access through the pointer compiling and running.
#     `WorkloadInfo` and `NodeWorkItem` ARE still passed as their ported
#     structs and are read FIELD BY FIELD through the pointer, which is
#     the established workaround; the dataset is unbundled because a
#     by-value struct ARGUMENT is the same shape and there is no reason to
#     find out the hard way.
#
#     THE LAYOUT, which is the API and is stated here so a caller can code
#     against it. Every array is indexed by
#
#         slot = nid * n_sampled_cols + fslot
#
#     where `nid` is the index of the node in THIS BATCH's `work_items`
#     (their `workload_info_cta.nodeid`, `:239`) and `fslot` is the index
#     of the feature in that node's sampled-column list, NOT the column id
#     -- the column id is `colids[nid * n_sampled_cols + fslot]`, their
#     `:250-255`. The four arrays are:
#
#         out_min[slot]        Float32  the range's minimum
#         out_max[slot]        Float32  the range's maximum
#         out_n_missing[slot]  Int32    NaN count over the node's rows
#         out_n_merges[slot]   Int32    how many BLOCKS published into
#                                       this cell
#
#     The first three are `FeatureRange` unbundled, and
#     `feature_range_at()` below reassembles one so callers keep using the
#     struct. The fourth has NO cuML counterpart and is ours: it is the
#     device's own report of which path it took, so a check can ASSERT
#     that a large node really was served by more than one block instead
#     of inferring it from host arithmetic. It is written unconditionally
#     and by the shipping kernel, not behind a flag, because a check that
#     runs a different binary from the one that ships proves nothing about
#     the one that ships.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 163 -- the empty range is carried IN the output as
#     `min > max`, and NaN never reaches the reduction
#
#     THEIRS. Not applicable: a histogram of no rows is a histogram of
#     zeros and needs no sentinel, and cuML has no missing-value branch in
#     this kernel at all.
#
#     SKLEARN'S. `find_min_max` (`_partitioner.pyx:143-163`) initializes
#     `min_feature_value = INFINITY, max_feature_value = -INFINITY`, seeds
#     BOTH from the first non-missing value, counts NaNs into `n_missing`,
#     and leaves the initial state untouched when every value is missing.
#     `node_feature_min_max` above transcribes that and returns `(1.0,
#     -1.0)` for the all-missing case, keeping the ORDERING property
#     (`min > max`) without needing an infinity.
#
#     OURS, on the device, and this is the part that is a design choice
#     rather than a transcription. A thread that has seen no non-missing
#     value holds `(+inf, -inf)`, which is sklearn's own initial state and
#     is the exact identity of `min`/`max`; NaNs are tested with `v != v`
#     and diverted to the counter BEFORE the comparison, so no NaN is ever
#     an operand of the reduction and `fmin`/`fmax`'s NaN rule -- the one
#     way min/max could fail to be commutative -- is unreachable. After
#     the block reduction, `blk_min > blk_max` is TRUE exactly when the
#     block contributed nothing, because `+inf > -inf`; the merge tests
#     that instead of carrying a separate valid-count, and writes the
#     host's `(1.0, -1.0)` form so that the OUTPUT CELL is a
#     correctly-formed `FeatureRange` after every merge, not only after
#     the last one.
#
#     WHY THAT MATTERS AND IS NOT A DETAIL. It is what makes DEVIATION
#     161's mutex merge closed over its own output: the merge reads the
#     cell, and the cell already says whether it is empty, so no block
#     needs to be identified as last in order to convert an internal
#     `(+inf, -inf)` accumulator into the published sentinel. A node with
#     no rows and an all-NaN column both land on that path, and the check
#     covers both.
#
#     THE ONE THING A CALLER MUST NOT DO: read `out_min`/`out_max` as
#     numbers without testing `min > max` first. `node_feature_is_constant`
#     already reports such a range constant, which is the behaviour
#     `_splitter.pyx:611-618` gives an all-missing column, so the normal
#     path is safe; a caller that draws a threshold from the raw pair is
#     not.
#     ==================================================================
# ============================================================================

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, inf
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    WorkloadInfo,
)


# Sabotage selectors. They are a kernel ARGUMENT rather than a comptime
# parameter on purpose: a sabotage compiled into a different binary proves
# nothing about the binary that ships, so every arm of
# `range_kernel_check.mojo` runs THIS kernel.
comptime RANGE_SAB_NONE = 0
"""No sabotage. The shipping path."""

comptime RANGE_SAB_ROW_MAJOR = 1
"""Index the feature matrix as `row * N + col` instead of cuML's column-major
`col * M + row` (`dataset.h:24`)."""

comptime RANGE_SAB_NO_ROW_IDS = 2
"""Drop the `row_ids` indirection and treat the slot index as the row id."""

comptime RANGE_SAB_BLOCK0_ONLY = 3
"""Publish only from `offset_blockid == 0`, so a large node sees just the
first block's slice. The block-wide reductions still run in every block --
they are collectives and every thread must reach them."""

comptime RANGE_SAB_NAN_AS_VALUE = 4
"""Skip the `v != v` test, so NaN is neither counted nor diverted."""

comptime RANGE_SAB_NO_SENTINEL = 5
"""Publish `(+inf, -inf)` for an empty contribution instead of the host's
`(1.0, -1.0)` sentinel."""


def node_feature_range_kernel[
    TPB: Int
](
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    mutexes: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    n_sampled_cols_in: Int32,
    sabotage_in: Int32,
):
    """Min, max and NaN count of one feature over one node's rows, on device.

    Step 1 of DEVIATION 137. The control plane -- the `WorkloadInfo`
    flattening, the `blockIdx.y == feature` grid, the `colids` lookup and the
    strided row loop -- is `computeSplitKernel`'s, `builder_kernels_impl.cuh`
    `:236-291`, transcribed. What replaces their body is the accumulator:
    theirs increments a shared histogram, ours keeps three registers.

    GRID. `grid_dim = (n_blocks_dimx, n_sampled_cols)`, `block_dim = TPB`,
    exactly their shape (`builder.cuh` builds `n_blocks_dimx` from
    `updateWorkloadInfo`, and `gridDim.y` is the feature). `block_idx.x`
    indexes `workload_info`, NOT the node -- a large node owns several
    consecutive entries.

    THE ANSWER IS ORDER-INDEPENDENT AND THEREFORE BIT-CHECKABLE. `min` and
    `max` select an input and return it unchanged, so they are associative and
    commutative EXACTLY in floating point and no regrouping can round; the NaN
    count is an exact integer sum. This kernel's block-strided reduction and
    `node_feature_min_max`'s sequential loop are obliged to produce identical
    bits, which is why the check compares `.to_bits()` with no tolerance. See
    DEVIATION BLOCK 160.

    SHARED MEMORY. This kernel declares none. The three block-wide reductions
    come from `max.gpu.primitives.block`, whose scratch is Modular's to size
    (VENDOR_LIBRARIES.md: call the vendor collective, do not hand-write it), so
    there is no budget for this file to query and no per-vendor size to guess.
    Their `extern __shared__ char smem[]` (`:235`) holds the histogram, the
    quantile borders and the done-flag -- all three of which DEVIATION 137
    deletes rather than makes cheaper.

    OUTPUT. Four parallel arrays at `nid * n_sampled_cols + fslot`; see
    DEVIATION BLOCK 162 for the layout and DEVIATION BLOCK 163 for the empty
    sentinel. The caller must seed them with
    `node_feature_range_init_kernel` and zero `mutexes` first.
    """
    # `:238-241` -- their `WorkloadInfo<IdxT> workload_info_cta =
    # workload_info[blockIdx.x]` is a whole-struct load, which is the shape
    # that kills the Metal compiler (DEVIATION 162). Same reads, one field at
    # a time, through the pointer.
    var wb = Int(block_idx.x)
    var fslot = Int(block_idx.y)
    var nid = Int(workload_info[unsafe_offset=wb].nodeid)
    var offset_blockid = Int(workload_info[unsafe_offset=wb].offset_blockid)
    var num_blocks = Int(workload_info[unsafe_offset=wb].num_blocks)

    # `:242-244` -- `const auto work_item = work_items[nid]`, likewise
    # field by field.
    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)

    var m = Int(m_in)
    var n = Int(n_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var sabotage = Int(sabotage_in)

    # `:250-255` -- the feature this block tests. Their `colStart` is the
    # batch offset into the sampled columns and is 0 for a single batch; the
    # `n_sampled_cols == N` arm of their `if` is the identity map, which
    # DEVIATION 156 already records the sampler as MATERIALIZING, so there is
    # one arm here rather than two and `colids` is always consulted.
    var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])

    # `:259, 264-265` -- their `end`, `stride` and `tid`. `blockDim.x` is TPB
    # by construction (the launch below is what sets it), and using the
    # comptime value keeps it in step with the block collectives, which need
    # it at compile time.
    var end = range_start + range_len
    var stride = TPB * num_blocks
    var tid = Int(thread_idx.x) + offset_blockid * TPB

    # sklearn's `min_feature_value = INFINITY, max_feature_value = -INFINITY`
    # (`_partitioner.pyx:143-144`), per thread. It is the identity of
    # min/max, so a thread whose stride never lands inside the node
    # contributes nothing -- and `+inf > -inf` is how the merge below
    # recognises exactly that state (DEVIATION 163).
    var local_min = inf[DType.float32]()
    var local_max = -inf[DType.float32]()
    var local_missing = Int32(0)

    # `:281-291` -- their row loop, with the histogram increment replaced.
    # `col_offset = size_t(col) * dataset.M` and `dataset.data[row +
    # col_offset]` (`:279, 284`) is the COLUMN-MAJOR read `dataset.h:24`
    # specifies; the sabotage arm is the row-major misread it is easy to
    # write instead.
    var col_offset = col * m
    var i = range_start + tid
    while i < end:
        var slot_row = Int(row_ids[unsafe_offset=i])
        if sabotage == RANGE_SAB_NO_ROW_IDS:
            slot_row = i
        var v: Float32
        if sabotage == RANGE_SAB_ROW_MAJOR:
            v = data[unsafe_offset = slot_row * n + col]
        else:
            v = data[unsafe_offset = col_offset + slot_row]

        # `_partitioner.pyx:146-163`, and the ORDER matters: the missing test
        # comes first, so a NaN is never an operand of a comparison. That is
        # what makes `fmin`/`fmax`'s NaN rule unreachable here and the
        # reduction exactly commutative (DEVIATION 163).
        if v != v and sabotage != RANGE_SAB_NAN_AS_VALUE:
            local_missing += 1
        else:
            # Two `if`s rather than the host's `if / elif`. With a seeded
            # +/-infinity the two forms agree on every input: the first value
            # a thread sees is below `+inf` AND above `-inf`, so it lands in
            # both, which is precisely what the host's "seed both from the
            # first non-missing value" does in one branch.
            if v < local_min:
                local_min = v
            if v > local_max:
                local_max = v
        i += stride

    # Their `__syncthreads()` at `:293` before the merge. The block
    # collectives replace their shared histogram; every thread must reach all
    # three, so no arm above may return early.
    var blk_min = block_min[block_size=TPB](local_min)
    barrier()
    var blk_max = block_max[block_size=TPB](local_max)
    barrier()
    var blk_missing = block_sum[block_size=TPB](local_missing)
    barrier()

    # `:295-313` -- the cross-block merge. Theirs is an atomicAdd into a
    # `large_nodeid`-keyed scratch plus `signalDone`; ours is a mutex merge
    # into the output cell itself, for the reasons in DEVIATION BLOCK 161.
    # Only thread 0 takes the lock, which is also their shape (`grid_sync.cuh
    # :240`, `if (threadIdx.x == 0)`).
    if Int(thread_idx.x) != 0:
        return
    if sabotage == RANGE_SAB_BLOCK0_ONLY and offset_blockid != 0:
        return

    var slot = nid * n_sampled_cols + fslot

    # Their `while (atomicCAS(mutex, 0, 1));` as an acquire-load spin plus a
    # weak relaxed claim -- the translation `ensemble/.../split.mojo:548-561`
    # established and enqueued, because `threadfence` is NVIDIA-only in Mojo
    # 1.0 and Metal rejects a strong compare-exchange by name.
    while True:
        if (
            Atomic.load[ordering = Ordering.ACQUIRE](mutexes.unsafe_offset(slot))
            != Int32(0)
        ):
            continue
        var expected = Int32(0)
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,
        ](mutexes.unsafe_offset(slot), expected, Int32(1)):
            break

    var cur_min = out_min[unsafe_offset=slot]
    var cur_max = out_max[unsafe_offset=slot]

    # `blk_min > blk_max` iff this block contributed no non-missing value,
    # because the seeds are `+inf` and `-inf`. Same test on the cell, which
    # holds the host's `(1.0, -1.0)` while empty. DEVIATION 163.
    var have_block = not (blk_min > blk_max)
    if sabotage == RANGE_SAB_NO_SENTINEL:
        have_block = True
    if have_block:
        if cur_min > cur_max:
            cur_min = blk_min
            cur_max = blk_max
        else:
            if blk_min < cur_min:
                cur_min = blk_min
            if blk_max > cur_max:
                cur_max = blk_max

    out_min[unsafe_offset=slot] = cur_min
    out_max[unsafe_offset=slot] = cur_max
    out_n_missing[unsafe_offset=slot] = (
        out_n_missing[unsafe_offset=slot] + blk_missing
    )
    out_n_merges[unsafe_offset=slot] = out_n_merges[unsafe_offset=slot] + 1

    # Their `__threadfence(); atomicExch(mutex, 0);` folded into one RELEASE
    # store, which orders the four writes above before the handback.
    Atomic.store[ordering = Ordering.RELEASE](
        mutexes.unsafe_offset(slot), Int32(0)
    )


def node_feature_range_init_kernel(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
    len: Int32,
):
    """Seed every output cell with the EMPTY range, before any merge.

    The counterpart of `initSplit` (`split.cuh:284-289`), which cuML calls
    once per batch for the same reason: the merge below reads the cell it is
    updating, so an unseeded cell is read garbage. A grid-stride write-only
    map, which is what their `raft::linalg::writeOnlyUnaryOp` is.

    The seed is the host function's own all-missing return, `(1.0, -1.0)`,
    NOT `(+inf, -inf)`: the invariant is that an output cell is a
    correctly-formed `FeatureRange` after EVERY merge including zero of them
    (DEVIATION 163).
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        out_min[unsafe_offset=idx] = Float32(1.0)
        out_max[unsafe_offset=idx] = Float32(-1.0)
        out_n_missing[unsafe_offset=idx] = Int32(0)
        out_n_merges[unsafe_offset=idx] = Int32(0)
        idx += stride


def feature_range_at(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    nid: Int,
    fslot: Int,
    n_sampled_cols: Int,
) -> FeatureRange:
    """Reassemble one cell of the kernel's output as a `FeatureRange`.

    DEVIATION 162 unbundles the struct for the device; this puts it back for
    a host caller, so nothing downstream has to know the layout.
    """
    var slot = nid * n_sampled_cols + fslot
    return FeatureRange(
        out_min[unsafe_offset=slot],
        out_max[unsafe_offset=slot],
        out_n_missing[unsafe_offset=slot],
    )


@fieldwise_init
struct WorkloadPlan(Copyable, Movable):
    """What `updateWorkloadInfo` returns, as a struct because Mojo will not
    return their `std::pair` plus an out-parameter."""

    var info: List[WorkloadInfo]
    """`n_blocks_dimx` entries, one per threadblock along x."""

    var n_blocks_dimx: Int
    """`gridDim.x` for the range kernel."""

    var n_large_nodes: Int
    """Nodes needing more than one block. Their scratch key; ours only reports
    it, since DEVIATION 161's merge needs no per-large-node scratch."""


def build_workload_info(
    work_items: List[NodeWorkItem], tpb: Int = TPB_DEFAULT
) -> WorkloadPlan:
    """Flatten a ragged batch of nodes into a list of threadblocks.

    `builder.cuh:365-385` (`Builder::updateWorkloadInfo`), transcribed:

        n_blocks_per_node = max(ceildiv(item.instances.count, TPB), 1)
        if (n_blocks_per_node > 1) ++n_large_nodes;
        for (b in 0..n_blocks_per_node)
          h_workload_info[n_blocks_dimx + b] =
            {int(i), n_large_nodes - 1, b, n_blocks_per_node};
        n_blocks_dimx += n_blocks_per_node;

    NOTE `n_large_nodes - 1` IS EVALUATED AFTER THE INCREMENT, so a small
    node inherits the PREVIOUS large node's `large_nodeid` (and -1 before any
    large node has been seen). That is theirs and is kept: a small node never
    uses the field, because only `num_blocks > 1` reaches the scratch.

    THE `max(..., 1)`: a node with zero rows still gets ONE block, which
    finds nothing and publishes the empty sentinel (DEVIATION 163).

    THIS FUNCTION IS `builder.cuh`'S, NOT `builder_kernels_impl.cuh`'S, and
    it is here because `builder.mojo` belongs to another session's lane this
    round. Moving it there is a merge-time action, the same convention
    `extratrees/DEVIATIONS.md` uses for its own placement.
    """
    var info = List[WorkloadInfo]()
    var n_large_nodes = 0
    var n_blocks_dimx = 0
    for i in range(len(work_items)):
        var count = Int(work_items[i].instances.count)
        var n_blocks_per_node = ceildiv(count, tpb)
        if n_blocks_per_node < 1:
            n_blocks_per_node = 1
        if n_blocks_per_node > 1:
            n_large_nodes += 1
        for b in range(n_blocks_per_node):
            info.append(
                WorkloadInfo(
                    Int32(i),
                    Int32(n_large_nodes - 1),
                    Int32(b),
                    Int32(n_blocks_per_node),
                )
            )
        n_blocks_dimx += n_blocks_per_node
    return WorkloadPlan(info^, n_blocks_dimx, n_large_nodes)
