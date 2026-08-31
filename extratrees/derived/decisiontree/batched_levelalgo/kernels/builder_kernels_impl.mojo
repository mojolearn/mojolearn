# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
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

from extratrees.original.pcg_rng import SplitKey, key_for, uniform_float
from extratrees.derived.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.derived.decisiontree.batched_levelalgo.split import Split
from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels import (
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
    `max`. `original/range_draw_check.mojo` counts how often -- it is not
    assumed either way.

    THIS IS THE HOST FORM AND IT IS THE AUTHORITY. A kernel must NOT call it:
    the rescale it delegates to is contracted into an FMA by the GPU backend
    and comes back up to 14 ulps away (measured -- DEVIATION BLOCK 173).
    `draw_threshold_device` below is the same draw with the rescale pinned to
    an EXPLICIT `fma`, which no backend may re-round, and
    `score_kernel_check.mojo` counts how far the two disagree every run.
    """
    var gen = key.generator()
    var threshold = uniform_float(gen, extent.min_value, extent.max_value)
    if threshold == extent.max_value:
        return extent.min_value
    return threshold


def draw_threshold_raw(key: SplitKey, extent: FeatureRange) -> Float32:
    """`draw_threshold` WITHOUT sklearn's `== max` guard.

    Exists so a check can count how often the guard fires. Never call it from
    the builder. Host form, like `draw_threshold`.
    """
    var gen = key.generator()
    return uniform_float(gen, extent.min_value, extent.max_value)


# ============================================================================
# STEP 1 OF DEVIATION 137, ON THE DEVICE: the per-(node, feature) RANGE pass.
#
#     Verified by:  extratrees/original/range_kernel_check.mojo
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
from std.memory import bitcast
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, inf
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier

from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

comptime BUILD_MODE = GLOBAL_NUMERIC_MODE
"""The repo-wide numeric mode (`original/numerics.mojo`). DEVIATION 452
branches the range pass's block fold on it; every `ftz` call below reads it
inside the helper. FAST -- the default -- compiles both to the code this file
always had, bit for bit."""

from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels import (
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
"""Publish `(+inf, -inf)` for an empty cell instead of the host's
`(1.0, -1.0)` sentinel. Selected in `node_feature_range_decode_kernel`, which
is where the sentinel lives once the merge is lock-free (DEVIATION 204)."""

comptime RANGE_SAB_EMPTY_NOT_IDENTITY = 7
"""An empty block publishes `range_key(0.0)` rather than its `+inf`/`-inf`
identities. DEVIATION 204 deleted 163's sentinel gate from the merge on the
grounds that those two keys are the identities of min and max; this is the arm
that makes that grounds observable instead of merely stated."""

comptime RANGE_SAB_SIGN_UNFLIPPED = 6
"""Skip `range_key`'s NEGATIVE branch: set the sign bit unconditionally
instead of inverting a negative's bits.

For a non-negative float the two spellings are the SAME expression, so this
arm is invisible on non-negative data and moves exactly the cells that carry
a negative value -- which is the whole content of DEVIATION 204's map, and
the reason the fixture has to carry negatives. The first attempt at this arm
merged the raw bit pattern instead, which the DECODE then misread as a key,
so all 84 cells moved and the arm predicted nothing; the check refused it,
which is what a shape check is for."""


def range_key(v: Float32) -> UInt32:
    """A float as an unsigned key whose INTEGER order is the float's order.

    ==================================================================
    DEVIATION BLOCK -- DEVIATION 204. THE CROSS-BLOCK RANGE MERGE IS
    LOCK-FREE. THIS CLOSES DEVIATION 161.

    161 said Mojo has no portable float `atomicMin`/`atomicMax`, which
    is true, and concluded that the merge had to take a lock. The
    conclusion was wrong: the standard order-preserving map turns the
    float compare into an INTEGER one, and integer `atomicMin`/`Max`
    exist on every backend this lane targets.

    THE MAP. For a non-negative float, set the sign bit; for a negative
    one, invert every bit. That is monotone on the whole of IEEE-754
    excluding NaN, and it is exactly invertible, so nothing is
    approximated -- `range_unkey(range_key(x))` is `x` BIT FOR BIT, and
    `range_key_check` asserts that over a scattered fixture rather than
    on a handful of round numbers. NaN never reaches it: the row loop
    counts NaNs separately and never lets one become an operand
    (DEVIATION 163).

    WHAT THE LOCK COST, MEASURED. At 581,012 rows the root level runs
    4,540 blocks against `n_sampled_cols` cells, and every one of them
    took the same spin lock. Instrumented per phase, the range pass was
    199-260 ms of a 250-330 ms tree at `max_features=5` -- 80% of the
    time -- while the SCORE pass, which reads the same rows and does
    strictly more arithmetic, took 27-36 ms. Six to seven times slower
    for less work is not the work; it is the lock.

    THE SENTINEL GATE IN THE MERGE GOES WITH IT; THE SENTINEL DOES NOT.
    163 gated the merge on `not (blk_min > blk_max)` so that a block
    which saw no value could not push `+inf`/`-inf` into the cell. Under
    an atomic min/max that gate is a NO-OP by construction:
    `range_key(+inf)` is the LARGEST key, so it cannot win a minimum,
    and `range_key(-inf)` is the smallest, so it cannot win a maximum.
    They are the identities. So the gate is deleted rather than left as
    a line that can never fire.

    The EMPTY-CELL rule itself survives one level up, and so does its
    sabotage: a cell no block ever touched still holds both seeds, which
    decode to `(+inf, -inf)`, and
    `node_feature_range_decode_kernel` turns that into the host
    function's own `(1.0, -1.0)`. `RANGE_SAB_NO_SENTINEL` now selects
    that kernel's arm instead of the merge's, and predicts exactly what
    it predicted before -- only the cells with no non-missing value
    move.

    THE MAP ITSELF NEEDED ITS OWN SABOTAGE, because nothing above would
    have caught a wrong one: `RANGE_SAB_SIGN_UNFLIPPED` drops the
    negative branch, which is the same expression on non-negative data
    and the wrong order on negative data, so it moves exactly the cells
    that carry a negative value and no others.

    AND DELETING THE GATE NEEDED ONE TOO. The gate is safe only
    BECAUSE `range_key` sends `+inf` to the largest key and `-inf` to
    the smallest; that is an argument, and an argument in a comment is
    not a check. `RANGE_SAB_EMPTY_NOT_IDENTITY` makes an empty block
    publish `range_key(0.0)` instead, and moves exactly the cells that
    HAVE an empty contribution -- the all-NaN column everywhere plus
    every column of the 0-row node, which is the same set the empty-cell
    sentinel moves and is computed the same way.
    ==================================================================
    """
    var b = rebind[UInt32](v.to_bits())
    if (b & UInt32(0x80000000)) != 0:
        return ~b
    return b | UInt32(0x80000000)


def range_unkey(k: UInt32) -> Float32:
    """The exact inverse of `range_key`."""
    if (k & UInt32(0x80000000)) != 0:
        return bitcast[DType.float32](k ^ UInt32(0x80000000))
    return bitcast[DType.float32](~k)


comptime RANGE_KEY_MIN_SEED: UInt32 = 0xFF800000
"""`range_key(+inf)`: the largest key, so an untouched cell loses every min."""
comptime RANGE_KEY_MAX_SEED: UInt32 = 0x007FFFFF
"""`range_key(-inf)`: the smallest key, so an untouched cell loses every max."""


def node_feature_range_decode_kernel(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    minkey: MutPointer[UInt32, MutAnyOrigin],
    maxkey: MutPointer[UInt32, MutAnyOrigin],
    len: Int32,
    sabotage_in: Int32,
):
    """Keys back into the `(min, max)` floats every later pass reads.

    A cell no block touched still holds both seeds; it decodes to
    `(+inf, -inf)`, and this writes the host function's all-missing return
    `(1.0, -1.0)` instead, so the invariant DEVIATION 163 states -- an output
    cell is a correctly-formed `FeatureRange` after zero merges as well as
    after many -- holds unchanged.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(len):
        var lo = range_unkey(minkey[unsafe_offset=idx])
        var hi = range_unkey(maxkey[unsafe_offset=idx])
        if lo > hi and sabotage_in != RANGE_SAB_NO_SENTINEL:
            out_min[unsafe_offset=idx] = Float32(1.0)
            out_max[unsafe_offset=idx] = Float32(-1.0)
        else:
            out_min[unsafe_offset=idx] = lo
            out_max[unsafe_offset=idx] = hi
        idx += stride


def node_feature_range_kernel[
    TPB: Int
](
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    out_n_merges: MutPointer[Int32, MutAnyOrigin],
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
    #
    # ==================================================================
    # DEVIATION BLOCK 452 -- under NUMERIC_IDENTICAL the block fold runs
    # in KEY space, because the library's FLOAT min/max fold is the one
    # order-dependent bit source left in this pass
    #
    # The library block collectives fold at the HARDWARE warp width -- 32
    # on Apple and NVIDIA, 64 on AMD wavefronts (IDENTITY_PATHS row 8's
    # residue class). For float `min`/`max` that grouping is invisible in
    # VALUE, and invisible in BITS too on all inputs except one: `-0.0`
    # and `+0.0` compare EQUAL, so which zero survives `min(-0.0, +0.0)`
    # is decided by operand ORDER, i.e. by the fold shape. A (node,
    # column) holding both zeros can therefore publish a min (or max)
    # whose SIGN BIT differs on the AMD column, and the sign bit of the
    # published range reaches `quesval` through the `== max -> min`
    # guard -- a bit in the MODEL.
    #
    # `range_key` already maps the floats onto a TOTAL integer order for
    # the cross-block atomics (DEVIATION 204); under IDENTICAL the
    # block-level fold uses the same keys, so any fold width and any
    # grouping returns the same bits, and the two zeros are ordered
    # (-0.0 below +0.0) instead of tied. The per-thread loop above is
    # untouched: its row order is fixed by the strided assignment, which
    # is a pure function of (count, TPB, num_blocks), all data-derived.
    #
    # FAST -- the default -- keeps the float fold, bit for bit.
    # ==================================================================
    var kmin: UInt32
    var kmax: UInt32
    comptime if BUILD_MODE == NUMERIC_IDENTICAL:
        kmin = block_min[block_size=TPB](range_key(local_min))
        barrier()
        kmax = block_max[block_size=TPB](range_key(local_max))
        barrier()
    else:
        var blk_min = block_min[block_size=TPB](local_min)
        barrier()
        var blk_max = block_max[block_size=TPB](local_max)
        barrier()
        kmin = range_key(blk_min)
        kmax = range_key(blk_max)
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

    # DEVIATION 204. Four independent atomics, no lock and no critical
    # section. `range_key` makes the float min/max an INTEGER min/max, and
    # the seeds are the identities, so a block that saw nothing cannot move
    # either extreme and needs no gate -- see the block on `range_key`.
    # `kmin > kmax` in key space is EXACTLY `blk_min > blk_max` in float
    # space -- the key order is the float order, and the empty block's
    # `(+inf, -inf)` pair maps to the two extreme keys.
    var empty_block = kmin > kmax
    if sabotage == RANGE_SAB_SIGN_UNFLIPPED and not empty_block:
        # FINITE VALUES ONLY. Applied to the `+/-inf` an empty block carries,
        # this would ALSO break the identity property, and the arm would move
        # the union of two mechanisms' cells instead of one's -- which is what
        # the first version did, and what the shape check rejected. The
        # decode-then-rebit round trip is exact (`range_unkey` inverts
        # `range_key` bit for bit), so this arm is unchanged under 452.
        kmin = rebind[UInt32](range_unkey(kmin).to_bits()) | UInt32(0x80000000)
        kmax = rebind[UInt32](range_unkey(kmax).to_bits()) | UInt32(0x80000000)
    if sabotage == RANGE_SAB_EMPTY_NOT_IDENTITY and empty_block:
        kmin = range_key(Float32(0.0))
        kmax = range_key(Float32(0.0))
    Atomic.min(out_minkey.unsafe_offset(slot), kmin)
    Atomic.max(out_maxkey.unsafe_offset(slot), kmax)
    _ = Atomic.fetch_add(out_n_missing.unsafe_offset(slot), blk_missing)
    _ = Atomic.fetch_add(out_n_merges.unsafe_offset(slot), Int32(1))


def node_feature_range_init_kernel(
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_minkey: MutPointer[UInt32, MutAnyOrigin],
    out_maxkey: MutPointer[UInt32, MutAnyOrigin],
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
        out_minkey[unsafe_offset=idx] = RANGE_KEY_MIN_SEED
        out_maxkey[unsafe_offset=idx] = RANGE_KEY_MAX_SEED
        out_n_missing[unsafe_offset=idx] = Int32(0)
        out_n_merges[unsafe_offset=idx] = Int32(0)
        idx += stride


def node_nonconstant_flag_kernel(
    out_flag: MutPointer[Int32, MutAnyOrigin],
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_n_missing: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    n_cells: Int32,
    n_sampled_cols: Int32,
):
    """Per node: did ANY of its sampled columns vary?

    DEVIATION 205 needs this and nothing else about the range cells. sklearn's
    loop keeps drawing while every draw has been constant
    (`_splitter.pyx:573-577`), so the quantity that decides whether a node is
    finished is "was a NON-CONSTANT feature evaluated", not "was a split
    found" -- a non-constant column rejected by `min_samples_leaf` still
    counts (`:665-666` is a `continue`).

    It reduces to ONE Int32 per node so the host reads back `n_nodes` values
    instead of the `3 * n_cells` the range cells would cost. `out_flag` must be
    zeroed first. The test is `node_feature_is_constant`, the SAME function the
    host trainer calls, because a second spelling of a `1e-7` band is a second
    answer.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(n_cells):
        var nid = idx // Int(n_sampled_cols)
        var extent = FeatureRange(
            out_min[unsafe_offset=idx],
            out_max[unsafe_offset=idx],
            out_n_missing[unsafe_offset=idx],
        )
        if not node_feature_is_constant(
            extent, work_items[unsafe_offset=nid].instances.count
        ):
            _ = Atomic.fetch_add(out_flag.unsafe_offset(nid), Int32(1))
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


# ============================================================================
# STEPS 2, 3 AND 4 OF DEVIATION 137, ON THE DEVICE: the constant test, the
# keyed threshold DRAW, and the SCORE pass.
#
#     Verified by:  extratrees/original/score_kernel_check.mojo
#
# Its INPUT is `node_feature_range_kernel`'s output above -- the three range
# arrays -- and its OUTPUT is one scored candidate per (node, feature), ready
# for the split reduction (`evalBestSplit`, step 5, which is theirs and is
# unchanged).
#
# The host functions above stay exactly where they are and none of them is
# replaced. `node_feature_is_constant` is not merely the oracle for step 2, it
# is THE SAME CODE: the kernels CALL it, so that test cannot disagree between
# host and device by transcription drift. Step 3 could not be shared that way
# and the reason is DEVIATION BLOCK 173 -- `draw_threshold`'s rescale is
# contracted into an FMA by the GPU backend and is not by the host compiler, so
# the kernels call `draw_threshold_device`, which pins the same arithmetic to
# an EXPLICIT `fma`. That leaves step 4, the accumulation, as the only body
# written twice. `node_feature_score_host` below is its oracle, written the way
# the range pass's is: sequential, in `row_ids` order, no block structure.
#
#     ==================================================================
#     DEVIATION BLOCK 170 -- the score pass is TWO kernel launches,
#     because their "last threadblock continues" cannot be expressed
#
#     THEIRS (`builder_kernels_impl.cuh:295-317`). A large node's blocks
#     accumulate a partial histogram in shared memory, `BinT::AtomicAdd`
#     it into global memory, `__threadfence()`, and then call
#     `MLCommon::signalDone` (`src_prims/common/grid_sync.cuh:238-247`).
#     The block that drives the counter to zero is "last", and that ONE
#     block reloads the merged histogram, scans it, scores it
#     (`:316-328`) and reduces it (`:340`). One launch does the whole
#     step.
#
#     OURS. Two launches. `node_feature_score_kernel` accumulates and
#     stops; `node_feature_score_finalize_kernel` -- one thread per
#     (node, feature) cell -- draws the threshold again, publishes it
#     with the cell's status, and forms the score from the finished
#     accumulators.
#
#     WHY, and it is the same platform wall DEVIATION 161 hit from the
#     other side. `signalDone` is a GRID-WIDE handshake and it is built
#     out of `__threadfence()`, which Mojo 1.0 comptime-asserts is "only
#     implemented on NVIDIA GPUs" (measured by the RF lane, DEVIATION
#     106). Without a fence there is no portable way for one block to
#     learn that another block's writes are visible, so no block can be
#     elected to continue. A kernel BOUNDARY is that fence: everything
#     the first launch wrote is visible to the second, on every vendor,
#     with no intrinsic at all.
#
#     WHY THE SECOND LAUNCH IS CHEAP RATHER THAN A SECOND PASS OVER THE
#     DATA, which is the objection a reader should raise. It reads no
#     dataset row. Its work per cell is: three range loads, the constant
#     test, one keyed draw, and -- for classification -- a loop over
#     `n_acc` accumulators. The row loop, which is the whole cost of this
#     step, happens exactly once, in the first launch. Their single
#     launch does the same scoring work; it just does it in a block that
#     is already resident.
#
#     WHY THE DRAW IS REPEATED RATHER THAN STORED. It is a pure function
#     of `(seed, tree_id, node_id, col)` (DEVIATION 130), so recomputing
#     it is a handful of integer operations and cannot disagree with the
#     first launch's value. Storing it instead would make the second
#     launch depend on the first having written it, which is a weaker
#     invariant than recomputing something that cannot vary.
#
#     PRICE. One extra launch per level per objective, with grid
#     `ceil(n_cells / 64)` -- a single block for a small batch. No timing
#     number is attached and none will be until the perf round.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 171 -- NOT a deviation: the cross-block merge here
#     IS their `atomicAdd`, and the range pass's mutex is not copied
#
#     A reader arriving from DEVIATION 161 will expect the spin mutex
#     again. It is not here, and the reason is worth stating rather than
#     leaving as a silent difference between two kernels in one file.
#
#     161's argument was never "mutexes are how this repository merges".
#     It was: the range pass's combine is `min`/`max` on `float32`, and
#     there is no portable device atomic for that -- the CUDA idiom is a
#     signed-magnitude bit twiddle into an integer atomic, which differs
#     per vendor. THE SCORE PASS'S COMBINE IS AN INTEGER ADD. `Int32`
#     `atomicAdd` exists on every column of the kernel matrix (measured
#     by the RF lane's `ensemble/original/atomic_width_probe.mojo`,
#     which also measured that the SIXTY-FOUR bit one is a hard compile
#     error on Apple GPU -- see DEVIATION 175 for what that costs here).
#     So this merge is transcribed from `builder_kernels_impl.cuh:301`
#     directly: one `Atomic.fetch_add` per accumulator per block, from
#     thread 0.
#
#     WHY IT CANNOT CHANGE AN ANSWER. Integer addition is associative and
#     exact, so the order the blocks happen to arrive in is not
#     observable in the total. That is DEVIATION 135's ruling used as it
#     was meant to be used: it is the reason the labels are quantized on
#     the host in the first place, and it is what lets this check demand
#     BIT-IDENTICAL agreement with a sequential host loop instead of a
#     tolerance.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 172 -- no dynamic shared memory: the accumulators
#     are a comptime-bounded PRIVATE array, reduced one at a time
#
#     THEIRS (`:233-235`, `:265-271`). `extern __shared__ char smem[]`,
#     sized by the host at launch from `max_n_bins * num_outputs *
#     sizeof(BinT) + ...`, carved with `alignPointer` into a histogram, a
#     quantile copy and a done-flag. Every thread of the block increments
#     the SHARED histogram directly (`:290`, `BinT::IncrementHistogram`,
#     a shared-memory `atomicAdd`).
#
#     OURS. Two `stack_allocation[MAX_ACC, Int32]` arrays PER THREAD, in
#     the default (private) address space, and one
#     `max.gpu.primitives.block.sum` per accumulator to combine them
#     across the block. `MAX_ACC` is a comptime parameter of the kernel;
#     the runtime `n_acc` (the class count, or 1 for regression) must not
#     exceed it, and a launch that would exceed it publishes nothing so
#     the caller sees `SCORE_STATUS_UNVISITED` rather than a corrupted
#     cell.
#
#     WHY. Mojo 1.0's `stack_allocation[..., address_space=SHARED]` is
#     STATIC -- the slot count is a comptime expression -- so a runtime
#     `num_outputs` cannot size a shared blob. The RF lane hit the same
#     wall and answered it the same way (their DEVIATION 103a, a comptime
#     carve-out). What is different here, and is why this costs less than
#     it does there, is that DEVIATION 137 deleted the bin dimension: the
#     shared array they size is `n_bins * n_classes` and ours is
#     `n_classes`, so a comptime bound of 8 or 16 covers the
#     configurations this lane supports rather than truncating them.
#
#     AND WHY PRIVATE RATHER THAN SHARED AT ALL. With one bin there is
#     nothing for the threads of a block to share: each thread's
#     contribution is a private running count that the block reduction
#     combines once, at the end. Their shared histogram exists because
#     `lower_bound` scatters each row into one of `n_bins` slots and the
#     threads must therefore write into each other's slots. Nothing
#     scatters here.
#
#     PRICE, and it is real. `MAX_ACC * 2 * 4` bytes of private memory
#     per thread (128 bytes at `MAX_ACC = 16`), which is memory and not
#     registers, and `n_acc` block reductions per block instead of one
#     shared-memory increment per row. Unmeasured, deliberately: this
#     lane takes no timing numbers. The knob if it ever matters is
#     `MAX_ACC`, which is already a parameter of the launch.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 173 -- every thread draws the threshold, and the
#     rescale is an EXPLICIT `fma` because no source-level barrier
#     survives the GPU backend -- measured, four ways, after the first
#     version of this block claimed the opposite
#
#     THEIRS. `computeSplitKernel` draws no random number at all; their
#     randomness is in the SAMPLER (`builder_kernels.cuh:165-176`), which
#     runs in a different kernel. There is no upstream to copy here.
#
#     OURS. Every thread of every block calls `key_for` and
#     `draw_threshold` -- the same host functions, above, unmodified --
#     and gets the same threshold, because the key is
#     `(seed, tree_id, node_id, col)` and nothing in it depends on the
#     thread, the block, or the arrival order (DEVIATION 130). The
#     alternative -- draw once in thread 0 and broadcast through shared
#     memory -- would need a shared slot, a `barrier()`, and a rule for
#     what the other threads do while they wait; a redundant PCG stream
#     of about forty integer operations is cheaper than the
#     synchronisation it would replace, and it is the reason the keyed
#     draw was chosen over a sequential one in the first place.
#
#     AND THE FIRST RUN OF THAT CHECK FOUND A DEFECT, WHICH IS WHY THIS
#     BLOCK IS LONGER THAN IT WOULD HAVE BEEN. The sentence written here
#     before the measurement was "the draw is bit-identical, DEVIATION
#     142's `@no_inline` barrier holds on the GPU too". It does not, and
#     no barrier written in Mojo source makes it hold.
#
#     WHAT WAS MEASURED, 2026-08-21, Apple GPU, each experiment inside
#     ONE binary so its arms compare:
#
#     (1) THE DEVICE CONTRACTS THE RESCALE. 9 of the 105 (node, feature)
#         cells of `score_kernel_check.mojo` came back with a threshold
#         different from the host's. Over a 2048-draw sweep the device's
#         value was EXACTLY `fma(res, span, min)` in the 332 draws where
#         fused and unfused differ, and the common value in the other
#         1716 -- never a third value. Worst disagreement 8 ulps, where
#         the sum cancels near zero and half an ulp of the product
#         becomes several ulps of the result. A threshold that moves
#         moves rows across the `<=`, so this is an answer difference.
#
#     (2) FOUR SOURCE-LEVEL BARRIERS, ALL USELESS. In one binary, four
#         spellings of the rescale -- plain `res*span + start`; an
#         `@no_inline` multiply, which is DEVIATION 142's fix; an integer
#         bitcast round-trip of the product; and both together -- gave
#         the SAME 332 fused cells on device and the SAME 2048 unfused
#         values on host. A private `stack_allocation` round-trip failed
#         too, and worse: it made the HOST fuse, moving 332 host draws
#         away from `uniform_float`.
#
#     (3) A SHARED-MEMORY ROUND-TRIP LOOKED LIKE A FIX AND WAS NOT. It
#         measured 2048 of 2048 unfused in a small probe, went into this
#         file, and the check still failed on the same 9 cells. The probe
#         had computed the product BEFORE a branch, giving the multiply
#         two uses; that -- not the store -- was what stopped the
#         contraction, and it does not survive being written as one
#         straight-line function. Recorded because it is exactly the kind
#         of measurement that looks like a result and is an artefact.
#
#     THE FIX: AN EXPLICIT `fma`, ON BOTH SIDES OF THE COMPARISON.
#     `draw_threshold_device` computes `res.fma(span, min)`. That is not
#     a barrier against an optimisation, which is the thing that kept
#     failing -- it is a single IEEE-754 operation with one correctly
#     rounded result, so the value is determined by the SOURCE on every
#     backend, and a host caller computing the same expression gets the
#     same bits. AFTER THE FIX every cell of `score_kernel_check.mojo`
#     agrees, on bit patterns, in both objectives.
#
#     AND IT IS ALSO WHAT THE UPSTREAM'S GPU DOES. RAFT writes the
#     rescale as one C++ expression and nvcc defaults to `--fmad=true`,
#     so `custom_next` on a CUDA device IS this fma. The unfused form is
#     an artefact of `tools/rng_oracle` being built with
#     `-ffp-contract=off` (DEVIATION 142) -- a property of that oracle
#     build, not of their kernel.
#
#     THE OPEN ITEM, WHICH IS NOT THIS SUB-LANE'S TO CLOSE.
#     `draw_threshold` (host, RAFT-unfused, what `host_splitter.mojo`
#     uses) and `draw_threshold_device` (fma) disagree in about one draw
#     in eight -- 9 of the 44 scored cells of the check's fixture. So a
#     tree grown on device and a tree grown by the host splitter can
#     differ, and the lane has to pick ONE draw. The recommendation is
#     the fma: it is the upstream's device behaviour, it is fixed by the
#     source rather than by a compiler flag, and it is the only one of
#     the two that a GPU can be made to produce. Taking it re-pins
#     `tools/rng_oracle` and amends DEVIATION 142, both outside these two
#     files. `node_feature_score_host` therefore takes `device_draw` as
#     an ARGUMENT with no hidden default answer, and
#     `score_kernel_check.mojo` measures the size of the disagreement
#     every run.
#
#     PRICE. The PCG stream is recomputed per thread and again in the
#     finalize kernel: arithmetic, no memory traffic, no synchronisation,
#     no shared memory. One `fma` instead of a multiply and an add.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 174 -- the device sees only INTEGERS, and the
#     status array carries what the host oracle raises
#
#     THEIRS. `computeSplitKernel` reads `dataset.labels[row]` as a
#     `LabelT` -- a float for regression -- and `BinT::AggregateBin`
#     accumulates it as a float (`objectives.cuh`). cuML's kernel has no
#     missing-value branch and nothing it can report but a `Split`.
#
#     OURS, two parts.
#
#     1. THE LABELS ARRIVE PRE-QUANTIZED. DEVIATION 135 rules that
#     regression accumulates in fixed point, with the scale chosen ONCE
#     on the host from the whole dataset's sum of magnitudes. This kernel
#     takes that literally: `labels_q` is an `Int32` array, the
#     quantization happens on the host before the launch, and the device
#     performs NO float-to-integer conversion. That matters beyond
#     tidiness -- a conversion on device would put a rounding mode into
#     the accumulation path, and a rounding mode is a per-vendor
#     question. For classification the same array holds the class id,
#     which cuML also stores as a float and casts at the accumulate site;
#     casting it on the host instead removes that cast from the kernel.
#
#     2. A KERNEL CANNOT RAISE, SO REFUSAL BECOMES A STATUS.
#     `host_splitter.mojo::_refuse_missing` raises on `n_missing != 0`
#     (DEVIATION 136: a NaN is an error, not a coin flip), and it does so
#     BEFORE the constant test. This kernel keeps that ORDER and turns
#     the raise into `SCORE_STATUS_MISSING_REFUSED`, which the host
#     caller must convert back into an error. A refusal that is only
#     visible as a status is still visible; a NaN silently sent right --
#     which is what `v <= threshold` does to it -- would not be.
#
#     THE OUTPUT LAYOUT, which is the API and is stated here so the split
#     reduction can be coded against it. With
#
#         slot = nid * n_sampled_cols + fslot
#
#     -- `nid` the index of the node in THIS BATCH's `work_items`, and
#     `fslot` the index of the feature in that node's sampled-column
#     list, NOT the column id (the column id is
#     `colids[nid * n_sampled_cols + fslot]`, their `:250-255`) -- the
#     arrays are:
#
#         out_status[slot]      Int32    one of SCORE_STATUS_*
#         out_threshold[slot]   Float32  the GUARDED draw; 0.0 unless
#                                        the status is SCORED or
#                                        REJECTED_MIN_SAMPLES_LEAF
#         out_n_left[slot]      Int32    rows with `value <= threshold`
#         out_n_total[slot]     Int32    rows the pass actually VISITED,
#                                        which must equal the node's row
#                                        count and is how a check sees a
#                                        strided loop that missed rows
#         out_acc_left[slot * n_acc + k]   Int32
#         out_acc_total[slot * n_acc + k]  Int32
#             classification: `k` is the class, the value is a count;
#             regression: `n_acc == 1` and the value is the FIXED-POINT
#             label sum. The right child is `total - left` in both
#             cases, exactly as cuML recovers it (`objectives.cuh:72-73`,
#             DEVIATION 143).
#         out_gini_num[slot]    Int64    classification only; see 175
#         out_gini_den[slot]    Int64    classification only; see 175
#         out_n_blocks[slot]    Int32    how many BLOCKS accumulated into
#                                        this cell
#
#     `out_n_blocks` has no cuML counterpart and is the same instrument
#     `out_n_merges` is for the range pass (DEVIATION 162): the device's
#     own report of the path it took, written unconditionally by the
#     shipping kernel and not behind a flag. It carries a second fact
#     here that it does not carry there -- it is ZERO for a cell that was
#     skipped, so a check can prove a constant feature was SKIPPED rather
#     than scored to zero, which is a different claim from "its
#     accumulators are zero".
#
#     THE EMPTY RANGE (DEVIATION 163) IS INTERCEPTED BEFORE THE DRAW, and
#     this is the caller obligation 163 names, discharged. A published
#     cell has `min > max` in exactly two situations: every value was
#     missing, which is caught by the refusal above because `n_missing !=
#     0`; and the node has no rows at all, which is caught by the
#     constant test's FIRST arm, because `n_rows == n_missing` reads
#     `0 == 0`. Neither reaches `draw_threshold`, so no threshold is ever
#     drawn from a `(1.0, -1.0)` pair.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 175 -- the published score is `Int64` for Gini and
#     is the ACCUMULATORS for MSE, and the Int64 row bound is 2^21
#
#     THEIRS. `objective.Gain(...)` (`:328`) returns a `Split` whose
#     `best_metric_val` is a single `DataT` float, and `evalBestSplit`
#     (`:340`) reduces on that float.
#
#     OURS. DEVIATIONS 144 and 145 already ruled that the classification
#     reduction key is sklearn's proxy as an EXACT RATIONAL and not a
#     float, so this kernel publishes the rational:
#
#         num = sq_L * nR + sq_R * nL        den = nL * nR
#
#     in two `Int64` arrays, which is exactly `GiniProxyExact`'s pair
#     (`objectives.mojo::ProxyImpurityExact`) so the reduction can hand
#     it to `CompareProxyExact` unchanged. cuML's float `GainPerSplit`
#     is NOT computed here: DEVIATION 145 makes it a REPORTING quantity
#     (`Split.best_metric_val`, `feature_importances_`), it is a pure
#     function of the accumulators this kernel already publishes, and
#     computing a float on device that a float on host must then match
#     bit for bit would put the FMA question of DEVIATION 142 into the
#     scoring path for a number that decides nothing.
#
#     THE ROW BOUND, DERIVED AND THEN MEASURED. `sq_L <= nL^2`, so
#     `num <= nL*nR*n <= n^3/4`. `Int64` holds `2^63`, so the published
#     rational is exact while `n^3/4 <= 2^63`, i.e.
#
#         n <= 2^21.67  ->  SCORE_MAX_ROWS_EXACT = 2^21 = 2,097,152
#
#     rows in ONE NODE, with two thirds of a bit of headroom.
#     `score_row_bound_ok` below is that test, and
#     `original/score_kernel_check.mojo` MEASURES the wrap at
#     `nL = nR = 2^25` rather than asserting the algebra.
#
#     **AND THAT IS A TIGHTER BOUND THAN `MAX_ROWS_EXACT`, WHICH IS THE
#     ONE THING A READER OF `objectives.mojo` MUST NOT ASSUME.** DEVIATION
#     144 sets `MAX_ROWS_EXACT = 2^26` from the `Int128` cross-multiply
#     in `CompareProxyExact`, which is the right bound for THAT step; but
#     `GiniProxyExact.num` is an `Int64` FIELD, so a node between 2^21.67
#     and 2^26 rows overflows the numerator before the comparison ever
#     sees it. The measurement is in the check. This block records it
#     where the kernel that publishes the field can be read; acting on it
#     in `objectives.mojo` is not this sub-lane's file to touch.
#
#     MSE IS DIFFERENT AND SKLEARN'S PROXY IS NOT WHAT IS PUBLISHED.
#     DEVIATION 135's own derivation is why: the MSE proxy's numerator is
#     `sum_L^2 * n_R + sum_R^2 * n_L` over FIXED-POINT sums bounded by
#     `2^SLOT_BITS = 2^30`, so it needs up to `2^60 * n` -- past `Int64`
#     at TEN ROWS IN A NODE, measured, which is precisely why
#     `fixed_point.mojo::mse_proxy_exact` returns `Int128`. There is no
#     `Int128` device buffer to publish it into and no `Int128` multiply
#     that compiles in a kernel (DEVIATION 167).
#
#     **THIS PARAGRAPH USED TO END "the regression instantiation leaves
#     `out_gini_num` and `out_gini_den` at their initialized zeros", AND
#     THAT IS NO LONGER TRUE.** DEVIATION BLOCKS 189-193 below publish a
#     regression key that DOES fit `Int64`, by publishing cuML's own MSE
#     gain rather than sklearn's proxy. The sentence is deleted rather
#     than annotated, per LANE_RULES 17. The same sentence still stands in
#     `extratrees/DEVIATIONS.md` entry 175, which is not this sub-lane's
#     file to edit and is reported as an OPEN item for its owner.
#
#     PRICE. A classification node above 2^21 rows must be refused or the
#     comparison silently wraps. `score_row_bound_ok` makes that a
#     checkable precondition rather than a comment.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 189 -- the REGRESSION key is cuML's OWN MSE gain as
#     an exact rational, not sklearn's proxy, and that is what makes it
#     fit `Int64`
#
#     THEIRS. `MSEObjectiveFunction::GainPerSplit`
#     (`objectives.cuh:225-244`) is
#
#         gain = (-sum_T^2/n - (-sum_L^2/n_L - sum_R^2/n_R)) * 0.5/n
#
#     i.e. sklearn's proxy MINUS the parent term `sum_T^2/n`, times the
#     node constant `0.5/n`. sklearn's `_criterion.pyx:944-973` proxy is
#     `sum_L^2/n_L + sum_R^2/n_R` with both of those constants dropped --
#     exactly the relation DEVIATION 144 records for classification
#     (`cuML_gain == parent_gini + sklearn_proxy / n`), one objective
#     over.
#
#     OURS. The published pair is cuML's gain with its node-constant
#     factor dropped, written over the fixed-point sums:
#
#         A   = sum_L * n_R - sum_R * n_L         (= n_L n_R (mean_L - mean_R))
#         num = (|A| >> j)^2                      den = n_L * n_R
#
#     `A^2/(n_L n_R)` is `n` times cuML's gain and `n` is a NODE constant,
#     so within one node this orders candidates EXACTLY as sklearn's proxy
#     does: the two differ by `sum_T^2/n`, a constant of the node, and a
#     constant shift cannot reorder anything. The reduction only ever
#     compares candidates of ONE node (`split_reduce_kernel`'s
#     `block_idx.y`), so nothing ever compares across the constant.
#
#     WHY THE CENTERED FORM AND NOT THE PROXY -- IT IS THE WIDTH, AND THE
#     WIDTH IS THE WHOLE PROBLEM. sklearn's numerator carries the parent
#     term, which is the biggest thing in it: at the shipped `2^30` slot
#     `sum_L^2 * n_R` alone reaches `2^60 * n`. cuML's gain has that term
#     SUBTRACTED OFF, and what is left is a perfect square of a quantity
#     bounded by `n * 2^30` -- so it can be shrunk by shrinking ONE
#     number before squaring it, instead of by shrinking the labels.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 190 -- MEASURED: sklearn's proxy wraps `Int64` at
#     TEN ROWS, and narrowing the accumulator instead loses 274 of 780
#     orderings
#
#     THE WRAP, computed in `Int128` and read back through `Int64`, not
#     asserted from algebra. With `|sum| <= 2^30 - 1` (what `choose_scale`
#     guarantees) and the worst case `sum_L = 2^30-1, sum_R = 0, n_L = 1`:
#
#         n =  9   num =  9223372019674906632   Int64 agrees
#         n = 10   num = 10376293522134269961   Int64 reads
#                        -8070450551575281655   *** WRAPPED, AND NEGATIVE ***
#
#     Negative matters twice over: `compare_exact_key`'s sign split then
#     ranks that candidate BELOW every other one, so the wrap does not
#     merely lose precision, it inverts the answer.
#
#     SO PUBLISHING SKLEARN'S PROXY IS NOT AVAILABLE AT ANY USEFUL ROW
#     COUNT, and the only lever left for it is a narrower fixed-point
#     scale: `2b + log2(n) <= 63` gives `b = 21` at a million rows against
#     the `b = 30` DEVIATION 135 chooses. **That was measured against a
#     `Float64` ground truth rather than argued**, and
#     `regression_score_check.mojo` arm G recomputes it every run rather
#     than quoting it. At n = 1,048,576, over 190 ordered pairs of hashed
#     candidates of ONE node, counting the pairs each key gets backwards:
#
#         b = 30 exact Int128 proxy (the ceiling)      0 of 190
#         THIS FILE'S published key                    0 of 190
#         narrower accumulator, b = 21 (route 1)      60 of 190
#         shifting the SUMS, keeping sklearn's proxy  42 of 190
#
#     and at n = 65,536, `0 / 0 / 2 / 0`. The published key is as good as
#     the `Int128` form it cannot afford to publish; the two narrowing
#     routes lose a third and a fifth of the orderings at a million rows.
#     **The number decided this, not taste.** The reason the narrowed
#     accumulator does worst is that truncation there costs up to ONE UNIT
#     PER ROW, while truncating `A` once costs one unit of `A`.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 191 -- the shift is NODE-UNIFORM and derived from
#     the row count ALONE, so no caller argument and no new buffer
#
#     `|A| <= max(n_L,n_R) * (|sum_L| + |sum_R|) <= n * 2^30`, so
#     `|A| >> j` fits `REGRESSION_KEY_BITS = 31` bits -- and its square
#     fits `Int64` with a bit to spare -- as soon as
#
#         j >= ceil_log2(n) + REGRESSION_SUM_BITS - REGRESSION_KEY_BITS
#
#     which at the shipped `2^30` slot is `j = ceil_log2(n) - 1`.
#
#     IT HAS TO BE THE SAME `j` FOR EVERY CANDIDATE OF A NODE or the
#     published ratios are not comparable, and that is the whole reason it
#     is derived from `range_len` -- a NODE constant every finalize thread
#     already reads out of `work_items` -- rather than from the cell's own
#     magnitudes, which would be sharper and would silently make two
#     candidates of one node incommensurable. A per-node minimum `j` would
#     need a pass over the node's cells; this needs nothing.
#
#     AND IT IS WHY THERE IS NO ROW CAP. `j` grows with the row count, so
#     the numerator never wraps at any `n`; what a bigger node costs is
#     resolution in `A`, and even that barely moves, because one unit of
#     `sum_L` is `n_R` units of `A` and `j` is only `log2(n) - 1`. A
#     one-unit change in a fixed-point sum therefore moves `A'` by about
#     two at EVERY node size -- which is why the check's adversarial arm,
#     which builds pairs specifically to collapse into one `A'` bucket,
#     reports ZERO ties at n = 1024, 65,536 and 1,048,576 across ~59,000
#     pairs. The truncation costs order, never resolution.
#
#     TRUNCATION IS ON `|A|`, so negating every label gives a
#     bit-identical key. An arithmetic shift of the signed `A` would not:
#     it floors, so `-5 >> 2` is `-2` where `5 >> 2` is `1`. `quantize`
#     truncates toward zero for the same reason (DEVIATION 135), and this
#     keeps that invariance one step further down the pipeline.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 192 -- WHAT THE KEY IS NOT: it is not exactly
#     order-preserving, and this is the honest size of that
#
#     `A' = |A| >> j` is a truncation, so two candidates whose exact
#     `Int128` proxies differ by less than it can tie or swap. That cannot
#     be argued away and it is not hidden behind a "should be fine":
#     `regression_score_check.mojo` arm C MEASURES it, on pairs built to
#     be as close as the arithmetic allows -- adjacent `n_left`, the second
#     candidate's `A` solved for so that it lands in the first one's
#     truncation bucket:
#
#         n =     1024:        0 of 19,955 adversarial pairs invert
#         n =    65,536:       3 of 18,520
#         n = 1,048,576:      18 of 20,000        (0.09%)
#
#     and on ordinary hashed candidates of one node, at every size, ZERO
#     of 1,128.
#
#     THE PHRASE "OF ONE NODE" IS LOAD-BEARING AND THE CHECK LEARNED IT BY
#     FAILING. Its first run drew an independent `sum_total` per candidate
#     and reported 66 of 1,128 inversions. Nothing was wrong with the key:
#     the parent term this form drops is a NODE constant, so candidates
#     from different nodes are not comparable through it -- and never are
#     compared, because `split_reduce_kernel` reduces one node per
#     `block_idx.y`. Arm C now carries that as a named CONTROL: the same
#     candidates with per-candidate totals must come out badly wrong, and
#     if they do not, the fixture cannot tell a node-local key from a
#     global one.
#
#     **NO KEY IN TWO `Int64` FIELDS CAN DO BETTER THAN "approximate"
#     HERE, AND THAT IS THE FINDING, NOT AN EXCUSE.** An exact
#     order-preserving encoding would have to be a rational equal to the
#     proxy, whose reduced denominator is `n_L n_R / gcd` -- unbounded --
#     or a common-denominator rescale, which needs MORE bits than the
#     proxy, not fewer: two proxies of one node that differ at all differ
#     by at least `16/n^4`, so a shared denominator resolving them is
#     `n^4/16`. The exact form is available only below ten rows. So the
#     question was never "exact or not", it was "which approximation",
#     and DEVIATION 190 is the measurement that answers it.
#
#     A tie here is not a coin flip: it falls through to `Split.update`'s
#     chain (DEVIATION 166), which is cuML's own reduction.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 193 -- the slot precondition is a REFUSAL STATUS,
#     and the gain gate is formed on the HOST, DEVIATION 183's shape
#
#     The derivation above needs `|sum_L| <= 2^30` and `|sum_R| <= 2^30`,
#     which is exactly what `choose_scale` guarantees for every partial
#     sum of every node (DEVIATION 135: the bound comes from the WHOLE
#     dataset once). A caller that quantized some other way would silently
#     get a wrapped numerator, so the kernel TESTS it -- and a kernel
#     cannot raise, so the cell becomes `SCORE_STATUS_REGRESSION_REFUSED`
#     and `score_to_candidate_kernel` turns it into the default `Split`,
#     which loses to every scored candidate. A refusal, never a
#     truncation; the same shape DEVIATION 174 gives the missing-value
#     refusal.
#
#     One test suffices for both, and that is derived rather than
#     hopeful: `|A| <= n_R |sum_L| + n_L |sum_R| <= n * 2^30 <= 2^(L+30)`,
#     so the slot test on the two sums IMPLIES `A' <= 2^31`.
#
#     THE GAIN GATE. `split_not_valid` needs cuML's float gain, and
#     DEVIATION 183 already settled where that is formed for
#     classification: on the HOST, in `Float64`, from the exact integers
#     the score pass published, because bringing three small arrays back
#     per level is cheaper than a 128-bit multiply in a kernel.
#     `mse_gain_from_exact_totals` below is the regression counterpart --
#     `0.5 * A^2 / (n^2 n_L n_R)` in the LABEL's units, so it takes
#     `inv_scale` the way `leaf_values_host` does (DEVIATION 179).
#
#     THE FIELD NAMES ARE NOW WRONG AND ARE DELIBERATELY NOT CHANGED.
#     `out_gini_num`, `out_gini_den` and `ScoredCandidate.gini_num` carry
#     the MSE key on this path. Renaming them would touch
#     `builder.mojo`'s `score_to_candidate_kernel` and
#     `score_kernel_check.mojo`, neither of which is this sub-lane's file.
#     Recorded here as an OPEN item rather than done badly across an
#     ownership line.
#     ==================================================================
# ============================================================================


# The five statuses a (node, feature) cell can end in. They are the branches of
# `_splitter.pyx:611-666` made into data, because a kernel cannot `continue` out
# of a loop the host wrote and cannot raise. `_empty_candidate` and
# `CandidateRecord` in `original/host_splitter.mojo` carry the same facts as
# three booleans; a single status is the same information in the width an
# `Int32` output array wants.
comptime SCORE_STATUS_UNVISITED: Int32 = 0
"""The initialized value, and it is not a legal outcome. A cell still holding
it after both launches means the finalize kernel never reached it -- a grid too
small, or `n_acc > MAX_ACC`. A check asserts there are none."""

comptime SCORE_STATUS_SCORED: Int32 = 1
"""Reached `_splitter.pyx:691`: not constant, not refused, not rejected. Only a
scored candidate can win."""

comptime SCORE_STATUS_CONSTANT: Int32 = 2
"""`_splitter.pyx:613-618` said constant, so `:619-626` skipped it. NO ROW WAS
READ: `out_n_blocks` stays 0, which is how a check tells this apart from a
candidate that was scored and happened to accumulate zeros."""

comptime SCORE_STATUS_MISSING_REFUSED: Int32 = 3
"""`n_missing != 0`. DEVIATION 136 refuses missing values rather than sending
them left or right on a coin flip (`_splitter.pyx:649`), and
`host_splitter.mojo::_refuse_missing` raises. A kernel cannot raise, so it
publishes this and the host caller raises. No row is read here either."""

comptime SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF: Int32 = 4
"""`_splitter.pyx:664-666`, a `continue` and NOT a redraw. The rows WERE read
and the accumulators ARE published -- `CandidateRecord` reports them for a
rejected candidate too -- but the score is invalid."""

comptime SCORE_STATUS_REGRESSION_REFUSED: Int32 = 5
"""REGRESSION ONLY, and OURS: a partial sum larger than the fixed-point slot,
so the published key would wrap. DEVIATION BLOCK 193. A caller that quantized
with `fixed_point.mojo::choose_scale` cannot reach this -- the whole point of
that function is that no node's partial sum can leave the slot -- so a cell in
this state means the labels did not come through the fixed-point contract. The
rows WERE read and the accumulators ARE published; only the key is withheld.

There were four statuses here until 2026-08-21, five that day, and six since
`SCORE_STATUS_PURE_NODE` (2026-08-22). A caller enumerating them
(`_status_name`, `score_to_candidate_kernel`'s `!= SCORED` test) needs no
change to be CORRECT -- anything not `SCORED` is already the default `Split`
-- but a caller PRINTING them will show `?` until it learns the new ones."""

comptime SCORE_STATUS_PURE_NODE: Int32 = 6
"""DEVIATION 216's companion (sklearn `_tree.pyx:240`): the node is PURE --
one class holds every row, Gini exactly 0 -- so it is a leaf before any
candidate can win. Classification only: regression purity (zero variance) is
not detectable from the sums the device accumulates, and its cost is bounded
by `min_samples_split` rather than guarded (the 216 entry prices this).
Published per cell because purity is tested where the counts live; every
cell of a pure node carries it."""


comptime SCORE_MAX_ACC_DEFAULT: Int = 16
"""Default comptime bound on `n_acc` (classes, or 1 for regression). See
DEVIATION BLOCK 172: Mojo's `stack_allocation` is comptime-sized, so the
accumulator array needs a bound; DEVIATION 137 deleted the bin dimension, so
the bound is over CLASSES alone and 16 covers this lane."""


comptime SCORE_MAX_ROWS_EXACT: Int = 1 << 21
"""Rows in ONE NODE above which the published `Int64` Gini numerator wraps.

`num = sq_L*nR + sq_R*nL <= nL*nR*n <= n^3/4`, and `Int64` holds `2^63`, so
`n <= 2^21.67`. See DEVIATION BLOCK 175 -- this is TIGHTER than
`objectives.mojo::MAX_ROWS_EXACT` (`2^26`), which bounds the `Int128`
cross-multiply and not the `Int64` field the numerator is stored in."""


def score_row_bound_ok(row_count: Int) -> Bool:
    """Is the published classification rational exact at this node size?

    Computed rather than asserted: the worst case `n^3/4` is formed in `Int128`
    and compared against `2^63 - 1`, so the algebra in DEVIATION BLOCK 175 is a
    checked claim. `fixed_point.mojo::comparator_product_fits` does the same
    thing one width up, for the same reason.
    """
    if row_count <= 0:
        return True
    var n = Int128(row_count)
    var worst = (n * n * n) // Int128(4)
    return worst <= (Int128(1) << Int128(63)) - Int128(1)


# ---------------------------------------------------------------------------
# THE REGRESSION KEY. DEVIATION BLOCKS 189-193 above are the derivation; this
# is the arithmetic, shared verbatim between the host oracle and the kernel so
# there is no second copy to drift.
# ---------------------------------------------------------------------------

comptime REGRESSION_SUM_BITS: Int = 30
"""The fixed-point slot a partial label sum must fit, `2^REGRESSION_SUM_BITS`.

The same number as `fixed_point.mojo::SLOT_BITS`, and it is DELIBERATELY not
imported from there: `fixed_point` is a host module whose functions `raise`,
and this constant is read inside a kernel. It is the caller's contract, not a
free choice -- `choose_scale` guarantees it for every partial sum of every node
of the whole fit, which is why no row cap is needed here (DEVIATION 191)."""

comptime REGRESSION_KEY_BITS: Int = 31
"""Bits `|A| >> j` is allowed to occupy, so `num = (|A| >> j)^2 <= 2^62`.

One bit under `Int64`'s `2^63`, held back the way `accumulator_bits_for` holds
two back, so that a later term added to `num` does not silently cross."""


def classification_key_shift(row_count: Int) -> Int:
    """`s`, the NODE-UNIFORM right shift on the classification squared sums.

    DEVIATION 218 -- deviation 191's scheme applied to the Gini rational,
    lifting DEVIATION 175's 2^21-row refusal. The published pair becomes
    `num = (sq_L >> s)*nR + (sq_R >> s)*nL`, `den = nL*nR`:

      * worst case at `nL = nR = n/2`: `num <= (n^3/4) >> s`, so
        `s = max(0, 3*ceil_log2(n) - 64)` keeps `num` under `2^62` --
        the `-64` is `-62` for the budget and `-2` for the `/4`, and
        getting it wrong by that 2 would shift AT `2^21` and break the
        bit-for-bit claim below (caught by arm E's shift assertions).
      * `den <= n^2/4` holds in Int64 past any Int32 row count.
      * the comparator's Int128 cross-multiply holds to `num*den <= 2^122`.

    `s == 0` for every node at or under 2^21 rows, so the entire formerly-
    legal regime is BIT-FOR-BIT unchanged and every existing identity gate
    still pins it. Above 2^21 the surrendered granularity is `2^s` on sums
    of squares of magnitude ~`2^(2*log2 n)` -- relative `<= 2^-40` at any
    `n` -- and candidates inside one granule tie into the total order, the
    same surrender deviation 191 already made for regression at its cap.
    The loop is `ceil_log2`'s, for its reason: a bound must not depend on a
    transcendental.
    """
    var bits = 0
    var acc = 1
    while acc < row_count:
        acc *= 2
        bits += 1
    var s = 3 * bits - 64
    return s if s > 0 else 0


def float_gain_key(gain: Float32) -> Int64:
    """The order-preserving SIGN-MAGNITUDE map of a Float32 gain onto
    `Int64`, the ENTROPY cell's `num` (with `den = 1`). DEVIATION 459.

    `key = mag` for a non-negative gain and `-mag` for a negative one, where
    `mag` is the low 31 bits of the float's pattern. For finite floats the
    magnitude bits order exactly as the magnitudes do, so the signed key
    orders exactly as the float does under `<`; `-0.0` and `+0.0` both map to
    `0`, which is the tie float `==` gives them and the tie cuML's
    `Split::update` first arm gives them. NaN never reaches this (every
    entropy term is `log` of a probability in `[1/n, 1]`). Stored in the
    `Int64` numerator, compared by `compare_exact_key` with `den == 1` on
    both sides, so the reduction's cross-multiply is this integer's order.
    Shared by the finalize kernel and the host oracle so there is no second
    copy to drift. NOT `range_key`: that map orders the two zeros, which is
    the right answer for a range fold and the wrong one for a gain tie.
    """
    var bits = Int64(Int(gain.to_bits()))
    var mag = bits & Int64(0x7FFFFFFF)
    if (bits & Int64(0x80000000)) != 0:
        return -mag
    return mag


def regression_key_shift(row_count: Int) -> Int:
    """`j`, the NODE-UNIFORM right shift on `|A|`. DEVIATION BLOCK 191.

    `j = ceil_log2(n) + REGRESSION_SUM_BITS - REGRESSION_KEY_BITS`, clamped at
    zero. Called from a kernel, so it cannot raise and it cannot use
    `fixed_point.mojo::ceil_log2`; the loop is that function's, and it is a
    loop for that function's reason -- a bound must not depend on a
    transcendental (`std.math.log` carries ~5e-8 of absolute error in this
    toolchain and has already re-decided a tie once in this repository).

    `row_count` is the NODE's row count, read from `work_items`, and NOT the
    accumulated `n_total`: it is the same number in every cell of the node by
    construction, which is the property the whole scheme rests on.
    """
    var bits = 0
    var acc = 1
    while acc < row_count:
        acc *= 2
        bits += 1
    var j = bits + REGRESSION_SUM_BITS - REGRESSION_KEY_BITS
    return j if j > 0 else 0


def regression_key_bound_ok(row_count: Int, max_abs_sum: Int) -> Bool:
    """Does the published `Int64` numerator hold at this node size and this
    largest partial-sum magnitude?

    HOST-side and computed in `Int128`, the way `score_row_bound_ok` and
    `fixed_point.mojo::comparator_product_fits` are, so DEVIATION BLOCK 191's
    algebra is a CHECKED claim: form the worst case `|A| = n * max_abs_sum`,
    shift it, square it in `Int128` and compare against `Int64`.
    """
    if row_count <= 0:
        return True
    var j = regression_key_shift(row_count)
    var worst = Int128(row_count) * Int128(max_abs_sum)
    var scaled = worst >> Int128(j)
    return scaled * scaled <= (Int128(1) << Int128(63)) - Int128(1)


def mse_gain_from_exact_totals(
    sum_left: Int64,
    sum_total: Int64,
    n_left: Int,
    n_right: Int,
    inv_scale: Float64,
) -> Float32:
    """cuML's `MSEObjectiveFunction::GainPerSplit` for the winning candidate,
    from EXACT INTEGERS, in the LABEL's own units.

    The regression counterpart of `builder.mojo::gain_from_exact_totals`, and
    it exists for DEVIATION 183's reason: `split_not_valid` compares this
    against `min_impurity_decrease`, the device does not compute a float gain,
    and forming it on the host from integers that carry no rounding is both
    cheaper and more exact than putting it in a kernel.

    `objectives.cuh:234-241` is `0.5/n * (sum_L^2/n_L + sum_R^2/n_R -
    sum_T^2/n)`, which is `0.5 * A^2 / (n^2 n_L n_R)`. `inv_scale` is
    `1 / scale` for the scale DEVIATION 135 chose, and it enters SQUARED
    because the gain is quadratic in the label -- the same dequantization
    `leaf_values_host` applies once (DEVIATION 179).

    Returns `0.0` for an empty child rather than cuML's `+inf`: their float
    form divides by `nLeft` (`objectives.cuh:57` for Gini, the same shape
    here), and a caller reaching this with an empty child has already been
    rejected by `min_samples_leaf`.
    """
    if n_left <= 0 or n_right <= 0:
        return Float32(0.0)
    var nl = Float64(n_left)
    var nr = Float64(n_right)
    var n = nl + nr
    var sum_right = sum_total - sum_left
    var a = Float64(sum_left) * nr - Float64(sum_right) * nl
    # Divided by `n` BEFORE squaring: `A` reaches 2^62 and `Float64` carries 53
    # bits, so squaring first would round away what the division then cannot
    # restore.
    var t = a / n * inv_scale
    return Float32(0.5 * t * t / (n * nl * nr))


# Sabotage selectors, one per MECHANISM. A kernel ARGUMENT and not a comptime
# parameter, for the reason `RANGE_SAB_*` above states: a sabotage compiled into
# a different binary proves nothing about the binary that ships, so every arm of
# `score_kernel_check.mojo` runs THIS kernel.
comptime SCORE_SAB_NONE: Int32 = 0
"""No sabotage. The shipping path."""

comptime SCORE_SAB_CONSTANT_STRICT: Int32 = 1
"""Test `max < min + FEATURE_THRESHOLD` instead of `max <= min + ...`
(`_splitter.pyx:616-617`). Moves ONLY a column whose spread is exactly
`FEATURE_THRESHOLD` in float32 -- which is why `fixtures.mojo`'s near-constant
straddle triple exists."""

comptime SCORE_SAB_NO_MAX_GUARD: Int32 = 2
"""Drop sklearn's `:653-654` `threshold == max -> min` guard, i.e. call
`draw_threshold_raw`. A threshold equal to `max` sends EVERY row left."""

comptime SCORE_SAB_SIDE_INVERTED: Int32 = 3
"""Accumulate a row into the LEFT child when `value > threshold`. The
assignment `value <= threshold` goes left is `builder_kernels_impl.cuh:65-66`
and `_partitioner.pyx:236`."""

comptime SCORE_SAB_STRICT_LESS: Int32 = 4
"""`value < threshold` goes left, so a row EQUAL to the threshold goes right.
Invisible unless some row sits exactly on the threshold -- which is guaranteed
in any cell where the `== max -> min` guard fired, because `min` is attained by
definition."""

comptime SCORE_SAB_NO_ROW_IDS: Int32 = 5
"""Drop the `row_ids` indirection and treat the slot index as the row id, for
BOTH the feature value and the label."""

comptime SCORE_SAB_SCALE_X2: Int32 = 6
"""Regression: accumulate `2 * labels_q[row]`, i.e. run the device at twice the
host's fixed-point scale. DEVIATION 135 chooses the scale ONCE on the host; this
is what a scale that does not reach the device intact looks like."""

comptime SCORE_SAB_FLOAT_ACCUM: Int32 = 7
"""Regression: accumulate the label sum in `Float32` -- per thread AND through
the block reduction -- and truncate once at the publish, instead of in fixed
point. This is the mechanism DEVIATION 135 exists to remove, and the sabotage
that makes its claim falsifiable. It must be the WHOLE reduction: a per-thread
float partial over a handful of rows is exact, so truncating there is a
sabotage with no target."""

comptime SCORE_SAB_BLOCK0_ONLY: Int32 = 8
"""Publish only from `offset_blockid == 0`, so a large node sees one block's
slice. The block collectives still run in every block -- they are collectives
and every thread must reach them."""

comptime SCORE_SAB_TOTAL_IS_LEFT: Int32 = 9
"""Accumulate the node TOTALS over left-going rows only, so `total - left` --
cuML's own recovery of the right child (`objectives.cuh:72-73`, DEVIATION 143)
-- is zero."""

# The REGRESSION KEY's mechanisms, one selector each. DEVIATION BLOCKS 189-193.
# They sit here, apart from the constants above, only because they arrived with
# the regression key; they are kernel ARGUMENTS for the same reason all of the
# others are -- a sabotage compiled into a different binary proves nothing
# about the binary that ships.
comptime SCORE_SAB_REG_NO_SHIFT: Int32 = 10
"""Regression: publish `|A|^2` with NO shift, i.e. `j = 0`. DEVIATION BLOCK
191's shift is what keeps the numerator inside `Int64`, so this is the wrap
itself, running in the shipping kernel."""

comptime SCORE_SAB_REG_NO_CENTER: Int32 = 11
"""Regression: `A = sum_L * n_R` instead of `sum_L * n_R - sum_R * n_L`, i.e.
cuML's gain without the right child's term (`objectives.cuh:237-238`). The
result is still a valid-looking rational, which is exactly why it needs a
sabotage rather than a bounds check."""

comptime SCORE_SAB_REG_DEN_ONE: Int32 = 12
"""Regression: publish `den = 1`, so the key becomes `A'^2` and the
`n_L n_R` weighting -- the difference between cuML's gain and the squared mean
gap -- disappears."""

comptime SCORE_SAB_REG_NO_SLOT_GUARD: Int32 = 13
"""Regression: skip the fixed-point slot precondition of DEVIATION BLOCK 193,
so a partial sum outside the slot publishes a WRAPPED numerator instead of
`SCORE_STATUS_REGRESSION_REFUSED`. Invisible unless the fixture actually
violates the contract, which is why `regression_score_check.mojo` carries a
second label plane that does."""


def regression_key(
    sum_left: Int64,
    sum_total: Int64,
    n_left: Int,
    n_right: Int,
    row_count: Int,
    sabotage: Int32,
    mut out_num: Int64,
    mut out_den: Int64,
) -> Bool:
    """cuML's MSE gain as an exact rational in `Int64`. DEVIATION BLOCK 189.

        A   = sum_L * n_R - sum_R * n_L        num = (|A| >> j)^2
                                               den = n_L * n_R

    Returns whether the fixed-point precondition held. On `False` the pair is
    `(0, 0)` -- an INVALID key, which `compare_exact_key` ranks below every
    valid one -- and the caller publishes `SCORE_STATUS_REGRESSION_REFUSED`.

    NO `Int128` ANYWHERE, deliberately: this runs inside a kernel and a kernel
    containing a 128-bit multiply does not build (DEVIATION 167). Every
    intermediate is bounded under the precondition -- `|sum| <= 2^30` and
    `n <= 2^31` give `|sum_L * n_R| <= 2^61` and `|A| <= 2^62` -- which is why
    the precondition is tested BEFORE the multiply and not after it.

    `sabotage` selects the arms; `SCORE_SAB_NONE` is the shipping path.
    """
    var sum_right = sum_total - sum_left
    if sabotage != SCORE_SAB_REG_NO_SLOT_GUARD:
        var abs_left = sum_left if sum_left >= 0 else -sum_left
        var abs_right = sum_right if sum_right >= 0 else -sum_right
        var slot = Int64(1) << Int64(REGRESSION_SUM_BITS)
        if abs_left > slot or abs_right > slot:
            out_num = Int64(0)
            out_den = Int64(0)
            return False

    var nl = Int64(n_left)
    var nr = Int64(n_right)
    # `objectives.cuh:236-239` with the parent term subtracted and the node
    # constant `0.5/n` dropped; `A = n_L n_R (mean_L - mean_R)`.
    var a = sum_left * nr - sum_right * nl
    if sabotage == SCORE_SAB_REG_NO_CENTER:
        a = sum_left * nr
    if a < 0:
        a = -a
    var j = regression_key_shift(row_count)
    if sabotage == SCORE_SAB_REG_NO_SHIFT:
        j = 0
    # The magnitude is taken FIRST, so the truncation is toward zero and the
    # key is invariant under negating every label. DEVIATION BLOCK 191.
    var scaled = a >> Int64(j)
    out_num = scaled * scaled
    out_den = nl * nr
    if sabotage == SCORE_SAB_REG_DEN_ONE:
        out_den = Int64(1)
    return True




def draw_threshold_device(
    key: SplitKey, extent: FeatureRange, guarded: Bool = True
) -> Float32:
    """`draw_threshold`'s draw, written so that NO backend can re-round it.

    Same PCG stream, same span-scaled uniform, same `:653-654` guard. One
    thing is different and it is the whole of DEVIATION BLOCK 173: the rescale
    is an EXPLICIT `fma` instead of a multiply followed by an add.

    WHY. A multiply-then-add in Mojo source is contracted into an FMA by the
    GPU backend and is not contracted by the host compiler, so the same source
    produced two different thresholds -- measured, up to 8 ulps apart, in 9 of
    105 (node, feature) cells. Four source-level barriers were tried and all
    four failed on device (see 173). An explicit `fma` is not a barrier
    against an optimisation; it is a single IEEE-754 operation with one
    correctly-rounded result, so every backend that has FMA must produce the
    same bits and one that does not must emulate them. The value is fixed by
    the SOURCE rather than by what a compiler chose to do with it.

    AND IT IS WHAT THE UPSTREAM'S GPU PRODUCES. RAFT writes the rescale as one
    C++ expression (`rng_device.cuh:173-183`) and nvcc defaults to
    `--fmad=true`, so `custom_next` on a CUDA device IS this fma. The unfused
    host form comes from `tools/rng_oracle` being built with
    `-ffp-contract=off` (DEVIATION 142), which is a property of that oracle
    build and not of their kernel.

    THE OPEN ITEM, stated because it is not this sub-lane's to close: this
    function and `draw_threshold` disagree in about one draw in eight, so a
    tree grown on device and a tree grown by `host_splitter.mojo` can differ.
    Closing it means choosing ONE of the two for the whole lane -- most likely
    this one, which then re-pins `tools/rng_oracle` and amends DEVIATION 142.
    `score_kernel_check.mojo` MEASURES the disagreement per cell every run so
    the size of it is never a guess.
    """
    # DEVIATION 453 (IDENTITY_PATHS row 10 applied): every float in this
    # draw goes through `numerics.ftz`. Metal flushes denormal OPERANDS,
    # intermediates and results of arithmetic; CUDA's default honors them,
    # so a denormal-scale range (a raw-data property -- the range pass
    # publishes SELECTED bits, never arithmetic results) would give the two
    # vendors different threshold bits and a different `== max` guard
    # decision. Under IDENTICAL the flushes align every backend to the
    # measured Metal model; under FAST `ftz` is a comptime no-op and this
    # is bit for bit the code that shipped. `res` is in [0, 1 - 2^-24] and
    # never denormal, so it is not flushed.
    var gen = key.generator()
    var res = gen.next_float()
    var lo = ftz(extent.min_value)
    var hi = ftz(extent.max_value)
    var span = ftz(hi - lo)
    var threshold = ftz(res.fma(span, lo))
    if guarded and threshold == hi:
        return lo
    return threshold


@fieldwise_init
struct ScoredCandidate(Copyable, Movable):
    """One (node, feature) cell of the score pass, as one value.

    DEVIATION 174 unbundles this into eight parallel arrays for the device;
    this puts it back for a host caller and for a per-cell comparison, the way
    `feature_range_at` does for the range pass.
    """

    var status: Int32
    var threshold: Float32
    var n_left: Int32
    var n_total: Int32
    var acc_left: List[Int32]
    var acc_total: List[Int32]
    var gini_num: Int64
    var gini_den: Int64

    def matches(self, other: Self) -> Bool:
        """Exact equality, with the threshold on FLOAT BIT PATTERNS.

        No tolerance anywhere, and the reason is DEVIATION 171 plus DEVIATION
        135: every accumulated quantity is an integer, integer addition is
        associative and exact, so the device's block-strided, cross-block
        reduction and a sequential host loop are OBLIGED to produce identical
        values rather than close ones. The threshold is not accumulated at all
        -- it is the same pure function of the same key on both sides
        (DEVIATION 173).
        """
        if self.status != other.status:
            return False
        if self.threshold.to_bits() != other.threshold.to_bits():
            return False
        if self.n_left != other.n_left or self.n_total != other.n_total:
            return False
        if self.gini_num != other.gini_num or self.gini_den != other.gini_den:
            return False
        if len(self.acc_left) != len(other.acc_left):
            return False
        if len(self.acc_total) != len(other.acc_total):
            return False
        for k in range(len(self.acc_left)):
            if self.acc_left[k] != other.acc_left[k]:
                return False
        for k in range(len(self.acc_total)):
            if self.acc_total[k] != other.acc_total[k]:
                return False
        return True


def empty_scored_candidate(status: Int32, n_acc: Int) -> ScoredCandidate:
    """A cell nothing was accumulated into: `_empty_candidate`'s shape
    (`host_splitter.mojo:408-430`), which reports `threshold = 0.0` and zero
    accumulators for a candidate that never reached the draw."""
    return ScoredCandidate(
        status,
        Float32(0.0),
        Int32(0),
        Int32(0),
        List[Int32](length=n_acc, fill=Int32(0)),
        List[Int32](length=n_acc, fill=Int32(0)),
        Int64(0),
        Int64(0),
    )


def node_feature_score_host(
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    m: Int,
    range_start: Int,
    range_len: Int,
    col: Int,
    extent: FeatureRange,
    key: SplitKey,
    n_acc: Int,
    is_classification: Bool,
    min_samples_leaf: Int,
    device_draw: Bool = True,
) -> ScoredCandidate:
    """Steps 2, 3 and 4 of DEVIATION 137 for ONE (node, feature), sequentially.

    THE ORACLE the device kernels are checked against, per `PORTING_RULES.md`
    0b-ii, and written the way `node_feature_min_max` is: their branches in
    their order, no block structure, `row_ids` order.

    It is deliberately NOT a re-transcription of steps 2 and 3: it calls
    `node_feature_is_constant` and `draw_threshold`, the same functions the
    kernel calls, so the only body being compared is step 4's accumulation.
    Transcribing them twice would let the two copies drift and would check the
    copy rather than the kernel.

    THE BRANCHES, in `_splitter.pyx`'s order as `host_splitter.mojo` fixed it:

    1. `n_missing != 0` -> refuse (DEVIATION 136, `_refuse_missing`, which runs
       BEFORE the constant test there and therefore here);
    2. `:613-618` constant -> skip, nothing read;
    3. `:632-637` + `:653-654` the guarded keyed draw;
    4. `:656-662` ONE pass, counting and accumulating, moving nothing
       (DEVIATION 152); `value <= threshold` goes LEFT
       (`_partitioner.pyx:236`, `builder_kernels_impl.cuh:65-66`);
    5. `:664-666` `min_samples_leaf` -> a `continue`, not a redraw, and the
       accumulators are still reported;
    6. `:691` the score, as the exact rational of DEVIATION 144 for
       classification and as cuML's own MSE gain, `(|A| >> j)^2 / (n_L n_R)`,
       for regression (DEVIATION BLOCKS 189-193). Both go into the same two
       fields, whose `gini_` names are now a misnomer on the regression path
       and are left alone deliberately -- see DEVIATION BLOCK 193.

    A label outside `[0, n_acc)` is SKIPPED rather than accumulated, on both
    sides, because on the device it would be an out-of-bounds write into a
    private array. Validating labels is the host caller's job and
    `host_splitter.mojo` raises on it; this is the bound that keeps the kernel
    memory-safe when it does not.
    """
    if extent.n_missing != Int32(0):
        return empty_scored_candidate(SCORE_STATUS_MISSING_REFUSED, n_acc)
    if node_feature_is_constant(extent, Int32(range_len)):
        return empty_scored_candidate(SCORE_STATUS_CONSTANT, n_acc)

    # `device_draw=True` is `draw_threshold_device`, the EXPLICIT fma the
    # kernels take; `False` is `draw_threshold`, `host_splitter.mojo`'s form.
    # They disagree in about one draw in eight and that disagreement is
    # DEVIATION BLOCK 173's open item -- an oracle has to say WHICH draw it is
    # the oracle of, so this argument exists and has no silent default answer
    # hidden inside the body.
    var threshold: Float32
    if device_draw:
        threshold = draw_threshold_device(key, extent)
    else:
        threshold = draw_threshold(key, extent)

    var acc_left = List[Int32](length=n_acc, fill=Int32(0))
    var acc_total = List[Int32](length=n_acc, fill=Int32(0))
    var n_left = 0
    var n_total = 0
    var col_offset = col * m

    for p in range(range_start, range_start + range_len):
        var row = Int(row_ids[unsafe_offset=p])
        var v = data[unsafe_offset = col_offset + row]
        var lab = Int(labels_q[unsafe_offset=row])
        n_total += 1
        var goes_left = v <= threshold
        if is_classification:
            if lab >= 0 and lab < n_acc:
                acc_total[lab] += 1
                if goes_left:
                    acc_left[lab] += 1
        else:
            acc_total[0] += Int32(lab)
            if goes_left:
                acc_left[0] += Int32(lab)
        if goes_left:
            n_left += 1

    var n_right = n_total - n_left
    var status = SCORE_STATUS_SCORED
    if (
        n_left < min_samples_leaf
        or n_right < min_samples_leaf
        or n_left == 0
        or n_right == 0
    ):
        status = SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
    # DEVIATION 216's companion, mirrored from the finalize kernel: a PURE
    # classification node is a leaf (sklearn `_tree.pyx:240`), tested after
    # the min_samples rejection exactly as the kernel tests it.
    if status == SCORE_STATUS_SCORED and is_classification and n_total > 0:
        for kc in range(n_acc):
            if Int(acc_total[kc]) == n_total:
                status = SCORE_STATUS_PURE_NODE

    var num = Int64(0)
    var den = Int64(0)
    if status == SCORE_STATUS_SCORED and is_classification:
        # `objectives.mojo::ProxyImpurityExact`, DEVIATION 144:
        # `num = sq_L*nR + sq_R*nL`, `den = nL*nR`, no rounding anywhere.
        var sq_left = Int64(0)
        var sq_right = Int64(0)
        for k in range(n_acc):
            var lv = Int64(Int(acc_left[k]))
            var rv = Int64(Int(acc_total[k] - acc_left[k]))
            sq_left += lv * lv
            sq_right += rv * rv
        var nl = Int64(n_left)
        var nr = Int64(n_right)
        # DEVIATION 218: mirrored from the finalize kernel.
        var cs = Int64(classification_key_shift(range_len))
        num = (sq_left >> cs) * nr + (sq_right >> cs) * nl
        den = nl * nr
    elif status == SCORE_STATUS_SCORED:
        # DEVIATION BLOCKS 189-193: cuML's MSE gain as an exact rational, from
        # the SAME function the kernel calls -- not a second transcription.
        if not regression_key(
            Int64(Int(acc_left[0])),
            Int64(Int(acc_total[0])),
            n_left,
            n_right,
            range_len,
            SCORE_SAB_NONE,
            num,
            den,
        ):
            status = SCORE_STATUS_REGRESSION_REFUSED

    return ScoredCandidate(
        status,
        threshold,
        Int32(n_left),
        Int32(n_total),
        acc_left^,
        acc_total^,
        num,
        den,
    )


def scored_candidate_at(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    nid: Int,
    fslot: Int,
    n_sampled_cols: Int,
    n_acc: Int,
) -> ScoredCandidate:
    """Reassemble one cell of the kernels' output. DEVIATION 174 unbundles the
    struct for the device; this puts it back, so nothing downstream has to know
    the layout."""
    var slot = nid * n_sampled_cols + fslot
    var left = List[Int32]()
    var total = List[Int32]()
    for k in range(n_acc):
        left.append(out_acc_left[unsafe_offset = slot * n_acc + k])
        total.append(out_acc_total[unsafe_offset = slot * n_acc + k])
    return ScoredCandidate(
        out_status[unsafe_offset=slot],
        out_threshold[unsafe_offset=slot],
        out_n_left[unsafe_offset=slot],
        out_n_total[unsafe_offset=slot],
        left^,
        total^,
        out_gini_num[unsafe_offset=slot],
        out_gini_den[unsafe_offset=slot],
    )


from std.memory import stack_allocation


def node_feature_score_init_kernel(
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    n_cells: Int32,
    n_acc_cells: Int32,
):
    """Seed every output cell, before any accumulation.

    The counterpart of `initSplit` (`split.cuh:284-289`) and the sibling of
    `node_feature_range_init_kernel` above, and it is not optional for the same
    reason theirs is not: the accumulate kernel ADDS into these cells, so an
    unseeded cell is garbage plus a partial sum.

    The status seed is `SCORE_STATUS_UNVISITED`, which is not a legal outcome
    -- the finalize kernel overwrites it in every cell it reaches, so a cell
    still holding it is a REACH failure the check can name rather than a
    plausible-looking answer.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var i = idx
    while i < Int(n_cells):
        out_status[unsafe_offset=i] = SCORE_STATUS_UNVISITED
        out_threshold[unsafe_offset=i] = Float32(0.0)
        out_n_left[unsafe_offset=i] = Int32(0)
        out_n_total[unsafe_offset=i] = Int32(0)
        out_gini_num[unsafe_offset=i] = Int64(0)
        out_gini_den[unsafe_offset=i] = Int64(0)
        out_n_blocks[unsafe_offset=i] = Int32(0)
        i += stride
    var j = idx
    while j < Int(n_acc_cells):
        out_acc_left[unsafe_offset=j] = Int32(0)
        out_acc_total[unsafe_offset=j] = Int32(0)
        j += stride


def node_feature_score_kernel[
    TPB: Int, MAX_ACC: Int, CLASSIFICATION: Bool
](
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    out_n_blocks: MutPointer[Int32, MutAnyOrigin],
    in_min: MutPointer[Float32, MutAnyOrigin],
    in_max: MutPointer[Float32, MutAnyOrigin],
    in_n_missing: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    sabotage_in: Int32,
):
    """Steps 2, 3 and 4 of DEVIATION 137: skip, draw, and accumulate.

    DEVIATION 211: the tree id is PER NODE (`tree_ids[nid]`), not per launch,
    because one batch's nodes may belong to different trees -- the forest
    trainer merges every in-flight tree's frontier into one launch. The key
    a cell draws with is unchanged: `(seed, tree_ids[nid], node_id, col)`,
    the same pure function of tree position it always was, so batch
    composition cannot move a threshold.

    The control plane is `computeSplitKernel`'s, `builder_kernels_impl.cuh`
    `:236-291`, transcribed and shared line for line with
    `node_feature_range_kernel` above: the same `WorkloadInfo` flattening, the
    same `blockIdx.y == feature` grid, the same `colids` lookup, the same
    strided row loop. What is replaced is the body -- theirs bins each value
    with `lower_bound` and increments a shared histogram; ours compares it
    against ONE drawn threshold and increments a private accumulator.

    GRID. `grid_dim = (n_blocks_dimx, n_sampled_cols)`, `block_dim = TPB`,
    exactly the range kernel's and exactly theirs. `block_idx.x` indexes
    `workload_info`, NOT the node.

    THE THREE EARLY RETURNS ARE BLOCK-UNIFORM AND THAT IS LOAD-BEARING. Every
    thread of the block computes the refusal, the `n_acc` bound and the
    constant test from the same three published range values and the same two
    kernel arguments, so either all of them return or none does. A block
    collective that some threads reach and others do not is undefined, and the
    three block reductions below are collectives.

    THIS KERNEL READS `work_item.idx`, WHICH THE RANGE KERNEL NEVER DOES. It is
    the `node_id` component of the draw key (DEVIATION 130), so the tree
    position of the node -- not its index in this batch -- is what the
    threshold depends on. `score_kernel_check.mojo` gives the batch node ids
    that are not their batch indices, so the two cannot be confused.

    OUTPUT. Five of the arrays of DEVIATION BLOCK 174, all ADDED into with
    `Atomic.fetch_add` (DEVIATION BLOCK 171). The other three are the finalize
    kernel's. The caller must run `node_feature_score_init_kernel` first.
    """
    # `:238-244` -- their two whole-struct loads, one field at a time through
    # the pointer instead (DEVIATION 162: a whole-struct load in a kernel kills
    # the Metal compiler).
    var wb = Int(block_idx.x)
    var fslot = Int(block_idx.y)
    var nid = Int(workload_info[unsafe_offset=wb].nodeid)
    var offset_blockid = Int(workload_info[unsafe_offset=wb].offset_blockid)
    var num_blocks = Int(workload_info[unsafe_offset=wb].num_blocks)
    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var node_id = UInt32(Int(work_items[unsafe_offset=nid].idx))

    var m = Int(m_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var n_acc = Int(n_acc_in)
    var sabotage = sabotage_in

    # `:250-255` -- the feature this block tests. See the range kernel: the
    # `n_sampled_cols == N` arm of their `if` is the identity map, which
    # DEVIATION 156 records the sampler as MATERIALIZING, so `colids` is always
    # consulted and there is one arm rather than two.
    var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])
    var slot = nid * n_sampled_cols + fslot

    # DEVIATION 172: the private array is comptime-sized, so a request past the
    # bound publishes NOTHING rather than writing out of bounds. The cell keeps
    # `SCORE_STATUS_UNVISITED` and the caller sees it.
    if n_acc > MAX_ACC or n_acc < 1:
        return

    # -------- step 2: the constant test, and DEVIATION 136's refusal --------
    # The three range values are read one at a time and only THEN assembled
    # into a local `FeatureRange`; the struct is never loaded from memory.
    var extent = FeatureRange(
        in_min[unsafe_offset=slot],
        in_max[unsafe_offset=slot],
        in_n_missing[unsafe_offset=slot],
    )
    # `host_splitter.mojo:552` refuses BEFORE testing constancy, so this does
    # too. It is also what discharges DEVIATION 163's caller obligation: the
    # only other way to hold `min > max` is a node with no rows, and the
    # constant test's first arm catches that as `0 == 0`.
    if extent.n_missing != Int32(0):
        return
    var constant: Bool
    if sabotage == SCORE_SAB_CONSTANT_STRICT:
        # `<` instead of `<=`, `_splitter.pyx:616-617`.
        constant = Int32(range_len) == extent.n_missing or (
            extent.max_value < extent.min_value + FEATURE_THRESHOLD
            and extent.n_missing == 0
        )
    else:
        constant = node_feature_is_constant(extent, Int32(range_len))
    if constant:
        return

    # -------- step 3: ONE keyed threshold, drawn in every thread -------------
    # The key is `(seed, tree_id, node_id, col)` and depends on nothing
    # thread-local, so every thread of every block serving this cell computes
    # the same bits and no broadcast is needed (DEVIATION BLOCK 173).
    #
    # THE DRAW GOES THROUGH `draw_threshold_device` AND NOT `draw_threshold`:
    # same PCG stream, same `:653-654` guard, but an EXPLICIT `fma` instead of
    # a multiply the GPU backend contracts into one anyway -- which is the one
    # place host and device could not share a function. DEVIATION BLOCK 173.
    var key = key_for(
        seed,
        tree_ids[unsafe_offset=nid].cast[DType.uint32](),
        node_id,
        UInt32(col),
    )
    var threshold = draw_threshold_device(
        key, extent, sabotage != SCORE_SAB_NO_MAX_GUARD
    )

    # -------- step 4: ONE pass over the node's rows -------------------------
    # DEVIATION 172: private, not shared. Two arrays because the right child is
    # recovered as `total - left` (`objectives.cuh:72-73`, DEVIATION 143) and
    # the totals therefore have to be accumulated over EVERY row, not only the
    # left-going ones.
    var priv_left = stack_allocation[MAX_ACC, Scalar[DType.int32]]()
    var priv_total = stack_allocation[MAX_ACC, Scalar[DType.int32]]()
    for k in range(MAX_ACC):
        priv_left[unsafe_offset=k] = Int32(0)
        priv_total[unsafe_offset=k] = Int32(0)

    var col_offset = col * m
    var end = range_start + range_len
    var stride = TPB * num_blocks
    var tid = Int(thread_idx.x) + offset_blockid * TPB
    var n_left = Int32(0)
    var n_seen = Int32(0)
    # Only SCORE_SAB_FLOAT_ACCUM reads these; they are the accumulator
    # DEVIATION 135 rejected, kept here so the ruling can be falsified.
    var f_left = Float32(0.0)
    var f_total = Float32(0.0)

    var i = range_start + tid
    while i < end:
        var row = Int(row_ids[unsafe_offset=i])
        if sabotage == SCORE_SAB_NO_ROW_IDS:
            row = i
        # `:279, 284` -- COLUMN MAJOR (`dataset.h:24`).
        var v = data[unsafe_offset = col_offset + row]
        var lab = Int(labels_q[unsafe_offset=row])
        n_seen += 1

        # `_partitioner.pyx:236` and `builder_kernels_impl.cuh:65-66`: a row
        # EQUAL to the threshold goes LEFT.
        var goes_left: Bool
        if sabotage == SCORE_SAB_SIDE_INVERTED:
            goes_left = v > threshold
        elif sabotage == SCORE_SAB_STRICT_LESS:
            goes_left = v < threshold
        else:
            goes_left = v <= threshold

        var counts_total = True
        if sabotage == SCORE_SAB_TOTAL_IS_LEFT:
            counts_total = goes_left

        comptime if CLASSIFICATION:
            # The label is the class id, cast on the HOST (DEVIATION 174). The
            # bound is what keeps the private write in range when a caller has
            # not validated its labels.
            if lab >= 0 and lab < n_acc:
                if counts_total:
                    priv_total[unsafe_offset=lab] += Int32(1)
                if goes_left:
                    priv_left[unsafe_offset=lab] += Int32(1)
        else:
            # DEVIATION 135: the label arrives ALREADY quantized, and the
            # accumulation is an integer add.
            var q = Int32(lab)
            if sabotage == SCORE_SAB_SCALE_X2:
                q = Int32(2 * lab)
            if sabotage == SCORE_SAB_FLOAT_ACCUM:
                if counts_total:
                    f_total += Float32(lab)
                if goes_left:
                    f_left += Float32(lab)
            else:
                if counts_total:
                    priv_total[unsafe_offset=0] += q
                if goes_left:
                    priv_left[unsafe_offset=0] += q

        if goes_left:
            n_left += 1
        i += stride

    # Their `__syncthreads()` at `:293`. `max.gpu.primitives.block.sum` is the
    # Mojo spelling of a CUB block reduce (PORTING_RULES 0b-i explicitly
    # exempts the block primitives), and every thread must reach every one of
    # them -- which is why none of the returns above is per-thread.
    var blk_n_left = block_sum[block_size=TPB](n_left)
    barrier()
    var blk_n_seen = block_sum[block_size=TPB](n_seen)
    barrier()

    # `:295-313`, and here it IS their `AtomicAdd` -- DEVIATION BLOCK 171.
    # Thread 0 publishes, which is also their shape.
    var publishes = not (
        sabotage == SCORE_SAB_BLOCK0_ONLY and offset_blockid != 0
    )
    if Int(thread_idx.x) == 0 and publishes:
        _ = Atomic.fetch_add(out_n_left.unsafe_offset(slot), blk_n_left)
        _ = Atomic.fetch_add(out_n_total.unsafe_offset(slot), blk_n_seen)
        _ = Atomic.fetch_add(out_n_blocks.unsafe_offset(slot), Int32(1))

    comptime if not CLASSIFICATION:
        # The accumulator DEVIATION 135 ruled out, in full: Float32 from the
        # per-thread partial all the way through the block reduction, truncated
        # once at the publish. Reducing in `Int32` after a per-thread float
        # partial would NOT be that -- at this fixture's magnitudes a
        # per-thread partial of a handful of rows is exact, so a sabotage that
        # truncates there is invisible and says nothing. The branch is on a
        # kernel argument, so it is block-uniform and the collectives inside it
        # are reached by every thread or by none.
        if sabotage == SCORE_SAB_FLOAT_ACCUM:
            var fl = block_sum[block_size=TPB](f_left)
            barrier()
            var ft = block_sum[block_size=TPB](f_total)
            barrier()
            if Int(thread_idx.x) == 0 and publishes:
                _ = Atomic.fetch_add(
                    out_acc_left.unsafe_offset(slot * n_acc), Int32(fl)
                )
                _ = Atomic.fetch_add(
                    out_acc_total.unsafe_offset(slot * n_acc), Int32(ft)
                )
            return

    for k in range(n_acc):
        var bl = block_sum[block_size=TPB](priv_left[unsafe_offset=k])
        barrier()
        var bt = block_sum[block_size=TPB](priv_total[unsafe_offset=k])
        barrier()
        if Int(thread_idx.x) == 0 and publishes:
            _ = Atomic.fetch_add(
                out_acc_left.unsafe_offset(slot * n_acc + k), bl
            )
            _ = Atomic.fetch_add(
                out_acc_total.unsafe_offset(slot * n_acc + k), bt
            )


def node_feature_score_finalize_kernel[
    MAX_ACC: Int, CLASSIFICATION: Bool
](
    out_status: MutPointer[Int32, MutAnyOrigin],
    out_threshold: MutPointer[Float32, MutAnyOrigin],
    out_gini_num: MutPointer[Int64, MutAnyOrigin],
    out_gini_den: MutPointer[Int64, MutAnyOrigin],
    out_n_left: MutPointer[Int32, MutAnyOrigin],
    out_n_total: MutPointer[Int32, MutAnyOrigin],
    out_acc_left: MutPointer[Int32, MutAnyOrigin],
    out_acc_total: MutPointer[Int32, MutAnyOrigin],
    in_min: MutPointer[Float32, MutAnyOrigin],
    in_max: MutPointer[Float32, MutAnyOrigin],
    in_n_missing: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    n_cells_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    min_samples_leaf_in: Int32,
    sabotage_in: Int32,
):
    """Publish each cell's status, threshold and score. One thread per cell.

    DEVIATION 211: `tree_ids[nid]` replaces the per-launch tree id, exactly
    as in the accumulate kernel above; the REDRAW below must key with the
    same tree the first draw used, and both now read it per node.

    This is the half of `computeSplitKernel` that their LAST BLOCK does
    (`:316-328`) after `signalDone` elects it, moved into a second launch
    because that election is not expressible -- DEVIATION BLOCK 170. It reads
    no dataset row: the row loop happened once, in the accumulate kernel.

    It REDRAWS the threshold rather than reading one the first launch stored,
    because the draw is a pure function of `(seed, tree_id, node_id, col)` and
    recomputing something that cannot vary is a stronger invariant than
    depending on a write.

    THE ORDER OF THE BRANCHES IS `host_splitter.mojo`'S, WHICH IS
    `_splitter.pyx`'S: refuse missing, then constant, then draw, then
    `min_samples_leaf`, then score. A `continue` in their loop is a status
    here (DEVIATION 174).

    BOTH OBJECTIVES PUBLISH A RATIONAL KEY. Classification's is sklearn's
    proxy (DEVIATION 144); regression's is cuML's own MSE gain with its
    node-constant factor dropped, `(|A| >> j)^2 / (n_L n_R)` (DEVIATION BLOCKS
    189-193). Regression is the ONE branch that can turn a scored cell back
    into a refusal, and only when the caller's labels left the fixed-point
    slot -- `SCORE_STATUS_REGRESSION_REFUSED`, DEVIATION BLOCK 193.
    """
    var n_cells = Int(n_cells_in)
    var n_sampled_cols = Int(n_sampled_cols_in)
    var n_acc = Int(n_acc_in)
    var min_samples_leaf = Int(min_samples_leaf_in)
    var sabotage = sabotage_in
    if n_acc > MAX_ACC or n_acc < 1:
        return

    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var slot = idx
    while slot < n_cells:
        var nid = slot // n_sampled_cols
        var fslot = slot % n_sampled_cols
        var range_len = Int(work_items[unsafe_offset=nid].instances.count)
        var node_id = UInt32(Int(work_items[unsafe_offset=nid].idx))
        var col = Int(colids[unsafe_offset = nid * n_sampled_cols + fslot])

        var extent = FeatureRange(
            in_min[unsafe_offset=slot],
            in_max[unsafe_offset=slot],
            in_n_missing[unsafe_offset=slot],
        )
        if extent.n_missing != Int32(0):
            out_status[unsafe_offset=slot] = SCORE_STATUS_MISSING_REFUSED
            slot += stride
            continue
        var constant: Bool
        if sabotage == SCORE_SAB_CONSTANT_STRICT:
            constant = Int32(range_len) == extent.n_missing or (
                extent.max_value < extent.min_value + FEATURE_THRESHOLD
                and extent.n_missing == 0
            )
        else:
            constant = node_feature_is_constant(extent, Int32(range_len))
        if constant:
            out_status[unsafe_offset=slot] = SCORE_STATUS_CONSTANT
            slot += stride
            continue

        var key = key_for(
            seed,
            tree_ids[unsafe_offset=nid].cast[DType.uint32](),
            node_id,
            UInt32(col),
        )
        var threshold = draw_threshold_device(
            key, extent, sabotage != SCORE_SAB_NO_MAX_GUARD
        )
        out_threshold[unsafe_offset=slot] = threshold

        var n_left = Int(out_n_left[unsafe_offset=slot])
        var n_total = Int(out_n_total[unsafe_offset=slot])
        var n_right = n_total - n_left

        # `_splitter.pyx:664-666`, plus the empty-child guard cuML's float form
        # does not have (`objectives.cuh:57` divides by `nLeft`) and
        # `ProxyImpurityExact` does.
        if (
            n_left < min_samples_leaf
            or n_right < min_samples_leaf
            or n_left == 0
            or n_right == 0
        ):
            out_status[unsafe_offset=slot] = (
                SCORE_STATUS_REJECTED_MIN_SAMPLES_LEAF
            )
            slot += stride
            continue

        # DEVIATION 216's companion: a PURE node (one class holds every row)
        # is a LEAF before any candidate can win -- sklearn `_tree.pyx:240`.
        # Tested here because this is where the node's class totals live;
        # under the old `<=` gate purity fell out of zero-gain rejection and
        # needed no test.
        comptime if CLASSIFICATION:
            var pure = False
            for kc in range(n_acc):
                if (
                    Int(out_acc_total[unsafe_offset = slot * n_acc + kc])
                    == n_total
                ):
                    pure = True
            if pure and n_total > 0:
                out_status[unsafe_offset=slot] = SCORE_STATUS_PURE_NODE
                slot += stride
                continue

        out_status[unsafe_offset=slot] = SCORE_STATUS_SCORED
        comptime if CLASSIFICATION:
            # DEVIATION 144 / BLOCK 175: sklearn's proxy as an exact rational,
            # `num = sq_L*nR + sq_R*nL`, `den = nL*nR`, in `Int64`.
            var sq_left = Int64(0)
            var sq_right = Int64(0)
            for k in range(n_acc):
                var lv = Int64(
                    Int(out_acc_left[unsafe_offset = slot * n_acc + k])
                )
                var rv = Int64(
                    Int(
                        out_acc_total[unsafe_offset = slot * n_acc + k]
                        - out_acc_left[unsafe_offset = slot * n_acc + k]
                    )
                )
                sq_left += lv * lv
                sq_right += rv * rv
            var nl = Int64(n_left)
            var nr = Int64(n_right)
            # DEVIATION 218: the node-uniform shift that lifts the 2^21 row
            # cap; s == 0 there and below, so the old regime is unchanged.
            var cs = Int64(classification_key_shift(range_len))
            out_gini_num[unsafe_offset=slot] = (
                (sq_left >> cs) * nr + (sq_right >> cs) * nl
            )
            out_gini_den[unsafe_offset=slot] = nl * nr
        else:
            # DEVIATION BLOCKS 189-193: cuML's own MSE gain
            # (`objectives.cuh:225-244`) as an exact rational, with its
            # node-constant factor dropped. This USED to publish nothing --
            # every regression candidate then carried an invalid key, every
            # comparison in `split_reduce_kernel` tied, and the winner fell to
            # `Split.update`'s `colid` arm, i.e. was decided by feature index.
            var key_num = Int64(0)
            var key_den = Int64(0)
            var key_ok = regression_key(
                Int64(Int(out_acc_left[unsafe_offset = slot * n_acc])),
                Int64(Int(out_acc_total[unsafe_offset = slot * n_acc])),
                n_left,
                n_right,
                range_len,
                sabotage,
                key_num,
                key_den,
            )
            if not key_ok:
                out_status[unsafe_offset=slot] = (
                    SCORE_STATUS_REGRESSION_REFUSED
                )
            out_gini_num[unsafe_offset=slot] = key_num
            out_gini_den[unsafe_offset=slot] = key_den
        slot += stride


# ============================================================================
# THE PARTITION AND THE LEAF PASS, ON THE DEVICE.
#
#     Verified by:  extratrees/original/partition_leaf_kernel_check.mojo
#
# These are the last two steps of a tree build that had no device
# implementation. `partition_samples` above and
# `builder.mojo::set_leaf_predictions_classification` stay exactly where they
# are and neither is replaced: they are the ORACLES these kernels are checked
# against, per `PORTING_RULES.md` 0b-ii.
#
#     ==================================================================
#     DEVIATION BLOCK 176 -- `partitionSamples` on the device:
#     `cub::BlockScan` becomes TWO block collectives, their `smem`
#     becomes one carved shared array, and their UNBOUNDED `while` gets a
#     DERIVED bound because a kernel cannot raise
#
#     THEIRS (`builder_kernels_impl.cuh:43-88`). Per iteration:
#
#         BlockScanT(temp1).ExclusiveSum(lflag, lidx, llen);
#         BlockScanT(temp2).ExclusiveSum(rflag, ridx, rlen);
#
#     -- ONE call per side that returns BOTH the per-thread exclusive
#     prefix AND the block aggregate, out of two separate
#     `TempStorage`s. The compaction buffers are carved by hand out of the
#     dynamic `extern __shared__ char smem[]` their launcher sizes at
#     `sizeof(IdxT) * TPB * 2` (`:39-41`, `:52-54`):
#     `lcomp = smem`, `rcomp = smem + sizeof(IdxT) * TPB`. The `while`
#     has no iteration bound.
#
#     OURS. Three differences, none of which can change an answer.
#
#     1. TWO COLLECTIVES PER SIDE INSTEAD OF ONE.
#        `max.gpu.primitives.block.prefix_sum[exclusive=True]` returns the
#        per-thread prefix and nothing else, and
#        `max.gpu.primitives.block.sum` returns the aggregate to EVERY
#        thread (`broadcast=True`, its default). Mojo 1.0 has no single
#        primitive returning both. The aggregate has to be block-uniform
#        and not merely thread-0's, because `llen` and `rlen` are what
#        drive `loffset`/`roffset` and therefore the LOOP CONDITION -- a
#        thread that disagreed about `llen` would leave the loop at a
#        different iteration from its neighbours, and every collective
#        inside the loop is undefined the moment that happens. This is
#        the vendor library, called rather than hand-written
#        (`VENDOR_LIBRARIES.md`); `PORTING_RULES.md` 0b-i exempts the
#        block primitives from the transcription rule for exactly this.
#
#     2. THE SHARED CARVE IS ONE ARRAY OF `2 * TPB`, NOT TWO OF `TPB`.
#        `lcomp` is `[0, TPB)` and `rcomp` is `[TPB, 2*TPB)`, which is
#        their pointer arithmetic written as an offset. It is ONE
#        `stack_allocation` on purpose: two allocations with identical
#        comptime parameters are two expressions the compiler is free to
#        treat as one, and a silently aliased `lcomp`/`rcomp` would
#        produce a partition that is wrong only when both sides have
#        misfits in the same iteration. Their own code carves one blob;
#        this is that, not an improvement on it.
#
#        It is also STATIC where theirs is dynamic. `stack_allocation`'s
#        slot count is a comptime expression in Mojo 1.0 (DEVIATION 172
#        hit the same wall from the other side), and `TPB` is already a
#        comptime parameter of this kernel, so nothing is lost: their
#        `smem_size` is a pure function of `TPB` too.
#
#     3. THE `while` CARRIES A DERIVED BOUND AND A SENTINEL. DEVIATION
#        159 names this obligation and hands it forward in as many words:
#        "an `Error` cannot be raised from a kernel, so the device
#        version needs the bound expressed some other way, or dropped
#        with the argument that the dispatch guarantees termination
#        written down." Here is the argument AND the bound.
#
#        THE ARGUMENT. `minlen = min(llen, rlen)`, so at least one of
#        `llen == minlen` and `rlen == minlen` is true every iteration,
#        so at least one of `loffset` and `roffset` advances by `TPB`
#        every iteration. `loffset` can advance at most
#        `ceildiv(n_left, TPB) + 1` times before `loffset >= part` ends
#        the loop, and `roffset` at most
#        `ceildiv(range_len - n_left, TPB) + 1`. The sum is the bound.
#
#        THE SENTINEL. If the bound is ever reached the kernel STOPS and
#        publishes `PARTITION_OVERRUN` in its iteration report, so the
#        failure is a value the host reads rather than a hang. A hang is
#        the one failure mode a check cannot report, which is DEVIATION
#        159's argument for the host bound and is not weaker on device.
#        `partition_leaf_kernel_check.mojo` asserts no cell holds it.
#
#     MEASURED, 2026-08-21, Apple GPU, `partition_leaf_kernel_check.mojo`.
#     Across two guard settings and block widths 32, 64 and 128, all 6144
#     `row_ids` slot comparisons against the host transcription agree
#     EXACTLY -- the device reproduces the permutation, not merely an
#     equivalent partition. `PARTITION_OVERRUN` was published by no item
#     in any arm, and every published iteration count was at or below
#     `partition_iteration_bound`.
#
#     AND ONE THING THE MEASUREMENT MADE STRONGER THAN THIS BLOCK FIRST
#     CLAIMED. The check's first draft printed that the output ORDER is
#     block-width dependent; it measured 0 of 1024 slots differing between
#     `TPB = 32` and `TPB = 128`, and the sentence was deleted rather than
#     annotated. Consumption is strictly in slot order on both sides -- the
#     compaction packs flagged misfits by ascending slot, the first
#     `minlen` are taken, and a side that was not fully consumed KEEPS its
#     flags -- so globally the k-th left misfit is always swapped with the
#     k-th right misfit and `TPB` decides only how many pairs are done per
#     iteration. Their header promises determinism GIVEN the width; the
#     permutation is in fact independent of it. That is reported, not
#     enforced: it is a fact about their algorithm, not a contract.
#
#     WHAT IS NOT CHANGED, and is the part a reader should look for and
#     not find. The comparison directions (`> quesval` is a LEFT misfit,
#     `<= quesval` is a RIGHT misfit, `:65-66`), the `llen == minlen`
#     guard that carries leftover flags into the next iteration
#     (`:65-66`), the exclusive-prefix compaction (`:75-77`), the
#     flag-clearing and the single-sided advance (`:79-83`), and the
#     PAIRED swap (`:84-89`) are transcribed unchanged from their file,
#     in their order, and `partition_samples` above is the host form of
#     the same lines.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 177 -- NOT a deviation: the partition stays ONE
#     BLOCK PER NODE, and the cost of that is recorded as an ITERATION
#     COUNT rather than argued about
#
#     THEIRS. Their own header note at `:39-41` says
#     "this should be called by only one block from all participating
#     blocks", and `nodeSplitKernel` (`:89-107`) is launched with one
#     block per WORK ITEM (`launchNodeSplitKernel`, `:109-134`,
#     `<<<work_items_size, TPB>>>`), each block partitioning its own
#     node. A node of a million rows is therefore served by ONE block
#     looping.
#
#     OURS. The same. `grid_dim = n_work_items`, `block_dim = TPB`, one
#     block per node, and the `while` loop is the same serial walk.
#
#     WHY IT IS NOT "IMPROVED" INTO A MULTI-BLOCK SCHEME, and this is a
#     ruling and not an oversight. The algorithm is a two-cursor
#     exchange: `loffset` and `roffset` are BLOCK-UNIFORM state that
#     every iteration reads and writes, and the flags a side keeps when
#     it did not advance are carried in the SAME threads' registers into
#     the next iteration. Splitting it across blocks needs a grid-wide
#     barrier per iteration, which is `__threadfence` plus a done-counter
#     -- the construction DEVIATIONS 161 and 170 have already measured as
#     inexpressible in Mojo 1.0 (`threadfence` comptime-asserts
#     NVIDIA-only). A different partition algorithm could be
#     multi-block; that is a rewrite, not a port, and this file's job is
#     the port.
#
#     THE COST, RECORDED AS ARITHMETIC AND AS A MEASUREMENT, NOT AS A
#     TIME. The kernel publishes `out_n_iters[b]`, the number of `while`
#     iterations its block actually ran -- the device's own report, in
#     the same spirit as `out_n_merges` (DEVIATION 162) and
#     `out_n_blocks` (DEVIATION 174), written unconditionally by the
#     shipping kernel and not behind a flag. The bound above says that
#     count is at most `ceildiv(n_left, TPB) + ceildiv(n_right, TPB) + 2`,
#     i.e. it grows LINEARLY in the node's row count at fixed `TPB` while
#     the rest of this lane's device work grows with `n / (TPB *
#     n_blocks)`. That is the shape of the cost. No time is attached to
#     it and none will be until the perf round; if it ever matters, the
#     number to look at is already being published every launch.
#
#     MEASURED, on the check's 11-item batch: at `TPB = 32` six items ran
#     more than one iteration, at `TPB = 64` four did, and at
#     `TPB = 128` two did -- i.e. the serial iteration count falls as the
#     block widens, which is what a one-block-per-node walk does and is
#     the number that would grow on a real node. The largest item there
#     is 300 rows, so this is the shape and not the magnitude.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 178 -- the leaf histogram is a comptime-bounded
#     PRIVATE array plus one block reduction per output, not their
#     dynamic shared `BinT[num_outputs]`
#
#     THEIRS (`builder_kernels_impl.cuh:391-417`). `extern __shared__
#     char shared_memory[]`, reinterpreted as `BinT* histogram` and sized
#     by the host at `sizeof(BinT) * dataset.num_outputs`
#     (`builder.cuh:581`). Every thread of the block increments the
#     SHARED histogram directly with
#     `BinT::IncrementHistogram(histogram, 1, 0, label)` -- note
#     `n_bins = 1` and `bin = 0`, so it is a plain per-class counter and
#     not a histogram over thresholds.
#
#     OURS. One `stack_allocation[MAX_OUT, Int32]` PER THREAD in the
#     default (private) address space, and one
#     `max.gpu.primitives.block.sum` per output to combine them across
#     the block. `MAX_OUT` is a comptime parameter; a launch whose
#     `num_outputs` exceeds it publishes NOTHING, so the caller sees the
#     cell's `LEAF_VISIT_NONE` rather than a corrupted leaf.
#
#     WHY. The same wall DEVIATION 172 hit in the score pass:
#     `stack_allocation`'s slot count is a comptime expression in Mojo
#     1.0, so a runtime `num_outputs` cannot size a shared blob. And the
#     same mitigation applies -- their shared array here is
#     `num_outputs` wide, not `n_bins * num_outputs`, so a comptime bound
#     of 16 covers the configurations this lane supports rather than
#     truncating them.
#
#     AND WHY PRIVATE RATHER THAN SHARED. With one bin there is nothing
#     for the threads to share: each thread's contribution is a private
#     running count that the block reduction combines once. Their shared
#     histogram exists because `IncrementHistogram` is a shared-memory
#     atomic that several threads hit at the same class; nothing needs to
#     be shared when the combine is a collective.
#
#     WHY IT CANNOT CHANGE AN ANSWER. Every accumulated quantity is an
#     INTEGER -- a class count for Gini, a fixed-point label sum for MSE
#     (DEVIATION 179) -- and integer addition is associative and exact,
#     so the block-strided private accumulation and a sequential host
#     loop are OBLIGED to produce identical values. That is DEVIATION
#     171's argument, reused where it applies again, and it is why the
#     check asserts on bit patterns with no tolerance.
#
#     PRICE. `MAX_OUT * 4` bytes of private memory per thread (64 bytes
#     at `MAX_OUT = 16`), which is memory and not registers, and
#     `num_outputs` block reductions per block instead of one shared
#     increment per row.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 179 -- the REGRESSION leaf accumulates in FIXED
#     POINT, so `builder.mojo`'s Float64 transcription is NOT this
#     kernel's oracle, and the gap between them is MEASURED per leaf
#
#     THEIRS. `MSEObjectiveFunction::SetLeafVector`
#     (`objectives.cuh:259-264`) is `out[i] = shist[i].label_sum /
#     shist[i].count`, and `label_sum` is an `AggregateBin<DataT>` field
#     accumulated on device in the label's own float type.
#
#     `builder.mojo`'s HOST transcription
#     (`set_leaf_predictions_regression`) accumulates in
#     `AggregateBin[DType.float64]`, and its own docstring says why and
#     hands the question forward: "the accumulator is
#     `AggregateBin[DType.float64]` here BECAUSE THIS IS THE HOST ORACLE
#     and DEVIATION 135 -- what the device accumulates in -- is open. The
#     device form must not silently inherit this choice." This block
#     closes that.
#
#     OURS. The labels arrive ALREADY QUANTIZED as `Int32`
#     (`labels_q`), exactly as they do in the score pass (DEVIATION 174,
#     part 1), the block accumulates an exact integer sum, and the leaf
#     value is formed once at the publish as
#
#         Float32(sum_q) / Float32(count) * inv_scale
#
#     `inv_scale` being `1 / scale` for the scale DEVIATION 135 chooses
#     ONCE on the host from the whole dataset. Classification passes
#     `inv_scale = 1.0` and never touches this arm.
#
#     WHY, in three parts.
#
#     1. THERE IS NO `float64` ON DEVICE in this toolchain, so the host
#        transcription's accumulator cannot be reproduced at all. That
#        is not a preference; the type does not exist on the target.
#     2. A `Float32` ACCUMULATOR WOULD NOT BE CHECKABLE. Float addition
#        is not associative, and a block reduction regroups the sum, so
#        the device answer would depend on the block width and on the
#        arrival order and no sequential host loop could be obliged to
#        match it. DEVIATION 135 ruled this out for the score pass on
#        exactly this argument and the leaf pass is the same shape.
#     3. IT IS THE SAME ARRAY THE SCORE PASS ALREADY TAKES. A build
#        whose split accumulation is fixed point and whose leaf
#        accumulation is float would carry two different notions of a
#        label through one tree.
#
#     WHAT THIS COSTS, AND IT IS NOT ZERO. The device's regression leaf
#     value is NOT bit-identical to
#     `set_leaf_predictions_regression`'s, and cannot be. Two things are
#     done about that rather than one:
#
#     * `leaf_values_host` below is the SEQUENTIAL form of exactly what
#       the kernel computes -- same quantized labels, same integer sum,
#       same final expression -- and it is what the check compares
#       against on BIT PATTERNS with no tolerance. The kernel is
#       therefore fully checked; it is just checked against the oracle
#       it is the oracle of.
#     * `partition_leaf_kernel_check.mojo` ALSO computes
#       `set_leaf_predictions_regression`'s Float64 answer for the same
#       fixture and MEASURES the largest disagreement, per leaf, every
#       run. The size of the quantization error is never a guess, and if
#       a later change to `choose_scale` widens it, the number moves.
#
#     MEASURED, 2026-08-21, Apple GPU. All 15 leaf slots are bit-identical
#     to `leaf_values_host`. Against `set_leaf_predictions_regression`'s
#     `Float64` answer, over 7 non-empty leaves, 0 agree bit for bit and
#     the largest disagreement is 7.63e-06 -- half of the `1 / 65536`
#     scale `choose_scale` picked for that fixture, i.e. exactly the
#     quantization step and nothing else.
#
#     AND THE FIRST VERSION OF THAT MEASUREMENT SAID NOTHING, WHICH IS
#     RECORDED BECAUSE IT IS THE KIND OF NUMBER THAT LOOKS LIKE A RESULT.
#     The check's regression labels were `hash / 64.0 - 64.0`, every one
#     of which is an exact multiple of `1 / 65536`, so the quantization
#     was LOSSLESS and the measurement came back "0.0 disagreement, 7 of 7
#     bit-identical" -- a fixture that could not produce a rounding
#     reporting that there was none. The divisor is now 97.
#
#     THE OPEN ITEM, stated because it is not this sub-lane's to close,
#     and it is the same shape as DEVIATION 173's: the lane has to pick
#     ONE regression leaf value. The recommendation is this one, because
#     it is the only one a GPU can produce and it is consistent with
#     DEVIATION 135; taking it means `set_leaf_predictions_regression`
#     becomes a REPORTING form rather than the authority, which is an
#     edit to `builder.mojo` and therefore another session's file.
#
#     THE `Int32` BOUND. The block's label sum is an `Int32`, so a leaf
#     whose quantized labels sum past `2^31` wraps. `fixed_point.mojo`'s
#     `choose_scale` bounds a single quantized label by `2^SLOT_BITS`
#     and sizes the accumulator from the ROW COUNT, which is the same
#     exposure the score pass already carries in `out_acc_total`
#     (DEVIATION 174) and is bounded by the same argument. It is stated
#     here rather than left implicit because this kernel does not see
#     the scale that made it true.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 180 -- one launch over ALL nodes instead of their
#     100,000-node batching, and both kernels REPORT the path they took
#
#     THEIRS. `SetLeafPredictions` (`builder.cuh:556-599`) loops over the
#     tree in batches of `min(100000, sparsetree.size())` nodes, copying
#     that slice of the node array and of the instance ranges to the
#     device, `cudaMemsetAsync`ing the leaf slice to ZERO, launching
#     `leafKernel` with `gridDim.x = batch_size`, and copying the slice
#     back. The comment says why: "do this in batch to reduce peak memory
#     usage in extreme cases".
#
#     OURS. `leaf_kernel` is launched once with `grid_dim = n_nodes` over
#     the whole tree.
#
#     WHY. Their batching is a HOST-SIDE memory-budget policy, not a step
#     of the algorithm: the kernel is identical, the node index is
#     `blockIdx.x` plus a batch offset they fold into the pointers, and a
#     batched caller and an unbatched one produce the same bytes. The
#     three device arrays it exists to cap are sized by the NODE COUNT,
#     not by the row count. Batching is one loop in the CALLER and
#     `builder.mojo` is another session's file this round; the kernel
#     signature below is already batch-ready -- pass the sliced pointers
#     and a smaller grid and it batches, with no change here.
#
#     AND THE TWO REPORTS, which have no cuML counterpart and are ours.
#     `out_n_iters` / `out_n_swaps` for the partition and `out_visit`
#     for the leaf pass are the device's own statement of which path it
#     took, written unconditionally by the SHIPPING kernel rather than
#     behind a flag, for the reason DEVIATION 162 gives: a check that
#     runs a different binary from the one that ships proves nothing
#     about the one that ships. They answer questions the OUTPUT cannot:
#
#     * an internal node's leaf slot is zero, and so is a node the
#       kernel never reached -- `out_visit` separates
#       `LEAF_VISIT_INTERNAL` from `LEAF_VISIT_NONE`;
#     * a node whose rows were already correctly ordered has an
#       UNCHANGED `row_ids`, and so does a node the partition skipped
#       and a node the partition never ran on -- `out_n_iters`
#       separates all three, and `out_n_swaps` says whether anything
#       actually moved.
#
#     THE CALLER OBLIGATIONS THIS CREATES, stated once here and repeated
#     in each kernel's docstring, because a sentinel nobody wrote is a
#     sentinel nobody can trust:
#     `out_visit` must be ZEROED and `out_n_iters` must be seeded with
#     `PARTITION_UNVISITED` before the launch. `out_leaves` must be
#     zeroed too, and that one is THEIRS -- `cudaMemsetAsync` at
#     `builder.cuh:582` -- because an internal node's slot is never
#     written by anything and the zero IS its value.
#     ==================================================================
#
#     ==================================================================
#     DEVIATION BLOCK 181 -- their `volatile IdxT* row_ids` has no Mojo
#     spelling, and `barrier()` is what carries it
#
#     THEIRS (`:51`):
#
#         volatile auto* row_ids =
#           reinterpret_cast<volatile IdxT*>(dataset.row_ids);
#
#     The whole partition reads and writes `row_ids` through that
#     `volatile` view. It is doing one job: forbidding nvcc from keeping
#     a slot in a register across the loop, because a slot one thread
#     reads at the top of an iteration is a slot ANOTHER thread may have
#     written at the bottom of the previous one.
#
#     OURS. `MutPointer`, with no volatile qualifier -- Mojo 1.0 has
#     none -- and an explicit `barrier()` at the end of every loop
#     iteration plus one between the flag reads and the swap writes.
#
#     WHY THAT IS STRONGER AND NOT WEAKER. `volatile` orders one
#     thread's accesses to one location; it does not synchronize
#     threads, and their code relies on `cub::BlockScan`'s internal
#     `__syncthreads()` for the actual cross-thread ordering. A
#     `barrier()` is that synchronization, named, and it makes every
#     write of the previous iteration visible to every thread of the
#     block before any read of the next. A register cached across a
#     `barrier()` would be a compiler defect, not an optimization.
#
#     THE ONE PLACE IT IS LOAD-BEARING, so a later reader does not
#     "tidy" a barrier away. Within an iteration the left window
#     `[loffset, part)` and the right window `[roffset, end)` are
#     disjoint (`part <= roffset` always), so the two sides cannot race
#     each other -- but a thread's own read at `loffset + tid` and
#     another thread's swap write at `lcomp[k]` are in the SAME window.
#     Reads-then-barrier-then-writes is what separates them. The
#     end-of-loop barrier separates this iteration's writes from the
#     next iteration's reads, and it also protects `lcomp`/`rcomp`:
#     the next iteration's compaction overwrites the very slots this
#     iteration's swap is reading.
#
#     A NOTE ON THEIR CODE, transcribed rather than fixed (rule 1). Their
#     iteration ends with the swap and no `__syncthreads()`; the next
#     iteration's flag reads are separated from it only by the fact that
#     a side which did NOT advance also does not RE-READ (the
#     `llen == minlen` guard), and a side which DID advance reads a
#     window disjoint from the one it just swapped in. That is a correct
#     argument and it is theirs. The barrier here does not depend on it.
#     ==================================================================
# ============================================================================

from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum as block_prefix_sum

from extratrees.derived.decisiontree.flatnode import (
    NODE_IS_LEAF,
    SparseTreeNode,
)
from extratrees.derived.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    split_not_valid,
)


# ---------------------------------------------------------------------------
# The partition kernel: `nodeSplitKernel` (`:89-107`) around
# `partitionSamples` (`:43-88`).
# ---------------------------------------------------------------------------

comptime PARTITION_UNVISITED: Int32 = -3
"""The seed the CALLER must write into `out_n_iters` before the launch. It is
not a legal outcome: a cell still holding it means no block served that work
item -- a grid too small. A check asserts there are none. DEVIATION 180."""

comptime PARTITION_SKIPPED: Int32 = -1
"""`nodeSplitKernel`'s early return, `:100-104`: `SplitNotValid` was true, so
NOTHING was partitioned and the node's `row_ids` are untouched. Distinct from
"partitioned and nothing needed to move", which reports `0` iterations."""

comptime PARTITION_OVERRUN: Int32 = -2
"""The derived iteration bound of DEVIATION 176 was reached. It cannot happen
-- the bound is proved, not guessed -- and it exists because a HANG is the one
failure a check cannot report and a kernel cannot raise."""


# Sabotage selectors, one per MECHANISM. A kernel ARGUMENT and not a comptime
# parameter, for the reason `RANGE_SAB_*` and `SCORE_SAB_*` above state: a
# sabotage compiled into a different binary proves nothing about the binary
# that ships, so every arm of `partition_leaf_kernel_check.mojo` runs THIS
# kernel.
comptime PART_SAB_NONE: Int32 = 0
"""No sabotage. The shipping path."""

comptime PART_SAB_LEFT_MISFIT_GE: Int32 = 1
"""A LEFT-side misfit becomes `col[row] >= quesval` instead of `> quesval`
(`:65`), so a row sitting exactly ON the threshold is dragged right. Invisible
unless some row equals the threshold -- which is why the fixture carries a
nine-level tied column; see `partition_check.mojo`'s scar."""

comptime PART_SAB_RIGHT_MISFIT_LT: Int32 = 2
"""A RIGHT-side misfit becomes `col[row] < quesval` instead of `<= quesval`
(`:66`), the other half of the same boundary rule and equally invisible with
distinct values."""

comptime PART_SAB_NO_SCAN: Int32 = 3
"""Compact at `thread_idx.x` instead of at the block scan's exclusive prefix
(`:69-71`, `:75-77`), so flagged threads scatter into holes instead of packing
down. The shared buffers are pre-seeded with `range_start` so this arm stays
memory-safe: every index it can produce is still a slot inside the node."""

comptime PART_SAB_UNPAIRED_SWAP: Int32 = 4
"""Write only the LEFT half of the swap (`:86-88`), so a row is duplicated and
another is dropped. Property (1) and (2) of `partition_check.mojo` can be
satisfied by that; property (4), the permutation, cannot."""

comptime PART_SAB_NO_ROW_IDS: Int32 = 5
"""Drop the `row_ids` indirection in the feature read: `col[row_ids[loff]]`
becomes `col[loff]` (`:65-66`). The SWAP still moves `row_ids`, so this is a
misread and not a different algorithm."""

comptime PART_SAB_NO_VALID_GUARD: Int32 = 6
"""Partition even when `SplitNotValid` says not to (`:100-104`).
`builder.mojo::train_classification` records that this guard is NOT observable
in `tree_check` -- a partition only permutes rows inside a range whose node
stays a leaf. It IS observable here, because this check compares `row_ids`
slot for slot rather than comparing a trained tree."""


def partition_iteration_bound(range_len: Int, n_left: Int, tpb: Int) -> Int:
    """The proved bound on `partitionSamples`' `while`. DEVIATION 176 part 3.

    `minlen = min(llen, rlen)`, so every iteration at least one of `loffset`
    and `roffset` advances by `tpb`. `loffset` needs at most
    `ceildiv(n_left, tpb) + 1` advances to reach `part` and `roffset` at most
    `ceildiv(range_len - n_left, tpb) + 1` to reach `end`; the loop stops when
    EITHER arrives, so the sum bounds the iteration count.

    Host and device call this same function, so the bound the kernel enforces
    and the bound a check asserts against cannot drift apart.
    """
    var n_right = range_len - n_left
    if n_right < 0:
        n_right = 0
    var nl = n_left
    if nl < 0:
        nl = 0
    return ceildiv(nl, tpb) + ceildiv(n_right, tpb) + 2


def node_split_kernel[
    TPB: Int
](
    row_ids: MutPointer[Int32, MutAnyOrigin],
    out_n_iters: MutPointer[Int32, MutAnyOrigin],
    out_n_swaps: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    m_in: Int32,
    min_impurity_decrease: Float32,
    min_samples_leaf_in: Int32,
    sabotage_in: Int32,
):
    """`nodeSplitKernel` (`:89-107`) and the `partitionSamples` it calls.

    `row_ids` IS MUTATED IN PLACE. That is theirs -- it is the one array the
    whole frontier partitions -- and it is the only output of this kernel that
    the algorithm reads again.

    GRID. `grid_dim = work_items_size`, `block_dim = TPB`, which is exactly
    `launchNodeSplitKernel`'s `<<<work_items_size, TPB, smem_size>>>`
    (`:109-134`). ONE BLOCK PER WORK ITEM, by their design and kept -- see
    DEVIATION BLOCK 177 for why it is not "improved" and for what it costs.
    `block_idx.x` indexes BOTH `work_items` and `splits`, which is their
    `work_items[blockIdx.x]` and `splits[blockIdx.x]` (`:96-97`).

    THE NODES OF ONE BATCH MUST HAVE DISJOINT RANGES. Theirs do -- `Push`
    (`builder.cuh:117-131`) tiles the children of a level -- and nothing here
    checks it: two blocks partitioning overlapping ranges would race with no
    ordering between them at all.

    CALLER OBLIGATIONS (DEVIATION 180):

    * `out_n_iters` -- `work_items_size` `Int32`s, SEEDED WITH
      `PARTITION_UNVISITED`. The kernel writes `PARTITION_SKIPPED` for the
      `SplitNotValid` early return, `PARTITION_OVERRUN` for the (unreachable)
      bound, and otherwise the number of `while` iterations its block ran.
    * `out_n_swaps` -- `work_items_size` `Int32`s. Written unconditionally, so
      it needs no seed; it is the total number of misfit PAIRS the block
      swapped, and it is `0` for a node whose rows were already ordered.
    * `data` is COLUMN MAJOR, `dataset.h:24`: `data[colid * m + row]`.
    * `m_in` is `dataset.M`, the ROW COUNT, which is the column stride.

    SHARED MEMORY. `2 * TPB` `Int32`s, declared statically here rather than
    sized by the launcher; see DEVIATION BLOCK 176 part 2. No caller argument.

    THE ANSWER IS THE HOST TRANSCRIPTION'S, SLOT FOR SLOT. Nothing in the
    algorithm depends on warp scheduling: the flags are per-thread, the two
    prefix sums and the two aggregates are block collectives, and `minlen`,
    `loffset` and `roffset` are block-uniform. Given `TPB`, the sequence of
    swaps is determined, so `partition_samples` above produces the identical
    `row_ids` ORDER and the check compares them slot for slot rather than
    comparing two partitions for equivalence.
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sabotage = sabotage_in

    # `:96-97` -- their two whole-struct loads (`const auto work_item =
    # work_items[blockIdx.x]`, `const auto split = splits[blockIdx.x]`), read
    # one field at a time through the pointer instead: DEVIATION 162, a
    # whole-struct load in a kernel kills the Metal compiler.
    var range_start = Int(work_items[unsafe_offset=b].instances.begin)
    var range_len = Int(work_items[unsafe_offset=b].instances.count)
    var quesval = splits[unsafe_offset=b].quesval
    var colid = splits[unsafe_offset=b].colid
    var best_metric = splits[unsafe_offset=b].best_metric_val
    var n_left = Int(splits[unsafe_offset=b].n_left)

    # `:98-104`. `split_not_valid` is `builder_kernels.cuh:59-67` and is the
    # SAME FUNCTION the host builder calls -- the four fields above are
    # assembled into a LOCAL `Split` here, which is a value and not a memory
    # load, so the transcription cannot drift between host and device. Same
    # trick `node_feature_score_kernel` uses for `FeatureRange`.
    var invalid = split_not_valid(
        Split(quesval, colid, best_metric, Int32(n_left)),
        min_impurity_decrease,
        min_samples_leaf_in,
        Int32(range_len),
    )
    if sabotage == PART_SAB_NO_VALID_GUARD:
        invalid = False
    if invalid:
        if tid == 0:
            out_n_iters[unsafe_offset=b] = PARTITION_SKIPPED
            out_n_swaps[unsafe_offset=b] = Int32(0)
        return

    # `:52-54` -- their one `smem` blob carved into `lcomp` and `rcomp`. One
    # allocation, two halves, for the aliasing reason in DEVIATION 176.
    var smem = stack_allocation[
        2 * TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    # Seed both halves with a slot INSIDE the node. Dead on the shipping path
    # -- the swap only ever reads indices below `minlen`, which compaction
    # has filled -- and it is what keeps PART_SAB_NO_SCAN memory-safe instead
    # of turning a sabotage into an out-of-bounds write.
    smem[unsafe_offset=tid] = Int32(range_start)
    smem[unsafe_offset = TPB + tid] = Int32(range_start)
    barrier()

    # `:55-58` -- their cursors. `col = dataset.data + split.colid * M`.
    var col_offset = Int(colid) * Int(m_in)
    var loffset = range_start
    var part = loffset + n_left
    var roffset = part
    var end = range_start + range_len

    # `:59` -- per-thread flags in registers, block-uniform lengths.
    var lflag = Int32(0)
    var rflag = Int32(0)
    var llen = 0
    var rlen = 0
    var minlen = 0

    var iters = 0
    var swaps = 0
    var bound = partition_iteration_bound(range_len, n_left, TPB)

    # `:61` -- and the condition is BLOCK-UNIFORM, which is what makes every
    # collective below legal: `loffset`, `part`, `roffset` and `end` are the
    # same in every thread, so the block leaves the loop together.
    while loffset < part and roffset < end:
        if iters >= bound:
            # DEVIATION 176 part 3. Unreachable; a hang is not.
            if tid == 0:
                out_n_iters[unsafe_offset=b] = PARTITION_OVERRUN
                out_n_swaps[unsafe_offset=b] = Int32(swaps)
            return

        # `:62-67` -- find the misfits. The `llen == minlen` guard is theirs:
        # a side whose cursor did NOT advance keeps the flags it already has,
        # which is what carries leftovers into the next iteration.
        var loff = loffset + tid
        var roff = roffset + tid
        if llen == minlen:
            if loff < part:
                var sl = Int(row_ids[unsafe_offset=loff])
                if sabotage == PART_SAB_NO_ROW_IDS:
                    sl = loff
                var v = data[unsafe_offset = col_offset + sl]
                # `:65`: a LEFT misfit is `> quesval`, so a row EQUAL to the
                # threshold stays LEFT. `_partitioner.pyx:236` agrees.
                var misfit: Bool
                if sabotage == PART_SAB_LEFT_MISFIT_GE:
                    misfit = v >= quesval
                else:
                    misfit = v > quesval
                lflag = Int32(1) if misfit else Int32(0)
            else:
                lflag = Int32(0)
        if rlen == minlen:
            if roff < end:
                var sr = Int(row_ids[unsafe_offset=roff])
                if sabotage == PART_SAB_NO_ROW_IDS:
                    sr = roff
                var v = data[unsafe_offset = col_offset + sr]
                # `:66`: a RIGHT misfit is `<= quesval`.
                var misfit: Bool
                if sabotage == PART_SAB_RIGHT_MISFIT_LT:
                    misfit = v < quesval
                else:
                    misfit = v <= quesval
                rflag = Int32(1) if misfit else Int32(0)
            else:
                rflag = Int32(0)

        # Every read of `row_ids` above is now done; the swap below writes
        # into the same windows. DEVIATION BLOCK 181.
        barrier()

        # `:69-72` -- their two `BlockScanT::ExclusiveSum(flag, idx, len)`
        # calls. Mojo 1.0 has no primitive returning both the prefix and the
        # aggregate, so each becomes a `prefix_sum[exclusive=True]` plus a
        # `sum`; the aggregate must be BLOCK-UNIFORM because it drives the
        # loop condition, which `sum`'s default `broadcast=True` gives.
        # DEVIATION BLOCK 176 part 1.
        var lidx = Int(
            block_prefix_sum[block_size=TPB, exclusive=True](lflag)
        )
        barrier()
        llen = Int(block_sum[block_size=TPB](lflag))
        barrier()
        var ridx = Int(
            block_prefix_sum[block_size=TPB, exclusive=True](rflag)
        )
        barrier()
        rlen = Int(block_sum[block_size=TPB](rflag))
        barrier()

        if sabotage == PART_SAB_NO_SCAN:
            lidx = tid
            ridx = tid

        # `:73` -- pair up only as many misfits as both sides can supply.
        minlen = llen if llen < rlen else rlen

        # `:75-77` -- compaction.
        if lflag != Int32(0):
            smem[unsafe_offset=lidx] = Int32(loff)
        if rflag != Int32(0):
            smem[unsafe_offset = TPB + ridx] = Int32(roff)
        barrier()

        # `:79-83` -- clear the flags about to be consumed, and advance only
        # the side that was fully consumed.
        if lidx < minlen:
            lflag = Int32(0)
        if ridx < minlen:
            rflag = Int32(0)
        if llen == minlen:
            loffset += TPB
        if rlen == minlen:
            roffset += TPB

        # `:84-89` -- swap the paired misfits.
        if tid < minlen:
            var lslot = Int(smem[unsafe_offset=tid])
            var rslot = Int(smem[unsafe_offset = TPB + tid])
            var a = row_ids[unsafe_offset=lslot]
            var bv = row_ids[unsafe_offset=rslot]
            row_ids[unsafe_offset=lslot] = bv
            if sabotage != PART_SAB_UNPAIRED_SWAP:
                row_ids[unsafe_offset=rslot] = a

        # This iteration's writes before the next iteration's reads, and this
        # iteration's `lcomp`/`rcomp` reads before the next iteration's
        # compaction overwrites them. DEVIATION BLOCK 181.
        barrier()
        iters += 1
        swaps += minlen

    if tid == 0:
        out_n_iters[unsafe_offset=b] = Int32(iters)
        out_n_swaps[unsafe_offset=b] = Int32(swaps)


# ---------------------------------------------------------------------------
# The leaf-value kernel: `leafKernel` (`:391-417`) and the half of
# `SetLeafPredictions` (`builder.cuh:556-599`) that is not host policy.
# ---------------------------------------------------------------------------

comptime LEAF_VISIT_NONE: Int32 = 0
"""The zero the CALLER must seed `out_visit` with. Not a legal outcome: a node
still holding it was never served by a block. DEVIATION 180."""

comptime LEAF_VISIT_INTERNAL: Int32 = 1
"""A block ran on this node and took `leafKernel`'s early return at `:403`,
because the node is not a leaf. Its `out_leaves` slot therefore keeps the
caller's zeros, and this value is how a check tells that apart from a slot
that is zero because nothing ran."""

comptime LEAF_VISIT_PUBLISHED: Int32 = 2
"""A block ran on this node, it is a leaf, and `SetLeafVector` wrote its
value."""

comptime LEAF_MAX_OUT_DEFAULT: Int = 16
"""Default comptime bound on `num_outputs`. DEVIATION BLOCK 178: Mojo's
`stack_allocation` is comptime-sized, so the per-thread histogram needs a
bound. Theirs is `num_outputs` wide too (`builder.cuh:581`), not
`n_bins * num_outputs`, so 16 covers this lane rather than truncating it."""


comptime LEAF_SAB_NONE: Int32 = 0
"""No sabotage. The shipping path."""

comptime LEAF_SAB_NO_ISLEAF: Int32 = 1
"""Drop `if (!node.IsLeaf()) return;` (`:403`), so every node's slot is filled.
A tree whose internal slots are filled still predicts correctly at every leaf,
which is why this needs its own arm."""

comptime LEAF_SAB_NO_ROW_IDS: Int32 = 2
"""Read `labels[i]` instead of `labels[row_ids[i]]` (`:409-411`), i.e. treat
the slot index as a row id. Invisible with an identity `row_ids`, which is why
the fixture shuffles it."""

comptime LEAF_SAB_STRIDE_ONE: Int32 = 3
"""Write at `node_id + c` instead of `node_id * num_outputs + c` (`:412`,
`decisiontree.cuh:218`). The stride is `num_outputs` for ALL nodes, leaves and
internal alike; a packed layout is right for node 0 and wrong by a growing
offset after it. Regression has `num_outputs == 1`, where the two spellings
agree and this arm is correctly invisible."""

comptime LEAF_SAB_NO_NORMALIZE: Int32 = 4
"""Publish the raw accumulator instead of dividing it: skip the `/ total` of
`SetLeafVector` for Gini (`objectives.cuh:97-107`) and the `/ count` for MSE
(`:259-264`). For a single-class leaf the probability is 1 and the count is 1,
so this arm is only visible where a leaf holds more than one row."""


def leaf_values_host(
    nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    instance_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    n_nodes: Int,
    num_outputs: Int,
    inv_scale: Float32,
    is_classification: Bool,
) -> List[Float32]:
    """THE ORACLE for `leaf_kernel`: the same pass, sequentially.

    Written the way `node_feature_min_max` and `node_feature_score_host` are
    -- their branches in their order, no block structure, `row_ids` order --
    and it is what `partition_leaf_kernel_check.mojo` compares the device
    against on BIT PATTERNS.

    IT IS NOT A SECOND TRANSCRIPTION OF THE CLASSIFICATION ARM.
    `builder.mojo::set_leaf_predictions_classification` is that, it is
    `builder.cuh:556-599` plus `leafKernel`, and the check compares against
    BOTH -- this function AND that one -- so a drift between the two host
    forms is itself visible. They are obliged to agree exactly:
    `GiniObjectiveFunction.SetLeafVector` is `Float32(count) / Float32(total)`
    and so is the expression below.

    THE REGRESSION ARM IS DIFFERENT AND DELIBERATELY SO. DEVIATION BLOCK 179:
    `set_leaf_predictions_regression` accumulates in `Float64`, which the
    device does not have; this accumulates the same `Int32` fixed-point labels
    the device does, in the same integer arithmetic, and forms the value with
    the same expression in the same order. So THIS is the regression oracle
    and that one is a reporting form. The check measures how far apart they
    land, per leaf, every run.

    A label outside `[0, num_outputs)` is SKIPPED rather than accumulated, on
    both sides, because on the device it would be an out-of-bounds write into
    a private array. `node_feature_score_host` carries the same bound for the
    same reason.
    """
    var out = List[Float32](length=n_nodes * num_outputs, fill=Float32(0.0))
    for node_id in range(n_nodes):
        # `:403`, the early return: an internal node's slot keeps its zeros.
        if nodes[unsafe_offset=node_id].left_child_id != NODE_IS_LEAF:
            continue
        var begin = Int(instance_ranges[unsafe_offset=node_id].begin)
        var count = Int(instance_ranges[unsafe_offset=node_id].count)
        var acc = List[Int32](length=num_outputs, fill=Int32(0))
        var seen = 0
        for i in range(begin, begin + count):
            var row = Int(row_ids[unsafe_offset=i])
            var lab = Int(labels_q[unsafe_offset=row])
            if is_classification:
                if lab >= 0 and lab < num_outputs:
                    acc[lab] += Int32(1)
            else:
                acc[0] += Int32(lab)
            seen += 1
        var base = node_id * num_outputs
        if is_classification:
            # `objectives.cuh:97-107`, and the shape of the conversion is
            # theirs: the numerator is cast and the `int` denominator is
            # promoted by the division.
            var total = Int32(0)
            for c in range(num_outputs):
                total += acc[c]
            for c in range(num_outputs):
                out[base + c] = Float32(Int(acc[c])) / Float32(Int(total))
        else:
            # `objectives.cuh:259-264` is `label_sum / count`; the `*
            # inv_scale` is DEVIATION 179's dequantization and is the last
            # operation, so no `+` sits next to a `*` for a backend to
            # contract into an FMA (DEVIATION BLOCK 173's defect cannot
            # recur here). `ftz` mirrors the kernel's publish (DEVIATION
            # 453) so the oracle computes the same flushed value.
            for c in range(num_outputs):
                out[base + c] = ftz(
                    Float32(Int(acc[c])) / Float32(seen) * inv_scale
                )
    return out^


def leaf_kernel[
    TPB: Int, MAX_OUT: Int, CLASSIFICATION: Bool
](
    out_leaves: MutPointer[Float32, MutAnyOrigin],
    out_visit: MutPointer[Int32, MutAnyOrigin],
    nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    instance_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    labels_q: MutPointer[Int32, MutAnyOrigin],
    num_outputs_in: Int32,
    inv_scale: Float32,
    sabotage_in: Int32,
):
    """`leafKernel`, `builder_kernels_impl.cuh:391-417`.

    GRID. `grid_dim = n_nodes`, `block_dim = TPB`. ONE BLOCK PER NODE, theirs
    (`launchLeafKernel`, `:419-432`, `<<<num_blocks, TPB_DEFAULT, smem_size>>>`
    with `num_blocks = batch_size`). `block_idx.x` is the node id and indexes
    `nodes`, `instance_ranges` and `out_visit` alike; `out_leaves` is indexed
    by `node_id * num_outputs`.

    DEVIATION 180: theirs launches this in batches of 100,000 nodes as a
    host-side memory-budget policy and ours launches it once over the whole
    tree. The signature is batch-ready -- slice the pointers and shrink the
    grid -- and nothing in the kernel changes.

    CALLER OBLIGATIONS, and neither is optional:

    * `out_leaves` -- `n_nodes * num_outputs` `Float32`s, ZEROED. That is
      THEIRS, `cudaMemsetAsync` at `builder.cuh:582`: an internal node's slot
      is written by nothing and the zero IS its value.
    * `out_visit` -- `n_nodes` `Int32`s, ZEROED (`LEAF_VISIT_NONE`). The
      kernel writes `LEAF_VISIT_INTERNAL` or `LEAF_VISIT_PUBLISHED` into
      every node it reaches, so a node still holding zero is a reach failure
      the check can NAME rather than a plausible-looking answer.
    * `labels_q` is an `Int32` array indexed by ROW ID, not by slot: the class
      id for classification, the fixed-point label for regression (DEVIATION
      179). The device performs no float-to-integer conversion at all.
    * `inv_scale` is `1 / scale` for regression and `1.0` for classification,
      which never reads it.

    `MAX_OUT` bounds `num_outputs`; a launch past it publishes NOTHING and the
    caller sees `LEAF_VISIT_NONE`. DEVIATION BLOCK 178.

    THE THREE WAYS THIS PASS IS QUIETLY WRONG, which is why the sabotages
    above are the ones they are: it reads its rows THROUGH `row_ids` over the
    node's own `InstanceRange` (`:409-412`), it must leave every INTERNAL
    node's slot zero (`:403`), and the `vector_leaf` stride is `num_outputs`
    for ALL nodes rather than packed by leaf (`decisiontree.cuh:218`).
    `original/leaf_check.mojo` names the same three for the host form.
    """
    var node_id = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var k = Int(num_outputs_in)
    var sabotage = sabotage_in

    # DEVIATION 178: the private array is comptime-sized, so a request past
    # the bound publishes nothing rather than writing out of bounds.
    if k > MAX_OUT or k < 1:
        return

    # `:398-403` -- `auto& node = tree[node_id]; ... if (!node.IsLeaf())
    # return;`. One FIELD through the pointer, not the struct: `IsLeaf()` is
    # `left_child_id == -1` and nothing else (`flatnode.h:69`).
    var is_leaf = (
        nodes[unsafe_offset=node_id].left_child_id == NODE_IS_LEAF
    )
    if sabotage == LEAF_SAB_NO_ISLEAF:
        is_leaf = True
    if not is_leaf:
        if tid == 0:
            out_visit[unsafe_offset=node_id] = LEAF_VISIT_INTERNAL
        return

    var begin = Int(instance_ranges[unsafe_offset=node_id].begin)
    var count = Int(instance_ranges[unsafe_offset=node_id].count)

    # `:404-407` -- their `histogram[i] = BinT()` over shared memory, as a
    # per-thread private array instead. DEVIATION BLOCK 178.
    var priv = stack_allocation[MAX_OUT, Scalar[DType.int32]]()
    for c in range(MAX_OUT):
        priv[unsafe_offset=c] = Int32(0)
    var seen = Int32(0)

    # `:409-412` -- their row loop. `dataset.labels[dataset.row_ids[i]]` is
    # the indirection this pass exists to get right.
    var i = begin + tid
    while i < begin + count:
        var row = Int(row_ids[unsafe_offset=i])
        if sabotage == LEAF_SAB_NO_ROW_IDS:
            row = i
        var lab = Int(labels_q[unsafe_offset=row])
        comptime if CLASSIFICATION:
            # `BinT::IncrementHistogram(histogram, 1, 0, label)` -- `n_bins`
            # is 1 and `bin` is 0, so it is a plain per-class counter.
            if lab >= 0 and lab < k:
                priv[unsafe_offset=lab] += Int32(1)
        else:
            # DEVIATION 179: the label arrives ALREADY quantized and the
            # accumulation is an exact integer add.
            priv[unsafe_offset=0] += Int32(lab)
        seen += 1
        i += TPB

    # Their `__syncthreads()` at `:413`. Every thread must reach every
    # collective, so no arm above returns per-thread: the two returns are
    # block-uniform (`k > MAX_OUT` and `IsLeaf`).
    var blk_seen = block_sum[block_size=TPB](seen)
    barrier()

    var tot = stack_allocation[MAX_OUT, Scalar[DType.int32]]()
    for c in range(MAX_OUT):
        tot[unsafe_offset=c] = Int32(0)
    for c in range(k):
        var v = block_sum[block_size=TPB](priv[unsafe_offset=c])
        barrier()
        tot[unsafe_offset=c] = v

    # `:414-416` -- `if (tid == 0) SetLeafVector(histogram, num_outputs,
    # leaves + num_outputs * node_id)`.
    if tid != 0:
        return

    var base = node_id * k
    if sabotage == LEAF_SAB_STRIDE_ONE:
        base = node_id

    comptime if CLASSIFICATION:
        # `objectives.cuh:97-107`, transcribed including the shape of the
        # conversion: `int total`, and `DataT(shist[i].x) / total`.
        var total = Int32(0)
        for c in range(k):
            total += tot[unsafe_offset=c]
        for c in range(k):
            var num = Float32(Int(tot[unsafe_offset=c]))
            if sabotage == LEAF_SAB_NO_NORMALIZE:
                out_leaves[unsafe_offset = base + c] = num
            else:
                out_leaves[unsafe_offset = base + c] = num / Float32(
                    Int(total)
                )
    else:
        # `objectives.cuh:259-264`, `label_sum / count`, over the fixed-point
        # sum, with DEVIATION 179's dequantization applied last. The publish
        # goes through `ftz` (DEVIATION 453, row 10): `sum/count` is never
        # denormal (both are integer-valued, quotient magnitude >= 2^-31),
        # but the `* inv_scale` product can be for a tiny-magnitude label
        # vector, and the flushed form is what Metal computes anyway.
        for c in range(k):
            var num = Float32(Int(tot[unsafe_offset=c]))
            if sabotage == LEAF_SAB_NO_NORMALIZE:
                out_leaves[unsafe_offset = base + c] = num * inv_scale
            else:
                out_leaves[unsafe_offset = base + c] = ftz(
                    num / Float32(Int(blk_seen)) * inv_scale
                )

    out_visit[unsafe_offset=node_id] = LEAF_VISIT_PUBLISHED
