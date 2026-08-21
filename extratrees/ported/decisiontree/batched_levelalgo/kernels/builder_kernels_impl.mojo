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


# ============================================================================
# STEPS 2, 3 AND 4 OF DEVIATION 137, ON THE DEVICE: the constant test, the
# keyed threshold DRAW, and the SCORE pass.
#
#     Verified by:  extratrees/mojo_only/score_kernel_check.mojo
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
#     by the RF lane's `ensemble/mojo_only/atomic_width_probe.mojo`,
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
#     `mojo_only/score_kernel_check.mojo` MEASURES the wrap at
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
#     MSE IS DIFFERENT AND IS NOT PUBLISHED AS A RATIONAL. DEVIATION
#     135's own derivation is why: the MSE proxy's numerator is
#     `sum_L^2 * n_R + sum_R^2 * n_L` over FIXED-POINT sums bounded by
#     `2^SLOT_BITS = 2^30`, so it needs up to `2*2^60*n` -- past `Int64`
#     for any node above four rows, which is precisely why
#     `fixed_point.mojo::mse_proxy_exact` returns `Int128`. There is no
#     `Int128` device buffer to publish it into, and there is no need for
#     one: `mse_proxy_exact(sum_left, n_left, sum_right, n_right)` forms
#     the pair from four small integers in one multiply each, so THE FOUR
#     ACCUMULATORS ARE THE SCORE. The regression instantiation leaves
#     `out_gini_num` and `out_gini_den` at their initialized zeros, and
#     the status -- not the rational -- is what says a candidate was
#     scored.
#
#     PRICE. A caller reducing regression candidates must call
#     `mse_proxy_exact` itself instead of comparing a published pair; and
#     a classification node above 2^21 rows must be refused or the
#     comparison silently wraps. `score_row_bound_ok` makes the second
#     one a checkable precondition rather than a comment.
#     ==================================================================
# ============================================================================


# The five statuses a (node, feature) cell can end in. They are the branches of
# `_splitter.pyx:611-666` made into data, because a kernel cannot `continue` out
# of a loop the host wrote and cannot raise. `_empty_candidate` and
# `CandidateRecord` in `mojo_only/host_splitter.mojo` carry the same facts as
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
    var gen = key.generator()
    var res = gen.next_float()
    var threshold = res.fma(
        extent.max_value - extent.min_value, extent.min_value
    )
    if guarded and threshold == extent.max_value:
        return extent.min_value
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
       classification and as the accumulators themselves for regression
       (DEVIATION BLOCK 175).

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
        num = sq_left * nr + sq_right * nl
        den = nl * nr

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
    m_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    tree_id_in: Int32,
    sabotage_in: Int32,
):
    """Steps 2, 3 and 4 of DEVIATION 137: skip, draw, and accumulate.

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
    var key = key_for(seed, UInt32(Int(tree_id_in)), node_id, UInt32(col))
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
    n_cells_in: Int32,
    n_sampled_cols_in: Int32,
    n_acc_in: Int32,
    seed: UInt64,
    tree_id_in: Int32,
    min_samples_leaf_in: Int32,
    sabotage_in: Int32,
):
    """Publish each cell's status, threshold and score. One thread per cell.

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

        var key = key_for(seed, UInt32(Int(tree_id_in)), node_id, UInt32(col))
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
            out_gini_num[unsafe_offset=slot] = sq_left * nr + sq_right * nl
            out_gini_den[unsafe_offset=slot] = nl * nr
        slot += stride
