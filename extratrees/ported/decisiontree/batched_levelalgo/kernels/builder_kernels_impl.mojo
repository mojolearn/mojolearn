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
    `:651-652` then applies a guard that is easy to miss and changes an
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
