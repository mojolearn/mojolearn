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
