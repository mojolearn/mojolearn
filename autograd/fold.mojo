# SPDX-License-Identifier: Apache-2.0
# The tape's blocked fold as a free function over an executor, so the tape
# (autograd/tape.mojo), the losses (autograd/losses.mojo) and the metrics
# (autograd/metrics.mojo) sum in one order: the same two launches with the
# same arguments, hence the same bits on every executor.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from autograd.ops import OP_AG_REDUCE_PART, OP_AG_REDUCE_FOLD, AXIS_ALL, AXIS_ROWS, ag_fold_block
from autograd.ops import OP_AGF_LEVEL, AUTOGRAD_FOLD_V2, AG_FOLD_LEAF

# Lane ag-pass3-fold (2026-10-19): under AUTOGRAD_FOLD_V2 (the default) the
# fold is a tree of fixed 64-term blocks (autograd/ops.mojo, "THE BLOCKED
# FOLD", geometry before and after): OP_AG_REDUCE_PART (or a first launch
# of the caller's that writes leaf partials, OP_AGK_BIN_VJP_PART), then
# OP_AGF_LEVEL while an output has more than 64 partials, then
# OP_AG_REDUCE_FOLD; a fold of at most 64 terms an output is one launch.
# Folds of at most 4096 terms keep v1's order and bits. A caller that
# writes the leaf partials itself takes its block length from fold_block
# (never ag_fold_block, which is v1's rule) and its partial count from
# fold_partials. The signatures below are unchanged.


@always_inline
def fold_extent(rows: Int, cols: Int, axis: Int) -> Int:
    """The number of terms each output of a fold over `axis` adds."""
    return rows * cols if axis == AXIS_ALL else (rows if axis == AXIS_ROWS else cols)


@always_inline
def fold_outputs(rows: Int, cols: Int, axis: Int) -> Int:
    return 1 if axis == AXIS_ALL else (cols if axis == AXIS_ROWS else rows)


def fold_partials(rows: Int, cols: Int, axis: Int) -> Int:
    """Floats of partial storage a fold of a rows x cols source needs: one
    per leaf block per output (v2's tree levels fold in place, no more)."""
    var extent = fold_extent(rows, cols, axis)
    var block = fold_block(rows, cols, axis)
    return ((extent + block - 1) // block) * fold_outputs(rows, cols, axis)


def fold_partials_bound(max_extent: Int) -> Int:
    """Partial storage that covers an AXIS_ALL fold of ANY extent up to
    `max_extent`: a fold of n terms has ceil(n / R(n)) <= R(n) blocks, and
    R = ag_fold_block is non-decreasing in n, so R(max_extent) blocks always
    suffice (the block count itself is not monotone in n). v2: the leaf
    length is a constant, so ceil(max_extent / 64) blocks."""
    comptime if AUTOGRAD_FOLD_V2:
        return max((max_extent + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF, 1)
    else:
        return ag_fold_block(max(max_extent, 1))


def fold_block(rows: Int, cols: Int, axis: Int) -> Int:
    """The block length of a fold of a rows x cols source over `axis`."""
    comptime if AUTOGRAD_FOLD_V2:
        return AG_FOLD_LEAF
    else:
        return ag_fold_block(fold_extent(rows, cols, axis))


def fold_single_launch(rows: Int, cols: Int, axis: Int) -> Bool:
    """True when the fold is one leaf block an output (v2), so its first
    launch can write the finished outputs (no partials, no second launch)."""
    comptime if AUTOGRAD_FOLD_V2:
        return fold_extent(rows, cols, axis) <= AG_FOLD_LEAF
    else:
        return False


def blocked_fold_finish[E: Exec](
    mut ex: E, dst: FP, partials: FP, rows: Int, cols: Int, axis: Int, factor: Float32,
    divide: Bool = False,
) raises:
    """The second launch of blocked_fold alone (OP_AG_REDUCE_FOLD): dst =
    the block partials of each output added ascending from +0.0, times
    `factor` (divided by it when `divide`). For a first launch that wrote
    the partials itself (OP_AGK_BIN_VJP_PART). v2: first the tree levels
    (OP_AGF_LEVEL, in place over `partials`) while an output has more than
    64 partials, then the last level's chain."""
    var f = Args()
    f.p0 = partials
    f.p1 = dst
    f.i0 = axis
    f.i1 = rows
    f.i2 = cols
    f.i3 = fold_block(rows, cols, axis)
    if divide:
        f.i4 = 1
    f.f0 = factor
    comptime if AUTOGRAD_FOLD_V2:
        var outputs = fold_outputs(rows, cols, axis)
        var live = (fold_extent(rows, cols, axis) + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF
        var stride = 1
        # driver-loop: over fold tree levels (ceil(log64(extent)) - 1 of them)
        while live > AG_FOLD_LEAF:
            var g = Args()
            g.p0 = partials
            g.i0 = axis
            g.i1 = rows
            g.i2 = cols
            g.i3 = AG_FOLD_LEAF
            g.i5 = stride
            g.i6 = live
            var groups = (live + AG_FOLD_LEAF - 1) // AG_FOLD_LEAF
            ex.launch[OP_AGF_LEVEL](g, groups * outputs)
            live = groups
            stride *= AG_FOLD_LEAF
        f.i5 = stride
        f.i6 = live
    ex.launch[OP_AG_REDUCE_FOLD](f, fold_outputs(rows, cols, axis))


def blocked_fold[E: Exec](
    mut ex: E, dst: FP, src: FP, rows: Int, cols: Int, axis: Int, factor: Float32, partials: FP,
    divide: Bool = False,
) raises:
    """dst = factor * (src summed over `axis`) as a blocked fold: blocks of
    ag_fold_block(extent) consecutive terms, each added ascending from +0.0
    (OP_AG_REDUCE_PART, parallel over blocks x outputs), then each output's
    block partials added ascending from +0.0 (OP_AG_REDUCE_FOLD, parallel
    over outputs). One order on every executor. `partials` holds at least
    fold_partials(rows, cols, axis) floats. `divide`: dst = the sum divided
    by `factor` (the fused linear node's bias gradient over its divisor)."""
    var extent = fold_extent(rows, cols, axis)
    var block = fold_block(rows, cols, axis)
    var blocks = (extent + block - 1) // block
    var outputs = fold_outputs(rows, cols, axis)
    var a = Args()
    a.p0 = src
    a.p1 = partials
    a.i0 = axis
    a.i1 = rows
    a.i2 = cols
    a.i3 = block
    if fold_single_launch(rows, cols, axis):
        # v2: one block an output; PART writes dst itself (same bits).
        # p1 stays bound to the (unused) partials: live storage.
        a.p2 = dst
        a.i4 = 1
        if divide:
            a.i5 = 1
        a.f0 = factor
        ex.launch[OP_AG_REDUCE_PART](a, outputs)
        return
    ex.launch[OP_AG_REDUCE_PART](a, blocks * outputs)
    blocked_fold_finish(ex, dst, partials, rows, cols, axis, factor, divide)
