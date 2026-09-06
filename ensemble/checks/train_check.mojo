# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the wired builder actually grow cuML's tree, end to end, on device?

    pixi run mojo run -I . ensemble/checks/train_check.mojo

THE FIRST CHECK IN THIS DIRECTORY THAT TRAINS. Everything before it held one
component to its own contract; this one runs quantiles -> feature sampling ->
histogram -> cdf -> gain -> split reduction -> partition -> leaf, on the
device, for real, and asks whether the TREE is right.

WHY THE FIXTURES ARE ANALYTIC AND NOT REAL DATA. This repository's rule is
that correctness gates on an analytic answer or on the competitor's own
output, never on a real dataset. cuML cannot run on this box, so the NVIDIA
column's per-cell dump is not available in this round -- which leaves
analytic, and analytic is enough IF the fixture is built so the right answer
is forced rather than merely likely.

So each fixture below is CONSTRUCTED so that one split is uniquely optimal
and every competing split is strictly worse, and the expected tree is then
written out by hand. A fixture where two splits tie would test nothing except
the tie-break, and a fixture drawn at random would require re-deriving Gini
to know the answer -- which is importing the formula under test.

  A. SEPARABLE. Feature 0 takes two values and separates the classes
     exactly; features 1 and 2 are hashed noise. Three nodes, a threshold
     of exactly 0.0, children of exactly 200/200, and leaf probabilities of
     exactly 1/0 and 0/1. Catches a wrong column, a reversed threshold
     direction, an unpartitioned child, or a leaf read off the wrong node.

  B. DEPTH TWO. A three-group staircase: one split with real gain, then a
     second that completes the separation. Five nodes. The two level-2
     leaves can only be pure if the PARTITION moved the right rows, which
     is the step arm A cannot see at all.

  C. `max_depth` and `min_samples_split` stop the device loop, not just the
     host queue -- including `max_depth = 0`, whose single leaf must hold
     the whole dataset's class fractions.

  D. LEAF VALUES, checked inside A, B and C rather than as its own arm.
     `SetLeafVector` writes class PROBABILITIES (`objectives.cuh:179-195`),
     so a pure leaf reads exactly 1.0/0.0 and the depth-0 leaf reads exactly
     0.5/0.5. Internal nodes must stay at 0.

  E. DETERMINISM. The same fit run twice must be bit-identical, leaf values
     included. This is the property the whole classification path was chosen
     for -- an integer histogram under an integer atomic, plus a total-order
     tie-break -- and it costs one extra fit to check.

TWO OF THESE FIXTURES WERE WRONG ON THE FIRST RUN, and both times the fit was
right:

  * arm A used `Float32(r)`, separable at 200 in the raw data, and got 9
    nodes instead of 3. Their quantiles come from a SUBSAMPLE of
    `min(n_rows, max_n_bins * oversampling_factor)` rows, so with 400 rows
    and 16 bins only 64 are sorted and the bin edge misses 200. The children
    were then slightly impure and split again -- faithful, and invisible to
    a fixture that could not tell that from a bug.
  * arm B was XOR, and returned a single leaf. Correct: under XOR neither
    feature alone carries information, every candidate gain is exactly 0,
    and `objectives.cuh:173` admits a split only when
    `gain > min_impurity_decrease` STRICTLY. No greedy tree splits XOR at
    the root -- not cuML's, not sklearn's, not this one.

Both are recorded at their arms, because a fixture that has been wrong once
is the kind a later reader should not have to re-derive.
"""

from std.math import ceildiv
from std.sys.info import size_of

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.builder import Builder
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    compute_quantiles,
    Quantiles,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI

comptime DT = DType.float32
comptime LT = DType.int32


def _params(
    max_depth: Int,
    min_samples_split: Int,
    max_n_bins: Int = 16,
    max_leaves: Int = -1,
) -> DecisionTreeParams:
    return DecisionTreeParams(
        max_depth=Int32(max_depth),
        max_leaves=Int32(max_leaves),
        max_features=Float32(1.0),
        max_n_bins=Int32(max_n_bins),
        min_samples_leaf=Int32(1),
        min_samples_split=Int32(min_samples_split),
        split_criterion=GINI,
        min_impurity_decrease=Float32(0.0),
        max_batch_size=Int32(128),
    )


@always_inline
def _mix(x: UInt64) -> UInt64:
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


struct Fit(Movable):
    """One trained tree plus everything the device needed, held together so
    nothing is freed at its last use while a kernel is still reading it."""

    var n_nodes: Int
    var colid: List[Int32]
    var quesval: List[Float32]
    var left_child: List[Int64]
    var count: List[Int32]
    var is_leaf: List[Bool]
    var leaf: List[Float32]
    var num_outputs: Int

    def __init__(out self):
        self.n_nodes = 0
        self.colid = List[Int32]()
        self.quesval = List[Float32]()
        self.left_child = List[Int64]()
        self.count = List[Int32]()
        self.is_leaf = List[Bool]()
        self.leaf = List[Float32]()
        self.num_outputs = 0


def _fit(
    ctx: DeviceContext,
    x: List[Float32],          # column-major, n_rows x n_cols
    y: List[Int32],
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    params: DecisionTreeParams,
) raises -> Fit:
    """Quantize, then build one tree. Column-major input, row_ids = identity
    (no bootstrap: `RowSampler` is not ported, so this is their
    `bootstrap=False` shape, `randomforest.cuh:159-160`'s
    `thrust::sequence`)."""
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    for i in range(n_rows * n_cols):
        hx.unsafe_ptr().unsafe_store(i, x[i])
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())

    var hy = ctx.enqueue_create_host_buffer[LT](n_rows)
    for i in range(n_rows):
        hy.unsafe_ptr().unsafe_store(i, y[i])
    var dy = ctx.enqueue_create_buffer[LT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var hr = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
    for i in range(n_rows):
        hr.unsafe_ptr().unsafe_store(i, Int32(i))
    var dr = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.enqueue_copy(dst_buf=dr, src_ptr=hr.unsafe_ptr())

    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var qr = compute_quantiles(
        ctx, dx, Int(params.max_n_bins), n_rows, n_cols, seed=UInt64(7)
    )
    var quantiles = Quantiles[DT](
        qr.quantiles_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        qr.n_bins_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var dataset = DatasetView[DT, LT](
        dx.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dsw.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_rows),
        Int64(n_cols),
        Int64(1),           # column-major: row_stride 1
        Int64(n_rows),      #               col_stride n_rows
        Int32(n_rows),
        Int32(n_cols),
        dr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(n_classes),
        False,
        # DEVIATION 314: raw path -- this check IS the searching loop's
        # oracle.
        dx.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8](),
        False,
    )

    var builder = Builder[
        ClassificationObjectiveFunction[DT, LT, ClassificationBin]
    ](
        ctx, params, Int32(0), UInt64(42), n_rows, n_cols,
        Int32(n_classes),
    )
    var tree = builder.train(ctx, dataset, quantiles)

    var out = Fit()
    out.n_nodes = len(tree.sparsetree)
    out.num_outputs = n_classes
    for i in range(out.n_nodes):
        ref n = tree.sparsetree[i]
        out.colid.append(n.ColumnId())
        out.quesval.append(n.QueryValue())
        out.left_child.append(n.LeftChildId())
        out.count.append(n.InstanceCount())
        out.is_leaf.append(n.IsLeaf())
    for i in range(len(tree.vector_leaf)):
        out.leaf.append(tree.vector_leaf[i])

    # Mojo frees a value at its LAST USE. Every buffer above was handed to a
    # kernel as a raw pointer, so each needs a use after the final
    # synchronize inside train(). Measured hazard, not a precaution.
    _ = dx^
    _ = dy^
    _ = dr^
    _ = dsw^
    _ = hx^
    _ = hy^
    _ = hr^
    _ = qr^
    _ = builder^
    return out^


def arm_a_separable(ctx: DeviceContext) raises -> Int:
    """Feature 0 separates perfectly at 100; features 1 and 2 are noise."""
    var n_rows = 400
    var n_cols = 3
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        # Feature 0 takes exactly TWO distinct values, and that is the
        # point. A first version used `Float32(r)` -- separable in the raw
        # data at 200 -- and the fit came back with 9 nodes instead of 3.
        # The fit was RIGHT: their quantiles come from a SUBSAMPLE of
        # `min(n_rows, max_n_bins * oversampling_factor)` rows
        # (`quantiles.cuh`, oversampling 4), so with 400 rows and 16 bins
        # only 64 rows are sorted and the bin edge does not land on 200.
        # The children were then slightly impure and split again --
        # faithful behaviour, and a fixture that could not tell that from a
        # bug. Two distinct values collapse to two bins under their
        # `thrust::unique`, so the boundary is exact whatever the subsample
        # draws.
        x[0 * n_rows + r] = Float32(0.0) if r < 200 else Float32(100.0)
        x[1 * n_rows + r] = Float32(Int(_mix(UInt64(r) + 11) % 1000))
        x[2 * n_rows + r] = Float32(Int(_mix(UInt64(r) + 977) % 1000))
        y.append(Int32(0) if r < 200 else Int32(1))

    var f = _fit(ctx, x, y, n_rows, n_cols, 2, _params(4, 2))
    var fails = 0

    if f.n_nodes != 3:
        fails += 1
        print(
            "  arm A: expected exactly 3 nodes (root + two PURE leaves),"
            " got", f.n_nodes,
        )
        return fails
    if f.is_leaf[0]:
        fails += 1
        print("  arm A: the root must split")
    if f.colid[0] != Int32(0):
        fails += 1
        print(
            "  arm A: the root must split on feature 0, the only informative"
            " one; got column", f.colid[0],
        )
    if not (f.is_leaf[1] and f.is_leaf[2]):
        fails += 1
        print(
            "  arm A: both children of a perfect split are PURE and must be"
            " leaves -- a pure node has zero gain at every threshold"
        )
    if f.count[0] != Int32(400):
        fails += 1
        print("  arm A: root count", f.count[0], "want 400")
    if Int(f.count[1]) + Int(f.count[2]) != 400:
        fails += 1
        print(
            "  arm A: children must conserve rows;",
            f.count[1], "+", f.count[2],
        )
    # Feature 0 has exactly two values, so the only separating threshold is
    # the lower one: `<= 0.0` goes left.
    if f.quesval[0] != Float32(0.0):
        fails += 1
        print(
            "  arm A: threshold", f.quesval[0],
            "-- with two distinct values the split must be at 0.0, and"
            " `<=` sends the class-0 rows left",
        )
    if f.count[1] != Int32(200) or f.count[2] != Int32(200):
        fails += 1
        print(
            "  arm A: children must be 200/200, got",
            f.count[1], "/", f.count[2],
        )
    # leaf probabilities: each leaf pure, so 1.0 for its class
    var l1a = f.leaf[1 * 2 + 0]
    var l1b = f.leaf[1 * 2 + 1]
    var l2a = f.leaf[2 * 2 + 0]
    var l2b = f.leaf[2 * 2 + 1]
    if not (l1a == Float32(1.0) and l1b == Float32(0.0)):
        fails += 1
        print("  arm A: left leaf probs", l1a, l1b, "want exactly 1.0 0.0")
    if not (l2a == Float32(0.0) and l2b == Float32(1.0)):
        fails += 1
        print("  arm A: right leaf probs", l2a, l2b, "want exactly 0.0 1.0")
    if f.leaf[0 * 2 + 0] != Float32(0.0) or f.leaf[0 * 2 + 1] != Float32(0.0):
        fails += 1
        print(
            "  arm A: the ROOT is not a leaf and its vector_leaf slots must"
            " stay 0; got", f.leaf[0], f.leaf[1],
        )

    if fails == 0:
        print(
            "  arm A OK: 3 nodes, split on the only informative column at",
            f.quesval[0],
            ", both children pure, leaf probabilities exactly 1/0 and 0/1,"
            " root slots still zero",
        )
    return fails


def arm_b_depth_two(ctx: DeviceContext) raises -> Int:
    """A three-group staircase: one split, then a second on the right.

    NOT XOR, AND THE FIRST VERSION OF THIS ARM WAS XOR, WHICH WAS WRONG.
    Under XOR neither feature alone carries any information, so every
    candidate split has a Gini gain of exactly 0, and `objectives.cuh:173`
    admits a split only when `gain > min_impurity_decrease` -- STRICTLY, and
    the default `min_impurity_decrease` is 0. So the fit returned a single
    leaf, and it was right: a greedy one-step-lookahead tree cannot split
    XOR at the root, and neither can cuML or sklearn. The fixture, not the
    port, was the defect.

    This one has a first split with real gain and a second that completes
    the separation:

        feature 0 == 0            -> class 0        (200 rows)
        feature 0 == 100, f1 == 0 -> class 1        (100 rows)
        feature 0 == 100, f1 ==100-> class 2        (100 rows)

    Expected tree: root splits on feature 0; the left child is pure class 0
    and is a leaf; the right child splits on feature 1 into two pure leaves.
    Five nodes, depth 2 -- and the two level-2 leaves can only be pure if
    the PARTITION actually moved the right rows, which is the step arm A
    cannot see.
    """
    var n_rows = 400
    var n_cols = 2
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        var left = r < 200
        var b = (r % 2) == 0
        x[0 * n_rows + r] = Float32(0.0) if left else Float32(100.0)
        x[1 * n_rows + r] = Float32(0.0) if b else Float32(100.0)
        if left:
            y.append(Int32(0))
        elif b:
            y.append(Int32(1))
        else:
            y.append(Int32(2))

    var f = _fit(ctx, x, y, n_rows, n_cols, 3, _params(3, 2))
    var fails = 0

    if f.n_nodes != 5:
        fails += 1
        print(
            "  arm B: expected exactly 5 nodes (root, a pure left leaf, a"
            " split right child and its two pure leaves), got", f.n_nodes,
            ". A builder that never PARTITIONS its children would stop"
            " at 3.",
        )
        return fails
    if f.is_leaf[0]:
        fails += 1
        print("  arm B: the root must split")
    if f.colid[0] != Int32(0):
        fails += 1
        print("  arm B: the root must split on feature 0, got", f.colid[0])
    if not f.is_leaf[1]:
        fails += 1
        print("  arm B: the left child is pure class 0 and must be a leaf")
    if f.is_leaf[2]:
        fails += 1
        print(
            "  arm B: the right child holds classes 1 and 2 and must SPLIT"
            " -- if it is a leaf, the partition never delivered its rows"
        )
    elif f.colid[2] != Int32(1):
        fails += 1
        print(
            "  arm B: the right child must split on feature 1, got",
            f.colid[2],
        )

    # every leaf must be PURE, because two splits separate XOR exactly
    var impure = 0
    var leaf_rows = 0
    for i in range(f.n_nodes):
        if f.is_leaf[i]:
            leaf_rows += Int(f.count[i])
            var best = Float32(0.0)
            for c in range(3):
                if f.leaf[i * 3 + c] > best:
                    best = f.leaf[i * 3 + c]
            if best != Float32(1.0):
                impure += 1
    if impure != 0:
        fails += 1
        print(
            "  arm B:", impure,
            "leaves are impure; two splits separate this staircase exactly,"
            " so every leaf must read a probability of exactly 1.0 for one"
            " class",
        )
    if leaf_rows != 400:
        fails += 1
        print(
            "  arm B: leaves hold", leaf_rows,
            "rows in total, want 400 -- the partition lost or duplicated"
            " rows",
        )

    if fails == 0:
        print(
            "  arm B OK:", f.n_nodes,
            "nodes, every leaf pure, leaves hold exactly 400 rows -- the"
            " level-2 nodes saw correctly partitioned data",
        )
    return fails


def arm_c_stopping(ctx: DeviceContext) raises -> Int:
    """`max_depth` and `min_samples_split` must stop the real loop."""
    var n_rows = 400
    var n_cols = 2
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        x[0 * n_rows + r] = Float32(r)
        x[1 * n_rows + r] = Float32(Int(_mix(UInt64(r) + 5) % 1000))
        y.append(Int32(r % 2))   # alternating: hard, so it keeps splitting

    var fails = 0
    var d1 = _fit(ctx, x, y, n_rows, n_cols, 2, _params(1, 2))
    if d1.n_nodes > 3:
        fails += 1
        print("  arm C: max_depth 1 gave", d1.n_nodes, "nodes, want at most 3")

    var d0 = _fit(ctx, x, y, n_rows, n_cols, 2, _params(0, 2))
    if d0.n_nodes != 1:
        fails += 1
        print("  arm C: max_depth 0 must give exactly 1 node, got", d0.n_nodes)
    if not d0.is_leaf[0]:
        fails += 1
        print("  arm C: max_depth 0's single node must be a leaf")
    # and its leaf value is the class distribution of the whole dataset:
    # 200/400 and 200/400 exactly.
    if d0.leaf[0] != Float32(0.5) or d0.leaf[1] != Float32(0.5):
        fails += 1
        print(
            "  arm C: the depth-0 leaf must hold the whole dataset's class"
            " fractions 0.5/0.5; got", d0.leaf[0], d0.leaf[1],
        )

    var ms = _fit(ctx, x, y, n_rows, n_cols, 2, _params(6, 1000))
    if ms.n_nodes != 1:
        fails += 1
        print(
            "  arm C: min_samples_split 1000 > 400 rows must give 1 node,"
            " got", ms.n_nodes,
        )

    if fails == 0:
        print(
            "  arm C OK: max_depth 1 capped the tree, max_depth 0 gave one"
            " leaf holding exactly 0.5/0.5, min_samples_split 1000 refused"
            " to split at all"
        )
    return fails


def arm_e_determinism(ctx: DeviceContext) raises -> Int:
    """Two fits of the same data must be BIT-identical.

    This is the property the classification path exists for: the histogram
    is an integer counter under an integer atomic, so it does not depend on
    the order blocks arrive, and `Split::update` breaks every tie on a total
    order. If this arm ever fails, the identity claim in `archive/plans/ensemble/PLAN.md`
    is void and the cause is upstream of anything a tolerance can hide.
    """
    var n_rows = 512
    var n_cols = 4
    var x = List[Float32]()
    x.resize(n_rows * n_cols, Float32(0))
    var y = List[Int32]()
    for r in range(n_rows):
        for c in range(n_cols):
            x[c * n_rows + r] = Float32(
                Int(_mix(UInt64(r) * 31 + UInt64(c)) % 4096)
            )
        y.append(Int32(Int(_mix(UInt64(r) + 3) % 3)))

    var p = _params(5, 2)
    var f1 = _fit(ctx, x, y, n_rows, n_cols, 3, p)
    var f2 = _fit(ctx, x, y, n_rows, n_cols, 3, p)

    var fails = 0
    if f1.n_nodes != f2.n_nodes:
        fails += 1
        print(
            "  arm E: node counts differ between two fits:",
            f1.n_nodes, "vs", f2.n_nodes,
        )
        return fails
    var diff = 0
    for i in range(f1.n_nodes):
        if (
            f1.colid[i] != f2.colid[i]
            or f1.quesval[i] != f2.quesval[i]
            or f1.left_child[i] != f2.left_child[i]
            or f1.count[i] != f2.count[i]
        ):
            diff += 1
    for i in range(len(f1.leaf)):
        if f1.leaf[i] != f2.leaf[i]:
            diff += 1
    if diff != 0:
        fails += 1
        print("  arm E: two fits differ in", diff, "places")
    if f1.n_nodes < 5:
        fails += 1
        print(
            "  arm E: the fixture produced only", f1.n_nodes,
            "nodes -- too small to be evidence of anything",
        )

    if fails == 0:
        print(
            "  arm E OK:", f1.n_nodes,
            "nodes and", len(f1.leaf),
            "leaf values, BIT-IDENTICAL across two fits of a 3-class,"
            " 4-feature hashed dataset",
        )
    return fails


def main() raises:
    print("train_check: ensemble/decisiontree/batched_levelalgo/builder.mojo")
    print("  end to end on device -- quantiles, sampling, histogram, cdf,")
    print("  gain, split reduction, partition, leaf")
    var ctx = DeviceContext()
    var fails = 0
    fails += arm_a_separable(ctx)
    fails += arm_b_depth_two(ctx)
    fails += arm_c_stopping(ctx)
    fails += arm_e_determinism(ctx)
    if fails == 0:
        print("train_check: ALL OK")
    else:
        raise Error("train_check: " + String(fails) + " failure(s)")
