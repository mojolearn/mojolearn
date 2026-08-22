"""DEVIATION 211's gate: the merged-frontier forest IS the per-tree forest.

`train_forest_classification_device` / `train_forest_regression_device` pop
work from EVERY in-flight tree's queue into one batch, so the launches that
ran once per tree per level run once per level for the whole forest. The
claim that makes this admissible is bit-identity BY CONSTRUCTION: every draw
is a pure function of `(seed, tree_id, node_id, feature_id)` and every row
range lives in its own tree's slot of one shared buffer. This file holds the
construction to its claim, node for node and leaf bit for leaf bit, and then
sabotages the two mechanisms that isolate the trees:

* **`FOREST_SAB_SCALAR_TREE`** stages every item in a merged batch with the
  FIRST item's tree id -- what a port that kept the per-launch scalar would
  silently do. The batch-first tree keeps its keys; every other tree draws
  the wrong thresholds and must MOVE.
* **`FOREST_SAB_SHARED_ROW_BASE`** roots every tree's ranges at slot 0, so
  their partitions overwrite each other -- what losing the slot offsets
  would do. The forest must MOVE.

Both sabotages are run on BOTH objectives (the twins are separate function
bodies, and reach is per branch), and both are asserted RED here rather than
applied and reverted by hand, so the gate stays live.

The multi-GROUP path (`row_slot_cap`) is also reached: a cap of two trees'
rows forces a five-tree forest through three sequential groups, and the
result must still be the per-tree forest.

The serial arm is `train_forest_*_device` called with ONE tree at a time --
one-tree batches on a fresh row buffer each -- and the merged arm for the
default configs is the PUBLIC `fit_*_device`, so the wiring from the fit
entry points through the merged trainer is what gets compared, not a private
path beside it.

NO DURATION IS TAKEN ANYWHERE IN THIS FILE.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from extratrees.estimator import quantize_labels
from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    hashed_classification,
    hashed_regression,
)
from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import TreeMetaDataNode
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    FOREST_SAB_SCALAR_TREE,
    FOREST_SAB_SHARED_ROW_BASE,
    train_forest_classification_device,
    train_forest_regression_device,
    upload_dataset,
)
from extratrees.ported.randomforest.randomforest import (
    class_ids_for,
    fit_classification_device,
    fit_regression_device,
)


def column_major(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def forest_diffs(
    got: List[TreeMetaDataNode[DType.float32]],
    want: List[TreeMetaDataNode[DType.float32]],
) raises -> Tuple[Int, Int]:
    """(nodes that differ, leaf values that differ), across the whole forest.

    Both arms are DEVICE builds, so the comparison is TOTAL: every
    `SparseTreeNode` field including `best_metric_val` (`==`), and every
    leaf value to the BIT. A tree-count or node-count mismatch counts every
    node of the larger tree as differing.
    """
    if len(got) != len(want):
        raise Error(
            "forest sizes differ: "
            + String(len(got))
            + " vs "
            + String(len(want))
        )
    var node_diffs = 0
    var leaf_diffs = 0
    for t in range(len(got)):
        if got[t].num_nodes() != want[t].num_nodes():
            node_diffs += got[t].num_nodes() + want[t].num_nodes()
            continue
        for i in range(got[t].num_nodes()):
            if not (got[t].sparsetree[i] == want[t].sparsetree[i]):
                node_diffs += 1
        if len(got[t].vector_leaf) != len(want[t].vector_leaf):
            leaf_diffs += len(got[t].vector_leaf) + len(want[t].vector_leaf)
            continue
        for i in range(len(got[t].vector_leaf)):
            if (
                got[t].vector_leaf[i].to_bits()
                != want[t].vector_leaf[i].to_bits()
            ):
                leaf_diffs += 1
    return (node_diffs, leaf_diffs)


def trees_mutually_differ(
    trees: List[TreeMetaDataNode[DType.float32]],
) -> Int:
    """How many trees differ from tree 0 -- the fixture-strength floor: a
    forest of copies could not detect the scalar-tree sabotage."""
    var moved = 0
    for t in range(1, len(trees)):
        if trees[t].num_nodes() != trees[0].num_nodes():
            moved += 1
            continue
        for i in range(trees[t].num_nodes()):
            if not (trees[t].sparsetree[i] == trees[0].sparsetree[i]):
                moved += 1
                break
    return moved


def serial_classification(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    """The serial arm: one-tree batches, a fresh device row buffer each."""
    var x = column_major(fixture)
    var class_ids = class_ids_for(
        fixture.y, Int32(fixture.n_rows), Int32(fixture.n_classes)
    )
    var dev = upload_dataset(
        ctx,
        x,
        class_ids,
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_classes),
    )
    var out = List[TreeMetaDataNode[DType.float32]]()
    for t in range(n_trees):
        var one = List[Int32]()
        one.append(Int32(t))
        var built = train_forest_classification_device(
            ctx, dev, params, one, seed
        )
        out.append(built[0].copy())
    return out^


def merged_classification(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
    sabotage: Int32,
    row_slot_cap: Int,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var x = column_major(fixture)
    var class_ids = class_ids_for(
        fixture.y, Int32(fixture.n_rows), Int32(fixture.n_classes)
    )
    var dev = upload_dataset(
        ctx,
        x,
        class_ids,
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_classes),
    )
    var ids = List[Int32]()
    for t in range(n_trees):
        ids.append(Int32(t))
    return train_forest_classification_device(
        ctx, dev, params, ids, seed, sabotage, row_slot_cap
    )


def serial_regression(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var x = column_major(fixture)
    var ql = quantize_labels(fixture.y, Int32(fixture.n_rows))
    var dev = upload_dataset(
        ctx, x, ql[0], Int32(fixture.n_rows), Int32(fixture.n_cols), 1
    )
    var out = List[TreeMetaDataNode[DType.float32]]()
    for t in range(n_trees):
        var one = List[Int32]()
        one.append(Int32(t))
        var built = train_forest_regression_device(
            ctx, dev, ql[1], params, one, seed
        )
        out.append(built[0].copy())
    return out^


def merged_regression(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    n_trees: Int,
    seed: UInt64,
    sabotage: Int32,
    row_slot_cap: Int,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    var x = column_major(fixture)
    var ql = quantize_labels(fixture.y, Int32(fixture.n_rows))
    var dev = upload_dataset(
        ctx, x, ql[0], Int32(fixture.n_rows), Int32(fixture.n_cols), 1
    )
    var ids = List[Int32]()
    for t in range(n_trees):
        ids.append(Int32(t))
    return train_forest_regression_device(
        ctx, dev, ql[1], params, ids, seed, sabotage, row_slot_cap
    )


def main() raises:
    var ctx = DeviceContext()
    print("[device] the batched forest on", ctx.name())
    var cells = 0
    comptime BIG_CAP = 1 << 26
    comptime SEED = UInt64(0x5EED_211)

    # ------------------------------------------------------------------
    # 1. CLASSIFICATION at the defaults, through the PUBLIC fit.
    # ------------------------------------------------------------------
    print("[classification] fit_classification_device vs one-tree builds")
    var clf = hashed_classification(0xA11CE, 240, 7, 3)
    var p = DecisionTreeParams()
    p.max_depth = 7
    var want = serial_classification(ctx, clf, p, 12, SEED)
    var strength = trees_mutually_differ(want)
    assert_true(
        strength >= 2,
        "fixture too weak: the 12 serial trees are near-copies and could"
        " not catch a lost tree id",
    )
    var got_forest = fit_classification_device(
        ctx,
        column_major(clf),
        clf.y,
        Int32(clf.n_rows),
        Int32(clf.n_cols),
        Int32(clf.n_classes),
        p,
        Int32(12),
        SEED,
    )
    var d = forest_diffs(got_forest.trees, want)
    cells += 12
    assert_equal(d[0], 0, "merged classification forest: nodes differ")
    assert_equal(d[1], 0, "merged classification forest: leaf bits differ")
    print(
        "    12 trees,",
        strength,
        "of 11 differ from tree 0; merged vs serial: 0 node diffs,"
        " 0 leaf-bit diffs",
    )

    # ------------------------------------------------------------------
    # 2. The scheduler under stress: a batch width SMALLER than the
    #    frontier, so cycles split trees across batches mid-level.
    # ------------------------------------------------------------------
    print("[batch width] max_batch_size=3 must not move a tree")
    var p3 = DecisionTreeParams()
    p3.max_depth = 6
    p3.max_batch_size = 3
    var want3 = serial_classification(ctx, clf, p3, 8, SEED)
    var got3 = merged_classification(
        ctx, clf, p3, 8, SEED, Int32(0), BIG_CAP
    )
    var d3 = forest_diffs(got3, want3)
    cells += 8
    assert_equal(d3[0], 0, "narrow-batch forest: nodes differ")
    assert_equal(d3[1], 0, "narrow-batch forest: leaf bits differ")
    print("    8 trees at width 3: identical")

    # ------------------------------------------------------------------
    # 3. max_leaves: the leaf budget is spent in each tree's own FIFO
    #    order, which merged scheduling must preserve.
    # ------------------------------------------------------------------
    print("[max_leaves] the budget's break order survives merging")
    var pl = DecisionTreeParams()
    pl.max_depth = 8
    pl.max_leaves = 9
    var wantl = serial_classification(ctx, clf, pl, 6, SEED)
    var gotl = merged_classification(
        ctx, clf, pl, 6, SEED, Int32(0), BIG_CAP
    )
    var dl = forest_diffs(gotl, wantl)
    cells += 6
    assert_equal(dl[0], 0, "max_leaves forest: nodes differ")
    assert_equal(dl[1], 0, "max_leaves forest: leaf bits differ")
    print("    6 trees at max_leaves=9: identical")

    # ------------------------------------------------------------------
    # 4. The multi-GROUP path: a row-slot cap of two trees' rows forces
    #    five trees through three sequential groups.
    # ------------------------------------------------------------------
    print("[groups] row_slot_cap at two trees: three groups, same forest")
    var wantg = serial_classification(ctx, clf, p, 5, SEED)
    var gotg = merged_classification(
        ctx, clf, p, 5, SEED, Int32(0), 2 * clf.n_rows
    )
    var dg = forest_diffs(gotg, wantg)
    cells += 5
    assert_equal(dg[0], 0, "grouped forest: nodes differ")
    assert_equal(dg[1], 0, "grouped forest: leaf bits differ")
    print("    5 trees in groups of 2: identical")

    # ------------------------------------------------------------------
    # 5. REGRESSION at the defaults, through the PUBLIC fit.
    # ------------------------------------------------------------------
    print("[regression] fit_regression_device vs one-tree builds")
    var reg = hashed_regression(0xB0B_BEE, 240, 6)
    var pr = DecisionTreeParams()
    pr.max_depth = 7
    var wantr = serial_regression(ctx, reg, pr, 10, SEED)
    var strengthr = trees_mutually_differ(wantr)
    assert_true(
        strengthr >= 2,
        "fixture too weak: the 10 serial regression trees are near-copies",
    )
    var ql = quantize_labels(reg.y, Int32(reg.n_rows))
    var gotr_forest = fit_regression_device(
        ctx,
        column_major(reg),
        ql[0],
        ql[1],
        Int32(reg.n_rows),
        Int32(reg.n_cols),
        pr,
        Int32(10),
        SEED,
    )
    var dr = forest_diffs(gotr_forest.trees, wantr)
    cells += 10
    assert_equal(dr[0], 0, "merged regression forest: nodes differ")
    assert_equal(dr[1], 0, "merged regression forest: leaf bits differ")
    print(
        "    10 trees,",
        strengthr,
        "of 9 differ from tree 0; merged vs serial: 0 node diffs,"
        " 0 leaf-bit diffs",
    )

    # ------------------------------------------------------------------
    # 6. THE SABOTAGES, asserted RED, both objectives, both mechanisms.
    # ------------------------------------------------------------------
    print("[sabotage] the two isolation mechanisms, seen to hold the gate")
    var sab_ct = merged_classification(
        ctx, clf, p, 12, SEED, FOREST_SAB_SCALAR_TREE, BIG_CAP
    )
    var dct = forest_diffs(sab_ct, want)
    assert_true(
        dct[0] > 0,
        "FOREST_SAB_SCALAR_TREE (classification) did not move the forest --"
        " the per-item tree ids are not reaching the kernels",
    )
    var sab_cb = merged_classification(
        ctx, clf, p, 12, SEED, FOREST_SAB_SHARED_ROW_BASE, BIG_CAP
    )
    var dcb = forest_diffs(sab_cb, want)
    assert_true(
        dcb[0] > 0,
        "FOREST_SAB_SHARED_ROW_BASE (classification) did not move the"
        " forest -- the slot offsets are not what isolates the trees",
    )
    var sab_rt = merged_regression(
        ctx, reg, pr, 10, SEED, FOREST_SAB_SCALAR_TREE, BIG_CAP
    )
    var drt = forest_diffs(sab_rt, wantr)
    assert_true(
        drt[0] > 0,
        "FOREST_SAB_SCALAR_TREE (regression) did not move the forest",
    )
    var sab_rb = merged_regression(
        ctx, reg, pr, 10, SEED, FOREST_SAB_SHARED_ROW_BASE, BIG_CAP
    )
    var drb = forest_diffs(sab_rb, wantr)
    assert_true(
        drb[0] > 0,
        "FOREST_SAB_SHARED_ROW_BASE (regression) did not move the forest",
    )
    cells += 4
    print(
        "    scalar-tree moved",
        dct[0],
        "/",
        drt[0],
        "nodes (clf/reg); shared-row-base moved",
        dcb[0],
        "/",
        drb[0],
    )

    print("device_batched: ", cells, "cells")
    print("device_batched_check: PASS")
