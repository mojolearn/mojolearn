"""A tree grown on the GPU, against the same tree grown on the host.

Until this check existed, every device kernel in this lane was reached only by
its own check. This one runs the whole split search on the device — range,
draw, score, reduce — inside the real `NodeQueue` frontier, and compares the
tree that comes out against `train_classification`'s.

**THE RESULT, AND IT IS THE POINT OF THE FILE: with cuML's zero-gain gate
disabled on the host, the two trees are BIT-IDENTICAL.** 9 configurations, 747
nodes, 0 differing in `(colid, quesval, left_child_id, instance_count)`, 0
differing row predictions. The device split search reproduces the host's
exactly — which it can, because deviation 142's amendment put the same explicit
`fma` on both sides of the draw and deviation 135's fixed point made the
accumulators integers.

**AND THE ONE THING THAT IS NOT IDENTICAL IS NAMED, MEASURED AND EXPLAINED.**
`split_not_valid` rejects a split whose `best_metric_val` is `<=
min_impurity_decrease`, and cuML's Gini gain is `>= 0` always, so at the
default `0.0` the HOST rejects a zero-gain split — one that does not reduce
impurity at all — and turns the node into a leaf. The DEVICE cannot: it does
not compute the float gain (deviations 175 and 182), so it publishes a
constant and splits anyway. That is deviation 183, it is the ONLY difference
between the two paths, and this file proves it is the only one by running both
arms:

* **arm 2a**, the gate disabled on the host: the trees must be identical, node
  for node. If anything else ever diverged, this arm is where it would show;
* **arm 2b**, the gate at its default: the difference is MEASURED, and the
  direction is asserted — the device can only ever have MORE nodes, never
  fewer, because the gate can only ever reject.

`best_metric_val` itself is excluded from the comparison in both arms, stated
rather than quietly skipped.

Analytic fixtures are asserted outright, because their answer does not depend
on any of that: a separable fixture must come out perfect on the device too.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    analytic_all_constant,
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_class,
    predict_leaf,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification,
    train_classification_device,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset


def column_major(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def row_of(fixture: FixtureDataset, r: Int) -> List[Float32]:
    var row = List[Float32]()
    for c in range(fixture.n_cols):
        row.append(fixture.value(r, c))
    return row^


def fit_device(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    tree_id: Int32,
) raises -> TreeMetaDataNode[DType.float32]:
    var x = column_major(fixture)
    var ids = List[Int32]()
    for r in range(fixture.n_rows):
        ids.append(Int32(Int(fixture.label[r])))
    var rows = List[Int32]()
    for r in range(fixture.n_rows):
        rows.append(Int32(r))
    var tree = train_classification_device(
        ctx,
        x,
        ids,
        rows,
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_classes),
        params,
        tree_id,
        seed,
    )
    _ = x.unsafe_ptr()
    _ = ids.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return tree^


def fit_host(
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    tree_id: Int32,
) raises -> TreeMetaDataNode[DType.float32]:
    var x = column_major(fixture)
    var labels = List[Float32]()
    for r in range(fixture.n_rows):
        labels.append(Float32(Int(fixture.label[r])))
    var rows = List[Int32]()
    for r in range(fixture.n_rows):
        rows.append(Int32(r))
    var ds = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](rows.unsafe_ptr()),
        Int32(fixture.n_classes),
    )
    var tree = train_classification(
        ds, params, tree_id, seed, Int32(fixture.n_classes)
    )
    _ = x.unsafe_ptr()
    _ = labels.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return tree^


def main() raises:
    var ctx = DeviceContext()
    print("[device] a tree grown on", ctx.name())
    var cells = 0

    # ---------------- 1. the analytic fixtures, asserted outright ---------
    print("[analytic] the device must fit a separable fixture exactly")
    var gap = analytic_separable_gap(0x5EED).data.copy()
    var p = DecisionTreeParams()
    p.max_depth = 8
    for seed_i in range(6):
        var seed = UInt64(seed_i * 7919 + 13)
        var dt = fit_device(ctx, gap, p, seed, 0)
        var wrong = 0
        for r in range(gap.n_rows):
            var row = row_of(gap, r)
            if predict_class(dt, row, 0) != Int(gap.label[r]):
                wrong += 1
        assert_equal(
            wrong,
            0,
            "seed "
            + String(seed_i)
            + ": the DEVICE split search must separate a fixture whose"
            " classes have an empty band between them",
        )
        assert_true(dt.num_nodes() >= 3, "and it must actually split")
        cells += 2
    print("    6 seeds, 0 of", gap.n_rows, "rows wrong")

    var flat = analytic_all_constant().data.copy()
    for seed_i in range(4):
        var dt = fit_device(ctx, flat, p, UInt64(seed_i * 31337 + 5), 0)
        assert_equal(
            dt.num_nodes(),
            1,
            "nothing is splittable, so the device must produce one leaf",
        )
        assert_equal(dt.depth_counter, 0)
        cells += 2
    print("    all-constant: 4 seeds, one node each")

    # ---------------- 2. device against host, node by node ----------------
    print("[identity] the device tree against the host tree, node by node")
    var hashed = hashed_classification(0xD3F1CE, 512, 6, 3)
    var total_nodes = 0
    var struct_diff = 0
    var shape_diff = 0
    var pred_diff = 0
    var configs = 0
    for depth in [Int32(3), Int32(5), Int32(7)]:
        for seed_i in range(3):
            var pc = DecisionTreeParams()
            pc.max_depth = depth
            pc.max_features = 0.5
            # ARM 2a: the host's zero-gain gate DISABLED. cuML's Gini gain is
            # >= 0 always and `split_not_valid` rejects `gain <=
            # min_impurity_decrease`, so a threshold below zero can never
            # reject. With the gate out of the way the two paths must agree
            # BIT FOR BIT -- that is the identity claim, and everything else
            # in the pipeline is upstream of this line.
            pc.min_impurity_decrease = -1.0
            var seed = UInt64(seed_i * 104729 + 11)
            var ht = fit_host(hashed, pc, seed, 2)
            var dt = fit_device(ctx, hashed, pc, seed, 2)
            configs += 1

            if ht.num_nodes() != dt.num_nodes():
                shape_diff += 1
            else:
                for i in range(ht.num_nodes()):
                    total_nodes += 1
                    var a = ht.sparsetree[i]
                    var b = dt.sparsetree[i]
                    # best_metric_val is EXCLUDED, deviation 182.
                    if (
                        a.ColumnId() != b.ColumnId()
                        or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                        or a.LeftChildId() != b.LeftChildId()
                        or a.InstanceCount() != b.InstanceCount()
                    ):
                        struct_diff += 1
                    cells += 1
            # and whether it matters to a prediction
            for r in range(hashed.n_rows):
                var row = row_of(hashed, r)
                if predict_class(ht, row, 0) != predict_class(dt, row, 0):
                    pred_diff += 1
    print(
        "    ",
        configs,
        "configurations;",
        shape_diff,
        "differ in node COUNT;",
        struct_diff,
        "of",
        total_nodes,
        "nodes differ in (colid, quesval, left_child, instance_count);",
        pred_diff,
        "row predictions differ",
    )
    assert_equal(
        shape_diff,
        0,
        "with the zero-gain gate disabled the device and host trees must have"
        " the SAME node count; a difference here is a broken frontier, not a"
        " tie-break",
    )
    assert_equal(
        struct_diff,
        0,
        "with the zero-gain gate disabled every node must agree in (colid,"
        " quesval, left_child_id, instance_count) -- the device split search"
        " reproduces the host's exactly",
    )
    assert_equal(
        pred_diff,
        0,
        "and therefore every row prediction must agree",
    )
    cells += 3

    # ---------------- 2b. the gate at its DEFAULT: the one difference ------
    # DEVIATION 183, measured rather than asserted away. The device cannot
    # apply cuML's `min_impurity_decrease` gate because it does not compute
    # the float gain, so it splits where the host makes a leaf. The gate can
    # only ever REJECT, so the device can only ever have MORE nodes.
    print("[deviation 183] the gate at its default, measured")
    var gated_configs = 0
    var device_more = 0
    var device_fewer = 0
    var same = 0
    var host_total = 0
    var dev_total = 0
    for depth in [Int32(3), Int32(5), Int32(7)]:
        for seed_i in range(3):
            var pg = DecisionTreeParams()
            pg.max_depth = depth
            pg.max_features = 0.5
            var seed = UInt64(seed_i * 104729 + 11)
            var ht = fit_host(hashed, pg, seed, 2)
            var dt = fit_device(ctx, hashed, pg, seed, 2)
            gated_configs += 1
            host_total += ht.num_nodes()
            dev_total += dt.num_nodes()
            if dt.num_nodes() > ht.num_nodes():
                device_more += 1
            elif dt.num_nodes() < ht.num_nodes():
                device_fewer += 1
            else:
                same += 1
    print(
        "    ",
        gated_configs,
        "configurations:",
        same,
        "identical in node count,",
        device_more,
        "where the device has MORE nodes,",
        device_fewer,
        "where it has FEWER;",
        host_total,
        "host nodes against",
        dev_total,
        "device nodes",
    )
    assert_equal(
        device_fewer,
        0,
        "the zero-gain gate can only ever REJECT a split, so the device --"
        " which does not apply it -- can only ever have MORE nodes. Fewer"
        " means something other than DEVIATION 183 is diverging.",
    )
    assert_true(
        device_more > 0,
        "if the device never has more nodes then this fixture never produces"
        " a zero-gain split, and arm 2b is measuring nothing -- DEVIATION"
        " 183's cost would be invisible here",
    )
    cells += 2

    # ---------------- 3. structural invariants on the device path ---------
    print("[invariants] the device tree on its own terms")
    var pc2 = DecisionTreeParams()
    pc2.max_depth = 6
    var dt2 = fit_device(ctx, hashed, pc2, 0xABCDEF, 4)
    var landed = List[Int](length=dt2.num_nodes(), fill=0)
    for r in range(hashed.n_rows):
        var row = row_of(hashed, r)
        var leaf = predict_leaf(dt2, row, 0)
        assert_true(
            dt2.sparsetree[leaf].IsLeaf(), "predict must land on a leaf"
        )
        landed[leaf] += 1
    var total = 0
    for i in range(dt2.num_nodes()):
        total += landed[i]
    assert_equal(total, hashed.n_rows, "every row in exactly one leaf")
    assert_true(dt2.depth_counter <= 6, "max_depth respected on device")
    # leaf probabilities must still sum to one
    for i in range(dt2.num_nodes()):
        if not dt2.sparsetree[i].IsLeaf():
            continue
        var s = Float32(0.0)
        for c in range(Int(dt2.num_outputs)):
            s += dt2.leaf_value(i, c)
        assert_true(
            s > 0.999 and s < 1.001,
            "device leaf " + String(i) + " probabilities must sum to one",
        )
        cells += 1
    print(
        "    ",
        dt2.num_nodes(),
        "nodes,",
        dt2.leaf_counter,
        "leaves, depth",
        dt2.depth_counter,
        "-- every row in one leaf, every leaf sums to one",
    )
    cells += 3

    # ---------------- 4. the device path is deterministic ------------------
    var again = fit_device(ctx, hashed, pc2, 0xABCDEF, 4)
    assert_equal(
        again.num_nodes(), dt2.num_nodes(), "same seed, same device tree"
    )
    for i in range(again.num_nodes()):
        assert_true(
            again.sparsetree[i] == dt2.sparsetree[i],
            "the device path must be deterministic at node " + String(i),
        )
        cells += 1
    var other = fit_device(ctx, hashed, pc2, 0x777777, 4)
    var moved = other.num_nodes() != dt2.num_nodes()
    if not moved:
        for i in range(other.num_nodes()):
            if not (other.sparsetree[i] == dt2.sparsetree[i]):
                moved = True
    assert_true(moved, "a different seed must give a different device tree")
    cells += 1

    print("device_tree: ", cells, "cells")
    print("device_tree_check: PASS")
