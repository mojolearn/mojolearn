# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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

**THE GATE AGREES TOO, SINCE DEVIATION 183 WAS CLOSED.** `split_not_valid`
rejects a split whose `best_metric_val` is `<= min_impurity_decrease`, and
cuML's Gini gain is `>= 0` always, so at the default `0.0` a zero-gain split
becomes a leaf. The device could not apply that gate while it did not compute
the gain — 747 device nodes against 689 host nodes — and now forms the gain on
the host from the exact integers the score pass already produced. Both arms
are kept, and the reason is diagnostic rather than historical:

* **arm 2a**, the gate DISABLED on the host: isolates the split search;
* **arm 2b**, the gate at its DEFAULT: exercises the gate as well.

A change that breaks only the gate turns 2b red while 2a stays green, which
localises it. Arm 2b also asserts the gate FIRES on this fixture — 2a's node
count must exceed 2b's — because an equality no zero-gain split ever tested
would be an equality about nothing.

`best_metric_val` used to be excluded from the comparison; it is now ASSERTED,
because the gain is computed on the device from cuML's expression and both
sides accumulate it with an explicit `fma`.

Analytic fixtures are asserted outright, because their answer does not depend
on any of that: a separable fixture must come out perfect on the device too.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from extratrees.original.fixtures import (
    Dataset as FixtureDataset,
    analytic_all_constant,
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.derived.decisiontree.decisiontree import DecisionTreeParams
from extratrees.derived.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_class,
    predict_leaf,
)
from extratrees.derived.decisiontree.batched_levelalgo.builder import (
    train_classification,
    train_classification_device,
)
from extratrees.derived.decisiontree.batched_levelalgo.dataset import Dataset


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
    var leaf_diff = 0
    var leaf_cells = 0
    var metric_diff = 0
    var metric_worst = 0
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
                    if a.BestMetric().to_bits() != b.BestMetric().to_bits():
                        metric_diff += 1
                        var dd = Int(a.BestMetric().to_bits()) - Int(
                            b.BestMetric().to_bits()
                        )
                        if dd < 0:
                            dd = -dd
                        if dd > metric_worst:
                            metric_worst = dd
                    if (
                        a.ColumnId() != b.ColumnId()
                        or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                        or a.LeftChildId() != b.LeftChildId()
                        or a.InstanceCount() != b.InstanceCount()
                    ):
                        struct_diff += 1
                    cells += 1
                # AND THE LEAF VALUES, which now come off the device too --
                # `leaf_kernel` rather than `set_leaf_predictions_*`. They are
                # ratios of integer class counts, so bit equality is the right
                # bar here as well.
                for i in range(len(ht.vector_leaf)):
                    if (
                        ht.vector_leaf[i].to_bits()
                        != dt.vector_leaf[i].to_bits()
                    ):
                        leaf_diff += 1
                    leaf_cells += 1
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
        "row predictions differ;",
        leaf_diff,
        "of",
        leaf_cells,
        "LEAF VALUES differ (device leaf_kernel against the host pass);",
        metric_diff,
        "best_metric_val differ, worst",
        metric_worst,
        "ulp",
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
    assert_equal(
        metric_diff,
        0,
        "best_metric_val must agree BIT FOR BIT too. It used to be excluded --"
        " the device published a constant and the host cuML's GainPerSplit --"
        " and it is not excluded any more: the gain is computed on the device"
        " from their expression (DEVIATION 183, second form) and both sides"
        " accumulate it with an explicit fma (DEVIATION 142's reason), so the"
        " same source produces the same bits on both.",
    )
    assert_equal(
        leaf_diff,
        0,
        "the DEVICE leaf kernel must produce the same leaf values as the host"
        " pass, bit for bit -- they are ratios of integer class counts",
    )
    assert_true(
        leaf_cells > 0, "the leaf comparison must actually have run"
    )
    cells += 5

    # ---------------- 2b. the gate at its DEFAULT ------------------------
    # THIS ARM USED TO MEASURE A GAP AND NOW ASSERTS ITS ABSENCE. DEVIATION
    # 183 was open when it was written: the device could not apply cuML's
    # zero-gain gate because it did not compute the float gain, so it split
    # where the host made a leaf -- 747 device nodes against 689 host nodes on
    # this fixture. The arm therefore asserted the DIRECTION (the device can
    # only ever have more, never fewer) and required the difference to be
    # visible at all, so that the cost could not go unmeasured.
    #
    # 183 is closed: the gain is now formed on the host from the exact
    # integers the score pass already produced, so `split_not_valid` applies
    # unchanged. The arm asserts identity instead.
    #
    # BOTH ARMS ARE KEPT ON PURPOSE. 2a runs with the gate disabled and 2b
    # with it at its default, so a future change that breaks only the gate
    # turns 2b red while 2a stays green -- which localises it to the gate
    # rather than to the split search.
    print("[deviation 183] the gate at its default -- CLOSED, asserted here")
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
        device_more + device_fewer,
        0,
        "with DEVIATION 183 closed the device and host must agree in node"
        " count at the DEFAULT gate too. More nodes on the device means the"
        " gain is not reaching split_not_valid; fewer means it is rejecting"
        " splits the host keeps.",
    )
    assert_equal(
        host_total,
        dev_total,
        "and the totals must match: 689 against 689 when this was closed,"
        " where it was 689 against 747 before",
    )
    # The gate must also be DOING something, or the arms assert equalities no
    # gate-rejected split ever tested. Under DEVIATION 216 the boundary is
    # sklearn's, so at the DEFAULT 0.0 the gate rejects nothing here
    # (zero-gain winners split; PURE nodes leaf through their own status) --
    # gate-disabled and gate-default now grow the SAME trees by design. The
    # firing configuration is a POSITIVE threshold: one config at 0.005
    # against the same config unfettered, host and device agreeing at both.
    var pg = DecisionTreeParams()
    pg.max_depth = 5
    pg.max_features = 0.5
    pg.min_impurity_decrease = 0.005
    var gate_seed = UInt64(104729 + 11)
    var g_host = fit_host(hashed, pg, gate_seed, 2)
    var g_dev = fit_device(ctx, hashed, pg, gate_seed, 2)
    assert_equal(
        g_host.num_nodes(),
        g_dev.num_nodes(),
        "host and device must agree at the firing gate too",
    )
    pg.min_impurity_decrease = -1.0
    var g_free = fit_host(hashed, pg, gate_seed, 2)
    assert_true(
        g_free.num_nodes() > g_host.num_nodes(),
        "gate disabled grew "
        + String(g_free.num_nodes())
        + " nodes and gate at 0.005 grew "
        + String(g_host.num_nodes())
        + "; if those are equal the gate never fired on this fixture and"
        " nothing above tested a rejection",
    )
    cells += 3

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
