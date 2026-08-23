"""A REGRESSION tree grown on the GPU, against the host regression tree.

The classification device path is bit-identical to its host counterpart
(`device_tree_check`). This is the same claim for regression, and it cannot be
the same equality, for one reason that is a ruling rather than a shortfall:

**DEVIATION 135 puts the device's regression accumulator in FIXED POINT and
leaves the host oracle in `Float64`.** The host trainer sums labels as
`Float64` to match sklearn; the device sums quantized integers, exactly and
order-independently. So the two agree on the SPLITS — which are decided by
cuML's MSE gain over integer sums (deviation 189) — and differ on the LEAF
VALUES by at most the quantization step, which deviation 179 measures at half a
step.

What is asserted here:

1. the analytic step fixture is fitted EXACTLY by the device, at every seed:
   `y` is one constant below an empty band and another above it, so every
   threshold the RNG can draw inside the band reproduces it and no
   quantization can blur a constant;
2. the device and host trees agree on STRUCTURE — `colid`, `quesval`,
   `left_child_id`, `instance_count` — because the split decision is made on
   integers on both sides;
3. the leaf values agree to within the quantization step, and the actual
   difference is REPORTED rather than assumed;
4. every row lands in exactly one leaf, `max_depth` is respected, the same seed
   reproduces the tree and a different seed does not;
5. the ESTIMATOR's device arm (`fit_extra_trees_regressor_device`, deviation
   188 closed) produces the host estimator arm's structure with leaves within
   one quantization step, at least one leaf MOVED (the reach proof: a device
   arm that silently served the host fit would return bit-equal leaves), and
   the shared `regressor_plan` refusals fire through the device arm.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext

from extratrees.mojo_only.fixed_point import choose_scale, quantize
from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    analytic_regression_step,
    hashed_regression,
)
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.ported.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_leaf,
    predict_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_regression,
    train_regression_device,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.estimator import (
    ExtraTreesConfig,
    fit_extra_trees_regressor,
    fit_extra_trees_regressor_device,
)


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


def quantized_labels(
    fixture: FixtureDataset,
) raises -> Tuple[List[Int32], Float64]:
    """The label vector as fixed point, and the scale that produced it.

    `choose_scale` takes the sum of magnitudes over the WHOLE dataset, because
    any node's rows are a subset of it -- deviation 135's bound, which is what
    makes overflow impossible rather than unlikely.
    """
    var mag = Float64(0.0)
    for r in range(fixture.n_rows):
        var v = Float64(fixture.y[r])
        mag += v if v >= 0.0 else -v
    var scale = choose_scale(mag, fixture.n_rows)
    var q = List[Int32]()
    for r in range(fixture.n_rows):
        q.append(Int32(quantize(Float64(fixture.y[r]), scale)))
    return (q^, scale)


def fit_device(
    ctx: DeviceContext,
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    tree_id: Int32,
) raises -> TreeMetaDataNode[DType.float32]:
    var x = column_major(fixture)
    var ql = quantized_labels(fixture)
    var rows = List[Int32]()
    for r in range(fixture.n_rows):
        rows.append(Int32(r))
    var tree = train_regression_device(
        ctx,
        x,
        ql[0],
        ql[1],
        rows,
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        params,
        tree_id,
        seed,
    )
    _ = x.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return tree^


def fit_host(
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    tree_id: Int32,
) raises -> TreeMetaDataNode[DType.float32]:
    var x = column_major(fixture)
    var y = List[Float32]()
    for r in range(fixture.n_rows):
        y.append(fixture.y[r])
    var rows = List[Int32]()
    for r in range(fixture.n_rows):
        rows.append(Int32(r))
    var ds = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](y.unsafe_ptr()),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](rows.unsafe_ptr()),
        Int32(1),
    )
    var tree = train_regression(ds, params, tree_id, seed)
    _ = x.unsafe_ptr()
    _ = y.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return tree^


def main() raises:
    var ctx = DeviceContext()
    print("[device] a REGRESSION tree grown on", ctx.name())
    var cells = 0

    # ---- 1. the analytic step, fitted exactly ---------------------------
    print("[analytic] a two-level step with an empty band must be exact")
    var step = analytic_regression_step(0xBEE5).data.copy()
    var p = DecisionTreeParams()
    p.max_depth = 8
    p.split_criterion = CRITERION_MSE
    for seed_i in range(6):
        var dt = fit_device(ctx, step, p, UInt64(seed_i * 104729 + 7), 0)
        var off = 0
        for r in range(step.n_rows):
            var row = row_of(step, r)
            if predict_regression(dt, row, 0) != step.y[r]:
                off += 1
        assert_equal(
            off,
            0,
            "seed "
            + String(seed_i)
            + ": the DEVICE must reproduce a step function whose two levels"
            " have an empty band between them -- a constant survives"
            " quantization exactly",
        )
        assert_true(dt.num_nodes() >= 3, "and it must actually split")
        cells += 2
    print("    6 seeds, every one of", step.n_rows, "rows exact")

    # ---- 2. structure against the host, leaves within a step -------------
    print("[identity] structure exact, leaves within the quantization step")
    var hashed = hashed_regression(0xD3C1DE, 512, 5)
    var ql = quantized_labels(hashed)
    var q_step = 1.0 / ql[1]
    print("    quantization step is", q_step)

    var total_nodes = 0
    var struct_diff = 0
    var shape_diff = 0
    var leaf_cells = 0
    var leaf_out = 0
    var worst_leaf = Float64(0.0)
    for depth in [Int32(3), Int32(5), Int32(7)]:
        for seed_i in range(3):
            var pc = DecisionTreeParams()
            pc.max_depth = depth
            pc.max_features = 0.5
            pc.split_criterion = CRITERION_MSE
            pc.min_impurity_decrease = -1.0
            var seed = UInt64(seed_i * 7919 + 3)
            var ht = fit_host(hashed, pc, seed, 2)
            var dt = fit_device(ctx, hashed, pc, seed, 2)
            if ht.num_nodes() != dt.num_nodes():
                shape_diff += 1
                continue
            for i in range(ht.num_nodes()):
                total_nodes += 1
                var a = ht.sparsetree[i]
                var b = dt.sparsetree[i]
                if (
                    a.ColumnId() != b.ColumnId()
                    or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                    or a.LeftChildId() != b.LeftChildId()
                    or a.InstanceCount() != b.InstanceCount()
                ):
                    struct_diff += 1
                cells += 1
                if not a.IsLeaf():
                    continue
                var d = Float64(ht.vector_leaf[i]) - Float64(
                    dt.vector_leaf[i]
                )
                if d < 0.0:
                    d = -d
                if d > worst_leaf:
                    worst_leaf = d
                # A leaf value is a MEAN of quantized labels, so it can be off
                # by at most one quantization step, and deviation 179 measures
                # the typical case at half a step.
                if d > q_step:
                    leaf_out += 1
                leaf_cells += 1
    print(
        "    ",
        shape_diff,
        "differ in node COUNT;",
        struct_diff,
        "of",
        total_nodes,
        "nodes differ in structure;",
        leaf_out,
        "of",
        leaf_cells,
        "leaf values are further than one quantization step apart, worst",
        worst_leaf,
    )
    assert_equal(
        shape_diff,
        0,
        "the regression split decision is made on INTEGER sums on both sides"
        " (deviations 135 and 189), so the trees must have the same shape",
    )
    assert_equal(
        struct_diff,
        0,
        "and every node must agree in (colid, quesval, left_child_id,"
        " instance_count)",
    )
    assert_equal(
        leaf_out,
        0,
        "a leaf value is a mean of quantized labels, so it cannot be further"
        " than ONE quantization step from the Float64 host mean; further than"
        " that is a defect, not a rounding",
    )
    assert_true(
        worst_leaf > 0.0,
        "if no leaf value differs at all then this fixture's labels are exact"
        " multiples of the scale and the comparison is measuring nothing --"
        " that exact fixture defect was found once already, in deviation 179",
    )
    cells += 4

    # ---- 3. invariants and determinism -----------------------------------
    var pd = DecisionTreeParams()
    pd.max_depth = 6
    pd.split_criterion = CRITERION_MSE
    var dt2 = fit_device(ctx, hashed, pd, 0xABCDEF, 4)
    var landed = List[Int](length=dt2.num_nodes(), fill=0)
    for r in range(hashed.n_rows):
        var row = row_of(hashed, r)
        var leaf = predict_leaf(dt2, row, 0)
        assert_true(dt2.sparsetree[leaf].IsLeaf())
        landed[leaf] += 1
    var total = 0
    for i in range(dt2.num_nodes()):
        total += landed[i]
    assert_equal(total, hashed.n_rows, "every row in exactly one leaf")
    assert_true(dt2.depth_counter <= 6, "max_depth respected")
    cells += 2

    var again = fit_device(ctx, hashed, pd, 0xABCDEF, 4)
    assert_equal(again.num_nodes(), dt2.num_nodes())
    for i in range(again.num_nodes()):
        assert_true(
            again.sparsetree[i] == dt2.sparsetree[i],
            "the device regression path must be deterministic",
        )
        cells += 1
    var other = fit_device(ctx, hashed, pd, 0x777777, 4)
    var moved = other.num_nodes() != dt2.num_nodes()
    if not moved:
        for i in range(other.num_nodes()):
            if not (other.sparsetree[i] == dt2.sparsetree[i]):
                moved = True
    assert_true(moved, "a different seed must give a different tree")
    cells += 1
    print(
        "    ",
        dt2.num_nodes(),
        "nodes, depth",
        dt2.depth_counter,
        "-- every row in one leaf, deterministic, seed-sensitive",
    )

    # ---- 4. the ESTIMATOR's device arm (deviation 188, closed) -----------
    # `fit_extra_trees_regressor_device` against `fit_extra_trees_regressor`,
    # through the sklearn surface rather than the raw trainers. The two arms
    # share `regressor_plan`, so what is asserted here is the part that CAN
    # drift: the device arm's own quantization and its trainer.
    #
    # REACH is proved by the leaf values, not by the structure: the device's
    # leaves are means of QUANTIZED labels (deviation 135) and the host's are
    # Float64 means, so at least one leaf MUST differ. An arm that silently
    # served the host fit under the device name would return bit-equal
    # leaves. Sabotaged once to prove the proof: with the body forwarding to
    # `fit_regression` instead, this section went red on exactly that
    # assertion, and with the quantization scale halved it went red on the
    # one-step bound. Both reverted.
    print("[estimator] fit_extra_trees_regressor_device vs the host arm")
    var config = ExtraTreesConfig().for_regression()
    config.n_estimators = 3
    config.max_depth = 5
    config.random_state = 0xE57
    var x_est = column_major(hashed)
    var y_est = List[Float32]()
    for r in range(hashed.n_rows):
        y_est.append(hashed.y[r])
    var host_fit = fit_extra_trees_regressor(
        x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), config
    )
    var dev_fit = fit_extra_trees_regressor_device(
        ctx, x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), config
    )
    assert_equal(
        Int(dev_fit.forest.n_trees),
        Int(host_fit.forest.n_trees),
        "same n_estimators both arms",
    )
    var est_struct_diff = 0
    var est_leaf_out = 0
    var est_leaf_moved = 0
    for t in range(len(host_fit.forest.trees)):
        assert_equal(
            host_fit.forest.trees[t].num_nodes(),
            dev_fit.forest.trees[t].num_nodes(),
            "tree " + String(t) + ": same node count both arms",
        )
        for i in range(host_fit.forest.trees[t].num_nodes()):
            var a = host_fit.forest.trees[t].sparsetree[i]
            var b = dev_fit.forest.trees[t].sparsetree[i]
            if (
                a.ColumnId() != b.ColumnId()
                or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                or a.LeftChildId() != b.LeftChildId()
                or a.InstanceCount() != b.InstanceCount()
            ):
                est_struct_diff += 1
            cells += 1
            if not a.IsLeaf():
                continue
            var d = Float64(
                host_fit.forest.trees[t].vector_leaf[i]
            ) - Float64(dev_fit.forest.trees[t].vector_leaf[i])
            if d < 0.0:
                d = -d
            if d > q_step:
                est_leaf_out += 1
            if d > 0.0:
                est_leaf_moved += 1
    assert_equal(
        est_struct_diff,
        0,
        "the estimator arms must agree on structure, as the raw trainers do",
    )
    assert_equal(
        est_leaf_out,
        0,
        "and every leaf value within one quantization step of the host mean",
    )
    assert_true(
        est_leaf_moved > 0,
        "REACH: no leaf value differs at all, so the device arm did not"
        " quantize -- it served the HOST fit under the device name, which is"
        " the defect deviation 188's refusal existed to prevent",
    )
    cells += 3

    # The plan is shared, so one refusal through the DEVICE arm proves the
    # plan is consulted there at all (rule 8's both-sides is in
    # estimator_check for the host arm).
    var refused = False
    var bad = ExtraTreesConfig().for_regression()
    bad.max_samples = 50  # WITHOUT bootstrap -- the DEVIATION 460 sabotage arm
    try:
        _ = fit_extra_trees_regressor_device(
            ctx, x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), bad
        )
    except:
        refused = True
    assert_true(
        refused, "max_samples without bootstrap refused by name on the device arm"
    )
    # DEVIATION 460 on the regressor: bootstrap=True is honoured on BOTH arms,
    # the device structure equals the host structure (the draw is integer,
    # the split key exact), and the bootstrap forest differs from the
    # no-bootstrap one.
    var boot_cfg = ExtraTreesConfig().for_regression()
    boot_cfg.bootstrap = True
    boot_cfg.n_estimators = 4
    boot_cfg.max_depth = 6
    boot_cfg.random_state = 7
    var plain_cfg = boot_cfg.copy()
    plain_cfg.bootstrap = False
    var dboot = fit_extra_trees_regressor_device(
        ctx, x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), boot_cfg
    )
    var hboot = fit_extra_trees_regressor(
        x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), boot_cfg
    )
    var dplain = fit_extra_trees_regressor_device(
        ctx, x_est, y_est, Int32(hashed.n_rows), Int32(hashed.n_cols), plain_cfg
    )
    var boot_struct_diff = 0
    var boot_nodes = 0
    var boot_vs_plain_same = True
    for t in range(len(dboot.forest.trees)):
        if dboot.forest.trees[t].num_nodes() != hboot.forest.trees[t].num_nodes():
            boot_struct_diff += 1
        else:
            for i in range(dboot.forest.trees[t].num_nodes()):
                boot_nodes += 1
                var a = dboot.forest.trees[t].sparsetree[i]
                var b = hboot.forest.trees[t].sparsetree[i]
                if not (
                    a.ColumnId() == b.ColumnId()
                    and a.QueryValue().to_bits() == b.QueryValue().to_bits()
                    and a.LeftChildId() == b.LeftChildId()
                    and a.InstanceCount() == b.InstanceCount()
                ):
                    boot_struct_diff += 1
        if dboot.forest.trees[t].num_nodes() != dplain.forest.trees[t].num_nodes():
            boot_vs_plain_same = False
        else:
            for i in range(dboot.forest.trees[t].num_nodes()):
                if not (
                    dboot.forest.trees[t].sparsetree[i]
                    == dplain.forest.trees[t].sparsetree[i]
                ):
                    boot_vs_plain_same = False
    assert_equal(
        boot_struct_diff,
        0,
        "the device bootstrap regression forest must have the host bootstrap"
        " forest's structure, node for node (DEVIATION 460)",
    )
    assert_true(
        not boot_vs_plain_same,
        "REACH: bootstrap=True must build a different regression forest"
        " than bootstrap=False on the device arm",
    )
    assert_equal(Int(dboot.plan.n_sampled_rows), hashed.n_rows)
    print(
        "     bootstrap regressor: device == host structure over",
        boot_nodes,
        "nodes; differs from the no-bootstrap forest",
    )
    cells += 3
    var wrong_crit = ExtraTreesConfig()  # Gini, the classifier default
    refused = False
    try:
        _ = fit_extra_trees_regressor_device(
            ctx,
            x_est,
            y_est,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            wrong_crit,
        )
    except:
        refused = True
    assert_true(refused, "a classification criterion refused on the device arm")
    cells += 2
    print(
        "    ",
        len(host_fit.forest.trees),
        "trees structure-identical;",
        est_leaf_moved,
        "leaf values moved by quantization, none past one step",
    )

    print("device_regression: ", cells, "cells")
    print("device_regression_check: PASS")
