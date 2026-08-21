"""A FOREST grown on the GPU, against the same forest grown on the host.

`device_tree_check` established the tree-level claim: with deviation 183 closed
the device tree and the host tree are BIT-IDENTICAL, gate and all. This file is
its forest-level counterpart, and it exists because three things can be wrong
one layer up that every single-tree check stays green for.

**THE TREES CAN BE COPIES OF EACH OTHER.** `fit_classification_device` passes
`i` as the tree id, and the tree id enters the split key (`key_for(seed,
tree_id, node_id, feature_id)`, deviation 130) exactly so that identical data
yields different thresholds. Pass a constant instead and you get `n_trees`
copies of one tree — which predicts fine, votes unanimously, and is not a
forest. The device path has its own way to lose the tree id, because the id
crosses a kernel boundary as a scalar argument rather than staying in a host
loop variable. So the trees are compared to each other, pairwise, ALL pairs.

**THE VOTE CAN BE THE LAST TREE INSTEAD OF THE AVERAGE.** `forest_vote` is
shared by both arms, and deviation 147's accumulating `predict_one` is what
makes it a forest sum. A zeroing wrapper there returns the last tree's leaf
vector and is invisible on any fixture where the trees agree — so the vote is
checked per row and per class against a tally written here, on a fixture whose
trees are first REQUIRED to disagree.

**THE DEVICE ARM CAN QUIETLY BE THE HOST ARM.** This is the failure this file
worried about most, and it is peculiar to a device path that is bit-identical
to its host path: THE OUTPUT CANNOT TELL THEM APART. If
`fit_extra_trees_classifier_device` called `fit_classification`, every identity
assertion below would still pass. Reach is therefore proved by MECHANISM, not
by output — see `[reach]`, which uses a refusal only the device trainer raises,
and the leaf array's single device writer.

**`best_metric_val` IS EXCLUDED from every node comparison**, stated here
rather than quietly skipped. Deviation 183's last paragraph is the reason: the
host's is cuML's `GainPerSplit` accumulated in `Float32` from float reciprocals
(`objectives.cuh:52-83`), the device path's is the same quantity formed in
`Float64` from exact integers. They agree far inside the gate's resolution but
not in the last bits. What is NOT excluded is the DECISION that field drives —
`split_not_valid`'s gate — which arm 1b runs at its default and requires to
agree.

SABOTAGED, ONE PER MECHANISM, AND THE ONE THAT STAYED GREEN
------------------------------------------------------------
Each was applied, seen, and reverted:

* **the tree id replaced by a constant** in `fit_classification_device` --
  RED: 13 of 24 trees differed in node count and 276 of 563 nodes differed;
* **the vote zeroed before each tree** in `forest_vote` -- RED at the first
  per-(row, class) cell;
* **the estimator device arm rewired to `fit_classification`** -- RED, and
  ONLY at `[reach]`. Every identity assertion above it passed while the device
  arm was secretly the host arm, which is the whole reason `[reach]` is not
  an output comparison;
* **the class-id cast permuted** (`+1 mod n_classes`) in `class_ids_for` --
  RED at the LEAF VALUES, with 0 nodes differing. A cyclic relabelling leaves
  Gini invariant, so the same splits win and only the leaves move; the
  structural comparison alone would have stayed green;
* **the device regressor forwarding to the host regressor** instead of
  refusing -- RED (deviation 188);
* **a shared `row_ids` across all trees -- STAYED GREEN, and that is a
  MEASURED FACT about the device path rather than a defect in this file.**
  `train_classification_device` uploads `row_ids` once and partitions
  `d_row_ids` on the DEVICE; nothing copies the permutation back, so its `mut`
  is currently vacuous and a shared buffer cannot corrupt anything. Rather
  than contrive a fixture that would fail, the file MEASURES the equivalence
  (a whole 12-tree forest fitted from one shared buffer, compared node by node
  against the per-tree forest) and PINS the fact that makes it safe -- that
  `row_ids` comes back unchanged. Sabotaging THAT pin (writing one element
  after the call, as a device write-back would) turns it red, so the pin is
  live. Deviation 185.

NO DURATION IS TAKEN ANYWHERE IN THIS FILE. Perf is deferred by the repo owner
and a timing number from this lane would be a rule violation.
"""

from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext
from std.sys import has_accelerator

from extratrees.estimator import (
    ExtraTreesConfig,
    MAX_FEATURES_ALL,
    fit_extra_trees_classifier,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor,
    fit_extra_trees_regressor_device,
)
from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    analytic_all_constant,
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.ported.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_leaf,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification_device,
)
from extratrees.ported.randomforest.randomforest import (
    Forest,
    class_ids_for,
    fit_classification,
    fit_classification_device,
    forest_vote,
    predict_class_forest,
)


def column_major(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def float_labels(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32]()
    for r in range(fixture.n_rows):
        out.append(Float32(Int(fixture.label[r])))
    return out^


def row_of(fixture: FixtureDataset, r: Int) -> List[Float32]:
    var row = List[Float32]()
    for c in range(fixture.n_cols):
        row.append(fixture.value(r, c))
    return row^


def nodes_agree(
    a: TreeMetaDataNode[DType.float32],
    b: TreeMetaDataNode[DType.float32],
    i: Int,
) -> Bool:
    """The four fields that ARE compared. `best_metric_val` is not one of them;
    see the module docstring."""
    var x = a.sparsetree[i]
    var y = b.sparsetree[i]
    return (
        x.ColumnId() == y.ColumnId()
        and x.QueryValue().to_bits() == y.QueryValue().to_bits()
        and x.LeftChildId() == y.LeftChildId()
        and x.InstanceCount() == y.InstanceCount()
    )


def trees_differ(forest: Forest, a: Int, b: Int) -> Bool:
    if forest.trees[a].num_nodes() != forest.trees[b].num_nodes():
        return True
    for i in range(forest.trees[a].num_nodes()):
        if not (forest.trees[a].sparsetree[i] == forest.trees[b].sparsetree[i]):
            return True
    return False


def device_refused(
    ctx: DeviceContext,
    x: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
) -> Bool:
    try:
        _ = fit_extra_trees_classifier_device(
            ctx, x, labels, n_rows, n_cols, n_classes, config
        )
        return False
    except:
        return True


def host_refused(
    x: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
) -> Bool:
    try:
        _ = fit_extra_trees_classifier(
            x, labels, n_rows, n_cols, n_classes, config
        )
        return False
    except:
        return True


def main() raises:
    comptime assert has_accelerator(), "the device forest needs a GPU"
    var ctx = DeviceContext()
    var cells = 0

    print("[device] a FOREST grown on", ctx.name())

    var hashed = hashed_classification(0xF0E57, 384, 5, 3)
    var xc = column_major(hashed)
    var labels = float_labels(hashed)

    # ================= 1. the device forest IS the host forest ============
    # Tree for tree, node for node, and every leaf value, bit for bit.
    #
    # ARM 1a runs with the zero-gain gate DISABLED on both arms
    # (`min_impurity_decrease = -1.0`, which cuML's non-negative Gini gain can
    # never trip) so that a difference can only come from the split search.
    # ARM 1b runs at the DEFAULT gate, which is deviation 183's claim: the
    # device forms the gain on the host from exact integers, so
    # `split_not_valid` applies unchanged. Keeping both localises a break --
    # 1b red with 1a green is the gate, both red is the search.
    print("[identity] the device forest against the host forest, node by node")
    var arm_names = ["1a gate DISABLED", "1b gate at its DEFAULT"]
    var gates = [Float32(-1.0), Float32(0.0)]
    var arm_nodes = [0, 0]
    for arm in range(2):
        var configs = 0
        var tree_count = 0
        var count_diff = 0
        var struct_diff = 0
        var leaf_diff = 0
        var leaf_cells = 0
        var leaf_nonzero = 0
        var pred_diff = 0
        var total_nodes = 0
        for depth in [Int32(4), Int32(6)]:
            for seed_i in range(2):
                var p = DecisionTreeParams()
                p.max_depth = depth
                p.max_features = 0.5
                p.min_impurity_decrease = gates[arm]
                var seed = UInt64(seed_i * 104729 + 11)
                var hf = fit_classification(
                    xc,
                    labels,
                    Int32(hashed.n_rows),
                    Int32(hashed.n_cols),
                    Int32(hashed.n_classes),
                    p,
                    6,
                    seed,
                )
                var df = fit_classification_device(
                    ctx,
                    xc,
                    labels,
                    Int32(hashed.n_rows),
                    Int32(hashed.n_cols),
                    Int32(hashed.n_classes),
                    p,
                    6,
                    seed,
                )
                configs += 1
                assert_equal(
                    len(df.trees),
                    len(hf.trees),
                    "the device forest must have as many trees as the host's",
                )
                assert_equal(df.n_trees, hf.n_trees)
                assert_equal(df.num_outputs, hf.num_outputs)
                for t in range(len(hf.trees)):
                    tree_count += 1
                    if hf.trees[t].num_nodes() != df.trees[t].num_nodes():
                        count_diff += 1
                        continue
                    for i in range(hf.trees[t].num_nodes()):
                        total_nodes += 1
                        if not nodes_agree(hf.trees[t], df.trees[t], i):
                            struct_diff += 1
                        cells += 1
                    for i in range(len(hf.trees[t].vector_leaf)):
                        if (
                            hf.trees[t].vector_leaf[i].to_bits()
                            != df.trees[t].vector_leaf[i].to_bits()
                        ):
                            leaf_diff += 1
                        if df.trees[t].vector_leaf[i] != Float32(0.0):
                            leaf_nonzero += 1
                        leaf_cells += 1
                # and whether any of it reaches a prediction
                for r in range(0, hashed.n_rows, 3):
                    var row = row_of(hashed, r)
                    if predict_class_forest(hf, row, 0) != predict_class_forest(
                        df, row, 0
                    ):
                        pred_diff += 1
                    cells += 1
        arm_nodes[arm] = total_nodes
        print(
            "    ",
            arm_names[arm],
            "--",
            configs,
            "forests,",
            tree_count,
            "trees,",
            count_diff,
            "differ in node COUNT;",
            struct_diff,
            "of",
            total_nodes,
            "nodes differ in (colid, quesval, left_child, instance_count);",
            leaf_diff,
            "of",
            leaf_cells,
            "leaf values differ (",
            leaf_nonzero,
            "nonzero );",
            pred_diff,
            "forest predictions differ",
        )
        assert_equal(
            count_diff,
            0,
            arm_names[arm]
            + ": every device tree must have the same node count as the host"
            " tree with the same id",
        )
        assert_equal(
            struct_diff,
            0,
            arm_names[arm]
            + ": the device forest and the host forest must be the SAME"
            " forest -- every node agreeing in (colid, quesval,"
            " left_child_id, instance_count)",
        )
        assert_equal(
            leaf_diff,
            0,
            arm_names[arm]
            + ": and every leaf value, bit for bit -- they are ratios of"
            " integer class counts",
        )
        assert_equal(
            pred_diff,
            0,
            arm_names[arm]
            + ": and therefore every averaged forest prediction",
        )
        assert_true(total_nodes > 0, "the comparison must have run")
        assert_true(
            leaf_nonzero > 0,
            arm_names[arm]
            + ": the device leaf array is memset to 0.0 and written by exactly"
            " one writer, leaf_kernel. A forest with no nonzero leaf value"
            " would be a forest whose leaf kernel never ran.",
        )
        cells += 6
    # The gate must be DOING something, or arm 1b asserts an equality no
    # zero-gain split ever tested.
    assert_true(
        arm_nodes[0] > arm_nodes[1],
        "arm 1a (gate disabled) grew "
        + String(arm_nodes[0])
        + " nodes and arm 1b (gate on) grew "
        + String(arm_nodes[1])
        + "; if those are equal the gate never fired on this fixture and 1b"
        " is measuring nothing",
    )
    cells += 1

    # ================= 2. REACH: this is not the host arm in disguise =====
    # The device forest is bit-identical to the host forest, so NO OUTPUT can
    # distinguish the two paths. Three mechanisms do.
    print("[reach] the device path is reached, proved by mechanism")

    # (a) A refusal ONLY the device trainer raises. `train_classification_device`
    #     refuses a class count above the score kernel's comptime shared width
    #     (deviation 172); the host trainer has no such bound. Same data, same
    #     config, opposite outcomes -- which is impossible if the device arm is
    #     forwarding to the host arm.
    var wide = hashed_classification(0xC1A55, 96, 3, 33)
    var wx = column_major(wide)
    var wl = float_labels(wide)
    var wcfg = ExtraTreesConfig()
    wcfg.n_estimators = 1
    wcfg.max_depth = 3
    wcfg.max_features_spec = MAX_FEATURES_ALL
    var host_took_33 = not host_refused(
        wx, wl, Int32(wide.n_rows), Int32(wide.n_cols), Int32(33), wcfg
    )
    var device_refused_33 = device_refused(
        ctx, wx, wl, Int32(wide.n_rows), Int32(wide.n_cols), Int32(33), wcfg
    )
    assert_true(
        host_took_33,
        "the HOST arm has no class-count bound and must accept 33 classes;"
        " if it refuses, this cell no longer distinguishes the two paths",
    )
    assert_true(
        device_refused_33,
        "33 classes must be REFUSED on the device arm (deviation 172, the"
        " score kernel's comptime shared width). This is the only thing that"
        " tells the device arm apart from the host arm, because their forests"
        " are bit-identical -- if this passes while the host accepts, the"
        " estimator's device option is calling the host trainer.",
    )
    # and the raw forest entry point, same mechanism
    var raw_refused = False
    try:
        _ = fit_classification_device(
            ctx,
            wx,
            wl,
            Int32(wide.n_rows),
            Int32(wide.n_cols),
            Int32(33),
            DecisionTreeParams(),
            1,
            7,
        )
    except:
        raw_refused = True
    assert_true(
        raw_refused,
        "fit_classification_device must propagate the device trainer's own"
        " refusal rather than swallowing it",
    )
    cells += 3
    print(
        "     host accepted 33 classes; the device arm and the raw device fit"
        " both refused it by name"
    )

    # (b) The leaf array. `d_leaves` is created, memset to 0.0, and written by
    #     exactly one writer -- `leaf_kernel`, launched on the GPU. Arm 1
    #     counted its nonzero values and required them, so a run in which no
    #     kernel executed cannot reach this line.
    # (c) The split decisions themselves. Every `colid`, `quesval` and
    #     `left_child_id` above came back out of `r_c`, `r_q` and the host
    #     partition of a DEVICE-partitioned `row_ids`; they matched the host
    #     forest over every node compared. Uninitialised memory does not match
    #     a host tree.
    print(
        "     leaf values and split decisions came off device buffers whose"
        " only writers are kernels"
    )

    # ================= 3. the trees must not be copies ====================
    print("[diversity] the trees of a DEVICE forest must differ from each other")
    var p12 = DecisionTreeParams()
    p12.max_depth = 6
    p12.max_features = 0.5
    var dforest = fit_classification_device(
        ctx,
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p12,
        12,
        0xABC123,
    )
    assert_equal(len(dforest.trees), 12)
    var identical_pairs = 0
    var pairs = 0
    for a in range(len(dforest.trees)):
        for b in range(a + 1, len(dforest.trees)):
            pairs += 1
            if not trees_differ(dforest, a, b):
                identical_pairs += 1
            cells += 1
    print("    ", pairs, "pairs,", identical_pairs, "identical")
    assert_equal(
        identical_pairs,
        0,
        "a forest whose trees are copies is one tree reported n_trees times --"
        " the tree id must reach the DEVICE split key (deviation 130)",
    )
    cells += 1

    # ================= 4. the trees actually DISAGREE =====================
    # Without this the vote check below is vacuous: if every tree returns the
    # same leaf vector, an average and a last-tree-wins are the same number.
    var disagreeing_rows = 0
    for r in range(hashed.n_rows):
        var row = row_of(hashed, r)
        var first = predict_leaf(dforest.trees[0], row, 0)
        var v0 = dforest.trees[0].leaf_value(first, 0)
        for t in range(1, len(dforest.trees)):
            var leaf = predict_leaf(dforest.trees[t], row, 0)
            if dforest.trees[t].leaf_value(leaf, 0) != v0:
                disagreeing_rows += 1
                break
    print(
        "    ",
        disagreeing_rows,
        "of",
        hashed.n_rows,
        "rows have DEVICE trees that disagree",
    )
    assert_true(
        disagreeing_rows * 2 > hashed.n_rows,
        "most rows must have disagreeing trees, or the vote check cannot"
        " distinguish an average from the last tree",
    )
    cells += 1

    # ================= 5. the vote is an average ==========================
    print("[vote] the device forest's vote, against an independent tally")
    var vote_cells = 0
    for r in range(0, hashed.n_rows, 11):
        var row = row_of(hashed, r)
        var got = forest_vote(dforest, row, 0)
        # INDEPENDENT: route the row through each tree by hand, take each
        # tree's leaf vector, and average them HERE rather than in the code
        # under test.
        var want = List[Float32](
            length=Int(dforest.num_outputs), fill=Float32(0.0)
        )
        for t in range(len(dforest.trees)):
            var leaf = predict_leaf(dforest.trees[t], row, 0)
            for k in range(Int(dforest.num_outputs)):
                want[k] += dforest.trees[t].leaf_value(leaf, k)
        var s = Float32(0.0)
        for k in range(Int(dforest.num_outputs)):
            want[k] = want[k] / Float32(Int(dforest.n_trees))
            assert_equal(
                got[k].to_bits(),
                want[k].to_bits(),
                "row "
                + String(r)
                + " class "
                + String(k)
                + ": the device forest's vote is not the average of its"
                " trees (deviation 147: predict_one ACCUMULATES and the only"
                " zeroing is one level up)",
            )
            s += got[k]
            vote_cells += 1
            cells += 1
        assert_true(
            s > 0.999 and s < 1.001,
            "the averaged vote must sum to one",
        )
        cells += 1
    print("    ", vote_cells, "per-(row, class) cells")

    # ================= 6. the analytic fixtures ===========================
    print("[analytic] a separable fixture, classified exactly by the device")
    var gap = analytic_separable_gap(0x5EED).data.copy()
    var gx = column_major(gap)
    var gl = float_labels(gap)
    var pg = DecisionTreeParams()
    pg.max_depth = 8
    var gforest = fit_classification_device(
        ctx,
        gx,
        gl,
        Int32(gap.n_rows),
        Int32(gap.n_cols),
        Int32(gap.n_classes),
        pg,
        10,
        0x24680,
    )
    var wrong = 0
    for r in range(gap.n_rows):
        var row = row_of(gap, r)
        if predict_class_forest(gforest, row, 0) != Int(gap.label[r]):
            wrong += 1
    assert_equal(
        wrong,
        0,
        "a 10-tree DEVICE forest on a fixture whose classes have an empty"
        " band between them must be exact",
    )
    for t in range(len(gforest.trees)):
        assert_true(
            gforest.trees[t].num_nodes() >= 3,
            "and every tree in it must actually split",
        )
        cells += 1
    print("    10 device trees, 0 of", gap.n_rows, "rows wrong")

    # nothing is splittable: every tree must be one leaf, and the forest must
    # still vote.
    var flat = analytic_all_constant().data.copy()
    var fx = column_major(flat)
    var fl = float_labels(flat)
    var fforest = fit_classification_device(
        ctx,
        fx,
        fl,
        Int32(flat.n_rows),
        Int32(flat.n_cols),
        Int32(flat.n_classes),
        pg,
        4,
        0x1357,
    )
    for t in range(len(fforest.trees)):
        assert_equal(
            fforest.trees[t].num_nodes(),
            1,
            "nothing is splittable, so every device tree is one leaf",
        )
        cells += 1
    var frow = row_of(flat, 0)
    var fvote = forest_vote(fforest, frow, 0)
    var fs = Float32(0.0)
    for k in range(Int(fforest.num_outputs)):
        fs += fvote[k]
    assert_true(fs > 0.999 and fs < 1.001, "a forest of leaves still votes")
    cells += 1
    print("    all-constant: 4 device trees, one node each, vote sums to one")

    # ================= 7. DEVIATION 185, pinned rather than argued ========
    # The device trainer declares `mut row_ids`, uploads it once, partitions
    # `d_row_ids` on the DEVICE across every level, and never copies the
    # permutation back. So the `mut` is currently vacuous. Both halves of that
    # are asserted, because the second one is what makes a shared buffer safe
    # TODAY and the first is what will make it unsafe LATER.
    print("[deviation 185] row_ids: the mut is vacuous, and that is pinned")
    var pin_rows = List[Int32]()
    for r in range(hashed.n_rows):
        pin_rows.append(Int32(r))
    var pin_ids = class_ids_for(
        labels, Int32(hashed.n_rows), Int32(hashed.n_classes)
    )
    var pin_tree = train_classification_device(
        ctx,
        xc,
        pin_ids,
        pin_rows,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p12,
        0,
        0xABC123,
    )
    var moved_rows = 0
    for r in range(hashed.n_rows):
        if pin_rows[r] != Int32(r):
            moved_rows += 1
    assert_equal(
        moved_rows,
        0,
        "train_classification_device must leave the HOST row_ids untouched --"
        " it partitions d_row_ids on the device and never copies back. When"
        " this goes red the device partition has become visible on the host"
        " and a SHARED row_ids across trees is now silent cross-tree"
        " corruption; deviation 185 in randomforest.mojo is then stale.",
    )
    assert_true(pin_tree.num_nodes() > 1, "and it must still have built a tree")
    cells += 2

    # ... and the measurement that says so: a whole forest fitted from ONE
    # shared buffer, compared against the per-tree forest, bit for bit.
    var shared_rows = List[Int32]()
    for r in range(hashed.n_rows):
        shared_rows.append(Int32(r))
    var shared_trees = List[TreeMetaDataNode[DType.float32]]()
    for tree_id in range(12):
        shared_trees.append(
            train_classification_device(
                ctx,
                xc,
                pin_ids,
                shared_rows,
                Int32(hashed.n_rows),
                Int32(hashed.n_cols),
                Int32(hashed.n_classes),
                p12,
                Int32(tree_id),
                0xABC123,
            )
        )
    var shared_diff = 0
    var shared_nodes = 0
    for t in range(12):
        if shared_trees[t].num_nodes() != dforest.trees[t].num_nodes():
            shared_diff += 1
            continue
        for i in range(shared_trees[t].num_nodes()):
            shared_nodes += 1
            if not nodes_agree(shared_trees[t], dforest.trees[t], i):
                shared_diff += 1
    print(
        "     a 12-tree forest from ONE shared row_ids differs from the"
        " per-tree forest in",
        shared_diff,
        "of",
        shared_nodes,
        "nodes -- the mut is vacuous, MEASURED",
    )
    assert_equal(
        shared_diff,
        0,
        "with the device partition invisible on the host, a shared row_ids"
        " must give the identical forest. A nonzero count here means the"
        " device trainer HAS started writing row_ids back and the per-tree"
        " buffer is now load-bearing rather than contractual.",
    )
    cells += 1

    # ================= 8. the estimator's device option ===================
    print("[estimator] the device option, and every refusal on it")
    var cfg = ExtraTreesConfig()
    cfg.n_estimators = 6
    cfg.max_depth = 6
    cfg.random_state = 0xABC123
    var dfit = fit_extra_trees_classifier_device(
        ctx,
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        cfg,
    )
    assert_equal(len(dfit.forest.trees), 6)
    # The estimator's forest must be the forest the raw device entry point
    # builds from the plan it resolved -- no policy of its own.
    var direct = fit_classification_device(
        ctx,
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        dfit.plan.params,
        dfit.plan.n_trees,
        cfg.random_state,
    )
    var est_diff = 0
    var est_nodes = 0
    for t in range(len(direct.trees)):
        if direct.trees[t].num_nodes() != dfit.forest.trees[t].num_nodes():
            est_diff += 1
            continue
        for i in range(direct.trees[t].num_nodes()):
            est_nodes += 1
            if not nodes_agree(direct.trees[t], dfit.forest.trees[t], i):
                est_diff += 1
        for i in range(len(direct.trees[t].vector_leaf)):
            if (
                direct.trees[t].vector_leaf[i].to_bits()
                != dfit.forest.trees[t].vector_leaf[i].to_bits()
            ):
                est_diff += 1
    assert_equal(
        est_diff,
        0,
        "the estimator's device option must produce exactly the forest"
        " fit_classification_device produces from the plan it resolved",
    )
    # ... and the same forest the HOST estimator arm produces, which is the
    # user-visible form of the identity claim.
    var hfit = fit_extra_trees_classifier(
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        cfg,
    )
    var arm_diff = 0
    for t in range(len(hfit.forest.trees)):
        if (
            hfit.forest.trees[t].num_nodes()
            != dfit.forest.trees[t].num_nodes()
        ):
            arm_diff += 1
            continue
        for i in range(hfit.forest.trees[t].num_nodes()):
            if not nodes_agree(hfit.forest.trees[t], dfit.forest.trees[t], i):
                arm_diff += 1
    assert_equal(
        arm_diff,
        0,
        "and the same forest the HOST estimator arm produces",
    )
    assert_equal(
        dfit.plan.max_features_count,
        hfit.plan.max_features_count,
        "both arms resolve max_features through the same resolve()",
    )
    assert_equal(
        dfit.depth_cap_bound,
        hfit.depth_cap_bound,
        "and both report the depth cap the same way",
    )
    print(
        "     ",
        est_nodes,
        "nodes: estimator-device == raw device == estimator-host",
    )
    cells += 5

    # Every refusal, on the DEVICE arm, one named case each. These are the
    # same eleven `estimator_check` enumerates on the host arm; the point here
    # is that the device arm applies them too, and applies them BEFORE it
    # reaches a device.
    var sx = column_major(hashed)
    var refused = 0
    var tried = 0

    var c1 = cfg.copy()
    c1.min_weight_fraction_leaf = 0.1
    var c2 = cfg.copy()
    c2.has_monotonic_cst = True
    var c3 = cfg.copy()
    c3.has_class_weight = True
    var c4 = cfg.copy()
    c4.bootstrap = True
    var c5 = cfg.copy()
    c5.oob_score = True
    var c6 = cfg.copy()
    c6.max_samples_set = True
    var c7 = cfg.copy()
    c7.warm_start = True
    var c8 = cfg.copy()
    c8.ccp_alpha = 0.01
    var c9 = cfg.copy()
    c9.max_leaf_nodes = 8
    var c10 = cfg.copy()
    c10.n_estimators = 0
    var c11 = cfg.copy()
    c11.criterion = CRITERION_ENTROPY
    var c12 = cfg.copy()
    c12.max_features_spec = Int(hashed.n_cols + 4)

    for bad in [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]:
        tried += 1
        if device_refused(
            ctx,
            sx,
            labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            bad,
        ):
            refused += 1
        cells += 1
    assert_equal(
        refused,
        tried,
        "every parameter the HOST arm refuses must be refused on the DEVICE"
        " arm too -- a parameter honoured on one path and inert on the other"
        " is what estimator.mojo exists to prevent",
    )
    print("    ", refused, "of", tried, "unported parameters refused by name")

    # The default config is ACCEPTED on the device arm -- rule 8 wants both
    # sides of every switch, and a refusal count is meaningless if everything
    # is refused.
    assert_true(
        not device_refused(
            ctx,
            sx,
            labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            cfg,
        ),
        "the default configuration must be ACCEPTED on the device arm",
    )
    cells += 1

    # Regression on the device is refused BY NAME (deviation 188), and the
    # HOST regressor still works -- so the refusal is about the device path,
    # not about regression.
    var rcfg = ExtraTreesConfig().for_regression()
    rcfg.n_estimators = 3
    rcfg.max_depth = 4
    var ry = List[Float32]()
    for r in range(hashed.n_rows):
        ry.append(hashed.y[r])
    var reg_refused = False
    try:
        _ = fit_extra_trees_regressor_device(
            ctx, xc, ry, Int32(hashed.n_rows), Int32(hashed.n_cols), rcfg
        )
    except:
        reg_refused = True
    assert_true(
        reg_refused,
        "there is no device regressor and it must be refused BY NAME rather"
        " than silently fitting on the host (deviation 188)",
    )
    var rfit = fit_extra_trees_regressor(
        xc, ry, Int32(hashed.n_rows), Int32(hashed.n_cols), rcfg
    )
    assert_equal(
        len(rfit.forest.trees),
        3,
        "and the HOST regressor must still fit, so the refusal is about the"
        " device path and not about regression",
    )
    assert_equal(rcfg.criterion, CRITERION_MSE)
    cells += 3

    # ================= 9. determinism, and that the seed does something ===
    var again = fit_classification_device(
        ctx,
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p12,
        12,
        0xABC123,
    )
    for t in range(12):
        assert_equal(
            again.trees[t].num_nodes(),
            dforest.trees[t].num_nodes(),
            "the same seed must give the same device forest",
        )
        for i in range(again.trees[t].num_nodes()):
            assert_true(
                again.trees[t].sparsetree[i] == dforest.trees[t].sparsetree[i],
                "the device forest must be deterministic",
            )
        cells += 1
    var other = fit_classification_device(
        ctx,
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p12,
        12,
        0x777777,
    )
    var seed_changed = False
    for t in range(12):
        if other.trees[t].num_nodes() != dforest.trees[t].num_nodes():
            seed_changed = True
        else:
            for i in range(other.trees[t].num_nodes()):
                if not (
                    other.trees[t].sparsetree[i] == dforest.trees[t].sparsetree[i]
                ):
                    seed_changed = True
    assert_true(
        seed_changed, "a different seed must give a different device forest"
    )
    cells += 1

    # ================= 10. the forest's own refusals ======================
    var forest_refusals = 0
    try:
        _ = fit_classification_device(
            ctx,
            xc,
            labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p12,
            4,
            1,
            True,  # bootstrap
        )
    except:
        forest_refusals += 1
    try:
        _ = fit_classification_device(
            ctx,
            xc,
            labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p12,
            0,
            1,
        )
    except:
        forest_refusals += 1
    try:
        _ = fit_classification_device(
            ctx,
            xc,
            labels,
            0,
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p12,
            4,
            1,
        )
    except:
        forest_refusals += 1
    # a label that truncates outside [0, n_classes) -- deviation 186's guard,
    # which the HOST arm does not have.
    var bad_labels = labels.copy()
    bad_labels[3] = Float32(Int(hashed.n_classes) + 4)
    try:
        _ = fit_classification_device(
            ctx,
            xc,
            bad_labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p12,
            2,
            1,
        )
    except:
        forest_refusals += 1
    assert_equal(
        forest_refusals,
        4,
        "bootstrap=True, n_trees=0, n_rows=0 and an out-of-range class id"
        " must each be refused BY NAME on the device forest",
    )
    cells += 4

    _ = xc.unsafe_ptr()
    _ = labels.unsafe_ptr()
    _ = gx.unsafe_ptr()
    _ = gl.unsafe_ptr()
    _ = fx.unsafe_ptr()
    _ = fl.unsafe_ptr()
    _ = wx.unsafe_ptr()
    _ = wl.unsafe_ptr()
    _ = sx.unsafe_ptr()
    _ = ry.unsafe_ptr()
    _ = bad_labels.unsafe_ptr()
    _ = pin_ids.unsafe_ptr()
    _ = pin_rows.unsafe_ptr()
    _ = shared_rows.unsafe_ptr()

    print("device_forest: ", cells, "cells")
    print("device_forest_check: PASS")
