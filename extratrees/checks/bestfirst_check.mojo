# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 466 to 469's gate: sklearn's `max_leaf_nodes`, and the four
things about it that could be quietly wrong.

THE FOUR CLAIMS, and how each is held to something that can go red.

1. **THE MODE IS OFF BY DEFAULT.** A caller who does not pass
   `max_leaf_nodes` must get the bits they got before this mode existed.
   Two arms hold it, and neither is a tautology. (a) `DEFAULT_FINGERPRINT`
   and `DEFAULT_FINGERPRINT_DEVICE` below are 64-bit digests of a default
   forest on each arm, PINNED FROM A PRE-PATCH RUN of
   `bestfirst_fingerprint.mojo` -- which imports nothing this change added
   and therefore runs on either side of it. Unpinned zeros FAIL this check
   with the observed values printed, so the gate cannot be accepted by
   leaving them blank. The DEVICE digest is the one that matters: the host
   arm gained only a dispatch at the top of `train_classification`, while
   the device driver's level cycle was edited in place with the depth-wise
   code moved inside an `else`. (b) The default queue's frontier must be
   EMPTY and `bestfirst_enabled()` False, with the SAME constructor
   returning True when the field is set -- the reach-negative for every
   line the mode added, and its complement.

2. **`max_leaf_nodes=k` YIELDS EXACTLY k LEAVES.** Counted off the fitted
   `sparsetree` -- `left_child_id == -1` -- on a fixture large enough that
   the frontier cannot run dry, on the host arm and the device arm, for
   classification and regression. `BESTFIRST_SAB_NO_BUDGET` is the sabotage:
   ignoring the budget at pop time must change the leaf count.

3. **BEST-FIRST IS NOT DEPTH-WISE.** The comparison is against cuML's
   `max_leaves` at the SAME BUDGET, which is the only comparison that
   isolates the thing under test: both arms stop at k leaves, so any
   difference in the tree is a difference in WHICH k nodes were expanded and
   in what order. `BESTFIRST_SAB_FIFO` is the sabotage -- it turns the
   priority queue back into cuML's deque, and the best-first tree must then
   stop differing from the depth-wise one.

4. **THE TIE RULE IS THE TIE RULE.** DEVIATION 468's order is exercised at
   the frontier itself, because an exact float tie between two nodes cannot
   be planted through a random split search: the draws are keyed per node,
   so two nodes of a real fit never have the same (count, gain) by
   construction. The frontier is a data structure with a total order on it,
   and this file admits records into one directly -- two with EQUAL keys and
   different node ids, and a size/gain pair built so the SCALED key and the
   RAW gain disagree. `BESTFIRST_SAB_TIE_MAX_IDX` must reverse the first;
   `BESTFIRST_SAB_UNSCALED_KEY` must reverse the second.

WHY ARM 4 IS A UNIT ARM AND SAYS SO. A gate that could only be reached
through a whole fit would be a gate on a path no fixture can steer. The
frontier's order is a pure function of the records in it, so admitting the
records directly tests exactly the seam and nothing else -- and the fit-level
arms above are what prove that seam is on the fit's path at all.

NO DURATION IS TAKEN ANYWHERE IN THIS FILE.
"""

from std.testing import assert_equal, assert_true
from std.sys.info import has_accelerator
from max.gpu.host import DeviceContext

from extratrees.checks.fixtures import (
    Dataset as FixtureDataset,
    hashed_classification,
    hashed_regression,
)
from extratrees.estimator import (
    ExtraTreesConfig,
    MAX_FEATURES_ALL,
    fit_extra_trees_classifier,
    fit_extra_trees_classifier_device,
    fit_extra_trees_regressor,
    fit_extra_trees_regressor_device,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    BESTFIRST_SAB_FIFO,
    BESTFIRST_SAB_NONE,
    BESTFIRST_SAB_NO_BUDGET,
    BESTFIRST_SAB_TIE_MAX_IDX,
    BESTFIRST_SAB_UNSCALED_KEY,
    NodeQueue,
    frontier_key,
    train_classification_bestfirst,
    train_regression_bestfirst,
)
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import Split
from extratrees.impl.randomforest.randomforest import Forest


# ==========================================================================
# CLAIM 1(a)'s PIN. Replace with the value `bestfirst_fingerprint.mojo`
# prints ON THE PRE-PATCH TREE, on the box that will run this check. Zero is
# the UNPINNED sentinel and this file fails on it by design: a default-bits
# gate that passes while unpinned is not a gate.
#
#   git stash                                   # or check out the parent
#   pixi run mojo run -I . \
#       extratrees/checks/bestfirst_fingerprint.mojo
#   git stash pop
#   <paste DEFAULT_FINGERPRINT_HOST and DEFAULT_FINGERPRINT_DEVICE here>
#
# BOTH ARE NEEDED AND THE DEVICE ONE IS THE ONE THAT MATTERS. The host arm
# gained only a dispatch at the top of `train_classification`; the device
# driver's LEVEL CYCLE was edited in place, with the depth-wise code moved
# inside an `else`. A host-only pin would gate the easier half.
#
# The probe imports nothing this change added, so it runs unmodified on
# either side. The fixture, the config and the seed are fixed in the probe
# and are repeated here; the two must not drift, which is why the probe
# prints them too.
# ==========================================================================
# PINNED 2026-09-01 on an Apple M4, from `git stash` of this change plus the
# untracked probe: 882 nodes over 6 trees, host and device equal (which is
# the pre-existing host/device identity claim, not something this pin adds).
comptime DEFAULT_FINGERPRINT: UInt64 = 14675911462422329739
comptime DEFAULT_FINGERPRINT_DEVICE: UInt64 = 14675911462422329739

comptime FP_SEED: UInt64 = 0xB3E5F1
comptime FP_ROWS: Int = 512
comptime FP_COLS: Int = 6
comptime FP_CLASSES: Int = 3
comptime FP_TREES: Int32 = 6
comptime FP_DEPTH: Int32 = 7

comptime K: Int32 = 11
"""The leaf budget every arm below uses. Chosen so the frontier CANNOT run
dry on a 512-row fixture at depth 7 -- a budget the tree could not spend
would make "exactly k" pass for the wrong reason -- and so the depth-wise
control has to stop mid-level, which is where the two growth orders part."""


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


def mix64(h_in: UInt64, v: UInt64) -> UInt64:
    """One splitmix64 step over an accumulator. A DIGEST, not a hash with a
    security claim: all it has to do is change when any input bit changes."""
    var h = h_in ^ v
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB
    return h ^ (h >> 31)


def forest_fingerprint(forest: Forest) -> UInt64:
    """Every node field and every leaf bit of every tree, in tree order.

    Fields are folded as their EXACT BITS (`to_bits`), never as printed
    floats, so a one-ULP move in a threshold or a leaf value moves the
    digest. That is the property claim 1(a) rests on.
    """
    var h = UInt64(0x243F6A8885A308D3)
    h = mix64(h, UInt64(len(forest.trees)))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        h = mix64(h, UInt64(Int(tree.treeid)))
        h = mix64(h, UInt64(Int(tree.depth_counter)))
        h = mix64(h, UInt64(Int(tree.leaf_counter)))
        h = mix64(h, UInt64(Int(tree.num_outputs)))
        h = mix64(h, UInt64(tree.num_nodes()))
        for i in range(tree.num_nodes()):
            ref n = tree.sparsetree[i]
            # The three integer fields are masked to 32 bits rather than
            # widened: `colid` and `left_child_id` are -1 on a leaf, and a
            # sign-extended -1 would fold in 32 bits of 1s that say nothing.
            # (Mojo int widening sign-extends -- the traps register.)
            h = mix64(h, UInt64(Int(n.colid) & 0xFFFFFFFF))
            h = mix64(h, UInt64(n.quesval.to_bits[DType.uint32]()))
            h = mix64(h, UInt64(n.best_metric_val.to_bits[DType.uint32]()))
            h = mix64(h, UInt64(Int(n.left_child_id) & 0xFFFFFFFF))
            h = mix64(h, UInt64(Int(n.instance_count) & 0xFFFFFFFF))
        h = mix64(h, UInt64(len(tree.vector_leaf)))
        for i in range(len(tree.vector_leaf)):
            h = mix64(h, UInt64(tree.vector_leaf[i].to_bits[DType.uint32]()))
    return h


def count_leaves(tree: TreeMetaDataNode[DType.float32]) -> Int:
    """Leaves counted off the STRUCTURE (`left_child_id == -1`), never off
    `leaf_counter`. The counter is what the builder maintains, so checking
    the budget against it would be checking the budget against itself."""
    var n = 0
    for i in range(tree.num_nodes()):
        if tree.sparsetree[i].left_child_id == -1:
            n += 1
    return n


def min_leaves(forest: Forest) -> Int:
    var m = -1
    for t in range(len(forest.trees)):
        var c = count_leaves(forest.trees[t])
        if m < 0 or c < m:
            m = c
    return m


def max_leaves_of(forest: Forest) -> Int:
    var m = 0
    for t in range(len(forest.trees)):
        var c = count_leaves(forest.trees[t])
        if c > m:
            m = c
    return m


def forest_node_diffs(got: Forest, want: Forest) raises -> Int:
    """Nodes that differ, across the forest. A tree-count or node-count
    mismatch counts every node of both trees as differing."""
    if len(got.trees) != len(want.trees):
        return len(got.trees) + len(want.trees)
    var d = 0
    for t in range(len(got.trees)):
        if got.trees[t].num_nodes() != want.trees[t].num_nodes():
            d += got.trees[t].num_nodes() + want.trees[t].num_nodes()
            continue
        for i in range(got.trees[t].num_nodes()):
            if not (
                got.trees[t].sparsetree[i] == want.trees[t].sparsetree[i]
            ):
                d += 1
    return d


def tree_node_diffs(
    got: TreeMetaDataNode[DType.float32],
    want: TreeMetaDataNode[DType.float32],
) -> Int:
    if got.num_nodes() != want.num_nodes():
        return got.num_nodes() + want.num_nodes()
    var d = 0
    for i in range(got.num_nodes()):
        if not (got.sparsetree[i] == want.sparsetree[i]):
            d += 1
    return d


def bf_config(k: Int32) -> ExtraTreesConfig:
    """The classification config every fit arm below shares, with sklearn's
    best-first budget set. `max_features=all` so the two growth modes are
    compared on the SAME candidate columns per node and the difference
    between them cannot be a sampling difference."""
    var c = ExtraTreesConfig()
    c.n_estimators = FP_TREES
    c.max_depth = FP_DEPTH
    c.max_features_spec = MAX_FEATURES_ALL
    c.random_state = FP_SEED
    c.max_leaf_nodes = k
    return c^


def cuml_cap_config(k: Int32) -> ExtraTreesConfig:
    """The same config with cuML's cap instead of sklearn's budget: the SAME
    leaf ceiling reached by the depth-wise frontier. Claim 3's control."""
    var c = bf_config(k)
    c.max_leaf_nodes = -1
    c.max_leaves = k
    return c^




@fieldwise_init
struct Fitted(Movable):
    """A single host tree plus the buffers `Dataset` points into.

    `Dataset` holds raw pointers, and Mojo frees a `List` at its LAST USE, so
    the three lists have to outlive the fit and be returned with the tree.
    `tree_check.mojo` carries the same struct for the same reason.
    """

    var tree: TreeMetaDataNode[DType.float32]
    var x_col: List[Float32]
    var labels: List[Float32]
    var row_ids: List[Int32]


def fit_one(
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    is_classification: Bool,
    bf_sabotage: Int32,
) raises -> Fitted:
    """ONE tree through the host best-first arm, on ITS OWN row list.

    A fresh `row_ids` per call is not tidiness: `train_*` partitions that
    list IN PLACE, so a second fit sharing it would start from the first
    fit's permutation and every arm below that compares two fits would be
    comparing two different problems.
    """
    var x_col = column_major(fixture)
    var labels = List[Float32]()
    for r in range(fixture.n_rows):
        if is_classification:
            labels.append(Float32(Int(fixture.label[r])))
        else:
            labels.append(fixture.y[r])
    var row_ids = List[Int32]()
    for r in range(fixture.n_rows):
        row_ids.append(Int32(r))
    var num_outputs = Int32(fixture.n_classes) if is_classification else Int32(1)
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x_col.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        Int32(fixture.n_rows),
        Int32(fixture.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        num_outputs,
    )
    var tree: TreeMetaDataNode[DType.float32]
    if is_classification:
        tree = train_classification_bestfirst(
            dataset, params, Int32(0), seed, Int32(fixture.n_classes),
            rescue=True, bf_sabotage=bf_sabotage,
        )
    else:
        tree = train_regression_bestfirst(
            dataset, params, Int32(0), seed,
            rescue=True, bf_sabotage=bf_sabotage,
        )
    return Fitted(tree^, x_col^, labels^, row_ids^)


def make_split(gain: Float32, n_left: Int32, colid: Int32) -> Split:
    """A candidate `split_not_valid` accepts at the defaults: a gain at or
    above `min_impurity_decrease = 0`, and both children non-empty."""
    return Split(Float32(0.5), colid, gain, n_left)


def admit_record(
    mut q: NodeQueue[DType.float32],
    idx: Int32,
    count: Int32,
    gain: Float32,
) raises -> Bool:
    """One synthetic frontier record. Only `bestfirst_expand` reads
    `node_instances`, and nothing here expands, so an item may name a node
    id the tree has not allocated."""
    var item = NodeWorkItem(idx, Int32(1), InstanceRange(Int32(0), count))
    return q.bestfirst_admit(
        item, make_split(gain, count // Int32(2), Int32(0)), Int32(0)
    )


def pop_order(mut q: NodeQueue[DType.float32]) raises -> List[Int32]:
    """Drain the frontier; node ids in pop order."""
    var out = List[Int32]()
    while len(q.frontier) > 0:
        out.append(q.bestfirst_pop().item.idx)
    return out^


def frontier_queue(
    sab: Int32
) raises -> NodeQueue[DType.float32]:
    """A queue whose frontier can be driven directly.

    `total_rows` is 1024, which is the denominator every key below is
    computed against, and `max_leaf_nodes` is large enough that
    `bestfirst_budget_left` never gates a pop -- so these arms measure the
    ORDER and only the order.
    """
    var p = DecisionTreeParams()
    p.max_depth = 8
    p.max_leaf_nodes = 1000
    var q = NodeQueue[DType.float32](p, Int32(1024), Int32(2), Int32(0))
    q.bf_sabotage = sab
    return q^


def order_string(ids: List[Int32]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i > 0:
            s += ","
        s += String(Int(ids[i]))
    return s + "]"


def main() raises:
    comptime assert has_accelerator(), "the device arms need a GPU"
    var ctx = DeviceContext()
    var cells = 0

    print("[bestfirst] DEVIATION 466-469 on", ctx.name())

    var clf = hashed_classification(FP_SEED, FP_ROWS, FP_COLS, FP_CLASSES)
    var xc = column_major(clf)
    var lab = float_labels(clf)
    var reg = hashed_regression(FP_SEED, FP_ROWS, FP_COLS)
    var xr = column_major(reg)

    # =====================================================================
    # 1. THE MODE IS OFF BY DEFAULT
    # =====================================================================
    var default_fit = fit_extra_trees_classifier(
        xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        bf_config(-1),
    )
    var default_dev = fit_extra_trees_classifier_device(
        ctx, xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        bf_config(-1),
    )
    var fp = forest_fingerprint(default_fit.forest)
    var fpd = forest_fingerprint(default_dev.forest)
    print("    default forest fingerprint: host", fp, "device", fpd)
    assert_true(
        DEFAULT_FINGERPRINT != 0 and DEFAULT_FINGERPRINT_DEVICE != 0,
        "DEFAULT_FINGERPRINT is UNPINNED. Run"
        " extratrees/checks/bestfirst_fingerprint.mojo on the PRE-PATCH"
        " tree, paste that number into this file, and re-run. Do NOT paste"
        " the value printed above: it is this tree's, and pinning it would"
        " make the gate say only that the fit is deterministic.",
    )
    assert_true(
        fp == DEFAULT_FINGERPRINT,
        "THE DEFAULT FIT MOVED. Every node field and every leaf bit of a"
        " six-tree forest is folded into this digest, so this assertion IS"
        " the claim that best-first growth is off unless it is asked for."
        " Nothing in DEVIATION 466 is admissible while it is red.",
    )
    assert_true(
        fpd == DEFAULT_FINGERPRINT_DEVICE,
        "THE DEFAULT DEVICE FIT MOVED. This is the half that matters: the"
        " device driver's level cycle was edited in place and the"
        " depth-wise code moved inside an `else`, so this digest is what"
        " says the move was textual and not behavioural.",
    )
    cells += 2

    # The reach-NEGATIVE for the same claim. The frontier IS the mode, so a
    # default fit must never put anything on it, and the SAME constructor
    # with the field set must.
    var pdef = DecisionTreeParams()
    pdef.max_depth = FP_DEPTH
    var qdef = NodeQueue[DType.float32](
        pdef, Int32(FP_ROWS), Int32(FP_CLASSES), Int32(0)
    )
    assert_true(
        not qdef.bestfirst_enabled(),
        "the default params must not select best-first growth",
    )
    assert_equal(
        len(qdef.frontier), 0, "the default queue's frontier starts empty"
    )
    assert_true(
        qdef.has_work(),
        "the default queue still puts the root on the FIFO frontier -- if"
        " this went red the two frontiers would have swapped roles",
    )
    var pbf = pdef.copy()
    pbf.max_leaf_nodes = 8
    var qbf = NodeQueue[DType.float32](
        pbf, Int32(FP_ROWS), Int32(FP_CLASSES), Int32(0)
    )
    assert_true(
        qbf.bestfirst_enabled(),
        "COMPLEMENT ARM: with max_leaf_nodes set, the SAME constructor must"
        " select the mode. If this and the assert above cannot disagree,"
        " the dispatch is not reading the field at all",
    )
    cells += 2

    # =====================================================================
    # 2. `max_leaf_nodes=k` YIELDS EXACTLY k LEAVES
    # =====================================================================
    var host_bf = fit_extra_trees_classifier(
        xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        bf_config(K),
    )
    assert_equal(
        min_leaves(host_bf.forest), Int(K),
        "host classification: every tree must have EXACTLY max_leaf_nodes"
        " leaves on a fixture this size",
    )
    assert_equal(
        max_leaves_of(host_bf.forest), Int(K),
        "host classification: and no tree may exceed it",
    )
    var dev_bf = fit_extra_trees_classifier_device(
        ctx, xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        bf_config(K),
    )
    assert_equal(
        min_leaves(dev_bf.forest), Int(K),
        "device classification: exactly k",
    )
    assert_equal(
        max_leaves_of(dev_bf.forest), Int(K),
        "device classification: no tree exceeds k",
    )
    var rbase = bf_config(K)
    var rcfg = rbase.for_regression()
    var host_bfr = fit_extra_trees_regressor(
        xr, reg.y, Int32(FP_ROWS), Int32(FP_COLS), rcfg
    )
    assert_equal(
        min_leaves(host_bfr.forest), Int(K), "host regression: exactly k"
    )
    var dev_bfr = fit_extra_trees_regressor_device(
        ctx, xr, reg.y, Int32(FP_ROWS), Int32(FP_COLS), rcfg
    )
    assert_equal(
        min_leaves(dev_bfr.forest), Int(K), "device regression: exactly k"
    )
    cells += 6

    # THE DEVICE ARM IS THE HOST ARM. `device_forest_check` makes this claim
    # for depth-wise growth; it has to be made again for the mode, because
    # the merged frontier now carries a priority queue per tree and a driver
    # that popped a different node would show here and nowhere else.
    assert_equal(
        forest_node_diffs(dev_bf.forest, host_bf.forest), 0,
        "the device best-first forest must be the host best-first forest,"
        " node for node",
    )
    cells += 1

    # SABOTAGE: ignore the leaf budget at pop time. The COUNT must move.
    var pbudget = DecisionTreeParams()
    pbudget.max_depth = FP_DEPTH
    pbudget.max_leaf_nodes = K
    pbudget.max_features = 1.0
    var f_budget = fit_one(clf, pbudget, FP_SEED, True, BESTFIRST_SAB_NONE)
    assert_equal(
        count_leaves(f_budget.tree), Int(K),
        "the single-tree host arm honours the budget too",
    )
    var f_nobudget = fit_one(
        clf, pbudget, FP_SEED, True, BESTFIRST_SAB_NO_BUDGET
    )
    assert_true(
        count_leaves(f_nobudget.tree) > Int(K),
        "BESTFIRST_SAB_NO_BUDGET did not grow the tree past the budget."
        " Then the budget is not what stops the expansion and 'exactly k'"
        " above is measuring something else",
    )
    print(
        "    budget sabotage:", count_leaves(f_budget.tree), "->",
        count_leaves(f_nobudget.tree), "leaves",
    )
    cells += 2

    # =====================================================================
    # 3. BEST-FIRST IS NOT DEPTH-WISE, AT THE SAME BUDGET
    # =====================================================================
    # The control is cuML's `max_leaves` at the SAME ceiling, which is the
    # only control that isolates the thing under test: both arms stop at k
    # leaves, so a difference in the tree is a difference in WHICH k nodes
    # were expanded and in what order, and cannot be a difference in size.
    var cap_fit = fit_extra_trees_classifier(
        xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        cuml_cap_config(K),
    )
    assert_equal(
        max_leaves_of(cap_fit.forest), Int(K),
        "the control arm must reach the SAME leaf ceiling, or the two"
        " forests would differ for a reason that is not the ORDER",
    )
    var mode_diff = forest_node_diffs(host_bf.forest, cap_fit.forest)
    assert_true(
        mode_diff > 0,
        "best-first and depth-wise produced the SAME forest at the same"
        " leaf budget. Then the priority queue decides nothing and the"
        " mode is accepted rather than load bearing",
    )
    print("    best-first vs cuML cap:", mode_diff, "nodes differ")
    cells += 2

    # SABOTAGE: order the frontier by ARRIVAL instead of by improvement --
    # cuML's deque wearing sklearn's name. The tree must move. This is the
    # arm that says the improvement KEY, and not merely the one-node-per-
    # cycle shape, is what separates the two modes.
    var f_fifo = fit_one(clf, pbudget, FP_SEED, True, BESTFIRST_SAB_FIFO)
    var fifo_diff = tree_node_diffs(f_fifo.tree, f_budget.tree)
    assert_true(
        fifo_diff > 0,
        "BESTFIRST_SAB_FIFO did not move the tree -- the improvement key is"
        " then not ordering the frontier and DEVIATION 467 is inert",
    )
    assert_equal(
        count_leaves(f_fifo.tree), Int(K),
        "and the FIFO sabotage must still spend the same budget, so the"
        " difference above is the ORDER and not the size",
    )
    print("    FIFO sabotage moved", fifo_diff, "nodes")
    cells += 2

    # =====================================================================
    # 4. THE TIE RULE AND THE KEY, AT THE FRONTIER
    # =====================================================================
    # (a) TWO EQUAL KEYS. Same count, same gain, different node ids: the
    #     smallest id pops first, and the sabotage reverses it.
    var qt = frontier_queue(BESTFIRST_SAB_NONE)
    assert_true(
        admit_record(qt, Int32(9), Int32(64), Float32(0.25)), "admit 9"
    )
    assert_true(
        admit_record(qt, Int32(4), Int32(64), Float32(0.25)), "admit 4"
    )
    assert_true(
        admit_record(qt, Int32(7), Int32(64), Float32(0.25)), "admit 7"
    )
    var ord_tie = pop_order(qt)
    assert_equal(len(ord_tie), 3, "three records in, three out")
    assert_equal(
        Int(ord_tie[0]), 4, "equal keys: the SMALLEST node id pops first"
    )
    assert_equal(Int(ord_tie[1]), 7, "then 7")
    assert_equal(Int(ord_tie[2]), 9, "then 9")
    var qt2 = frontier_queue(BESTFIRST_SAB_TIE_MAX_IDX)
    _ = admit_record(qt2, Int32(9), Int32(64), Float32(0.25))
    _ = admit_record(qt2, Int32(4), Int32(64), Float32(0.25))
    _ = admit_record(qt2, Int32(7), Int32(64), Float32(0.25))
    var ord_sab = pop_order(qt2)
    assert_equal(
        Int(ord_sab[0]), 9,
        "BESTFIRST_SAB_TIE_MAX_IDX must reverse the tie arm. If it cannot,"
        " the third arm of DEVIATION 468's order is dead code and the"
        " assertion above is passing for some other reason",
    )
    assert_equal(Int(ord_sab[2]), 4, "and 4 must go last under it")
    print(
        "    tie order", order_string(ord_tie), "-> sabotaged",
        order_string(ord_sab),
    )
    cells += 3

    # (b) THE SCALING IS LOAD BEARING. A big node with a small gain against
    #     a small node with a big gain, built so the two orders disagree:
    #       big:   800 of 1024 rows, gain 0.10  ->  improvement 0.078125
    #       small:  16 of 1024 rows, gain 0.90  ->  improvement 0.0140625
    #     On the RAW gain the small node wins; on sklearn's improvement the
    #     big one does. That is exactly what `impurity_improvement`'s
    #     `w_node / w_total` factor is for.
    var k_big = frontier_key(Float32(0.10), Int32(800), Int32(1024))
    var k_small = frontier_key(Float32(0.90), Int32(16), Int32(1024))
    assert_true(
        k_big > k_small,
        "the scaled-key fixture is not built as described: the BIG node"
        " must win on the improvement",
    )
    assert_true(
        Float32(0.90) > Float32(0.10),
        "and the SMALL node must win on the raw gain",
    )
    var qk = frontier_queue(BESTFIRST_SAB_NONE)
    _ = admit_record(qk, Int32(2), Int32(800), Float32(0.10))
    _ = admit_record(qk, Int32(3), Int32(16), Float32(0.90))
    var ord_key = pop_order(qk)
    assert_equal(
        Int(ord_key[0]), 2,
        "the frontier must pop by sklearn's IMPROVEMENT, which is the gain"
        " scaled by the node's share of the tree's rows",
    )
    var qk2 = frontier_queue(BESTFIRST_SAB_UNSCALED_KEY)
    _ = admit_record(qk2, Int32(2), Int32(800), Float32(0.10))
    _ = admit_record(qk2, Int32(3), Int32(16), Float32(0.90))
    var ord_key_sab = pop_order(qk2)
    assert_equal(
        Int(ord_key_sab[0]), 3,
        "BESTFIRST_SAB_UNSCALED_KEY must flip it. If dropping the"
        " count/total factor changes nothing, the factor is not in the key"
        " and DEVIATION 467 is a comment rather than an implementation",
    )
    cells += 4

    # (c) AN INVALID SPLIT IS NOT ADMITTED. Their `is_leaf` frontier records
    #     have no counterpart here (DEVIATION BLOCK 466), so the rule that
    #     keeps such nodes off the heap has to hold, and its complement has
    #     to fire on the SAME node with a valid split.
    var qv = frontier_queue(BESTFIRST_SAB_NONE)
    var bad_item = NodeWorkItem(
        Int32(5), Int32(1), InstanceRange(Int32(0), Int32(32))
    )
    assert_true(
        not qv.bestfirst_admit(
            bad_item, make_split(Float32(0.5), Int32(0), Int32(0)), Int32(0)
        ),
        "a split with an EMPTY left child must be refused admission",
    )
    assert_equal(len(qv.frontier), 0, "and must leave the frontier empty")
    assert_true(
        qv.bestfirst_admit(
            bad_item, make_split(Float32(0.5), Int32(16), Int32(0)), Int32(0)
        ),
        "COMPLEMENT ARM: the SAME node with a VALID split must be admitted,"
        " or the arm above passes because nothing is ever admitted",
    )
    cells += 2

    # =====================================================================
    # 5. REGRESSION GROWS BEST-FIRST TOO. Reach is per branch, and
    #    `train_regression_bestfirst` is a separate function body.
    # =====================================================================
    var preg = DecisionTreeParams()
    preg.max_depth = FP_DEPTH
    preg.max_leaf_nodes = K
    preg.max_features = 1.0
    preg.split_criterion = Int32(CRITERION_MSE)
    var f_reg = fit_one(reg, preg, FP_SEED, False, BESTFIRST_SAB_NONE)
    assert_equal(
        count_leaves(f_reg.tree), Int(K),
        "the regression best-first arm honours the budget",
    )
    var f_reg_fifo = fit_one(reg, preg, FP_SEED, False, BESTFIRST_SAB_FIFO)
    assert_true(
        tree_node_diffs(f_reg_fifo.tree, f_reg.tree) > 0,
        "BESTFIRST_SAB_FIFO must move the REGRESSION tree too; the two"
        " trainers are separate bodies and a sabotage that only reaches"
        " one of them gates only one of them",
    )
    cells += 2

    print("bestfirst: ", cells, "cells")
    print("bestfirst_check: PASS")
