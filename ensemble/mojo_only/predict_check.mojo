"""Does the ported flat-tree walk route every row to the node cuML's
walk routes it to -- and would this check notice if it did not?

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/mojo_only/predict_check.mojo

WHAT IS UNDER TEST. `DecisionTree.predict_one`
(`decisiontree.mojo`, mirroring `decisiontree.cuh:370-389`) and
`RandomForest.predict` (mirroring `randomforest.cuh:382-436`), against
hand-built flat trees and hand-written expected leaf indices. No
training, no dataset, no fitted model -- STANDING_ORDERS rule 4: gate
correctness on an analytic answer, never on a real dataset.

## HOW THIS CHECK IS BUILT, and why it is built that way

This repository has a scar that dictates the shape. A kernel once read
`0 wrong of 512` with uniform test data and `490 wrong of 512` with
hashed data -- same kernel, same parameters -- and two earlier checks
had reported it correct at exactly the failing configuration. A check
whose expected value is the same in every cell verifies the TOTAL and
nothing about PLACEMENT.

So:

  * **Leaf values are HASHED and SCATTERED**, one distinct value per
    (tree, node, output) triple, and the check ASSERTS they are pairwise
    distinct before it uses them (`assert_distinct_leaf_values`). If
    every leaf held the same number, a walk that terminated anywhere at
    all would pass.
  * **Internal-node leaf slots are POISONED.** `vector_leaf` is indexed
    by NODE id (`decisiontree.cuh:387`), so the slots belonging to split
    nodes are allocated and never read. They are filled with values no
    leaf holds, so a walk that stops one hop early reports poison rather
    than a plausible number.
  * **The comparison is per ROW against a hand-written expected leaf
    index**, not against a total, a mean or a digest.
  * **Every mechanism is SABOTAGED** and the check is required to MOVE.
    A digest cannot distinguish a working change from a no-op, and reach
    is per branch.

## THE FIXTURES, and the bug shape each one exists to catch

  A. BALANCED depth-3, 7 nodes, 2 columns. The ordinary case, and the
     carrier for the rows that land EXACTLY on `quesval`.
  B. SINGLE LEAF. `sparsetree = [leaf]`. The loop must test the leaf
     flag before reading any threshold, so this row's feature vector is
     never touched.
  C. UNBALANCED, 5 nodes, where node 1 is a LEAF and node 2 is a SPLIT.
     Left and right subtrees have different depths. An implementation
     that assumes a complete level-order tree -- children at `2i+1` and
     `2i+2` -- gets fixture A right and fixture C wrong.
  D. MULTI-OUTPUT (num_outputs = 3) over fixture A's shape, exercising
     the `idx * num_outputs + k` leaf indexing and the forest-level
     class-vote aggregation.

## THE ROWS THAT MATTER MOST

`decisiontree.cuh:379` is `row[n.ColumnId()] <= n.QueryValue()` and the
LEFT branch is the `<=` branch, so a row landing EXACTLY on `quesval`
goes LEFT. That is the single most common silent difference between two
tree implementations, and it is invisible to random test data because a
random float never lands on a threshold. Rows 0, 1, 2, 6 and 9 of
fixture A and rows 0 and 3 of fixture C are placed EXACTLY on a
threshold on purpose, at both levels of the tree in row 0's case. The
`SAB_COMPARE` sabotage below flips `<=` to `<` and NOTHING ELSE, so the
number of rows it moves is exactly the number of rows sitting on a
threshold -- the check prints that number, and if it were zero this file
would be testing nothing about the tie.

Rows are also placed BELOW the smallest threshold and ABOVE the largest
(+/- 1e30), and one row (fixture A row 7) differs from its neighbour
only in the SECOND column so that a walk reading the wrong `ColumnId()`
is separated from one reading the right one.

## THE HASH

`fnv1a32` is duplicated here rather than imported from
`ensemble/decisiontree/batched_levelalgo/random_utils.mojo`. It is doing
a different job -- scattering fixture values so a mis-route is visible,
not reproducing cuML's seed chain -- and a check that shares a helper
with the code around it is a check that can be broken by an edit to
that helper. Duplication is the deliberate choice.
"""

from std.ffi import external_call
from std.math import log2 as mojo_log2

from ensemble.decisiontree.decisiontree import (
    CRITERION_END,
    GINI,
    MAE,
    MSE,
    DecisionTree,
    DecisionTreeParams,
    TreeMetaDataNode,
    get_tree_json,
    get_tree_summary_text,
)
from ensemble.flatnode import SparseTreeNode
from ensemble.randomforest import (
    CLASSIFICATION,
    INT32_MAX,
    MAX_FEATURES_LOG2,
    MAX_FEATURES_NONE,
    MAX_FEATURES_SQRT,
    REGRESSION,
    RandomForest,
    RandomForestMetaData,
    compute_feature_importances,
    compute_max_features,
    compute_max_features_float,
    compute_max_features_int,
    default_rf_params_classifier,
    default_rf_params_regressor,
    n_sampled_cols,
    print_metrics,
    set_all_rf_metrics,
    set_rf_params,
)


comptime F32 = DType.float32
comptime I32 = DType.int32


# ---------------------------------------------------------------------------
# Fixture value generator -- hashed and scattered, never uniform
# ---------------------------------------------------------------------------


def fixture_hash(a: Int, b: Int, c: Int) -> UInt32:
    """FNV-1a32 over three small integers. Duplicated on purpose; see
    the module docstring."""
    var h: UInt32 = 2166136261
    var vals = [UInt32(a), UInt32(b), UInt32(c)]
    for i in range(len(vals)):
        var txt = vals[i]
        for byte in range(4):
            h ^= (txt >> UInt32(byte * 8)) & 0xFF
            h *= 16777619
    return h


def leaf_value(tree_id: Int, node_id: Int, k: Int) -> Float32:
    """A distinct, scattered, EXACTLY representable float32 per cell.

    `% 8388593` is the largest prime below 2^23, so every value is an
    integer float32 with no rounding -- the check compares for exact
    equality and must not be measuring the formatter.
    """
    return Float32(Int(fixture_hash(tree_id, node_id, k) % 8388593))


def poison_value(tree_id: Int, node_id: Int, k: Int) -> Float32:
    """A value no leaf can hold: above the prime modulus, so it can only
    appear if an internal node's `vector_leaf` slot was read."""
    return Float32(9000000 + tree_id * 10000 + node_id * 100 + k)


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------


def build_tree_from_spec(
    tree_id: Int,
    colids: List[Int32],
    quesvals: List[Float32],
    left_children: List[Int64],
    instance_counts: List[Int32],
    num_outputs: Int,
) raises -> TreeMetaDataNode[F32]:
    """Hand-build a flat tree through THEIR factories only.

    `left_children[i] == -1` means leaf, which is `flatnode.h:54-57`'s
    `CreateLeafNode`; anything else is `CreateSplitNode`
    (`flatnode.h:48-53`). Leaf slots of `vector_leaf` get hashed values,
    internal slots get poison.
    """
    var sparsetree = List[SparseTreeNode[F32]]()
    var vector_leaf = List[Float32]()
    var n_leaves: Int32 = 0
    var max_depth: Int32 = 0
    for i in range(len(colids)):
        if left_children[i] == -1:
            sparsetree.append(
                SparseTreeNode[F32].CreateLeafNode(instance_counts[i])
            )
            n_leaves += 1
            for k in range(num_outputs):
                vector_leaf.append(leaf_value(tree_id, i, k))
        else:
            sparsetree.append(
                SparseTreeNode[F32].CreateSplitNode(
                    colids[i],
                    quesvals[i],
                    Float32(1.0),
                    left_children[i],
                    instance_counts[i],
                )
            )
            for k in range(num_outputs):
                vector_leaf.append(poison_value(tree_id, i, k))
            max_depth = 1
    return TreeMetaDataNode[F32](
        treeid=Int32(tree_id),
        depth_counter=max_depth,
        leaf_counter=n_leaves,
        train_time=0.0,
        vector_leaf=vector_leaf^,
        sparsetree=sparsetree^,
        num_outputs=Int32(num_outputs),
    )


def fixture_a(tree_id: Int, num_outputs: Int) raises -> TreeMetaDataNode[F32]:
    """BALANCED, depth 3, 7 nodes, 2 columns.

        0: col0 <= 10.0 ? 1 : 2
        1: col1 <=  5.0 ? 3 : 4
        2: col1 <= 20.0 ? 5 : 6
        3,4,5,6: leaves

    The two subtrees split on the SAME column (1) at DIFFERENT
    thresholds, so a walk that routes correctly at the root but reads
    the wrong node's `quesval` at level 2 is separated from one that
    does not.
    """
    var colids: List[Int32] = [0, 1, 1, 0, 0, 0, 0]
    var quesvals: List[Float32] = [10.0, 5.0, 20.0, 0.0, 0.0, 0.0, 0.0]
    var left_children: List[Int64] = [1, 3, 5, -1, -1, -1, -1]
    var counts: List[Int32] = [400, 250, 150, 130, 120, 90, 60]
    return build_tree_from_spec(
        tree_id, colids, quesvals, left_children, counts, num_outputs
    )


def fixture_b(tree_id: Int, num_outputs: Int) raises -> TreeMetaDataNode[F32]:
    """SINGLE LEAF. One node, `left_child_id == -1`."""
    var colids: List[Int32] = [0]
    var quesvals: List[Float32] = [0.0]
    var left_children: List[Int64] = [-1]
    var counts: List[Int32] = [400]
    return build_tree_from_spec(
        tree_id, colids, quesvals, left_children, counts, num_outputs
    )


def fixture_c(tree_id: Int, num_outputs: Int) raises -> TreeMetaDataNode[F32]:
    """UNBALANCED, 5 nodes. Node 1 is a LEAF, node 2 is a SPLIT.

        0: col0 <= 5.0 ? 1 : 2
        1: LEAF                     <- depth 1
        2: col1 <= 3.0 ? 3 : 4
        3,4: leaves                 <- depth 2

    A complete-tree assumption (`children at 2i+1, 2i+2`) gives the same
    answers as the sibling rule on fixture A and DIFFERENT answers here,
    because node 2's children are at 3 and 4 rather than at 5 and 6.
    """
    var colids: List[Int32] = [0, 0, 1, 0, 0]
    var quesvals: List[Float32] = [5.0, 0.0, 3.0, 0.0, 0.0]
    var left_children: List[Int64] = [1, -1, 3, -1, -1]
    var counts: List[Int32] = [400, 200, 200, 110, 90]
    return build_tree_from_spec(
        tree_id, colids, quesvals, left_children, counts, num_outputs
    )


# ---------------------------------------------------------------------------
# The reference walk, and its four sabotages
# ---------------------------------------------------------------------------

comptime SAB_NONE: Int = 0
comptime SAB_LEAF_TEST: Int = 1
comptime SAB_CHILD_INDEX: Int = 2
comptime SAB_COMPARE: Int = 3
comptime SAB_COLUMN: Int = 4

comptime ESCAPED: Int = -1


def sabotage_name(mode: Int) -> String:
    if mode == SAB_NONE:
        return "SAB_NONE (faithful)"
    if mode == SAB_LEAF_TEST:
        return "SAB_LEAF_TEST (IsLeaf := left_child_id == 0)"
    if mode == SAB_CHILD_INDEX:
        return "SAB_CHILD_INDEX (right := left + 2)"
    if mode == SAB_COMPARE:
        return "SAB_COMPARE (<= becomes <)"
    return "SAB_COLUMN (always read column 0)"


def walk_with_sabotage(
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    row_offset: Int,
    mode: Int,
) -> Int:
    """The walk of `decisiontree.cuh:370-385`, with ONE mechanism
    optionally broken, returning the LEAF NODE INDEX it lands on.

    `SAB_NONE` must agree with `DecisionTree.predict_one` on every row of
    every fixture -- that agreement is what makes this function a valid
    stand-in for the real walk, and it is asserted, not assumed. Each
    other mode must DISAGREE on at least one row, or the corresponding
    mechanism is not actually being exercised by these fixtures.

    Returns `ESCAPED` if the sabotaged walk leaves the node array, so a
    sabotage that runs off the end is a reported result rather than a
    crash.
    """
    var idx: Int = 0
    var steps: Int = 0
    while True:
        if idx < 0 or idx >= len(tree.sparsetree):
            return ESCAPED
        steps += 1
        if steps > 4 * len(tree.sparsetree) + 8:
            return ESCAPED
        var n = tree.sparsetree[idx].copy()

        var is_leaf: Bool
        if mode == SAB_LEAF_TEST:
            # The "0 means no child" convention other flat-tree formats
            # use. Under cuML's layout nothing is ever a leaf under it.
            is_leaf = n.LeftChildId() == 0
        else:
            is_leaf = n.IsLeaf()
        if is_leaf:
            return idx

        var col: Int
        if mode == SAB_COLUMN:
            col = 0
        else:
            col = Int(n.ColumnId())

        var go_left: Bool
        if mode == SAB_COMPARE:
            go_left = rows[row_offset + col] < n.QueryValue()
        else:
            go_left = rows[row_offset + col] <= n.QueryValue()

        if go_left:
            idx = Int(n.LeftChildId())
        else:
            if mode == SAB_CHILD_INDEX:
                idx = Int(n.LeftChildId()) + 2
            else:
                idx = Int(n.RightChildId())


# ---------------------------------------------------------------------------
# Fixture assertions
# ---------------------------------------------------------------------------


def assert_distinct_leaf_values(
    tree: TreeMetaDataNode[F32], label: String
) raises:
    """A check whose cells all hold the same value verifies the total and
    nothing about placement. This is the guard that says these cells do
    not."""
    var num_outputs = Int(tree.num_outputs)
    var seen = List[Float32]()
    for i in range(len(tree.sparsetree)):
        if tree.sparsetree[i].IsLeaf():
            for k in range(num_outputs):
                var v = tree.vector_leaf[i * num_outputs + k]
                for j in range(len(seen)):
                    if seen[j] == v:
                        raise Error(
                            label
                            + ": leaf value "
                            + String(v)
                            + " appears twice; the fixture cannot"
                            " distinguish a mis-routed row from a"
                            " correctly routed one"
                        )
                seen.append(v)
    if len(seen) == 0:
        raise Error(label + ": fixture has no leaves at all")


def check_single_tree(
    label: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    n_rows: Int,
    n_cols: Int,
    expected_leaf: List[Int],
    mut failures: Int,
) raises:
    """Per-ROW comparison against a HAND-WRITTEN expected leaf index.

    The ported walk is run through the real entry point,
    `DecisionTree.predict`, so the offsets, the `predict_all` loop and
    the `+=` accumulation are all in the path -- not just `predict_one`.
    """
    assert_distinct_leaf_values(tree, label)
    var num_outputs = Int(tree.num_outputs)
    var preds = List[Float32]()
    for _ in range(n_rows * num_outputs):
        preds.append(0.0)
    DecisionTree.predict(tree, rows, n_rows, n_cols, preds, num_outputs)

    for r in range(n_rows):
        for k in range(num_outputs):
            var want = tree.vector_leaf[expected_leaf[r] * num_outputs + k]
            var got = preds[r * num_outputs + k]
            if got != want:
                failures += 1
                print(
                    "  FAIL",
                    label,
                    "row",
                    r,
                    "output",
                    k,
                    ": expected leaf",
                    expected_leaf[r],
                    "value",
                    want,
                    "but got",
                    got,
                )
        # The sabotage harness must agree with the real walk when it is
        # not sabotaging anything, or every sabotage result below is
        # measuring the harness instead of the port.
        var mirrored = walk_with_sabotage(tree, rows, r * n_cols, SAB_NONE)
        if mirrored != expected_leaf[r]:
            failures += 1
            print(
                "  FAIL",
                label,
                "row",
                r,
                ": the SAB_NONE mirror routed to",
                mirrored,
                "but the hand-written expectation is",
                expected_leaf[r],
            )


def run_sabotages(
    label: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    n_rows: Int,
    n_cols: Int,
    expected_leaf: List[Int],
    modes: List[Int],
    mut failures: Int,
) raises:
    """Break one mechanism at a time and require the check to MOVE.

    A sabotage that moves nothing means the fixture does not reach that
    mechanism, which is a defect in the CHECK and is reported as a
    failure, not as a pass.
    """
    for m in range(len(modes)):
        var mode = modes[m]
        var moved: Int = 0
        var escaped: Int = 0
        for r in range(n_rows):
            var got = walk_with_sabotage(tree, rows, r * n_cols, mode)
            if got == ESCAPED:
                escaped += 1
                moved += 1
            elif got != expected_leaf[r]:
                moved += 1
        if moved == 0:
            failures += 1
            print(
                "  FAIL",
                label,
                sabotage_name(mode),
                "moved 0 of",
                n_rows,
                "rows -- this fixture does not reach that mechanism, so"
                " the check has no teeth there",
            )
        else:
            print(
                "  sabotage",
                label,
                sabotage_name(mode),
                "moved",
                moved,
                "of",
                n_rows,
                "rows (" + String(escaped) + " escaped the node array)",
            )



def importance_tree(
    tree_id: Int,
    root_col: Int32,
    root_metric: Float32,
    root_count: Int32,
    inner_col: Int32,
    inner_metric: Float32,
    inner_count: Int32,
) raises -> TreeMetaDataNode[F32]:
    """Fixture C's SHAPE with hand-chosen BestMetric / InstanceCount, for
    `compute_feature_importances`.

        0: split (root_col,  root_metric,  root_count)  -> children 1, 2
        1: leaf
        2: split (inner_col, inner_metric, inner_count) -> children 3, 4
        3, 4: leaves

    Leaf values are irrelevant here -- importances read only
    `BestMetric()`, `InstanceCount()` and `ColumnId()` off SPLIT nodes
    (`randomforest.cu:820-836`), which is itself worth pinning: a version
    that walked leaves too would count nothing, and a version that
    skipped the leaf test would count a leaf's placeholder colid 0.
    """
    var sparsetree = List[SparseTreeNode[F32]]()
    sparsetree.append(
        SparseTreeNode[F32].CreateSplitNode(
            root_col, 0.0, root_metric, 1, root_count
        )
    )
    sparsetree.append(SparseTreeNode[F32].CreateLeafNode(50))
    sparsetree.append(
        SparseTreeNode[F32].CreateSplitNode(
            inner_col, 0.0, inner_metric, 3, inner_count
        )
    )
    sparsetree.append(SparseTreeNode[F32].CreateLeafNode(30))
    sparsetree.append(SparseTreeNode[F32].CreateLeafNode(20))
    var vector_leaf = List[Float32]()
    for i in range(5):
        vector_leaf.append(leaf_value(tree_id, i, 0))
    return TreeMetaDataNode[F32](
        treeid=Int32(tree_id),
        depth_counter=2,
        leaf_counter=3,
        train_time=0.0,
        vector_leaf=vector_leaf^,
        sparsetree=sparsetree^,
        num_outputs=1,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var failures: Int = 0

    # ---------------- Fixture A: balanced, 2 columns -------------------
    #
    # row : (col0, col1)          -> expected leaf, and why
    #  0  : (10.0,        5.0)    -> 3  EXACTLY on quesval at BOTH levels
    #  1  : (10.0,        5.0001) -> 4  on quesval at the root only
    #  2  : (10.0001,    20.0)    -> 5  on quesval at level 2 only
    #  3  : ( 1e30,       1e30)   -> 6  above every threshold
    #  4  : (-1e30,      -1e30)   -> 3  below every threshold
    #  5  : (12.0,        7.0)    -> 5  right then left; sibling rule
    #  6  : ( 1e30,      20.0)    -> 5  on quesval, right subtree
    #  7  : ( 3.0,        7.0)    -> 4  differs from row 8 in col1 ONLY
    #  8  : ( 3.0,        2.0)    -> 3
    #  9  : ( 9.999999,   5.0)    -> 3  on quesval at level 2 only
    var rows_a: List[Float32] = [
        10.0,
        5.0,
        10.0,
        5.0001,
        10.0001,
        20.0,
        1e30,
        1e30,
        -1e30,
        -1e30,
        12.0,
        7.0,
        1e30,
        20.0,
        3.0,
        7.0,
        3.0,
        2.0,
        9.999999,
        5.0,
    ]
    var expected_a: List[Int] = [3, 4, 5, 6, 3, 5, 5, 4, 3, 3]
    var n_rows_a = 10
    var n_cols_a = 2

    var tree_a = fixture_a(0, 1)
    print("fixture A (balanced depth-3, 7 nodes, num_outputs=1)")
    check_single_tree(
        "A", tree_a, rows_a, n_rows_a, n_cols_a, expected_a, failures
    )
    var all_modes: List[Int] = [
        SAB_LEAF_TEST,
        SAB_CHILD_INDEX,
        SAB_COMPARE,
        SAB_COLUMN,
    ]
    run_sabotages(
        "A",
        tree_a,
        rows_a,
        n_rows_a,
        n_cols_a,
        expected_a,
        all_modes,
        failures,
    )

    # SAB_COMPARE moves exactly the rows sitting ON a threshold. Rows 0,
    # 1, 2, 6 and 9 of fixture A are on one; count them explicitly, so a
    # future edit that removes the on-threshold rows fails here rather
    # than quietly reducing this file to a random-data check.
    var on_threshold: Int = 0
    for r in range(n_rows_a):
        if (
            walk_with_sabotage(tree_a, rows_a, r * n_cols_a, SAB_COMPARE)
            != expected_a[r]
        ):
            on_threshold += 1
    print(
        "  rows landing EXACTLY on a quesval (moved by <= -> <):",
        on_threshold,
    )
    if on_threshold < 4:
        failures += 1
        print(
            "  FAIL A: fewer than 4 rows land exactly on a threshold; the"
            " tie behaviour at quesval is the thing this file exists to"
            " pin down"
        )

    # ---------------- Fixture B: single leaf ---------------------------
    #
    # A one-node tree. The loop must test the leaf flag BEFORE reading a
    # threshold, so both rows return node 0 and neither feature is read.
    var rows_b: List[Float32] = [1e30, -1e30, 0.0, 0.0]
    var expected_b: List[Int] = [0, 0]
    var tree_b = fixture_b(1, 1)
    print("fixture B (single leaf, 1 node)")
    check_single_tree("B", tree_b, rows_b, 2, 2, expected_b, failures)
    # Only the leaf test is reachable in a one-node tree: there is no
    # child index, no comparison and no column read. Sabotaging the
    # other three here would be a sabotage with nothing to reach, which
    # this file reports as a failure -- so only the leaf test is run.
    var b_modes: List[Int] = [SAB_LEAF_TEST]
    run_sabotages(
        "B", tree_b, rows_b, 2, 2, expected_b, b_modes, failures
    )

    # ---------------- Fixture C: unbalanced ----------------------------
    #
    # row : (col0, col1)     -> expected leaf, and why
    #  0  : ( 5.0,   99.0)   -> 1  EXACTLY on quesval; leaf at depth 1
    #  1  : ( 1.0,    1e30)  -> 1  left subtree is one node deep
    #  2  : ( 9.0,    1.0)   -> 3  right subtree is two nodes deep
    #  3  : ( 9.0,    3.0)   -> 3  EXACTLY on quesval at level 2
    #  4  : ( 9.0,    9.0)   -> 4
    #  5  : (-1e30,  -1e30)  -> 1
    var rows_c: List[Float32] = [
        5.0,
        99.0,
        1.0,
        1e30,
        9.0,
        1.0,
        9.0,
        3.0,
        9.0,
        9.0,
        -1e30,
        -1e30,
    ]
    var expected_c: List[Int] = [1, 1, 3, 3, 4, 1]
    var tree_c = fixture_c(2, 1)
    print("fixture C (unbalanced, 5 nodes; node 1 leaf, node 2 split)")
    check_single_tree("C", tree_c, rows_c, 6, 2, expected_c, failures)
    run_sabotages(
        "C", tree_c, rows_c, 6, 2, expected_c, all_modes, failures
    )

    # ---------------- Fixture D: multi-output + the forest -------------
    #
    # Two trees of fixture A's shape with num_outputs = 3, hashed leaf
    # vectors, poisoned internal slots. The per-row expectation is
    # computed from the HAND-WRITTEN leaf indices above and the trees'
    # own `vector_leaf`, then averaged and argmaxed here -- an
    # independent tally of the aggregation, sharing no code with
    # `RandomForest.predict`.
    var num_outputs = 3
    var trees = List[TreeMetaDataNode[F32]]()
    trees.append(fixture_a(10, num_outputs))
    trees.append(fixture_a(11, num_outputs))
    for t in range(len(trees)):
        assert_distinct_leaf_values(trees[t], "D tree " + String(t))

    var params = set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=1.0,
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=2,
        max_samples=1.0,
        seed=0,
        split_criterion=GINI,
        cfg_n_streams=1,
        max_batch_size=4096,
    )
    params.check()

    var forest = RandomForestMetaData[F32, I32](
        trees=trees^, rf_params=params, n_features=2
    )
    var rf = RandomForest[F32, I32](rf_params=params, rf_type=CLASSIFICATION)
    var preds_d = List[Scalar[I32]]()
    for _ in range(n_rows_a):
        preds_d.append(0)
    rf.predict(rows_a, n_rows_a, n_cols_a, preds_d, forest)

    print("fixture D (2 trees, num_outputs=3, classifier vote)")
    for r in range(n_rows_a):
        # Independent tally: both trees route row r to `expected_a[r]`,
        # because both have fixture A's shape.
        var summed = List[Float64]()
        for k in range(num_outputs):
            var s: Float64 = 0.0
            for t in range(len(forest.trees)):
                s += Float64(
                    forest.trees[t].vector_leaf[
                        expected_a[r] * num_outputs + k
                    ]
                )
            summed.append(s / 2.0)
        # `randomforest.cuh:417-427`: best_class 0, best_prob 0.0,
        # STRICT `>`, so a tie keeps the LOWER index.
        var want_class: Int = 0
        var best: Float64 = 0.0
        for k in range(num_outputs):
            if summed[k] > best:
                want_class = k
                best = summed[k]
        var got_class = Int(preds_d[r])
        if got_class != want_class:
            failures += 1
            print(
                "  FAIL D row",
                r,
                ": expected class",
                want_class,
                "but got",
                got_class,
            )

    # The class vote must not be a pass-by-accident: if the hashed leaf
    # vectors happened to make every row vote for the same class, this
    # arm would verify a constant.
    var classes_seen = List[Int]()
    for r in range(n_rows_a):
        var c = Int(preds_d[r])
        var found = False
        for j in range(len(classes_seen)):
            if classes_seen[j] == c:
                found = True
        if not found:
            classes_seen.append(c)
    print("  distinct classes predicted:", len(classes_seen))
    if len(classes_seen) < 2:
        failures += 1
        print(
            "  FAIL D: every row voted for the same class, so this arm"
            " verifies a constant and not the argmax"
        )

    # ---------------- Aggregation rules, constructed exactly -----------
    #
    # Two rules in `randomforest.cuh:417-427` that hashed data will not
    # reach on purpose: the STRICT `>` (a tie keeps the LOWER class) and
    # the all-non-positive fallthrough (answer is class 0, not an error).
    # Both are built by hand.
    var tie_tree = fixture_b(20, 3)  # single leaf, 3 outputs
    # Overwrite the leaf vector with an exact tie between classes 0 and 2.
    tie_tree.vector_leaf[0] = 7.0
    tie_tree.vector_leaf[1] = 3.0
    tie_tree.vector_leaf[2] = 7.0
    var tie_trees = List[TreeMetaDataNode[F32]]()
    tie_trees.append(tie_tree.copy())
    var tie_params = set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=1.0,
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=1,
        max_samples=1.0,
        seed=0,
        split_criterion=GINI,
        cfg_n_streams=1,
        max_batch_size=4096,
    )
    var tie_forest = RandomForestMetaData[F32, I32](
        trees=tie_trees^, rf_params=tie_params, n_features=2
    )
    var tie_rf = RandomForest[F32, I32](
        rf_params=tie_params, rf_type=CLASSIFICATION
    )
    var tie_preds = List[Scalar[I32]]()
    tie_preds.append(0)
    tie_rf.predict(rows_b, 1, 2, tie_preds, tie_forest)
    if Int(tie_preds[0]) != 0:
        failures += 1
        print(
            "  FAIL tie: classes 0 and 2 both average 7.0; the strict `>`"
            " at randomforest.cuh:421 must keep class 0, got",
            Int(tie_preds[0]),
        )
    else:
        print("  tie between class 0 and class 2 resolved to class 0")

    var neg_tree = fixture_b(21, 3)
    neg_tree.vector_leaf[0] = -5.0
    neg_tree.vector_leaf[1] = -1.0
    neg_tree.vector_leaf[2] = -3.0
    var neg_trees = List[TreeMetaDataNode[F32]]()
    neg_trees.append(neg_tree.copy())
    var neg_forest = RandomForestMetaData[F32, I32](
        trees=neg_trees^, rf_params=tie_params, n_features=2
    )
    var neg_preds = List[Scalar[I32]]()
    neg_preds.append(0)
    tie_rf.predict(rows_b, 1, 2, neg_preds, neg_forest)
    if Int(neg_preds[0]) != 0:
        failures += 1
        print(
            "  FAIL non-positive: no class averages above best_prob's"
            " initial 0.0, so randomforest.cuh:418-427 falls through to"
            " class 0; got",
            Int(neg_preds[0]),
        )
    else:
        print(
            "  all-negative row falls through to class 0 (NOT class 1,"
            " which has the largest value)"
        )

    # ---------------- Regressor arm ------------------------------------
    #
    # `randomforest.cuh:429` takes output 0 only, and the value is the
    # MEAN of the per-tree leaf values because of the `/= n_trees` at
    # `:415`. Three trees, so the mean is not the sum and not one tree's
    # value.
    var reg_trees = List[TreeMetaDataNode[F32]]()
    reg_trees.append(fixture_a(30, 1))
    reg_trees.append(fixture_c(31, 1))
    reg_trees.append(fixture_a(32, 1))
    var reg_params = set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=1.0,
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=3,
        max_samples=1.0,
        seed=0,
        split_criterion=MSE,
        cfg_n_streams=1,
        max_batch_size=4096,
    )
    var reg_forest = RandomForestMetaData[F32, F32](
        trees=reg_trees^, rf_params=reg_params, n_features=2
    )
    var reg_rf = RandomForest[F32, F32](
        rf_params=reg_params, rf_type=REGRESSION
    )
    var reg_preds = List[Float32]()
    for _ in range(n_rows_a):
        reg_preds.append(0.0)
    reg_rf.predict(rows_a, n_rows_a, n_cols_a, reg_preds, reg_forest)

    print("regressor arm (3 trees, two shapes, mean of leaf values)")
    for r in range(n_rows_a):
        # Independent per-row tally: tree 0 and tree 2 have fixture A's
        # shape, tree 1 has fixture C's, so their expected leaves differ
        # and are looked up separately.
        var leaf_ac = expected_a[r]
        var leaf_c = walk_with_sabotage(
            reg_forest.trees[1], rows_a, r * n_cols_a, SAB_NONE
        )
        var want = (
            Float32(reg_forest.trees[0].vector_leaf[leaf_ac])
            + Float32(reg_forest.trees[1].vector_leaf[leaf_c])
            + Float32(reg_forest.trees[2].vector_leaf[leaf_ac])
        ) / Float32(3)
        if reg_preds[r] != want:
            failures += 1
            print(
                "  FAIL regressor row",
                r,
                ": expected",
                want,
                "got",
                reg_preds[r],
            )

    # ---------------- n_streams refusal --------------------------------
    #
    # DEVIATION 117: the one field a caller can set today and have
    # silently ignored. The refusal must fire.
    var streamy = params.copy()
    streamy.n_streams = 4
    var refused = False
    try:
        streamy.check()
    except e:
        refused = True
    if not refused:
        failures += 1
        print(
            "  FAIL: RF_params.check() accepted n_streams=4, which this"
            " port does not honor (DEVIATION 117)"
        )
    else:
        print("n_streams=4 refused by name")

    # `set_rf_params` must CLAMP rather than refuse, because that is what
    # their own non-OpenMP build does (randomforest.cu:584 with
    # randomforest.cuh:41-42).
    var clamped = set_rf_params(
        max_depth=INT32_MAX,
        max_leaves=-1,
        max_features=1.0,
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        bootstrap=True,
        n_trees=100,
        max_samples=1.0,
        seed=0,
        split_criterion=GINI,
        cfg_n_streams=4,
        max_batch_size=4096,
    )
    if clamped.n_streams != 1:
        failures += 1
        print(
            "  FAIL: set_rf_params(cfg_n_streams=4) gave n_streams =",
            clamped.n_streams,
            "-- their min(cfg, omp_get_max_threads()) is 1 without"
            " OpenMP",
        )
    else:
        print("set_rf_params clamped cfg_n_streams=4 to 1, as theirs does")

    # ---------------- fit is PORTED now --------------------------------
    # This block used to assert that `RandomForest.fit()` RAISES. It no
    # longer does: the method forwards to `fit_forest`, the port of
    # `randomforest.cuh:286-370`. The assertion is deleted rather than
    # inverted, because what `fit` DOES is checked where it can be checked
    # properly -- `forest_check` (classification, bagged, per cell against
    # RAFT's own row sample), `regression_check` (the method itself, plus
    # predict's regression arm) and `criteria_check` (all six criteria). A
    # one-line "it did not raise" here would add nothing and go stale again.
    print(
        "RandomForest.fit is PORTED; behaviour is checked in forest_check,"
        " regression_check and criteria_check"
    )

    # ---------------- The default table --------------------------------
    #
    # The table at the top of `randomforest.mojo` is the deliverable of
    # this lane as much as the walk is, so it is asserted rather than
    # only written down. Every value below is the PYTHON default, which
    # is what ships; the four that disagree with the C++ header are
    # marked.
    var clf = default_rf_params_classifier(16)
    var reg = default_rf_params_regressor()
    print("shipped defaults (randomforestclassifier.py / regressor.py)")

    def want_i(label: String, got: Int, expect: Int, mut f: Int):
        if got != expect:
            f += 1
            print("  FAIL default", label, ": expected", expect, "got", got)

    want_i("clf.n_trees", Int(clf.n_trees), 100, failures)
    want_i("clf.max_depth", Int(clf.tree_params.max_depth), 2147483647, failures)
    want_i("clf.max_leaves", Int(clf.tree_params.max_leaves), -1, failures)
    want_i("clf.max_n_bins", Int(clf.tree_params.max_n_bins), 128, failures)
    want_i(
        "clf.min_samples_leaf", Int(clf.tree_params.min_samples_leaf), 1, failures
    )
    want_i(
        "clf.min_samples_split",
        Int(clf.tree_params.min_samples_split),
        2,
        failures,
    )
    want_i(
        "clf.max_batch_size", Int(clf.tree_params.max_batch_size), 4096, failures
    )
    want_i("clf.split_criterion", clf.tree_params.split_criterion, GINI, failures)
    want_i("clf.n_streams (clamped from 4)", Int(clf.n_streams), 1, failures)
    want_i("reg.split_criterion", reg.tree_params.split_criterion, MSE, failures)
    if clf.tree_params.min_impurity_decrease != 0.0:
        failures += 1
        print("  FAIL default clf.min_impurity_decrease")
    if not clf.bootstrap or not reg.bootstrap:
        failures += 1
        print("  FAIL default bootstrap should be True")
    if clf.max_samples != 1.0 or reg.max_samples != 1.0:
        failures += 1
        print("  FAIL default max_samples should be 1.0")
    if clf.seed != 0 or reg.seed != 0:
        failures += 1
        print("  FAIL default seed should be 0 (random_state=None)")
    # THE DISAGREEMENT THAT MATTERS: the classifier's max_features is
    # 'sqrt' and the regressor's is 1.0. A port that took one value for
    # both would pass every other assertion above.
    if reg.tree_params.max_features != 1.0:
        failures += 1
        print(
            "  FAIL default reg.max_features should be 1.0"
            " (randomforestregressor.py:162), got",
            reg.tree_params.max_features,
        )
    if clf.tree_params.max_features >= 1.0:
        failures += 1
        print(
            "  FAIL default clf.max_features should be sqrt(16)/16 = 0.25"
            " (randomforestclassifier.py:218), got",
            clf.tree_params.max_features,
        )
    else:
        print(
            "  clf max_features (sqrt, n_cols=16) =",
            clf.tree_params.max_features,
            "; reg max_features =",
            reg.tree_params.max_features,
        )

    # ---------------- max_features truncation knife-edge ---------------
    #
    # `builder.cuh:240` is `max(1, IdxT(max_features * n_cols))` -- a
    # TRUNCATION. One ULP low anywhere in the ratio chain takes a column
    # away. This is the check that says libm's log2 (and not
    # std.math.log2, which this repository has measured at ~5e-8
    # absolute error) is what feeds it.
    print("max_features -> column count (builder.cuh:240 truncation)")
    var squares: List[Int] = [4, 9, 16, 25, 36, 49, 64, 100, 144, 256, 1024]
    var roots: List[Int] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32]
    for i in range(len(squares)):
        var n = squares[i]
        var ratio = Float32(compute_max_features(MAX_FEATURES_SQRT, n))
        var k = Int(n_sampled_cols(ratio, n))
        if k != roots[i]:
            failures += 1
            print(
                "  FAIL sqrt at n_cols =",
                n,
                ": expected",
                roots[i],
                "columns, got",
                k,
                "-- the ratio truncated low",
            )
    var pow2: List[Int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    var logs: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(pow2)):
        var n = pow2[i]
        var ratio = Float32(compute_max_features(MAX_FEATURES_LOG2, n))
        var k = Int(n_sampled_cols(ratio, n))
        if k != logs[i]:
            failures += 1
            print(
                "  FAIL log2 at n_cols =",
                n,
                ": expected",
                logs[i],
                "columns, got",
                k,
                "-- Mojo's log2 would do exactly this; libm's must not",
            )
    # THE PERFECT SQUARES AND POWERS OF TWO ABOVE CANNOT TELL TRUNCATION
    # FROM ROUNDING, because their ratio chain lands on an integer. That
    # is exactly the failure the "uniform data hides permutation" rule
    # describes -- the first version of this file asserted 21 sizes and a
    # sabotage replacing `IdxT(...)` with `round(...)` moved NOTHING. The
    # sizes below have a fractional part of at least 0.5, so truncation
    # and rounding give DIFFERENT column counts and only truncation
    # matches `builder.cuh:240`.
    var frac_sqrt_n: List[Int] = [8, 32, 512]
    var frac_sqrt_k: List[Int] = [2, 5, 22]  # 2.83, 5.66, 22.63 truncated
    for i in range(len(frac_sqrt_n)):
        var n = frac_sqrt_n[i]
        var k = Int(
            n_sampled_cols(
                Float32(compute_max_features(MAX_FEATURES_SQRT, n)), n
            )
        )
        if k != frac_sqrt_k[i]:
            failures += 1
            print(
                "  FAIL sqrt TRUNCATION at n_cols =",
                n,
                ": builder.cuh:240 truncates to",
                frac_sqrt_k[i],
                "columns, got",
                k,
            )
    var frac_log_n: List[Int] = [3, 6, 12, 100]
    var frac_log_k: List[Int] = [1, 2, 3, 6]  # 1.58, 2.58, 3.58, 6.64
    for i in range(len(frac_log_n)):
        var n = frac_log_n[i]
        var k = Int(
            n_sampled_cols(
                Float32(compute_max_features(MAX_FEATURES_LOG2, n)), n
            )
        )
        if k != frac_log_k[i]:
            failures += 1
            print(
                "  FAIL log2 TRUNCATION at n_cols =",
                n,
                ": builder.cuh:240 truncates to",
                frac_log_k[i],
                "columns, got",
                k,
            )

    # Is the libm `log2` in `compute_max_features_log2` actually needed,
    # or is `std.math.log2` identical here? This repository has measured
    # `std.math.log` at ~5e-8 absolute error against libm and had that
    # error silently re-decide plateau ties, so the libm call was made
    # defensively. This measures whether the defence has anything to
    # defend against, and PRINTS the answer either way rather than
    # assuming it.
    var log2_disagreements: Int = 0
    var worst_n: Int = 0
    for n in range(2, 4097):
        var libm_v = external_call["log2", Float64](Float64(n))
        var mojo_v = mojo_log2(Float64(n))
        if libm_v != mojo_v:
            log2_disagreements += 1
            if worst_n == 0:
                worst_n = n
        var kl = Int(n_sampled_cols(Float32(libm_v / Float64(n)), n))
        var km = Int(n_sampled_cols(Float32(mojo_v / Float64(n)), n))
        if kl != km:
            failures += 1
            print(
                "  FAIL log2 SOURCE changes the column count at n_cols =",
                n,
                ": libm gives",
                kl,
                "and std.math gives",
                km,
            )
    print(
        "  std.math.log2 vs libm log2 over n_cols 2..4096:",
        log2_disagreements,
        "bit-level disagreements, first at n_cols =",
        worst_n,
    )

    if Int(n_sampled_cols(Float32(compute_max_features(MAX_FEATURES_NONE, 7)), 7)) != 7:
        failures += 1
        print("  FAIL max_features=None must give every column")
    if Int(n_sampled_cols(Float32(compute_max_features_int(3, 10)), 10)) != 3:
        failures += 1
        print("  FAIL max_features=3 of 10 columns must give 3")
    # The `max(1, ...)` floor: a ratio small enough to truncate to zero
    # must still sample one column.
    if Int(n_sampled_cols(Float32(compute_max_features_float(0.01)), 10)) != 1:
        failures += 1
        print("  FAIL max(1, ...) floor at builder.cuh:240 not honored")
    print(
        "  sqrt/log2 column counts exact at 21 integer sizes + 7"
        " fractional ones; max(1,..) floor holds"
    )

    # ---------------- score ---------------------------------------------
    #
    # Analytic, both arms. Classification: raft scores.cuh:110 is
    # `correctly_predicted * 1.0f / n`. Regression: scores.cuh:203-216,
    # with the EVEN-n median averaging the two middle elements.
    print("score (raft 661a3b8 stats/detail/scores.cuh)")
    var s_pred: List[Scalar[I32]] = [0, 1, 1, 0, 2]
    var s_ref: List[Scalar[I32]] = [0, 1, 0, 0, 2]
    var acc = RandomForest[F32, I32].score(s_ref, 5, s_pred, CLASSIFICATION)
    if acc.accuracy != Float32(4.0) / Float32(5.0):
        failures += 1
        print("  FAIL accuracy: expected 0.8, got", acc.accuracy)
    if acc.mean_squared_error != -1.0:
        failures += 1
        print("  FAIL classification metrics must leave MSE at the -1.0 sentinel")
    # diffs   = [-0.5, 0.0, -2.0, 0.0]
    # abs     = [ 0.5, 0.0,  2.0, 0.0]  -> MAE  = 2.5 / 4 = 0.625
    # squares = [0.25, 0.0,  4.0, 0.0]  -> MSE  = 4.25 / 4 = 1.0625
    # sorted  = [ 0.0, 0.0,  0.5, 2.0]  -> med  = (0.5 + 0.0) / 2 = 0.25
    var r_pred: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var r_ref: List[Float32] = [1.5, 2.0, 5.0, 4.0]
    var reg_m = RandomForest[F32, F32].score(r_ref, 4, r_pred, REGRESSION)
    if reg_m.mean_abs_error != 0.625:
        failures += 1
        print("  FAIL mean_abs_error: expected 0.625, got", reg_m.mean_abs_error)
    if reg_m.mean_squared_error != 1.0625:
        failures += 1
        print(
            "  FAIL mean_squared_error: expected 1.0625, got",
            reg_m.mean_squared_error,
        )
    if reg_m.median_abs_error != 0.25:
        failures += 1
        print(
            "  FAIL median (EVEN n averages the two middle elements,"
            " scores.cuh:215): expected 0.25, got",
            reg_m.median_abs_error,
        )
    # ODD n takes the middle element outright (scores.cuh:213).
    var o_pred: List[Float32] = [1.0, 2.0, 3.0]
    var o_ref: List[Float32] = [1.0, 5.0, 4.0]
    var odd_m = RandomForest[F32, F32].score(o_ref, 3, o_pred, REGRESSION)
    if odd_m.median_abs_error != 1.0:
        failures += 1
        print(
            "  FAIL odd-n median: sorted |diff| is [0, 1, 3], middle is 1;"
            " got",
            odd_m.median_abs_error,
        )
    if reg_m.accuracy != -1.0:
        failures += 1
        print("  FAIL regression metrics must leave accuracy at -1.0")
    print("  accuracy 0.8, MAE 0.625, MSE 1.0625, median 0.25 (even) / 1.0 (odd)")

    # ---------------- compute_feature_importances -----------------------
    #
    # Hand-built, hand-computed, and DELIBERATELY NON-UNIFORM across the
    # two features -- an importances check whose expected value is the
    # same in both cells would verify the normalization and nothing about
    # which feature got the credit.
    #
    # tree 0:  node0 split col0, metric 2.0, count 100 -> 200
    #          node2 split col1, metric 1.0, count  40 ->  40
    #          tree sum 240 -> f0 = 200/240, f1 = 40/240
    # tree 1:  node0 split col1, metric 4.0, count  80 -> 320
    #          node2 split col1, metric 0.5, count  20 ->  10
    #          tree sum 330 -> f0 = 0,       f1 = 330/330 = 1
    # forest:  f0 = 200/240, f1 = 40/240 + 1; total = 2
    print("compute_feature_importances (randomforest.cu:799-860)")
    var imp_trees = List[TreeMetaDataNode[F32]]()
    imp_trees.append(importance_tree(0, 0, 2.0, 100, 1, 1.0, 40))
    imp_trees.append(importance_tree(1, 1, 4.0, 80, 1, 0.5, 20))
    var imp_forest = RandomForestMetaData[F32, F32](
        trees=imp_trees^, rf_params=reg_params.copy(), n_features=2
    )
    var importances: List[Float32] = [0.0, 0.0]
    compute_feature_importances(imp_forest, importances)
    var acc0 = 200.0 / 240.0
    var acc1 = 40.0 / 240.0 + 330.0 / 330.0
    var tot = acc0 + acc1
    var want0 = Float32(acc0 / tot)
    var want1 = Float32(acc1 / tot)
    if importances[0] != want0 or importances[1] != want1:
        failures += 1
        print(
            "  FAIL importances: expected [",
            want0,
            ",",
            want1,
            "] got [",
            importances[0],
            ",",
            importances[1],
            "]",
        )
    else:
        print("  per-feature importances", importances[0], importances[1])

    # The +inf arm (`randomforest.cu:826-836`) is a RARE branch and a
    # rare branch is an unchecked one until something reaches it. Here
    # feature 0's split metric is +inf and feature 1's is a large FINITE
    # number, so the two arms give OPPOSITE answers: the infinite arm
    # credits feature 0, the finite arm would credit feature 1.
    var zero: Float32 = 0.0
    var inf32 = Float32(1.0) / zero
    var inf_trees = List[TreeMetaDataNode[F32]]()
    inf_trees.append(importance_tree(2, 0, inf32, 10, 1, 5.0, 100))
    var inf_forest = RandomForestMetaData[F32, F32](
        trees=inf_trees^, rf_params=reg_params.copy(), n_features=2
    )
    var inf_imp: List[Float32] = [0.0, 0.0]
    compute_feature_importances(inf_forest, inf_imp)
    if inf_imp[0] != 1.0 or inf_imp[1] != 0.0:
        failures += 1
        print(
            "  FAIL +inf arm: has_infinite_importance must DISCARD the"
            " finite vector entirely (randomforest.cu:838-839), giving"
            " [1, 0]; got [",
            inf_imp[0],
            ",",
            inf_imp[1],
            "]",
        )
    else:
        print("  +inf arm discards the finite vector, credits feature 0")

    # ---------------- refusals and the remaining surface ----------------
    var mae_params = clf.copy()
    mae_params.tree_params.split_criterion = MAE
    var mae_refused = False
    try:
        mae_params.check()
    except e:
        mae_refused = True
    if not mae_refused:
        failures += 1
        print("  FAIL: split_criterion=MAE must be refused (decisiontree.cu:28)")

    var json_refused = False
    try:
        _ = get_tree_json(tree_a)
    except e:
        json_refused = True
    if not json_refused:
        failures += 1
        print("  FAIL: get_tree_json is NOT PORTED and must raise")

    var fitp_refused = False
    try:
        var dp = clf.tree_params.copy()
        dp.check_fit_supported()
    except e:
        fitp_refused = True
    # INVERTED, deliberately: `check_fit_supported` used to refuse all nine
    # training fields because none had a consumer. Every one of them is
    # read now -- max_depth/max_leaves/min_samples_split by NodeQueue,
    # max_features by the round schedule, max_n_bins by the quantiles,
    # min_samples_leaf and min_impurity_decrease by the gains,
    # split_criterion by their switch, max_batch_size by Pop. So it must
    # NOT raise. The method stays, so a future unported field has somewhere
    # to be refused BY NAME.
    if fitp_refused:
        failures += 1
        print(
            "  FAIL: DecisionTreeParams.check_fit_supported raised, but"
            " every training field it names now has a consumer"
        )

    print(get_tree_summary_text(tree_a), end="")
    print_metrics(set_all_rf_metrics(CLASSIFICATION, 0.8, -1.0, -1.0, -1.0))
    print("MAE refused, get_tree_json refused, training params honored")

    print("")
    if failures == 0:
        print("predict_check: PASS")
    else:
        raise Error(
            "predict_check: " + String(failures) + " failure(s)"
        )
