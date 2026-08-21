"""`extratrees/ported/decisiontree/flatnode.mojo`, per node and per row.

    cd /Users/andrewhendel/CascadeProjects/mojolearn && \\
        pixi run mojo run -I . extratrees/mojo_only/flatnode_check.mojo

WHAT THIS IS GATING, and why it is shaped the way it is.

Rule 8, earned expensively in this repository: **a check whose expected value
is the same in every cell verifies the TOTAL and nothing about PLACEMENT.** A
uniform fixture once reported 0 wrong of 512 on a kernel that a hashed fixture
showed was 490 wrong of 512. A predict traversal is exactly that hazard: every
leaf holding the same number, or leaf `i` holding `i`, or leaf `i` holding
`i % k`, makes a wrong-leaf landing either invisible or off by a pattern the
eye completes. So:

* **Every leaf cell is DISTINCT and hashed.** `leaf_cell(node, out)` is
  `node * 16 + out` in the integer part — injective, so two different cells can
  never collide — plus `fnv1a32(node, out) & 1023` over 1024 in the fraction,
  so the value carries no usable gradient and landing one node away is not one
  step away in value. Both parts are exactly representable in `float32` at
  these magnitudes, so every comparison below is exact `==`, never a tolerance.
* **`best_metric_val` and `instance_count` are hashed per node too**, for the
  same reason: a field-transcription check whose expected value is 0 in every
  cell proves the field exists and nothing else.
* **Rows are routed by an INDEPENDENT traversal in a different style.**
  `walk_recursive` is recursive where the port is iterative, tests
  `x > quesval` for RIGHT where the port tests `x <= quesval` for LEFT, reads
  the raw fields where the port goes through `ColumnId()`/`LeftChildId()`, and
  computes the right child as `left + 1` itself rather than calling
  `RightChildId()`. It shares no line with the thing it checks.
* **Comparison is PER ROW and PER NODE**, never a sum. The counters below
  count cells, and a mismatch prints the row, the node it should have reached,
  and the node it did.

FIXTURES, and what each one exists to catch.

| fixture | shape | catches |
|---|---|---|
| T0 | a single node that is only a leaf | a traversal that assumes node 0 splits |
| T1 | a stump, `col0 <= 5.0` | the base case, and boundary rows AT 5.0 |
| T2 | asymmetric, depth 3 left / depth 1 right | a builder that assumes balance |
| T3 | boundary chain, every threshold IS a data value | `<` for `<=` |
| T4 | 3 outputs, exact argmax ties + an all-zero leaf | the tie rule and the `best_prob = 0.0` init |
| T5 | 4 outputs, all 28 leaf cells hashed-distinct | a `vector_leaf` stride that is off by one |

T2 and T4 both contain a node whose two children are one leaf and one split,
which is the case a builder that allocates children in pairs gets wrong first.

Rows are hand-written boundary cases first, then 256 hashed rows per tree
drawn from the value set `{-1.5, 0, 1, 2, 3, 4, 5, 7}` — a set chosen so that
every threshold in every fixture is IN it, which puts roughly an eighth of the
generated rows exactly on a boundary rather than leaving boundaries to the
hand-written handful.

SABOTAGE (rule 8: a check never seen to fail is not evidence). One per
MECHANISM, applied to the ported file, run, observed red, restored bit for
bit. MEASURED 2026-08-21, against a clean run of 5706 cells / 0 wrong:

| # | mechanism | sabotage | result |
|---|---|---|---|
| 1 | the boundary decision | `<=` -> `<` in `predict_leaf` | 864 of 5706 wrong |
| 2 | the adjacency convention | left/right swapped in `predict_leaf` | 4304 of 5706 wrong |
| 3 | the `vector_leaf` stride | `leaf_base` returns `id * k + 1` | red on the FIRST cell (`FAIL T0 leaf_base got 1 expected 0`), then the end-to-end read trips Mojo's own bounds assert; exit 1 |
| 4 | the -1 sentinel | `IsLeaf` also returns true when `colid == 0` | 4448 of 5706 wrong |

**T0 is 0-wrong under all four and that is not a gap in the sabotage, it is
what T0 IS:** a tree of one leaf has no comparison, no adjacency and no
second node, so nothing in it can be gotten wrong except by a traversal that
assumes node 0 splits — which is the one failure T0 exists to catch, and
which none of these four is.
"""

from extratrees.ported.decisiontree.flatnode import (
    NODE_IS_LEAF,
    SparseTreeNode,
    TreeMetaDataNode,
    predict_all,
    predict_all_accumulate,
    predict_class,
    predict_leaf,
    predict_one_accumulate,
    predict_regression,
)

comptime F32 = DType.float32

def value_set() -> List[Float32]:
    """The eight values a generated row can hold.

    Every threshold used by every fixture is a member, so a generated row
    lands exactly on a boundary about one time in eight.
    """
    var v: List[Float32] = [-1.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    return v^

comptime N_GENERATED = 256
comptime N_COLS = 3


# ----------------------------------------------------------------------
# Hashing. Nothing here is a port; it exists so that no two cells in any
# fixture hold the same number.
# ----------------------------------------------------------------------


def fnv1a32(a: Int, b: Int) -> UInt32:
    """Fnv1a32 over the eight bytes of `(a, b)`. Both must be >= 0."""
    var h: UInt32 = 2166136261
    var t = UInt32(a)
    for _ in range(4):
        h = (h ^ (t & 0xFF)) * 16777619
        t >>= 8
    var u = UInt32(b)
    for _ in range(4):
        h = (h ^ (u & 0xFF)) * 16777619
        u >>= 8
    return h


def leaf_cell(node_id: Int, output_id: Int) -> Float32:
    """`node_id * 16 + output_id` (injective for `output_id < 16`) plus a
    hashed fraction in `[0, 1)`. Distinct for every `(node, output)` pair by
    construction, and exact in `float32` for these magnitudes."""
    var whole = Float32(node_id * 16 + output_id)
    var frac = Float32(Int(fnv1a32(node_id, output_id) & 1023)) / 1024.0
    return whole + frac


def hashed_metric(node_id: Int) -> Float32:
    """A distinct `best_metric_val` per node. The traversal never reads this
    field, so a check that left it 0 would prove only that it compiles."""
    return Float32(Int(fnv1a32(node_id, 7717) & 4095)) / 64.0


def hashed_count(node_id: Int) -> Int32:
    """A distinct `instance_count` per node, always >= 1."""
    return Int32(Int(fnv1a32(node_id, 9319) & 32767) + 1)


# ----------------------------------------------------------------------
# The independent traversal. Recursive, complementary predicate, raw
# fields, right child computed here. Shares no line with `predict_leaf`.
# ----------------------------------------------------------------------


def walk_recursive(
    tree: TreeMetaDataNode[F32],
    row: List[Float32],
    row_offset: Int,
    node: Int,
) -> Int:
    var nd = tree.sparsetree[node]
    if nd.left_child_id == -1:
        return node
    var goes_right = row[row_offset + Int(nd.colid)] > nd.quesval
    if goes_right:
        return walk_recursive(tree, row, row_offset, Int(nd.left_child_id) + 1)
    return walk_recursive(tree, row, row_offset, Int(nd.left_child_id))


# ----------------------------------------------------------------------
# Fixture construction
# ----------------------------------------------------------------------


def build_tree(
    treeid: Int32,
    colid: List[Int32],
    quesval: List[Float32],
    left: List[Int32],
    num_outputs: Int,
    var vector_leaf: List[Float32],
) -> TreeMetaDataNode[F32]:
    """Nodes are built through the PORTED factories, never through the
    fieldwise constructor, so `CreateSplitNode` / `CreateLeafNode` are on the
    path of every fixture."""
    var nodes = List[SparseTreeNode[F32]]()
    var n_leaves: Int32 = 0
    for i in range(len(colid)):
        if left[i] == NODE_IS_LEAF:
            # `CreateLeafNode` zeroes colid/quesval/best_metric_val, theirs.
            nodes.append(
                SparseTreeNode[F32].CreateLeafNode(hashed_count(i))
            )
            n_leaves += 1
        else:
            nodes.append(
                SparseTreeNode[F32].CreateSplitNode(
                    colid[i],
                    quesval[i],
                    hashed_metric(i),
                    Int64(left[i]),
                    hashed_count(i),
                )
            )
    var depth: Int32 = 0
    var t = TreeMetaDataNode[F32](
        treeid, depth, n_leaves, Int32(num_outputs), vector_leaf^, nodes^
    )
    return t^


def hashed_vector_leaf(n_nodes: Int, num_outputs: Int) -> List[Float32]:
    """Every node gets a slot, internal ones included, exactly as cuML does, internal ones included
    (`decisiontree.cuh:218` indexes by raw node id). Filling the internal
    slots with hashed values too means a stride error that reads an internal
    node's slot produces a wrong number rather than a plausible one."""
    var v = List[Float32]()
    for i in range(n_nodes):
        for k in range(num_outputs):
            v.append(leaf_cell(i, k))
    return v^


def generated_rows(salt: Int) -> List[Float32]:
    var vals = value_set()
    var rows = List[Float32]()
    for r in range(N_GENERATED):
        for c in range(N_COLS):
            rows.append(vals[Int(fnv1a32(r * 131 + salt, c * 17 + 3) % 8)])
    return rows^


# ----------------------------------------------------------------------
# Counters
# ----------------------------------------------------------------------


@fieldwise_init
struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var cells: Int
    var bad: Int

    @staticmethod
    def zero() -> Self:
        return Self(0, 0)

    def eq_i(mut self, got: Int, expected: Int, what: String) raises:
        self.cells += 1
        if got != expected:
            self.bad += 1
            if self.bad <= 20:
                print(
                    "    FAIL",
                    what,
                    "got",
                    got,
                    "expected",
                    expected,
                )

    def eq_f(mut self, got: Float32, expected: Float32, what: String) raises:
        self.cells += 1
        if got != expected:
            self.bad += 1
            if self.bad <= 20:
                print(
                    "    FAIL",
                    what,
                    "got",
                    String(got),
                    "expected",
                    String(expected),
                )

    def eq_b(mut self, got: Bool, expected: Bool, what: String) raises:
        self.cells += 1
        if got != expected:
            self.bad += 1
            if self.bad <= 20:
                print(
                    "    FAIL", what, "got", String(got), "expected",
                    String(expected),
                )


# ----------------------------------------------------------------------
# Per-node structural transcription: every field, every accessor, every
# pointer reader, per node, against a hashed hand table.
# ----------------------------------------------------------------------


def check_structure(
    name: String,
    tree: TreeMetaDataNode[F32],
    colid: List[Int32],
    quesval: List[Float32],
    left: List[Int32],
    mut tally: Tally,
) raises:
    var p = tree.sparsetree.unsafe_ptr()
    for i in range(tree.num_nodes()):
        var nd = tree.sparsetree[i]
        var is_leaf = left[i] == NODE_IS_LEAF
        # `CreateLeafNode` writes {0, 0, 0, -1, count} (`flatnode.h:67`).
        var want_col: Int32 = 0 if is_leaf else colid[i]
        var want_ques: Float32 = 0.0 if is_leaf else quesval[i]
        var want_metric: Float32 = 0.0 if is_leaf else hashed_metric(i)

        tally.eq_i(Int(nd.ColumnId()), Int(want_col), name + " ColumnId")
        tally.eq_f(nd.QueryValue(), want_ques, name + " QueryValue")
        tally.eq_f(nd.BestMetric(), want_metric, name + " BestMetric")
        tally.eq_i(Int(nd.LeftChildId()), Int(left[i]), name + " LeftChildId")
        tally.eq_i(
            Int(nd.RightChildId()), Int(left[i]) + 1, name + " RightChildId"
        )
        tally.eq_i(
            Int(nd.InstanceCount()),
            Int(hashed_count(i)),
            name + " InstanceCount",
        )
        tally.eq_b(nd.IsLeaf(), is_leaf, name + " IsLeaf")

        # The field-by-field pointer readers must agree with the accessors;
        # this is the path device code takes and nothing else exercises it.
        tally.eq_i(
            Int(SparseTreeNode[F32].ColumnIdAt(p, i)),
            Int(nd.ColumnId()),
            name + " ColumnIdAt",
        )
        tally.eq_f(
            SparseTreeNode[F32].QueryValueAt(p, i),
            nd.QueryValue(),
            name + " QueryValueAt",
        )
        tally.eq_f(
            SparseTreeNode[F32].BestMetricAt(p, i),
            nd.BestMetric(),
            name + " BestMetricAt",
        )
        tally.eq_i(
            Int(SparseTreeNode[F32].LeftChildIdAt(p, i)),
            Int(nd.LeftChildId()),
            name + " LeftChildIdAt",
        )
        tally.eq_i(
            Int(SparseTreeNode[F32].RightChildIdAt(p, i)),
            Int(nd.RightChildId()),
            name + " RightChildIdAt",
        )
        tally.eq_i(
            Int(SparseTreeNode[F32].InstanceCountAt(p, i)),
            Int(nd.InstanceCount()),
            name + " InstanceCountAt",
        )
        tally.eq_b(
            SparseTreeNode[F32].IsLeafAt(p, i),
            nd.IsLeaf(),
            name + " IsLeafAt",
        )

        # The `vector_leaf` stride, checked DIRECTLY rather than only
        # end-to-end, so an off-by-one stride is caught as a wrong base
        # instead of as an out-of-range read.
        tally.eq_i(
            tree.leaf_base(i),
            i * Int(tree.num_outputs),
            name + " leaf_base",
        )
        tally.eq_b(
            nd == tree.sparsetree[i], True, name + " operator=="
        )


# ----------------------------------------------------------------------
# Per-row routing
# ----------------------------------------------------------------------


def check_routing(
    name: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    n_rows: Int,
    mut tally: Tally,
) raises -> Int:
    """Returns the number of rows that landed on a boundary (a row whose
    value at some tested node equalled that node's `quesval`), so the
    report can say the boundary case was actually REACHED and not merely
    written down. Rule: reach, not output."""
    var boundary_rows = 0
    for r in range(n_rows):
        var off = r * N_COLS
        var expected = walk_recursive(tree, rows, off, 0)
        var got = predict_leaf(tree, rows, off)
        tally.cells += 1
        if got != expected:
            tally.bad += 1
            if tally.bad <= 20:
                print(
                    "    FAIL",
                    name,
                    "row",
                    r,
                    "landed on node",
                    got,
                    "expected node",
                    expected,
                )
        # Did this row touch a threshold exactly?
        var idx = 0
        var touched = False
        while not tree.sparsetree[idx].IsLeaf():
            var nd = tree.sparsetree[idx]
            var x = rows[off + Int(nd.ColumnId())]
            if x == nd.QueryValue():
                touched = True
            if x <= nd.QueryValue():
                idx = Int(nd.LeftChildId())
            else:
                idx = Int(nd.RightChildId())
        if touched:
            boundary_rows += 1
    return boundary_rows


def check_hand_routing(
    name: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    expected_leaf: List[Int],
    mut tally: Tally,
) raises:
    """The hand-written table. `walk_recursive` and `predict_leaf` could in
    principle both be wrong in the same way; a table typed out from the tree
    on paper cannot be."""
    for r in range(len(expected_leaf)):
        var got = predict_leaf(tree, rows, r * N_COLS)
        tally.cells += 1
        if got != expected_leaf[r]:
            tally.bad += 1
            if tally.bad <= 20:
                print(
                    "    FAIL",
                    name,
                    "HAND row",
                    r,
                    "landed on node",
                    got,
                    "expected node",
                    expected_leaf[r],
                )


def check_regression_values(
    name: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    n_rows: Int,
    mut tally: Tally,
) raises:
    for r in range(n_rows):
        var off = r * N_COLS
        var node = walk_recursive(tree, rows, off, 0)
        tally.eq_f(
            predict_regression(tree, rows, off),
            leaf_cell(node, 0),
            name + " predict_regression row " + String(r),
        )


def check_multi_output_values(
    name: String,
    tree: TreeMetaDataNode[F32],
    rows: List[Float32],
    n_rows: Int,
    mut tally: Tally,
) raises:
    """`predict_all` cell by cell, plus the two DEVIATION 147 behaviours:
    the zeroing form overwrites a dirty buffer, and the accumulating form
    adds to it (which is what makes a forest sum)."""
    var k = Int(tree.num_outputs)
    var preds = List[Float32]()
    for i in range(n_rows * k):
        # Deliberately dirty, and hashed rather than constant.
        preds.append(leaf_cell(1000 + i, 3))
    predict_all(tree, rows, n_rows, N_COLS, preds, k)
    for r in range(n_rows):
        var node = walk_recursive(tree, rows, r * N_COLS, 0)
        for c in range(k):
            tally.eq_f(
                preds[r * k + c],
                leaf_cell(node, c),
                name + " predict_all row " + String(r) + " out " + String(c),
            )
    # DEVIATION 147: the accumulating form ADDS. Two passes must double.
    predict_all_accumulate(tree, rows, n_rows, N_COLS, preds, k)
    for r in range(n_rows):
        var node = walk_recursive(tree, rows, r * N_COLS, 0)
        for c in range(k):
            tally.eq_f(
                preds[r * k + c],
                leaf_cell(node, c) * 2.0,
                name + " accumulate row " + String(r) + " out " + String(c),
            )


# ----------------------------------------------------------------------


def main() raises:
    print("flatnode: cuML SparseTreeNode / TreeMetaDataNode / predict_one")
    print("upstream cuML 00094f7; `x <= quesval` goes LEFT")
    print("  (decisiontree.cuh:403-404, kLE at :214-215;")
    print("   sklearn 1.9.0 _partitioner.pyx:238 agrees)")
    print("")

    var total = Tally.zero()

    # ---------------- T0: a single node that is only a leaf -----------
    var t0_col: List[Int32] = [0]
    var t0_ques: List[Float32] = [0.0]
    var t0_left: List[Int32] = [NODE_IS_LEAF]
    var t0 = build_tree(0, t0_col, t0_ques, t0_left, 1, hashed_vector_leaf(1, 1))
    var t0_tally = Tally.zero()
    check_structure("T0", t0, t0_col, t0_ques, t0_left, t0_tally)
    var t0_rows = generated_rows(11)
    _ = check_routing("T0", t0, t0_rows, N_GENERATED, t0_tally)
    check_regression_values("T0", t0, t0_rows, N_GENERATED, t0_tally)
    # Every row must land on node 0; there is nowhere else.
    var t0_hand = List[Int]()
    for _ in range(N_GENERATED):
        t0_hand.append(0)
    check_hand_routing("T0", t0, t0_rows, t0_hand, t0_tally)
    print(
        "T0 leaf-only tree      cells",
        t0_tally.cells,
        " wrong",
        t0_tally.bad,
    )
    total.cells += t0_tally.cells
    total.bad += t0_tally.bad

    # ---------------- T1: a stump, col0 <= 5.0 ------------------------
    var t1_col: List[Int32] = [0, 0, 0]
    var t1_ques: List[Float32] = [5.0, 0.0, 0.0]
    var t1_left: List[Int32] = [1, NODE_IS_LEAF, NODE_IS_LEAF]
    var t1 = build_tree(1, t1_col, t1_ques, t1_left, 1, hashed_vector_leaf(3, 1))
    var t1_tally = Tally.zero()
    check_structure("T1", t1, t1_col, t1_ques, t1_left, t1_tally)
    # Hand rows. 5.0 is EXACTLY the threshold and must go LEFT.
    var t1_rows: List[Float32] = [
        4.9, 0.0, 0.0,
        5.0, 0.0, 0.0,
        5.001, 0.0, 0.0,
        10.0, 0.0, 0.0,
        -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        5.0, 9.0, 9.0,
        5.0, -9.0, -9.0,
    ]
    var t1_hand: List[Int] = [1, 1, 2, 2, 1, 1, 1, 1]
    check_hand_routing("T1", t1, t1_rows, t1_hand, t1_tally)
    var t1_hand_rows = len(t1_hand)
    var t1_hand_boundary = check_routing(
        "T1", t1, t1_rows, t1_hand_rows, t1_tally
    )
    check_regression_values("T1", t1, t1_rows, t1_hand_rows, t1_tally)
    var t1_gen = generated_rows(23)
    var t1_boundary = check_routing("T1", t1, t1_gen, N_GENERATED, t1_tally)
    check_regression_values("T1", t1, t1_gen, N_GENERATED, t1_tally)
    print(
        "T1 stump               cells",
        t1_tally.cells,
        " wrong",
        t1_tally.bad,
        " boundary rows reached",
        t1_boundary + t1_hand_boundary,
    )
    total.cells += t1_tally.cells
    total.bad += t1_tally.bad

    # ---------------- T2: asymmetric, mixed siblings ------------------
    #   0: col0 <= 1.0   -> 1 (split) / 2 (leaf)     MIXED SIBLINGS
    #   1: col1 <= 2.0   -> 3 (split) / 4 (leaf)     MIXED SIBLINGS
    #   3: col0 <= 0.0   -> 5 (leaf)  / 6 (leaf)
    var t2_col: List[Int32] = [0, 1, 0, 0, 0, 0, 0]
    var t2_ques: List[Float32] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var t2_left: List[Int32] = [
        1,
        3,
        NODE_IS_LEAF,
        5,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
    ]
    var t2 = build_tree(2, t2_col, t2_ques, t2_left, 1, hashed_vector_leaf(7, 1))
    var t2_tally = Tally.zero()
    check_structure("T2", t2, t2_col, t2_ques, t2_left, t2_tally)
    var t2_rows: List[Float32] = [
        1.0, 2.0, 0.0,     # both boundaries, both left -> 3 -> 1.0>0.0 -> 6
        0.0, 2.0, 0.0,     # left, left, 0.0<=0.0 left  -> 5
        1.0, 2.001, 0.0,   # left, right                -> 4
        1.001, 0.0, 0.0,   # right                      -> 2
        -3.0, -3.0, 0.0,   # left, left, left           -> 5
        1.0, -5.0, 0.0,    # left, left, 1.0>0.0 right  -> 6
    ]
    var t2_hand: List[Int] = [6, 5, 4, 2, 5, 6]
    check_hand_routing("T2", t2, t2_rows, t2_hand, t2_tally)
    var t2_hand_boundary = check_routing(
        "T2", t2, t2_rows, len(t2_hand), t2_tally
    )
    check_regression_values("T2", t2, t2_rows, len(t2_hand), t2_tally)
    var t2_gen = generated_rows(37)
    var t2_boundary = check_routing("T2", t2, t2_gen, N_GENERATED, t2_tally)
    check_regression_values("T2", t2, t2_gen, N_GENERATED, t2_tally)
    print(
        "T2 asymmetric depth 3  cells",
        t2_tally.cells,
        " wrong",
        t2_tally.bad,
        " boundary rows reached",
        t2_boundary + t2_hand_boundary,
    )
    total.cells += t2_tally.cells
    total.bad += t2_tally.bad

    # ---------------- T3: every threshold IS a data value -------------
    #   0: col0 <= 2.0    -> 1 / 2
    #   1: col1 <= -1.5   -> 3 / 4
    #   2: col1 <= 4.0    -> 5 / 6
    var t3_col: List[Int32] = [0, 1, 1, 0, 0, 0, 0]
    var t3_ques: List[Float32] = [2.0, -1.5, 4.0, 0.0, 0.0, 0.0, 0.0]
    var t3_left: List[Int32] = [
        1,
        3,
        5,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
    ]
    var t3 = build_tree(3, t3_col, t3_ques, t3_left, 1, hashed_vector_leaf(7, 1))
    var t3_tally = Tally.zero()
    check_structure("T3", t3, t3_col, t3_ques, t3_left, t3_tally)
    var t3_rows: List[Float32] = [
        2.0, -1.5, 0.0,   # BOTH exact -> left, left  -> 3
        2.0, -1.4, 0.0,   # exact, right              -> 4
        2.5, 4.0, 0.0,    # right, exact -> left      -> 5
        2.5, 4.5, 0.0,    # right, right             -> 6
        1.0, -2.0, 0.0,   # left, left               -> 3
        3.0, -1.5, 0.0,   # right, -1.5 <= 4.0 left  -> 5
        2.0, 4.0, 0.0,    # exact left, 4.0 > -1.5 right -> 4
    ]
    var t3_hand: List[Int] = [3, 4, 5, 6, 3, 5, 4]
    check_hand_routing("T3", t3, t3_rows, t3_hand, t3_tally)
    var t3_hand_boundary = check_routing(
        "T3", t3, t3_rows, len(t3_hand), t3_tally
    )
    check_regression_values("T3", t3, t3_rows, len(t3_hand), t3_tally)
    var t3_gen = generated_rows(53)
    var t3_boundary = check_routing("T3", t3, t3_gen, N_GENERATED, t3_tally)
    check_regression_values("T3", t3, t3_gen, N_GENERATED, t3_tally)
    print(
        "T3 boundary chain      cells",
        t3_tally.cells,
        " wrong",
        t3_tally.bad,
        " boundary rows reached",
        t3_boundary + t3_hand_boundary,
    )
    total.cells += t3_tally.cells
    total.bad += t3_tally.bad

    # ---------------- T4: 3 outputs, exact argmax ties ----------------
    #   0: col0 <= 0.0 -> 1 (split) / 2 (leaf)   MIXED SIBLINGS
    #   1: col1 <= 0.0 -> 3 (leaf)  / 4 (leaf)
    var t4_col: List[Int32] = [0, 1, 0, 0, 0]
    var t4_ques: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0]
    var t4_left: List[Int32] = [
        1,
        3,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
    ]
    # Leaf 2: all zero. `best_prob` starts at 0.0 and the test is STRICTLY
    #         greater (randomforest.cuh:245-247), so nothing ever wins and
    #         class 0 is returned by default.
    # Leaf 3: classes 0 and 2 tie at 0.75 -> LOWEST index wins -> 0.
    # Leaf 4: classes 1 and 2 tie at 0.875 -> LOWEST index wins -> 1.
    # Nodes 0 and 1 are internal; their slots exist (cuML gives every node
    # one) and are hashed so a stride error reading them is visible.
    var t4_leaf: List[Float32] = [
        leaf_cell(0, 0), leaf_cell(0, 1), leaf_cell(0, 2),
        leaf_cell(1, 0), leaf_cell(1, 1), leaf_cell(1, 2),
        0.0, 0.0, 0.0,
        0.75, 0.5, 0.75,
        0.125, 0.875, 0.875,
    ]
    var t4 = build_tree(4, t4_col, t4_ques, t4_left, 3, t4_leaf^)
    var t4_tally = Tally.zero()
    check_structure("T4", t4, t4_col, t4_ques, t4_left, t4_tally)
    var t4_rows: List[Float32] = [
        0.0, 0.0, 0.0,     # both exact -> left, left  -> node 3
        0.0, 0.5, 0.0,     # exact left, right         -> node 4
        -1.0, -1.0, 0.0,   # left, left                -> node 3
        0.5, 0.0, 0.0,     # right                     -> node 2
        7.0, 7.0, 7.0,     # right                     -> node 2
        -1.5, 0.0, 0.0,    # left, exact left          -> node 3
        -1.5, 1.0, 0.0,    # left, right               -> node 4
    ]
    var t4_hand: List[Int] = [3, 4, 3, 2, 2, 3, 4]
    var t4_class: List[Int] = [0, 1, 0, 0, 0, 0, 1]
    var t4_n = len(t4_hand)
    check_hand_routing("T4", t4, t4_rows, t4_hand, t4_tally)
    var t4_hand_boundary = check_routing("T4", t4, t4_rows, t4_n, t4_tally)
    for r in range(t4_n):
        t4_tally.eq_i(
            predict_class(t4, t4_rows, r * N_COLS),
            t4_class[r],
            "T4 predict_class row " + String(r),
        )
    # And the whole leaf vector, cell by cell, against a hand table.
    var t4_expected_vec: List[Float32] = [
        0.0, 0.0, 0.0,
        0.75, 0.5, 0.75,
        0.125, 0.875, 0.875,
    ]
    for r in range(t4_n):
        var node = t4_hand[r]
        for c in range(3):
            t4_tally.eq_f(
                t4.leaf_value(node, c),
                t4_expected_vec[(node - 2) * 3 + c],
                "T4 leaf_value node " + String(node) + " out " + String(c),
            )
    var t4_gen = generated_rows(71)
    var t4_boundary = check_routing("T4", t4, t4_gen, N_GENERATED, t4_tally)
    for r in range(N_GENERATED):
        var node = walk_recursive(t4, t4_gen, r * N_COLS, 0)
        var want = 0
        if node == 4:
            want = 1
        t4_tally.eq_i(
            predict_class(t4, t4_gen, r * N_COLS),
            want,
            "T4 gen predict_class row " + String(r),
        )
    print(
        "T4 argmax ties, k=3    cells",
        t4_tally.cells,
        " wrong",
        t4_tally.bad,
        " boundary rows reached",
        t4_boundary + t4_hand_boundary,
    )
    total.cells += t4_tally.cells
    total.bad += t4_tally.bad

    # ---------------- T5: 4 outputs, every leaf cell distinct ---------
    #   0: col0 <= 3.0 -> 1 / 2
    #   1: col1 <= 3.0 -> 3 / 4
    #   2: col2 <= 3.0 -> 5 / 6
    var t5_col: List[Int32] = [0, 1, 2, 0, 0, 0, 0]
    var t5_ques: List[Float32] = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    var t5_left: List[Int32] = [
        1,
        3,
        5,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
        NODE_IS_LEAF,
    ]
    var t5 = build_tree(5, t5_col, t5_ques, t5_left, 4, hashed_vector_leaf(7, 4))
    var t5_tally = Tally.zero()
    check_structure("T5", t5, t5_col, t5_ques, t5_left, t5_tally)
    var t5_gen = generated_rows(97)
    var t5_boundary = check_routing("T5", t5, t5_gen, N_GENERATED, t5_tally)
    check_multi_output_values("T5", t5, t5_gen, N_GENERATED, t5_tally)
    print(
        "T5 stride k=4          cells",
        t5_tally.cells,
        " wrong",
        t5_tally.bad,
        " boundary rows reached",
        t5_boundary,
    )
    total.cells += t5_tally.cells
    total.bad += t5_tally.bad

    # ---------------- the empty-tree assert ---------------------------
    var empty = TreeMetaDataNode[F32](
        9, 0, 0, 1, List[Float32](), List[SparseTreeNode[F32]]()
    )
    var empty_rows: List[Float32] = [0.0, 0.0, 0.0]
    var empty_preds: List[Float32] = [0.0]
    var raised = False
    try:
        predict_all(empty, empty_rows, 1, N_COLS, empty_preds, 1)
    except:
        raised = True
    total.cells += 1
    if not raised:
        total.bad += 1
        print("    FAIL empty tree did not raise (decisiontree.cuh:374-376)")
    print("empty-tree assert      cells 1  wrong", 0 if raised else 1)

    print("")
    print("TOTAL cells", total.cells, " wrong", total.bad)
    if total.bad != 0:
        raise Error(
            "flatnode_check: "
            + String(total.bad)
            + " of "
            + String(total.cells)
            + " cells wrong"
        )
    print("flatnode_check: OK")
