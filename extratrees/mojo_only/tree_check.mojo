"""An ExtraTree, end to end: fit, then predict, against hand-computable answers.

This is the first check in the lane that runs the WHOLE learner rather than a
piece of it — sampler, range pass, keyed draw, score pass, total-order select,
partition, frontier, leaf values — and it is the check that can catch the class
of bug none of the piece-wise checks can: a piece that is individually correct
and wired to the wrong neighbour.

The specific failure it exists for: every node in a batch must be partitioned
before the NEXT `pop`, because that is when its children become work items and
start reading `row_ids` over the ranges `Push` recorded. Defer the partitions
past the loop and every child of every node holds the wrong rows — while the
tree stays structurally perfect, every count conserves, and every piece-wise
check stays green. The pure-leaf assertions below are what see it.

Two NARROWER claims that a sabotage killed, kept here because a check should
say what it does not cover: partitioning after `queue.push` instead of before
is EQUIVALENT (`Push` records ranges, the partition moves rows, and inside one
batch iteration they commute), and partitioning an INVALID split is equivalent
too (a partition only permutes rows inside the node's own range, and an invalid
split leaves the node a leaf whose value depends on the set of rows and not
their order). Both are kept as cuML has them; neither is observable here.

What is asserted:

1. **The analytic separable-gap fixture**: class 0 lives in `[0,1)`, class 1 in
   `(9,10]`, nothing between, and a second column is pure noise. A correct
   learner splits on the separable column and every leaf comes out PURE, for
   every seed — because every threshold the RNG can draw inside the gap induces
   the same partition. Training accuracy must be 1.0, exactly.
2. **The regression step fixture**: `y` is one constant below the gap and
   another above it, so a correct learner's leaves have zero variance and
   predictions are exactly one of the two constants.
3. **The all-constant fixture**: nothing is splittable, so the tree is ONE
   LEAF. A learner that splits anyway is producing a tree from noise.
4. **`max_batch_size` cannot change the tree.** The frontier width is a
   scheduling parameter; the keyed draws are counter-based precisely so that it
   is (DEVIATION 130). Five widths, bit-identical trees, per node.
5. **The same seed gives the same tree and different seeds do not**, so the
   determinism above is not the determinism of a learner that ignores its seed.
6. **`max_depth` is respected** and every row lands in exactly one leaf.
"""

from std.testing import assert_equal, assert_true

from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    analytic_all_constant,
    analytic_regression_step,
    analytic_separable_gap,
    hashed_classification,
)
from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_class,
    predict_leaf,
    predict_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification,
    train_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset


@fieldwise_init
struct Fitted(Movable):
    """A fit plus the buffers it borrows, kept alive together.

    `Dataset`'s pointers are `MutUntrackedOrigin`, so the compiler tracks no
    relationship between them and the `List`s behind them and would free a
    buffer after its last syntactic use. Bundling them makes the lifetime
    explicit instead of depending on where the last mention happens to be.
    """

    var tree: TreeMetaDataNode[DType.float32]
    var x_col: List[Float32]
    var labels: List[Float32]
    var row_ids: List[Int32]


def column_major(fixture: FixtureDataset) -> List[Float32]:
    """Row-major fixture into cuML's column-major layout (`dataset.h:24`)."""
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def fit(
    fixture: FixtureDataset,
    params: DecisionTreeParams,
    seed: UInt64,
    tree_id: Int32,
    is_classification: Bool,
) raises -> Fitted:
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

    var num_outputs = Int32(fixture.n_classes) if is_classification else Int32(
        1
    )
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
        tree = train_classification(
            dataset, params, tree_id, seed, Int32(fixture.n_classes)
        )
    else:
        tree = train_regression(dataset, params, tree_id, seed)
    return Fitted(tree^, x_col^, labels^, row_ids^)


def row_of(fixture: FixtureDataset, r: Int) -> List[Float32]:
    var row = List[Float32]()
    for c in range(fixture.n_cols):
        row.append(fixture.value(r, c))
    return row^


def leaf_purity(
    fitted: Fitted, fixture: FixtureDataset
) raises -> Tuple[Int, Int]:
    """(pure leaves, total leaves), by routing every row and tallying classes
    per landing leaf — independently of anything the builder recorded."""
    var n_nodes = fitted.tree.num_nodes()
    var counts = List[Int](
        length=n_nodes * fixture.n_classes, fill=0
    )
    for r in range(fixture.n_rows):
        var row = row_of(fixture, r)
        var leaf = predict_leaf(fitted.tree, row, 0)
        counts[leaf * fixture.n_classes + Int(fixture.label[r])] += 1
    var pure = 0
    var total = 0
    for node in range(n_nodes):
        if not fitted.tree.sparsetree[node].IsLeaf():
            continue
        var nonzero = 0
        var seen = 0
        for c in range(fixture.n_classes):
            if counts[node * fixture.n_classes + c] > 0:
                nonzero += 1
                seen += counts[node * fixture.n_classes + c]
        if seen == 0:
            continue  # a leaf no row reaches is not impure
        total += 1
        if nonzero == 1:
            pure += 1
    return (pure, total)


def main() raises:
    var cells = 0

    # ---------------- 1. the separable-gap fixture ------------------------
    print("[separable gap] every leaf must be pure, at every seed")
    var gap = analytic_separable_gap(0x5EED)
    var p = DecisionTreeParams()
    p.max_depth = 8
    for seed_i in range(12):
        var f = fit(gap.data, p, UInt64(seed_i * 7919 + 13), 0, True)
        var pt = leaf_purity(f, gap.data)
        assert_equal(
            pt[0],
            pt[1],
            "seed "
            + String(seed_i)
            + ": "
            + String(pt[1] - pt[0])
            + " of "
            + String(pt[1])
            + " reached leaves are impure -- with a clean gap between the"
            " classes every threshold the RNG can draw separates them",
        )
        # and the model must actually predict them
        var wrong = 0
        for r in range(gap.data.n_rows):
            var row = row_of(gap.data, r)
            if predict_class(f.tree, row, 0) != Int(gap.data.label[r]):
                wrong += 1
        assert_equal(wrong, 0, "training accuracy on a separable fixture must be exact")
        assert_true(
            f.tree.num_nodes() >= 3, "a separable fixture must actually split"
        )
        cells += 3
    print("    12 seeds, all leaves pure, 0 rows misclassified")

    # ---------------- 2. the regression step fixture ----------------------
    print("[regression step] leaves must be exactly the two constants")
    var step = analytic_regression_step(0xBEE5)
    for seed_i in range(8):
        var f = fit(step.data, p, UInt64(seed_i * 104729 + 7), 0, False)
        var off = 0
        for r in range(step.data.n_rows):
            var row = row_of(step.data, r)
            var got = predict_regression(f.tree, row, 0)
            if got != step.data.y[r]:
                off += 1
        assert_equal(
            off,
            0,
            "seed "
            + String(seed_i)
            + ": a step function with a clean gap must be fitted exactly",
        )
        cells += 1
    print("    8 seeds, every row predicted exactly")

    # ---------------- 3. the all-constant fixture -------------------------
    print("[all constant] the tree must be a single leaf")
    var flat = analytic_all_constant()
    for seed_i in range(8):
        var f = fit(flat.data, p, UInt64(seed_i * 31337 + 5), 0, True)
        assert_equal(
            f.tree.num_nodes(),
            1,
            "nothing is splittable, so a split is a tree built from noise",
        )
        assert_true(f.tree.sparsetree[0].IsLeaf())
        assert_equal(f.tree.depth_counter, 0)
        assert_equal(f.tree.leaf_counter, 1)
        cells += 4
    print("    8 seeds, one node each")

    # ---------------- 4. the batch width cannot change the tree -----------
    print("[batch width] a scheduling parameter must not change the model")
    var hashed = hashed_classification(0xC0FFEE, 1024, 6, 3)
    var widths = [Int32(1), Int32(2), Int32(3), Int32(17), Int32(4096)]
    var ref_p = DecisionTreeParams()
    ref_p.max_depth = 6
    ref_p.max_batch_size = 4096
    var reference = fit(hashed, ref_p, 0xABCDEF, 3, True)
    for w in widths:
        var pw = DecisionTreeParams()
        pw.max_depth = 6
        pw.max_batch_size = w
        var f = fit(hashed, pw, 0xABCDEF, 3, True)
        assert_equal(
            f.tree.num_nodes(),
            reference.tree.num_nodes(),
            "batch width " + String(w) + " changed the node count",
        )
        for i in range(f.tree.num_nodes()):
            assert_true(
                f.tree.sparsetree[i] == reference.tree.sparsetree[i],
                "batch width " + String(w) + " changed node " + String(i),
            )
            cells += 1
        for i in range(len(f.tree.vector_leaf)):
            assert_equal(
                f.tree.vector_leaf[i].to_bits(),
                reference.tree.vector_leaf[i].to_bits(),
                "batch width " + String(w) + " changed a leaf value",
            )
            cells += 1
    print(
        "    ",
        len(widths),
        "widths x",
        reference.tree.num_nodes(),
        "nodes, bit-identical",
    )

    # ---------------- 5. the seed does something --------------------------
    var again = fit(hashed, ref_p, 0xABCDEF, 3, True)
    var same = True
    for i in range(again.tree.num_nodes()):
        if not (again.tree.sparsetree[i] == reference.tree.sparsetree[i]):
            same = False
    assert_true(same, "the same seed must give the same tree")
    var other = fit(hashed, ref_p, 0x123456, 3, True)
    var differs = other.tree.num_nodes() != reference.tree.num_nodes()
    if not differs:
        for i in range(other.tree.num_nodes()):
            if not (other.tree.sparsetree[i] == reference.tree.sparsetree[i]):
                differs = True
    assert_true(
        differs,
        "a different seed must give a different tree -- otherwise the"
        " determinism above is the determinism of a learner ignoring its seed",
    )
    # and the TREE ID is part of the key too, so a second tree at the same seed
    # must differ: that is what makes a forest a forest rather than 100 copies.
    var tree1 = fit(hashed, ref_p, 0xABCDEF, 4, True)
    var tree_differs = tree1.tree.num_nodes() != reference.tree.num_nodes()
    if not tree_differs:
        for i in range(tree1.tree.num_nodes()):
            if not (tree1.tree.sparsetree[i] == reference.tree.sparsetree[i]):
                tree_differs = True
    assert_true(
        tree_differs,
        "a different tree_id at the same seed must give a different tree, or"
        " a forest is 100 copies of one tree",
    )
    cells += 3

    # ---------------- 6. depth, and every row in exactly one leaf ---------
    for depth in [Int32(0), Int32(1), Int32(2), Int32(4), Int32(8)]:
        var pd = DecisionTreeParams()
        pd.max_depth = depth
        var f = fit(hashed, pd, 0x99, 1, True)
        assert_true(
            f.tree.depth_counter <= depth,
            "max_depth " + String(depth) + " exceeded",
        )
        if depth == 0:
            assert_equal(
                f.tree.num_nodes(), 1, "depth 0 is a single leaf"
            )
        var landed = List[Int](length=f.tree.num_nodes(), fill=0)
        for r in range(hashed.n_rows):
            var row = row_of(hashed, r)
            var leaf = predict_leaf(f.tree, row, 0)
            assert_true(
                f.tree.sparsetree[leaf].IsLeaf(),
                "predict_leaf must land on a leaf",
            )
            landed[leaf] += 1
        var total = 0
        for i in range(f.tree.num_nodes()):
            total += landed[i]
        assert_equal(
            total, hashed.n_rows, "every row must land in exactly one leaf"
        )
        cells += 3
        print(
            "    depth",
            depth,
            "->",
            f.tree.num_nodes(),
            "nodes,",
            f.tree.leaf_counter,
            "leaves, depth reached",
            f.tree.depth_counter,
        )

    # ---------------- 7. max_features < 1, so the SAMPLER actually runs ----
    # Until this section existed the whole feature-sampling path was dead in
    # this check: `max_features` defaults to 1.0, `n_sampled_cols_for` returns
    # every column, and `plan_feature_sampling` takes the all-features arm at
    # every node. A sabotage giving every node work-item 0's sampled columns
    # left the check GREEN, because with every column sampled there is nothing
    # per-node to get wrong.
    #
    # What discriminates: with k = 1 of 8 columns, each node draws ITS OWN
    # single column, so the internal nodes at one depth split on many
    # different columns. If nodes shared a batch's sample, a whole depth level
    # would collapse onto ONE column -- levels are exactly what a batch is.
    print("[max_features] k = 1 of 8: each node draws its own column")
    var wide = hashed_classification(0x77AA, 2048, 8, 2)
    var pf = DecisionTreeParams()
    pf.max_depth = 6
    pf.max_features = 0.125  # 1 of 8
    var ff = fit(wide, pf, 0xFEEDBEEF, 2, True)

    # depth of every node, by walking from the root
    var depth_of = List[Int](length=ff.tree.num_nodes(), fill=-1)
    depth_of[0] = 0
    for node in range(ff.tree.num_nodes()):
        if depth_of[node] < 0 or ff.tree.sparsetree[node].IsLeaf():
            continue
        var l = Int(ff.tree.sparsetree[node].LeftChildId())
        depth_of[l] = depth_of[node] + 1
        depth_of[l + 1] = depth_of[node] + 1

    var best_level_distinct = 0
    for d in range(1, 6):
        var seen = List[Int](length=Int(wide.n_cols), fill=0)
        var n_at_level = 0
        for node in range(ff.tree.num_nodes()):
            if depth_of[node] != d or ff.tree.sparsetree[node].IsLeaf():
                continue
            n_at_level += 1
            seen[Int(ff.tree.sparsetree[node].ColumnId())] = 1
        var distinct = 0
        for c in range(Int(wide.n_cols)):
            distinct += seen[c]
        if n_at_level >= 4:
            print(
                "    depth",
                d,
                ":",
                n_at_level,
                "internal nodes split on",
                distinct,
                "distinct columns",
            )
            if distinct > best_level_distinct:
                best_level_distinct = distinct
            cells += 1
    assert_true(
        best_level_distinct >= 2,
        "with k = 1 per node, some depth level must split on more than one"
        " column -- if every level collapses to one column then nodes are"
        " sharing a batch's sample instead of drawing their own",
    )
    # and the sampler must actually be changing the model
    var pfull = DecisionTreeParams()
    pfull.max_depth = 6
    var ffull = fit(wide, pfull, 0xFEEDBEEF, 2, True)
    var sampling_matters = ff.tree.num_nodes() != ffull.tree.num_nodes()
    if not sampling_matters:
        for i in range(ff.tree.num_nodes()):
            if not (ff.tree.sparsetree[i] == ffull.tree.sparsetree[i]):
                sampling_matters = True
    assert_true(
        sampling_matters,
        "max_features = 0.125 must give a different tree from 1.0",
    )
    cells += 2

    # THE FLOOR OF 1 IN `n_sampled_cols_for`. `builder.cuh:222` is
    # `max(1, IdxT(params.max_features * n_cols))`, and the `max` is not
    # decoration: at max_features = 0.05 on 8 columns the product truncates to
    # ZERO, and without the floor a node would get no candidate columns and
    # every tree would be a single leaf. Sabotaging the floor away left this
    # check green until this case existed, because every other case here uses
    # a max_features whose product is already >= 1.
    var ptiny = DecisionTreeParams()
    ptiny.max_depth = 4
    ptiny.max_features = 0.05  # 0.05 * 8 = 0.4, truncates to 0
    var ftiny = fit(wide, ptiny, 0x2468, 2, True)
    assert_true(
        ftiny.tree.num_nodes() > 1,
        "max_features below one column must still sample ONE column, not"
        " zero -- builder.cuh:222's max(1, ...)",
    )
    print(
        "    max_features 0.05 of 8 cols ->",
        ftiny.tree.num_nodes(),
        "nodes (the floor of 1 held)",
    )
    cells += 1

    print("tree: ", cells, "cells")
    print("tree_check: PASS")
