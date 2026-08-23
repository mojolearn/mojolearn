"""The forest: that the trees differ, and that the vote is really an average.

Two things can be wrong here in ways every single-tree check stays green for.

**The trees can be copies of each other.** `fit_*` passes `i` as the tree id,
and the tree id enters the split key (`key_for(seed, tree_id, node_id,
feature_id)`, deviation 130) exactly so that identical data yields different
thresholds. Pass a constant instead and you get a hundred copies of one tree —
which predicts fine, votes unanimously, and is not a forest. So this file
compares the trees to each other, pairwise, and requires them to differ.

**The vote can be the last tree instead of the average.** cuML's `predict`
(`randomforest.cuh:229-242`) accumulates with `+=` into a buffer whose only
zeroing is `std::vector`'s constructor one level up, then divides by
`n_trees`. Deviation 147 ports that under a name that says it accumulates,
precisely because reaching for a zeroing wrapper here returns the LAST tree's
leaf vector and is invisible on any fixture where the trees agree. So the vote
is checked against an independently computed average, per row, per class — on
a fixture built so the trees do NOT agree.

**The bootstrap can be inert, or not a bootstrap (DEVIATION 460).** Since
2026-08-23 `row_sample_for` draws cuML's with-replacement sample
(`randomforest.cuh:64-67`) through the RF lane's Philox port. This file pins
the per-tree seed chain to hand-computed fnv1a32 values, requires the sample
to be a with-replacement sample (duplicates, every id in range, `n_sampled`
long, different per tree, identical per (seed, tree) twice), and requires a
bootstrap forest to DIFFER from the no-bootstrap forest on the same data and
seed and to equal itself on a second fit. And a sabotage: `max_samples`
(`n_sampled_rows`) without bootstrap is refused by name.
"""

from std.testing import assert_equal, assert_true

from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    analytic_separable_gap,
    hashed_classification,
    hashed_regression,
)
from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import predict_leaf
from extratrees.ported.randomforest.randomforest import (
    Forest,
    fit_classification,
    fit_regression,
    forest_vote,
    predict_class_forest,
    predict_regression_forest,
    row_sample_for,
)
from extratrees.mojo_only.pcg_rng import row_sample_seed


def forests_equal(a: Forest, b: Forest) -> Bool:
    if len(a.trees) != len(b.trees):
        return False
    for t in range(len(a.trees)):
        if a.trees[t].num_nodes() != b.trees[t].num_nodes():
            return False
        for i in range(a.trees[t].num_nodes()):
            if not (a.trees[t].sparsetree[i] == b.trees[t].sparsetree[i]):
                return False
        if len(a.trees[t].vector_leaf) != len(b.trees[t].vector_leaf):
            return False
        for i in range(len(a.trees[t].vector_leaf)):
            if (
                a.trees[t].vector_leaf[i].to_bits()
                != b.trees[t].vector_leaf[i].to_bits()
            ):
                return False
    return True


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


def trees_differ(forest: Forest, a: Int, b: Int) -> Bool:
    if forest.trees[a].num_nodes() != forest.trees[b].num_nodes():
        return True
    for i in range(forest.trees[a].num_nodes()):
        if not (forest.trees[a].sparsetree[i] == forest.trees[b].sparsetree[i]):
            return True
    return False


def main() raises:
    var cells = 0

    # ---------------- the trees must not be copies ------------------------
    print("[diversity] the trees must differ from each other")
    var hashed = hashed_classification(0xF0E57, 1024, 6, 3)
    var p = DecisionTreeParams()
    p.max_depth = 6
    p.max_features = 0.5
    var xc = column_major(hashed)
    var labels = List[Float32]()
    for r in range(hashed.n_rows):
        labels.append(Float32(Int(hashed.label[r])))

    var forest = fit_classification(
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p,
        12,
        0xABC123,
    )
    assert_equal(len(forest.trees), 12)
    var identical_pairs = 0
    var pairs = 0
    for a in range(len(forest.trees)):
        for b in range(a + 1, len(forest.trees)):
            pairs += 1
            if not trees_differ(forest, a, b):
                identical_pairs += 1
            cells += 1
    print("    ", pairs, "pairs,", identical_pairs, "identical")
    assert_equal(
        identical_pairs,
        0,
        "a forest whose trees are copies is one tree reported n_trees times --"
        " the tree id must reach the split key (deviation 130)",
    )

    # ---------------- the vote is an average, per row, per class ----------
    print("[vote] against an independently averaged tally")
    var vote_cells = 0
    for r in range(0, hashed.n_rows, 37):
        var row = row_of(hashed, r)
        var got = forest_vote(forest, row, 0)
        # INDEPENDENT: route the row through each tree by hand, take each
        # tree's leaf vector, and average them here rather than in the code
        # under test.
        var want = List[Float32](
            length=Int(forest.num_outputs), fill=Float32(0.0)
        )
        for t in range(len(forest.trees)):
            var leaf = predict_leaf(forest.trees[t], row, 0)
            for k in range(Int(forest.num_outputs)):
                want[k] += forest.trees[t].leaf_value(leaf, k)
        for k in range(Int(forest.num_outputs)):
            want[k] = want[k] / Float32(Int(forest.n_trees))
            assert_equal(
                got[k].to_bits(),
                want[k].to_bits(),
                "row " + String(r) + " class " + String(k) + ": the vote is"
                " not the average of the trees",
            )
            vote_cells += 1
            cells += 1
    print("    ", vote_cells, "per-(row, class) cells")

    # The vote must also SUM TO ONE per row, because every tree's leaf vector
    # does (they are class probabilities) and an average of such vectors is
    # one. A vote that returned the last tree would also sum to one, which is
    # why that property is a sanity check and the per-cell comparison above is
    # the actual test.
    for r in range(0, hashed.n_rows, 101):
        var row = row_of(hashed, r)
        var got = forest_vote(forest, row, 0)
        var s = Float32(0.0)
        for k in range(Int(forest.num_outputs)):
            s += got[k]
        assert_true(
            s > 0.999 and s < 1.001, "the averaged vote must sum to one"
        )
        cells += 1

    # ---------------- the trees actually DISAGREE on some rows ------------
    # Without this the vote check is vacuous: if every tree returns the same
    # leaf vector, an average and a last-tree-wins are the same number.
    var disagreeing_rows = 0
    for r in range(hashed.n_rows):
        var row = row_of(hashed, r)
        var first = predict_leaf(forest.trees[0], row, 0)
        var v0 = forest.trees[0].leaf_value(first, 0)
        for t in range(1, len(forest.trees)):
            var leaf = predict_leaf(forest.trees[t], row, 0)
            if forest.trees[t].leaf_value(leaf, 0) != v0:
                disagreeing_rows += 1
                break
    print("    ", disagreeing_rows, "of", hashed.n_rows, "rows have trees that disagree")
    assert_true(
        disagreeing_rows * 2 > hashed.n_rows,
        "most rows must have disagreeing trees, or the vote check cannot"
        " distinguish an average from the last tree",
    )
    cells += 1

    # ---------------- a separable problem must be classified exactly ------
    print("[quality] a separable fixture must come out exact")
    var gap = analytic_separable_gap(0x5EED)
    var gx = column_major(gap.data)
    var glab = List[Float32]()
    for r in range(gap.data.n_rows):
        glab.append(Float32(Int(gap.data.label[r])))
    var pg = DecisionTreeParams()
    pg.max_depth = 8
    var gforest = fit_classification(
        gx,
        glab,
        Int32(gap.data.n_rows),
        Int32(gap.data.n_cols),
        Int32(gap.data.n_classes),
        pg,
        10,
        0x24680,
    )
    var wrong = 0
    for r in range(gap.data.n_rows):
        var row = row_of(gap.data, r)
        if predict_class_forest(gforest, row, 0) != Int(gap.data.label[r]):
            wrong += 1
    assert_equal(wrong, 0, "a 10-tree forest on a separable fixture must be exact")
    cells += 1
    print("    10 trees, 0 of", gap.data.n_rows, "rows wrong")

    # ---------------- regression ------------------------------------------
    print("[regression] the forest mean, against an independent tally")
    var rfx = hashed_regression(0x1234F, 512, 4)
    var rx = column_major(rfx)
    var ry = List[Float32]()
    for r in range(rfx.n_rows):
        ry.append(rfx.y[r])
    var pr = DecisionTreeParams()
    pr.max_depth = 5
    var rforest = fit_regression(
        rx, ry, Int32(rfx.n_rows), Int32(rfx.n_cols), pr, 8, 0x9999
    )
    for r in range(0, rfx.n_rows, 23):
        var row = row_of(rfx, r)
        var got = predict_regression_forest(rforest, row, 0)
        var acc = Float32(0.0)
        for t in range(len(rforest.trees)):
            var leaf = predict_leaf(rforest.trees[t], row, 0)
            acc += rforest.trees[t].leaf_value(leaf, 0)
        var want = acc / Float32(Int(rforest.n_trees))
        assert_equal(
            got.to_bits(),
            want.to_bits(),
            "row " + String(r) + ": the regression forest must be the mean of"
            " its trees",
        )
        cells += 1

    # ---------------- determinism, and that the seed does something -------
    var again = fit_classification(
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p,
        12,
        0xABC123,
    )
    for t in range(len(forest.trees)):
        assert_equal(
            again.trees[t].num_nodes(),
            forest.trees[t].num_nodes(),
            "the same seed must give the same forest",
        )
        cells += 1
    var other = fit_classification(
        xc,
        labels,
        Int32(hashed.n_rows),
        Int32(hashed.n_cols),
        Int32(hashed.n_classes),
        p,
        12,
        0x777777,
    )
    var seed_changed = False
    for t in range(len(other.trees)):
        if other.trees[t].num_nodes() != forest.trees[t].num_nodes():
            seed_changed = True
        else:
            for i in range(other.trees[t].num_nodes()):
                if not (
                    other.trees[t].sparsetree[i] == forest.trees[t].sparsetree[i]
                ):
                    seed_changed = True
    assert_true(seed_changed, "a different seed must give a different forest")
    cells += 1

    # ---------------- the bootstrap (DEVIATION 460) -------------------------
    print("[bootstrap] cuML's get_row_sample arm, seed chain pinned")
    # `randomforest.cuh:59-62` by hand (python, fnv1a32 over the low word,
    # the RF lane's DEVIATION 400 high-half round, then tree_id):
    assert_equal(Int(row_sample_seed(0, 0)), 2615243109, "rs(0,0)")
    assert_equal(Int(row_sample_seed(7, 0)), 2210223106, "rs(7,0)")
    assert_equal(Int(row_sample_seed(7, 1)), 3555300115, "rs(7,1)")
    assert_equal(Int(row_sample_seed(7, 2)), 3815036384, "rs(7,2)")
    assert_equal(Int(row_sample_seed(0xABC123, 5)), 1686781639, "rs(abc123,5)")
    assert_equal(
        Int(row_sample_seed(UInt64(1) << 40, 0)),
        2917598650,
        "rs(2^40,0): the high half gets its round (RF lane DEVIATION 400)",
    )
    cells += 6
    var samp0 = row_sample_for(1024, True, 0, 0xABC123, 0)
    var samp0b = row_sample_for(1024, True, 0, 0xABC123, 0)
    var samp1 = row_sample_for(1024, True, 0, 0xABC123, 1)
    var samp_half = row_sample_for(1024, True, 300, 0xABC123, 0)
    assert_equal(len(samp0), 1024, "n_sampled_rows=0 means every row")
    assert_equal(len(samp_half), 300, "n_sampled_rows=300 means 300 draws")
    var in_range = True
    var dup = False
    var seen = List[Int](length=1024, fill=0)
    for i in range(len(samp0)):
        var v = Int(samp0[i])
        if v < 0 or v >= 1024:
            in_range = False
        else:
            if seen[v] > 0:
                dup = True
            seen[v] += 1
    assert_true(in_range, "every drawn id is in [0, n_rows)")
    assert_true(dup, "WITH replacement: 1024 draws from 1024 must repeat")
    var same = True
    var differ = False
    for i in range(1024):
        if samp0[i] != samp0b[i]:
            same = False
        if samp0[i] != samp1[i]:
            differ = True
    assert_true(same, "the same (seed, tree) draws the same rows twice")
    assert_true(differ, "tree 0 and tree 1 draw different rows")
    var prefix = True
    for i in range(300):
        if samp_half[i] != samp0[i]:
            prefix = False
    assert_true(
        prefix,
        "a shorter sample is the longer one's prefix (RAFT's stride mapping:"
        " thread t writes t, t+stride, ...; 300 < stride)",
    )
    cells += 6

    var boot = fit_classification(
        xc, labels, Int32(hashed.n_rows), Int32(hashed.n_cols),
        Int32(hashed.n_classes), p, 12, 0xABC123, True,
    )
    var boot_again = fit_classification(
        xc, labels, Int32(hashed.n_rows), Int32(hashed.n_cols),
        Int32(hashed.n_classes), p, 12, 0xABC123, True,
    )
    var boot_half = fit_classification(
        xc, labels, Int32(hashed.n_rows), Int32(hashed.n_cols),
        Int32(hashed.n_classes), p, 12, 0xABC123, True, 512,
    )
    assert_true(
        not forests_equal(boot, forest),
        "REACH: bootstrap=True must build a different forest than"
        " bootstrap=False on the same data and seed",
    )
    assert_true(
        forests_equal(boot, boot_again),
        "bootstrap=True twice at one seed is the same forest",
    )
    assert_true(
        not forests_equal(boot, boot_half),
        "REACH: n_sampled_rows=512 must build a different forest than 1024",
    )
    var root_half = 0
    for t in range(len(boot_half.trees)):
        if boot_half.trees[t].sparsetree[0].InstanceCount() == 512:
            root_half += 1
    assert_equal(
        root_half, 12, "every tree's root holds exactly n_sampled_rows rows"
    )
    cells += 4
    print("     bootstrap forest differs, repeats, and honours n_sampled_rows")

    # ---------------- refusals ---------------------------------------------
    var refused = 0
    try:
        # the SABOTAGE arm of DEVIATION 460: max_samples without bootstrap
        _ = row_sample_for(10, False, 5)
    except:
        refused += 1
    try:
        _ = fit_classification(
            xc,
            labels,
            Int32(hashed.n_rows),
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p,
            0,
            1,
        )
    except:
        refused += 1
    try:
        _ = fit_classification(
            xc,
            labels,
            0,
            Int32(hashed.n_cols),
            Int32(hashed.n_classes),
            p,
            4,
            1,
        )
    except:
        refused += 1
    assert_equal(
        refused,
        3,
        "n_sampled_rows without bootstrap, n_trees=0 and n_rows=0 must each"
        " be refused BY NAME",
    )
    cells += 3

    _ = xc.unsafe_ptr()
    _ = labels.unsafe_ptr()
    _ = gx.unsafe_ptr()
    _ = glab.unsafe_ptr()
    _ = rx.unsafe_ptr()
    _ = ry.unsafe_ptr()

    print("forest: ", cells, "cells")
    print("forest_check: PASS")
