"""The split record's total order, checked against an independent tally.

`Split.update` (`split.cuh:76-90`) is the ONLY thing that decides which
candidate survives a reduction whose order is unspecified. If it is not a
total order, two runs of the same fit can disagree while every other check
stays green, because every other check compares a chosen split against the
same chosen split.

So this check does not test `update` against itself. It builds a pile of
SCATTERED, HASHED candidates -- deliberately seeded with exact ties in the
metric, and exact ties in (metric, colid), because those are the branches that
exist -- reduces them in MANY different orders, and requires:

1. the reduction result is the same for every order (this is what makes the
   parallel builder reproducible), and
2. it equals the maximum under an INDEPENDENT comparator written as a plain
   lexicographic triple, which is a different expression of the same intent.

Rule 8: hashed values, per-cell comparison, and a sabotage per mechanism.
"""

from std.testing import assert_equal, assert_true

from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    split_not_valid,
)


def mix32(x: UInt32) -> UInt32:
    """A scatter hash. Any avalanche mix does; this is Murmur3's finalizer,
    used here only so the fixture values are unpredictable and distinct."""
    var h = x
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def independent_better(a: Split, b: Split) -> Bool:
    """Is `a` strictly better than `b`, by a plain lexicographic triple?

    Deliberately NOT written like `Split.update`: no early-out flag, no nested
    ifs, no assignment. If both expressions of the order agree on every pair,
    a transcription slip in one of them has to have been made twice.
    """
    if a.best_metric_val != b.best_metric_val:
        return a.best_metric_val > b.best_metric_val
    if a.colid != b.colid:
        return a.colid > b.colid
    return a.quesval > b.quesval


def make_candidates(n: Int, seed: UInt32) -> List[Split]:
    """Hashed candidates, with ties PLANTED at both tie branches.

    - every 7th candidate shares its metric with candidate 0, so the
      `colid` branch is reached;
    - every 11th shares BOTH metric and colid with candidate 1, so the
      `quesval` branch is reached.
    A fixture with no ties would leave two of the three branches dead, which
    is the "uniform data hides the defect" failure this repo has already paid
    for once.
    """
    var out = List[Split]()
    for i in range(n):
        var h = mix32(seed ^ UInt32(i))
        var metric = Float32(Int(h % 1000)) / 8.0
        var colid = Int32(Int(mix32(h) % 97))
        var quesval = Float32(Int(mix32(h ^ 0x9E3779B9) % 4096)) / 16.0
        out.append(Split(quesval, colid, metric, Int32(i + 1)))
    if n > 2:
        for i in range(n):
            if i % 7 == 0:
                out[i].best_metric_val = out[0].best_metric_val
            if i % 11 == 0:
                out[i].best_metric_val = out[1].best_metric_val
                out[i].colid = out[1].colid
    return out^


def reduce_in_order(candidates: List[Split], stride: Int) -> Split:
    """Reduce with `update`, visiting the candidates in a strided order.

    Striding by a value coprime with the count visits every candidate exactly
    once in a different sequence -- a stand-in for the arbitrary order a
    device reduction produces.
    """
    var best = Split()
    var n = len(candidates)
    var i = 0
    for _ in range(n):
        _ = best.update(candidates[i])
        i = (i + stride) % n
    return best^


def main() raises:
    var n = 233  # prime, so every stride below is coprime with it
    var candidates = make_candidates(n, 0xC0FFEE)

    # --- the independent tally: linear max under the plain comparator -----
    var expected = candidates[0]
    for i in range(1, n):
        if independent_better(candidates[i], expected):
            expected = candidates[i]

    # --- order independence, per order, per field -------------------------
    var orders = [1, 2, 3, 5, 7, 11, 13, 29, 101, 232]
    var checked = 0
    for stride in orders:
        var got = reduce_in_order(candidates, stride)
        assert_equal(got.best_metric_val, expected.best_metric_val)
        assert_equal(got.colid, expected.colid)
        assert_equal(got.quesval, expected.quesval)
        assert_equal(got.n_left, expected.n_left)
        checked += 4
    print("order independence:", len(orders), "orders x 4 fields =", checked, "cells")

    # --- the tie branches were actually reached ---------------------------
    var metric_ties = 0
    var colid_ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            if candidates[i].best_metric_val == candidates[j].best_metric_val:
                metric_ties += 1
                if candidates[i].colid == candidates[j].colid:
                    colid_ties += 1
    print("planted ties: metric", metric_ties, " metric+colid", colid_ties)
    assert_true(metric_ties > 0, "fixture planted no metric tie: two of three branches are dead")
    assert_true(colid_ties > 0, "fixture planted no (metric,colid) tie: the quesval branch is dead")

    # --- update() agrees with the comparator PAIRWISE, per cell -----------
    # Not a digest: every ordered pair is one cell, and both directions are
    # taken, so an asymmetric slip cannot hide.
    var pairs = 0
    for i in range(n):
        for j in range(n):
            var acc = candidates[i]
            var moved = acc.update(candidates[j])
            assert_equal(moved, independent_better(candidates[j], candidates[i]))
            pairs += 1
    print("pairwise agreement:", pairs, "ordered pairs")

    # --- a fresh Split never wins, and a real one always beats it ---------
    var empty = Split()
    assert_true(not empty.is_valid(), "default Split must report colid == -1")
    for i in range(n):
        var acc = Split()
        assert_true(acc.update(candidates[i]), "any real candidate must beat the default")

    # --- split_not_valid, the rejection branches, one cell each -----------
    # `builder_kernels.cuh:59-67`. Each assertion isolates ONE clause.
    var good = Split(1.0, 3, 0.5, 40)
    assert_true(not split_not_valid(good, 0.0, 1, 100), "a good split must pass")
    # DEVIATION 216: gain EQUAL to min_impurity_decrease PASSES -- sklearn's
    # boundary, replacing cuML's `<=` (this line pinned "REJECTED (their <=)"
    # until year's test MSE paid for the pruned zero-gain splits).
    assert_true(not split_not_valid(Split(1.0, 3, 0.0, 40), 0.0, 1, 100), "gain == min_impurity_decrease PASSES (sklearn boundary, 216)")
    assert_true(split_not_valid(Split(1.0, 3, -1.0, 40), 0.0, 1, 100), "negative gain rejected")
    assert_true(split_not_valid(Split(1.0, 3, Float32.MIN_FINITE, 40), 0.0, 1, 100), "the invalid-candidate sentinel rejected")
    assert_true(split_not_valid(good, 0.0, 41, 100), "left child below min_samples_leaf rejected")
    assert_true(split_not_valid(good, 0.0, 61, 100), "right child below min_samples_leaf rejected")
    print("split_not_valid: 5 clauses, one cell each")

    print("split_check: PASS")
