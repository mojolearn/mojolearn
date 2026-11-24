# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Bagging's draws and member weights as scikit-learn 1.9.1 makes them (lane w3f3-h, core/bag_draws.mojo):
sklearn/ensemble/_bagging.py:82-93 (a weighted draw: row i with probability w[i] / sum(w)), :164-172 (a member that
takes sample_weight gets `np.bincount(draw, minlength=n)`), sklearn/ensemble/_bootstrap.py:49-59 (a float max_samples
of a weighted fit is a fraction of sum(w)) and _bagging.py:468-472 (its refusal without replacement).

Fixtures: hand-written weights over 8 rows; no generator outside the module, no pack.

Fail-proof edits (each makes the named case fail): in `bag_counts_into` add 1.0 at `i` instead of `picks[i]` (case 1);
skip the class-row product (case 2); in `bag_draw_weighted` give a zero-weight row the key 0 instead of the largest
(case 3: a zero-weight row is drawn); write the rows in selection order instead of ascending (case 3); drop the
"Fewer non-zero entries" refusal (case 4: no raise); use `n` instead of `total` in `bag_rows_count` (case 5);
drop the negative-weight refusal in `bag_weights` (case 6: no raise); draw unit i on stream 2i instead of
2(first + i) in `bag_draw_units` (case 7).
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from core.bag_draws import bag_weights, bag_rows_count, bag_draw_weighted, bag_draw_units, bag_counts_into


def _weights() -> List[Float32]:
    var w: List[Float32] = [1.0, 0.0, 2.0, 0.5, 0.0, 3.0, 1.0, 0.25]
    return w^


def test_counts() raises:
    """Case 1: the draw [5, 0, 5, 7, 5] over 8 rows counts [1, 0, 0, 0, 0, 3, 0, 1]."""
    var draw: List[Int32] = [5, 0, 5, 7, 5]
    var c = List[Float32](length=8, fill=0.0)
    bag_counts_into(Int(c.unsafe_ptr()), Int(draw.unsafe_ptr()), 5, 8, 0)
    var want: List[Float32] = [1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0]
    for r in range(8):
        assert_equal(c[r], want[r])
    _ = len(draw)


def test_counts_times_class_rows() raises:
    """Case 2: the counts times the tree's class rows (row r: r + 1)."""
    var draw: List[Int32] = [5, 0, 5, 7, 5]
    var cw = List[Float32](length=8, fill=0.0)
    for r in range(8):
        cw[r] = Float32(r + 1)
    var c = List[Float32](length=8, fill=0.0)
    bag_counts_into(Int(c.unsafe_ptr()), Int(draw.unsafe_ptr()), 5, 8, Int(cw.unsafe_ptr()))
    assert_equal(c[0], Float32(1.0))
    assert_equal(c[5], Float32(18.0))
    assert_equal(c[7], Float32(8.0))
    assert_equal(c[3], Float32(0.0))
    _ = len(draw)
    _ = len(cw)


def test_draws_skip_zero_weights() raises:
    """Case 3: with and without replacement, no zero-weight row (1, 4) is drawn; without replacement the rows are
    distinct and ascending."""
    var w = _weights()
    var bw = bag_weights(Int(w.unsafe_ptr()), 8)
    assert_equal(bw.positive, 6)
    var rows = List[Int32](length=64, fill=Int32(-1))
    bag_draw_weighted(bw, 8, Int(rows.unsafe_ptr()), 64, True, 3, 0)
    for i in range(64):
        var r = Int(rows[i])
        assert_true(r >= 0 and r < 8 and r != 1 and r != 4)
    var four = List[Int32](length=4, fill=Int32(-1))
    bag_draw_weighted(bw, 8, Int(four.unsafe_ptr()), 4, False, 3, 2)
    for i in range(4):
        var r = Int(four[i])
        assert_true(r >= 0 and r < 8 and r != 1 and r != 4)
        if i > 0:
            assert_true(Int(four[i - 1]) < r)
    var six = List[Int32](length=6, fill=Int32(-1))
    bag_draw_weighted(bw, 8, Int(six.unsafe_ptr()), 6, False, 3, 4)
    var every: List[Int32] = [0, 2, 3, 5, 6, 7]
    for i in range(6):
        assert_equal(six[i], every[i])
    _ = len(w)


def test_fewer_positive_than_draw() raises:
    """Case 4: seven rows without replacement from six positive weights: numpy's words."""
    var w = _weights()
    var bw = bag_weights(Int(w.unsafe_ptr()), 8)
    var rows = List[Int32](length=7, fill=Int32(0))
    with assert_raises(contains="Fewer non-zero entries in p than size"):
        bag_draw_weighted(bw, 8, Int(rows.unsafe_ptr()), 7, False, 3, 0)
    _ = len(w)
    _ = len(rows)


def test_rows_count() raises:
    """Case 5: max_samples as scikit-learn 1.9.1 resolves it: None = n; an int as given, past n with replacement
    only; a fraction of sum(w) = 7.75 (weighted) or of n, at least 1."""
    var w = _weights()
    var bw = bag_weights(Int(w.unsafe_ptr()), 8)
    assert_equal(bw.total, 7.75)
    assert_equal(bag_rows_count(True, True, 0, 0.0, 8, bw.total, True), 8)
    assert_equal(bag_rows_count(False, True, 20, 0.0, 8, 8.0, True), 20)
    assert_equal(bag_rows_count(False, False, 0, 0.5, 8, bw.total, True), 3)
    assert_equal(bag_rows_count(False, False, 0, 0.01, 8, 8.0, True), 1)
    with assert_raises(contains="Effective max_samples=20 must be <= n_samples=8"):
        _ = bag_rows_count(False, True, 20, 0.0, 8, 8.0, False)
    with assert_raises(contains="max_samples resolves to 0"):
        _ = bag_rows_count(False, True, 0, 0.0, 8, 8.0, True)
    _ = len(w)


def test_units_equal_one_by_one() raises:
    """Case 7: `bag_draw_units` (units on the host cores) writes what each unit's own draw writes: unit i's rows on
    stream 2(first + i), its features on stream 2(first + i) + 1."""
    var w = _weights()
    var bw = bag_weights(Int(w.unsafe_ptr()), 8)
    var rows = List[List[Int32]]()
    var cols = List[List[Int32]]()
    for _ in range(3):
        rows.append(List[Int32](length=5, fill=Int32(-1)))
        cols.append(List[Int32](length=2, fill=Int32(-1)))
    var rows_at = List[Int]()
    var cols_at = List[Int]()
    for u in range(3):
        rows_at.append(Int(rows[u].unsafe_ptr()))
        cols_at.append(Int(cols[u].unsafe_ptr()))
    bag_draw_units(rows_at, cols_at, 4, bw, True, 8, 5, True, 3, 2, False, 11)
    for u in range(3):
        var one = List[Int32](length=5, fill=Int32(-1))
        bag_draw_weighted(bw, 8, Int(one.unsafe_ptr()), 5, True, 11, 2 * (4 + u))
        for i in range(5):
            assert_equal(rows[u][i], one[i])
        for j in range(2):
            var c = Int(cols[u][j])
            assert_true(c >= 0 and c < 3)
        assert_true(cols[u][0] != cols[u][1])
    _ = len(w)


def test_weight_refusals() raises:
    """Case 6: a negative weight and a zero total, numpy's words."""
    var neg: List[Float32] = [1.0, -0.5, 1.0]
    with assert_raises(contains="probabilities are not non-negative"):
        _ = bag_weights(Int(neg.unsafe_ptr()), 3)
    var zero: List[Float32] = [0.0, 0.0, 0.0]
    with assert_raises(contains="probabilities contain NaN"):
        _ = bag_weights(Int(zero.unsafe_ptr()), 3)
    _ = len(neg)
    _ = len(zero)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
