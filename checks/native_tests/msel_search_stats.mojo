# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed RUN of the search's score tables
(bindings/msel_search_stats_run.mojo), run natively (device wave, lane
dw-msel-parallel, 2026-11-21): row means and deviations, a fold's column,
the 'min' ranking (ties, NaN scores last), the best candidate and the
permutation p-value, against literal expected values.

Each case is a `test_*` of this TestSuite file (one process, its own binary). No Python participates.
"""
from std.testing import TestSuite
from bindings.msel_search_stats_run import (
    msel_pvalue, msel_rank_best, msel_rank_scores, msel_row_means, msel_row_stds, msel_score_column,
)
from checks.native_tests.testsuite_cases import require


def _table() -> List[List[Float64]]:
    # three candidates over four folds; every value is a binary fraction
    var rows = List[List[Float64]]()
    var a: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var b: List[Float64] = [0.5, 0.5, 0.5, 0.5]
    var c: List[Float64] = [-1.0, 1.0, -1.0, 1.0]
    rows.append(a^)
    rows.append(b^)
    rows.append(c^)
    return rows^


def test_means_and_deviations() raises:
    var rows = _table()
    var means = msel_row_means(rows)
    require(len(means) == 3 and means[0] == 2.5 and means[1] == 0.5 and means[2] == 0.0, "row means")
    var stds = msel_row_stds(rows, means)
    # population deviations: sqrt(1.25), 0, 1
    require(len(stds) == 3 and stds[1] == 0.0 and stds[2] == 1.0, "row deviations of constant and unit rows")
    require(stds[0] * stds[0] > 1.2499999 and stds[0] * stds[0] < 1.2500001, "row deviation of 1..4")
    var empty = List[List[Float64]]()
    require(len(msel_row_means(empty)) == 0 and len(msel_row_stds(empty, means)) == 0, "empty table")


def test_column() raises:
    var rows = _table()
    var column = msel_score_column(rows, 2)
    require(len(column) == 3 and column[0] == 3.0 and column[1] == 0.5 and column[2] == -1.0, "fold column")


def test_ranks() raises:
    var scores: List[Float64] = [0.25, 0.75, 0.25, 0.5, 0.75]
    var ranks = msel_rank_scores(scores)
    require(ranks[0] == 4 and ranks[1] == 1 and ranks[2] == 4 and ranks[3] == 3 and ranks[4] == 1, "min ranks with ties")
    require(msel_rank_best(ranks) == 1, "the first candidate of rank 1")
    var zeros: List[Float64] = [0.0, -0.0]
    var tied = msel_rank_scores(zeros)
    require(tied[0] == 1 and tied[1] == 1, "0.0 and -0.0 are one score")


def test_nan_ranks() raises:
    var zero = 0.0
    var nan = zero / zero
    var scores: List[Float64] = [0.5, 0.0, 0.25, 0.0]
    scores[1] = nan
    scores[3] = nan
    var ranks = msel_rank_scores(scores)
    require(ranks[0] == 1 and ranks[2] == 2 and ranks[1] == 3 and ranks[3] == 3, "NaN scores rank last, tied")
    var all_nan: List[Float64] = [0.0, 0.0]
    all_nan[0] = nan
    all_nan[1] = nan
    var last = msel_rank_scores(all_nan)
    require(last[0] == 1 and last[1] == 1 and msel_rank_best(last) == 0, "all NaN: every candidate ranks 1")
    var none = List[Int]()
    var refused = False
    try:
        _ = msel_rank_best(none)
    except problem:
        refused = "1 is not in list" in String(problem)
    require(refused, "no candidate of rank 1 is refused")


def test_pvalue() raises:
    var scores: List[Float64] = [0.25, 0.5, 0.75, 1.0]
    require(msel_pvalue(scores, 0.5, 4) == 0.8, "p-value counts scores >= the score")
    require(msel_pvalue(scores, 2.0, 4) == 0.2, "p-value with no hit")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
