# SPDX-License-Identifier: Apache-2.0
"""RUN of the search's score tables and of the permutation test's p-value
(device wave, lane dw-msel-parallel, 2026-11-21): the per-candidate mean and
standard deviation over the folds, the candidates' ranks, the index of the
best candidate and the p-value, on native Float64 tables.

The scores are the door's (bindings/msel_search_door.mojo: each is one
member call's result); the door packs what is computed here into
`cv_results_`. The means and deviations are the host passes `row_means_f64`
and `row_stds_f64` (core/array_helpers_impl.mojo), unchanged.

Nothing here is split across cores: a table is candidates x folds Float64
words (a few kilobytes), its passes are linear and the fits that produced it
are what the search's workers run in parallel.

No Python object is read, built or called here.
"""
from std.builtin.sort import sort
from core.array_helpers_impl import row_means_f64, row_stds_f64


def _score_ptr(values: List[Float64]) -> MutPointer[Float64, MutUntrackedOrigin]:
    """The address of a score buffer the caller keeps alive past the pass."""
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def _score_table(rows: List[List[Float64]], m: Int, k: Int) -> List[Float64]:
    """The m x k table of `rows` in one C-order buffer."""
    var flat = List[Float64](length=m * k, fill=0.0)
    for i in range(m):  # driver-loop: over candidates
        for j in range(k):  # driver-loop: over folds
            flat[i * k + j] = rows[i][j]
    return flat^


def msel_row_means(rows: List[List[Float64]]) raises -> List[Float64]:
    """`fsum(row) / len(row)` of each row (`row_means_f64`)."""
    var m = len(rows)
    var k = len(rows[0]) if m else 0
    var res = List[Float64](length=m, fill=0.0)
    if m and k:
        var flat = _score_table(rows, m, k)
        if row_means_f64(_score_ptr(flat), _score_ptr(res), m, k):
            raise Error("-inf + inf in fsum")
        _ = flat  # keep the storage alive through the pass above
    return res^


def msel_row_stds(rows: List[List[Float64]], means: List[Float64]) raises -> List[Float64]:
    """Each row's population standard deviation about its mean
    (`row_stds_f64`)."""
    var m = len(rows)
    var res = List[Float64](length=m, fill=0.0)
    if m == 0:
        return res^
    var k = len(rows[0])
    if k == 0:
        return res^
    var flat = _score_table(rows, m, k)
    var mu = means.copy()
    row_stds_f64(_score_ptr(flat), _score_ptr(mu), _score_ptr(res), m, k)
    _ = flat  # keep the storage alive through the pass above
    _ = mu
    return res^


def msel_score_column(rows: List[List[Float64]], fold: Int) -> List[Float64]:
    """Fold `fold`'s score of every candidate (one `split<i>_test_score`
    column)."""
    var column = List[Float64](capacity=len(rows))
    for c in range(len(rows)):  # driver-loop: over candidates
        column.append(rows[c][fold])
    return column^


def msel_rank_scores(values: List[Float64]) -> List[Int]:
    """scikit-learn's 'min' ranking, descending (1 is best): a candidate's
    rank is one more than the number of candidates with a greater score, so
    equal scores share a rank. A NaN score (a failed candidate under
    `error_score=nan`) is greater than nothing: every NaN ranks after every
    number, tied with the other NaNs (scikit-learn's rule)."""
    var m = len(values)
    var numbers = List[Float64](capacity=m)
    for i in range(m):  # driver-loop: over candidates
        if values[i] == values[i]:
            numbers.append(values[i])
    sort(numbers)
    var count = len(numbers)
    var ranks = List[Int](length=m, fill=count + 1)
    for i in range(m):  # driver-loop: over candidates (one binary search each)
        var v = values[i]
        if v != v:
            continue
        # the first position past every number <= v: the numbers above it are greater
        var lo = 0
        var hi = count
        while lo < hi:  # driver-loop: binary search over the sorted scores
            var mid = (lo + hi) // 2
            if numbers[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        ranks[i] = count - lo + 1
    return ranks^


def msel_rank_best(ranks: List[Int]) raises -> Int:
    """The first candidate of rank 1 (`ranks.index(1)`)."""
    for i in range(len(ranks)):  # driver-loop: over candidates
        if ranks[i] == 1:
            return i
    raise Error("ValueError: 1 is not in list")


def msel_pvalue(scores: List[Float64], score: Float64, n_permutations: Int) -> Float64:
    """`(count(permutation score >= score) + 1) / (n_permutations + 1)`."""
    var hits = 0
    for i in range(len(scores)):  # driver-loop: over permutations
        if scores[i] >= score:
            hits += 1
    return (Float64(hits) + 1.0) / Float64(n_permutations + 1)
