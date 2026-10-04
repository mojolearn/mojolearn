# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE fam2-prep-metrics (2026-10-04): THE LABEL METRICS' FLOAT EPILOGUE ON
THE DEVICE, IN BINARY64, THE SAME BITS ON EVERY VENDOR AND THE HOST COLUMN.

DEVIATION 650 kept the float epilogue of entropy, mutual_info_score and
adjusted_rand_index on the host (the Apple GPU has no Float64), reading the
integer histogram (k ints) or contingency matrix (k^2 ints) back. DEVIATION
651 made the IDENTICAL log-bearing epilogues Float32 through
`identical_log`, about seven digits against the Float64 reference.

Under IDN_METRIC_EPI (IDENTICAL, ON by default, -D
MOJOLEARN_IDN_METRIC_EPI_OFF or MOJOLEARN_IDN_ALL_OFF restores both
deviations' code) the epilogue is integer arithmetic on the binary64
encoding (checks/soft_f64.mojo: correctly rounded add, mul, div, fma and
the portable log), run in device kernels next to the counts, and one
binary64 word comes back instead of k or k^2 ints:

  entropy   term_i = (-p_i) * log(p_i), p_i = count_i / size (zero skipped)
  MI        term_ij = c_ij * (log(size * c_ij) - log(a_i * b_j)) (zero cells
            skipped), the fold divided by size
  ARI       the three pair-count sums as exact Int64, then the five Float64
            operations of adjusted_rand_index.mojo as soft binary64: the
            IEEE results the host computed, so ARI's value does not move.

THE FOLD ORDER (entropy, MI), one order everywhere: terms are added
ascending from zero inside chunks of EPI_CH, and the chunk sums are folded
by levels the same way (EPI_CH sums into one, at least one level, until one
sum is left). On the device one thread owns a chunk (it computes the chunk's
terms too) and one thread owns each sum of a level; the last level applies
the final division: no serial chain over all k^2 cells. The host column
(`*_sf_list`, called by metrics/host/metrics_oracle.mojo and by the traced
entries) runs the same term functions in the same order.

BITS: entropy and MI change (binary64 terms and the chunked order replace
the Float32 serial fold); ACCURACY improves from Float32 to binary64
against the reference. ARI is unchanged.

THIS FILE is the arithmetic (no GPU import: the host oracle and the CPU-only
binding import it); the kernels and device entries are sf_epilogue.mojo.
"""
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.soft_f64 import (
    SF64_ZERO, SF64_ONE, _norm_round_pack, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_fma, sf64_neg, sf64_log,
)

comptime IDN_METRIC_EPI = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_METRIC_EPI_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

#: terms a chunk folds (a constant of the source, never of a launch)
comptime EPI_CH = 64


comptime _SF_HALF = UInt64(0x3FE0000000000000)
comptime _SF_TWO = UInt64(0x4000000000000000)


@always_inline
def sf64_from_i64(i: Int) -> UInt64:
    """SoftFloat's `i64_to_f64`: the correctly rounded binary64 of any
    |i| < 2^63 (checks/soft_f64.mojo `sf64_from_int` is exact only below
    2^53; a_i * b_j and the pair-count sums reach 2^62)."""
    if i == 0:
        return SF64_ZERO
    var s = UInt64(1) if i < 0 else UInt64(0)
    var u = UInt64(-i) if i < 0 else UInt64(i)
    return _norm_round_pack(s, 0x43C, u)


@always_inline
def sf_is0(v: UInt64) -> Bool:
    return (v << UInt64(1)) == UInt64(0)


@always_inline
def entropy_term(count: Int, size: Int) -> UInt64:
    """(-p) * log(p), p = count / size; +0.0 for an empty class."""
    if count == 0:
        return SF64_ZERO
    var pr = sf64_div(sf64_from_i64(count), sf64_from_i64(size))
    if sf_is0(pr):
        return SF64_ZERO
    return sf64_mul(sf64_neg(pr), sf64_log(pr))


@always_inline
def mi_term(cij: Int, ab: Int, size: Int) -> UInt64:
    """c * (log(size * c) - log(a * b)); +0.0 for an empty cell."""
    if ab == 0 or cij == 0:
        return SF64_ZERO
    var dc = sf64_from_i64(cij)
    var l1 = sf64_log(sf64_mul(sf64_from_i64(size), dc))
    var l2 = sf64_log(sf64_from_i64(ab))
    return sf64_mul(dc, sf64_sub(l1, l2))


@always_inline
def n_c_two_i(v: Int) -> Int:
    """`nCTwo` (adjusted_rand_index.mojo `n_c_two`)."""
    if v % 2 != 0:
        return ((v - 1) >> 1) * v
    return (v >> 1) * (v - 1)


@always_inline
def ari_value(n_choose_two_sum: Int, a_c_two_sum: Int, b_c_two_sum: Int, size: Int) -> UInt64:
    """adjusted_rand_index.mojo's five Float64 operations, as soft binary64
    (the same IEEE results; the fused `(a + b) * 0.5 - expected` included)."""
    var n_choose_two = sf64_div(sf64_mul(sf64_from_i64(size), sf64_from_i64(size - 1)), _SF_TWO)
    var expected = sf64_div(sf64_mul(sf64_from_i64(a_c_two_sum), sf64_from_i64(b_c_two_sum)), n_choose_two)
    var span = sf64_fma(
        sf64_add(sf64_from_i64(b_c_two_sum), sf64_from_i64(a_c_two_sum)), _SF_HALF, sf64_neg(expected)
    )
    if sf_is0(span):
        return SF64_ZERO
    return sf64_div(sf64_sub(sf64_from_i64(n_choose_two_sum), expected), span)


# ---------------------------------------------------------------- the host column (lists)
def entropy_sf_parts(counts: List[Int32], size: Int) -> List[UInt64]:
    """The chunk sums of the entropy terms (EPI_CH classes a chunk)."""
    var k = len(counts)
    var parts = List[UInt64]()
    var c0 = 0
    while c0 < k:
        var hi = min(c0 + EPI_CH, k)
        var acc = SF64_ZERO
        for i in range(c0, hi):
            acc = sf64_add(acc, entropy_term(Int(counts[i]), size))
        parts.append(acc)
        c0 += EPI_CH
    return parts^


def mi_sf_parts(c: List[Int32], a: List[Int64], b: List[Int64], k: Int, size: Int) -> List[UInt64]:
    """The chunk sums of the MI terms over the row-major cells."""
    var m = k * k
    var parts = List[UInt64]()
    var c0 = 0
    while c0 < m:
        var hi = min(c0 + EPI_CH, m)
        var acc = SF64_ZERO
        for t in range(c0, hi):
            var i = t // k
            var j = t - i * k
            acc = sf64_add(acc, mi_term(Int(c[t]), Int(a[i]) * Int(b[j]), size))
        parts.append(acc)
        c0 += EPI_CH
    return parts^


def sf_fold_list(parts: List[UInt64]) -> UInt64:
    """The fold of the chunk sums, by levels: EPI_CH sums are added
    ascending from zero into one sum of the next level; at least one level,
    then until one sum is left (sf_epilogue.mojo `sf_level_kernel` runs one
    thread per sum of a level)."""
    var cur = parts.copy()
    while True:
        var nxt = List[UInt64]()
        var c0 = 0
        while c0 < len(cur):
            var hi = min(c0 + EPI_CH, len(cur))
            var acc = SF64_ZERO
            for i in range(c0, hi):
                acc = sf64_add(acc, cur[i])
            nxt.append(acc)
            c0 += EPI_CH
        cur = nxt^
        if len(cur) <= 1:
            break
    if len(cur) == 0:
        return SF64_ZERO
    return cur[0]


def sf_to_f64(bits: UInt64) -> Float64:
    return bitcast[DType.float64](bits)


def entropy_sf_list(counts: List[Int32], size: Int) -> Float64:
    return sf_to_f64(sf_fold_list(entropy_sf_parts(counts, size)))


def mi_sf_list(c: List[Int32], a: List[Int64], b: List[Int64], k: Int, size: Int) -> Float64:
    return sf_to_f64(sf64_div(sf_fold_list(mi_sf_parts(c, a, b, k, size)), sf64_from_i64(size)))
