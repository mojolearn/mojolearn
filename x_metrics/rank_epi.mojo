# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l7-metrics (2026-10-04): op 61 `rank_epi`, the ranking and
probability metric tails, and op 62 `cl_epi`, the clustering tail, on the
device.

python/mojolearn/_expansion_metrics.py finished these in Python binary64
after the device folds (`_row_mean`'s division, the d2 scores' class
frequency denominators, the DCG discount table, ndcg's per-row ratios and
mean). Here they run in the program that made the folds, in binary64 as
integer arithmetic (checks/soft_f64.mojo: correctly rounded + - * / and
`sf64_log` = `_portable_math.log`), so the Apple GPU, NVIDIA, AMD and the
host column write the same words. Folds over k classes run in ascending
class order in ONE unit (k is the class count, a handful); maps over rows or
columns run one unit per item. Both numeric modes; the only route.

rank_epi q[0] = KIND:
- 0 MEAN (one unit): q = [0, TOT, N, SW, NORM, HALF, OUT]. v = TOT (a Float32
  PairSum, widened), divided by N (SW < 0) or by the Float32 weight total at
  SW when NORM, times 0.5 when HALF; binary64 at OUT.
- 1 NDCG_ROW (unit per row): q = [1, GAIN, IDEAL, n, OUT]. OUT[t] = the
  Float32 nearest gain / ideal (binary64 quotient), 0 when ideal is 0.
- 2 D2_LOG (one unit): q = [2, NUM, PER, WEIGHTED, k, OUT]. total = sum of
  the k class sums, den = sum over nonzero wc of wc * -log(clip(wc / total,
  eps, 1 - eps)), eps = 2^-23; OUT = 1 - NUM / den (binary64), NaN when the
  total or den is 0.
- 3 D2_BRIER (one unit): q = [3, TOT, N, SW, PER, WEIGHTED, k, OUT]. num =
  the mean Brier term (MEAN's arithmetic), freq_j = wc_j / total, den =
  (sum_c wc_c * sum_j (delta_cj - freq_j)^2) / total; OUT = 1 - num / den.
- 4 DCG_DISC (unit per column): q = [4, c, B, OUT]. OUT[t] = the Float32
  nearest 1 / (log(t + 2) / log(b)), b the binary64 log base at B.
"""
from checks.soft_f64 import (
    SF64_NAN, SF64_ZERO, SF64_ONE, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_log, sf64_neg, sf64_lt,
    sf64_gt, sf64_from_int, sf64_to_f32,
)
from x_metrics.common import FP, IP, p, st
from x_metrics.tail import st64, ld64, is0, sum_at, f32_64

comptime RANK_MEAN = 0
comptime RANK_NDCG_ROW = 1
comptime RANK_D2_LOG = 2
comptime RANK_D2_BRIER = 3
comptime RANK_DCG_DISC = 4

#: 2^-23 (the Float32 epsilon) and 1 - 2^-23, binary64
comptime EPS32 = UInt64(0x3E80000000000000)
comptime ONE_M_EPS32 = UInt64(0x3FEFFFFFC0000000)
comptime HALF64 = UInt64(0x3FE0000000000000)


@always_inline
def _mean(f: FP, tot: Int, n: Int, sw: Int, norm: Bool) -> UInt64:
    """`_row_mean`'s finish: total / n, or / the weight total at sw >= 0."""
    var v = f32_64(f, tot)
    if not norm:
        return v
    if sw >= 0:
        return sf64_div(v, f32_64(f, sw))
    return sf64_div(v, sf64_from_int(n))


@always_inline
def _total(f: FP, per: Int, weighted: Bool, k: Int) -> UInt64:
    var s = SF64_ZERO
    for c in range(k):
        s = sf64_add(s, sum_at(f, per, c, weighted))
    return s


@always_inline
def _d2_log(f: FP, q: IP):
    var per = p(q, 2)
    var weighted = p(q, 3) != 0
    var k = p(q, 4)
    var out = p(q, 5)
    var total = _total(f, per, weighted, k)
    if is0(total):
        st64(f, out, SF64_NAN)
        return
    var den = SF64_ZERO
    for c in range(k):
        var wc = sum_at(f, per, c, weighted)
        if is0(wc):
            continue
        var r = sf64_div(wc, total)
        if sf64_lt(r, EPS32):
            r = EPS32
        if sf64_gt(r, ONE_M_EPS32):
            r = ONE_M_EPS32
        den = sf64_add(den, sf64_mul(wc, sf64_neg(sf64_log(r))))
    if is0(den):
        st64(f, out, SF64_NAN)
        return
    st64(f, out, sf64_sub(SF64_ONE, sf64_div(f32_64(f, p(q, 1)), den)))


@always_inline
def _d2_brier(f: FP, q: IP):
    var per = p(q, 4)
    var weighted = p(q, 5) != 0
    var k = p(q, 6)
    var out = p(q, 7)
    var num = _mean(f, p(q, 1), p(q, 2), p(q, 3), True)
    var total = _total(f, per, weighted, k)
    if is0(total):
        st64(f, out, SF64_NAN)
        return
    var den = SF64_ZERO
    for c in range(k):
        var inner = SF64_ZERO
        for j in range(k):
            var d = sf64_sub(SF64_ONE if j == c else SF64_ZERO, sf64_div(sum_at(f, per, j, weighted), total))
            inner = sf64_add(inner, sf64_mul(d, d))
        den = sf64_add(den, sf64_mul(sum_at(f, per, c, weighted), inner))
    den = sf64_div(den, total)
    if is0(den):
        st64(f, out, SF64_NAN)
        return
    st64(f, out, sf64_sub(SF64_ONE, sf64_div(num, den)))


def rank_epi_unit(t: Int, f: FP, q: IP):
    var kind = p(q, 0)
    if kind == RANK_MEAN:
        if t != 0:
            return
        var v = _mean(f, p(q, 1), p(q, 2), p(q, 3), p(q, 4) != 0)
        if p(q, 5) != 0:
            v = sf64_mul(v, HALF64)
        st64(f, p(q, 6), v)
    elif kind == RANK_NDCG_ROW:
        if t >= p(q, 3):
            return
        var g = f32_64(f, p(q, 1) + t)
        var i = f32_64(f, p(q, 2) + t)
        var r = Float32(0)
        if not is0(i):
            r = sf64_to_f32(sf64_div(g, i))
        st(f, p(q, 4) + t, r)
    elif kind == RANK_D2_LOG:
        if t != 0:
            return
        _d2_log(f, q)
    elif kind == RANK_D2_BRIER:
        if t != 0:
            return
        _d2_brier(f, q)
    elif kind == RANK_DCG_DISC:
        if t >= p(q, 1):
            return
        var lb = sf64_log(ld64(f, p(q, 2)))
        var d = sf64_div(sf64_log(sf64_from_int(t + 2)), lb)
        st(f, p(q, 3) + t, sf64_to_f32(sf64_div(SF64_ONE, d)))


def cl_epi_unit(t: Int, f: FP, q: IP):
    """Op 62, the clustering tail (filled by the lane; empty until then)."""
    pass
