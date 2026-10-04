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

Lane cpu2-l7-metrics S3b (2026-10-04), the ranking averages:
- 5 ROW_SUM (unit per row): q = [5, S, n, k, FLAG]. Row t's binary64 sum
  (ascending columns, correctly rounded adds of the exact widenings of the
  raw Float32 words) farther than 1e-8 + 1e-5 from one sets FLAG to 1
  (roc_auc_score's multiclass probability check; the host
  `x_metrics_row_sum_range` walk left the route).
- 6 CURVE_SCORE (unit per problem): q = [6, CF, P, FOLD, SC, FL, PER,
  WEIGHTED, SUP]. Problem t's curve-fold words at CF + CF_OUT*t (s hi, lo,
  F, T, the count word k, ..., the point count c at 9) become its score,
  binary64 at SC + 2t: FOLD 0 (AUC) s / (2F * T), NaN with FL[t] = 1 when
  c <= 0 or F or T is not positive; FOLD 1 (AP) max(0, s / T), 0 when
  c <= 0 or T == 0, FL[t] = 2 (score 0) when k > 0 (a zero precision
  denominator: the caller's curve rule). PER >= 0: SUP + 2t = the class
  support, `sum_at(PER, t, WEIGHTED)` (the per-class sums the program made).
- 7 AVERAGE (one unit): q = [7, SC, SUP, m, MODE, OUT]. MODE 0 (macro): the
  m binary64 scores at SC summed ascending, / m; MODE 1 (weighted): total =
  the ascending sum of the m binary64 weights at SUP, 0 when it is 0, else
  (the ascending sum of SC_i * SUP_i) / total. Binary64 at OUT.
- 8 OVO_MASK (unit per row): q = [8, C, n, a, b, W]. W[t] = 1 when the Int32
  code C[t] is a or b, else 0 (Float32): the pair's rows as curve weights,
  so the curve drops every other row on the device.
- 9 OVO_PAIR (unit per pair): q = [9, SC, OFF, k, n, G0, P, PS, PREV]. Pair
  G0 + t (pairs a < b in ascending order): PS + 2t = (SC_2t + SC_2t+1) / 2,
  PREV + 2t = (count_a + count_b) / n (OFF: the class group_sort table).

cl_epi q[0] = KIND (the clustering tail, from the per-cluster sums the same
program made; binary64 throughout, products correctly rounded, sums in
ascending order):
- 0 CENT (unit per cluster x column): q = [0, OFF, SUMS, k, d, C32, CB].
  cs = SUMS[t] / count_i; CB + 2t = cs, C32[t] = the Float32 nearest cs.
- 1 CH_ROW (unit per cluster): q = [1, CB, GSUM, n, d, k, INNER]. INNER + 2i
  = sum over c of (cs_ic - GSUM_c / n)^2.
- 2 CH_FIN (one unit): q = [2, OFF, INNER, PER, n, k, OUT]. extra = sum_i
  count_i * INNER_i, intra = sum_i PER_i (Float32 row-distance sums); OUT =
  1 when intra is 0, else extra * (n - k) / (intra * (k - 1)).
- 3 DB_DIST (unit per cluster pair i*k + j): q = [3, CB, k, d, DIST]. DIST +
  2t = sqrt(sum over c of (cs_ic - cs_jc)^2).
- 4 DB_FIN (one unit): q = [4, OFF, PER, DIST, k, OUT]. intra_i = PER_i /
  count_i; 0 when every intra or every dist is within 1e-8 of 0, else the
  ascending sum over i of max_j (intra_i + intra_j) / dist_ij (inf for a
  zero dist), / k.
"""
from checks.soft_f64 import (
    SF64_NAN, SF64_ZERO, SF64_ONE, SF64_INF, SF64_SIGN, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_log,
    sf64_neg, sf64_lt, sf64_gt, sf64_from_int, sf64_from_f32, sf64_to_f32, sf64_sqrt,
)
from x_metrics.common import FP, IP, p, st, ldi, sti
from x_metrics.par import CF_OUT
from x_metrics.tail import st64, ld64, is0, abs64, cnt_at, sum_at, f32_64

comptime RANK_MEAN = 0
comptime RANK_NDCG_ROW = 1
comptime RANK_D2_LOG = 2
comptime RANK_D2_BRIER = 3
comptime RANK_DCG_DISC = 4
comptime RANK_ROW_SUM = 5
comptime RANK_CURVE_SCORE = 6
comptime RANK_AVERAGE = 7
comptime RANK_OVO_MASK = 8
comptime RANK_OVO_PAIR = 9

comptime CL_CENT = 0
comptime CL_CH_ROW = 1
comptime CL_CH_FIN = 2
comptime CL_DB_DIST = 3
comptime CL_DB_FIN = 4

#: words per problem of the curve fold
comptime CF_WORDS = CF_OUT
#: 1e-8 + 1e-5 (scikit-learn's row-sum tolerance, as Python adds it) and 1e-8, binary64
comptime ROW_SUM_TOL = UInt64(0x3EE4FE13EC9BF514)
comptime TINY8 = UInt64(0x3E45798EE2308C3A)

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


@always_inline
def _le_tiny(v: UInt64, tiny: UInt64) -> Bool:
    """|v| <= tiny (a positive constant); False for NaN. Non-negative
    binary64 words order as their bits, and every NaN's bits exceed tiny's."""
    return abs64(v) <= tiny


def _row_sum(t: Int, f: FP, q: IP):
    var n = p(q, 2)
    if t >= n:
        return
    var k = p(q, 3)
    var base = p(q, 1) + t * k
    var s = SF64_ZERO
    for c in range(k):
        s = sf64_add(s, sf64_from_f32(f.unsafe_load(base + c)))
    # Python's `abs(s - 1) > tol` (a NaN row passes, as it did there)
    if sf64_gt(abs64(sf64_sub(s, SF64_ONE)), ROW_SUM_TOL):
        sti(f, p(q, 4), 1)


def _curve_score(t: Int, f: FP, q: IP):
    if t >= p(q, 2):
        return
    var o = p(q, 1) + CF_WORDS * t
    var sc = p(q, 4) + 2 * t
    var fl = p(q, 5) + t
    var per = p(q, 6)
    if per >= 0:
        st64(f, p(q, 8) + 2 * t, sum_at(f, per, t, p(q, 7) != 0))
    var s = sf64_add(f32_64(f, o), f32_64(f, o + 1))
    var F = f32_64(f, o + 2)
    var T = f32_64(f, o + 3)
    var k = ldi(f, o + 4)
    var c = ldi(f, o + 9)
    if p(q, 3) == 0:
        if c <= 0 or not (sf64_gt(F, SF64_ZERO) and sf64_gt(T, SF64_ZERO)):
            st64(f, sc, SF64_NAN)
            sti(f, fl, 1)
            return
        st64(f, sc, sf64_div(s, sf64_mul(sf64_mul(sf64_from_int(2), F), T)))
        sti(f, fl, 0)
        return
    sti(f, fl, 0)
    if c <= 0 or is0(T):
        st64(f, sc, SF64_ZERO)
        return
    if k > 0:
        st64(f, sc, SF64_ZERO)
        sti(f, fl, 2)
        return
    var v = sf64_div(s, T)
    st64(f, sc, v if sf64_gt(v, SF64_ZERO) else SF64_ZERO)


def _average(f: FP, q: IP):
    var SC = p(q, 1)
    var SUP = p(q, 2)
    var m = p(q, 3)
    var out = p(q, 5)
    if p(q, 4) == 0:
        var s = SF64_ZERO
        for i in range(m):
            s = sf64_add(s, ld64(f, SC + 2 * i))
        st64(f, out, sf64_div(s, sf64_from_int(m)))
        return
    var total = SF64_ZERO
    for i in range(m):
        total = sf64_add(total, ld64(f, SUP + 2 * i))
    if is0(total):
        st64(f, out, SF64_ZERO)
        return
    var num = SF64_ZERO
    for i in range(m):
        num = sf64_add(num, sf64_mul(ld64(f, SC + 2 * i), ld64(f, SUP + 2 * i)))
    st64(f, out, sf64_div(num, total))


def _ovo_pair(t: Int, f: FP, q: IP):
    if t >= p(q, 6):
        return
    var k = p(q, 3)
    var g = p(q, 5) + t
    var a = 0
    var b = 0
    for i in range(k):
        var cnt = k - 1 - i
        if g < cnt:
            a = i
            b = i + 1 + g
            break
        g -= cnt
    var SC = p(q, 1) + 4 * t
    var both = sf64_add(ld64(f, SC), ld64(f, SC + 2))
    st64(f, p(q, 7) + 2 * t, sf64_mul(both, HALF64))
    var off = p(q, 2)
    var m = cnt_at(f, off, a) + cnt_at(f, off, b)
    st64(f, p(q, 8) + 2 * t, sf64_div(sf64_from_int(m), sf64_from_int(p(q, 4))))


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
    elif kind == RANK_ROW_SUM:
        _row_sum(t, f, q)
    elif kind == RANK_CURVE_SCORE:
        _curve_score(t, f, q)
    elif kind == RANK_AVERAGE:
        if t != 0:
            return
        _average(f, q)
    elif kind == RANK_OVO_MASK:
        if t >= p(q, 2):
            return
        var c = ldi(f, p(q, 1) + t)
        st(f, p(q, 5) + t, Float32(1) if (c == p(q, 3) or c == p(q, 4)) else Float32(0))
    elif kind == RANK_OVO_PAIR:
        _ovo_pair(t, f, q)


@always_inline
def _sqdiff(a: UInt64, b: UInt64) -> UInt64:
    var e = sf64_sub(a, b)
    return sf64_mul(e, e)


def _cl_cent(t: Int, f: FP, q: IP):
    var k = p(q, 3)
    var d = p(q, 4)
    if t >= k * d:
        return
    var i = t // d
    var cs = sf64_div(f32_64(f, p(q, 2) + t), sf64_from_int(cnt_at(f, p(q, 1), i)))
    st64(f, p(q, 6) + 2 * t, cs)
    st(f, p(q, 5) + t, sf64_to_f32(cs))


def _cl_ch_row(t: Int, f: FP, q: IP):
    var d = p(q, 4)
    if t >= p(q, 5):
        return
    var CB = p(q, 1) + 2 * t * d
    var G = p(q, 2)
    var nn = sf64_from_int(p(q, 3))
    var s = SF64_ZERO
    for c in range(d):
        s = sf64_add(s, _sqdiff(ld64(f, CB + 2 * c), sf64_div(f32_64(f, G + c), nn)))
    st64(f, p(q, 6) + 2 * t, s)


def _cl_ch_fin(f: FP, q: IP):
    var off = p(q, 1)
    var INNER = p(q, 2)
    var PER = p(q, 3)
    var n = p(q, 4)
    var k = p(q, 5)
    var extra = SF64_ZERO
    var intra = SF64_ZERO
    for i in range(k):
        extra = sf64_add(extra, sf64_mul(sf64_from_int(cnt_at(f, off, i)), ld64(f, INNER + 2 * i)))
        intra = sf64_add(intra, f32_64(f, PER + i))
    if is0(intra):
        st64(f, p(q, 6), SF64_ONE)
        return
    var num = sf64_mul(extra, sf64_from_int(n - k))
    st64(f, p(q, 6), sf64_div(num, sf64_mul(intra, sf64_from_int(k - 1))))


def _cl_db_dist(t: Int, f: FP, q: IP):
    var k = p(q, 2)
    var d = p(q, 3)
    if t >= k * k:
        return
    var i = t // k
    var j = t - i * k
    var CB = p(q, 1)
    var s = SF64_ZERO
    for c in range(d):
        s = sf64_add(s, _sqdiff(ld64(f, CB + 2 * (i * d + c)), ld64(f, CB + 2 * (j * d + c))))
    st64(f, p(q, 4) + 2 * t, sf64_sqrt(s))


def _cl_db_fin(f: FP, q: IP):
    var off = p(q, 1)
    var PER = p(q, 2)
    var DIST = p(q, 3)
    var k = p(q, 4)
    var out = p(q, 5)
    var all_i = True
    for i in range(k):
        var intra = sf64_div(f32_64(f, PER + i), sf64_from_int(cnt_at(f, off, i)))
        if not _le_tiny(intra, TINY8):
            all_i = False
    var all_d = True
    for u in range(k * k):
        if not _le_tiny(ld64(f, DIST + 2 * u), TINY8):
            all_d = False
    if all_i or all_d:
        st64(f, out, SF64_ZERO)
        return
    var s = SF64_ZERO
    for i in range(k):
        var ii = sf64_div(f32_64(f, PER + i), sf64_from_int(cnt_at(f, off, i)))
        var best = SF64_INF | SF64_SIGN
        for j in range(k):
            var ij = sf64_div(f32_64(f, PER + j), sf64_from_int(cnt_at(f, off, j)))
            var dd = ld64(f, DIST + 2 * (i * k + j))
            var den = SF64_INF if is0(dd) else dd
            var v = sf64_div(sf64_add(ii, ij), den)
            if sf64_gt(v, best):  # Python's max(best, v): v only when it is larger
                best = v
        s = sf64_add(s, best)
    st64(f, out, sf64_div(s, sf64_from_int(k)))


def cl_epi_unit(t: Int, f: FP, q: IP):
    """Op 62, the clustering tail (module docstring, cl_epi kinds)."""
    var kind = p(q, 0)
    if kind == CL_CENT:
        _cl_cent(t, f, q)
    elif kind == CL_CH_ROW:
        _cl_ch_row(t, f, q)
    elif kind == CL_CH_FIN:
        if t != 0:
            return
        _cl_ch_fin(f, q)
    elif kind == CL_DB_DIST:
        _cl_db_dist(t, f, q)
    elif kind == CL_DB_FIN:
        if t != 0:
            return
        _cl_db_fin(f, q)
