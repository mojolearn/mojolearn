# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Regression units: per-element loss terms (scikit-learn 1.9
`sklearn/metrics/_regression.py`), a per-column stable sort, the weighted
percentile (`sklearn/utils/stats.py::_weighted_percentile`) and a column
maximum. The per-column (weighted) means of the terms are x_metrics/group.mojo's
`group_sum_unit` over one group; the ratios are the Python epilogue.
"""
from std.memory import bitcast
from x_metrics.common import FP, IP, p, ld, st, ldi, sti, key, fadd
from checks.numerics import (
    ftz, identical_mul, identical_div, identical_log, identical_log1p, identical_pow,
)

comptime TERM_SQ = 0          # (y - p)^2
comptime TERM_ABS = 1         # |y - p|
comptime TERM_SQLOG = 2       # (log1p y - log1p p)^2
comptime TERM_APE = 3         # |p - y| / max(|y|, eps64)
comptime TERM_PINBALL = 4     # alpha * max(d, 0) - (1 - alpha) * min(d, 0), d = y - p
comptime TERM_TWEEDIE = 5     # the unit Tweedie deviance at power S
comptime TERM_DIFF = 6        # y - p

#: numpy's float64 machine epsilon, which scikit-learn's MAPE floors |y| at;
#: a normal Float32, so it survives the flush. DEVIATION 6103: the floor is
#: also why APE never computes 0/0 (y = p = 0 gives +0.0).
comptime EPS64 = Float32(2.220446049250313e-16)


def _tweedie(y: Float32, mu: Float32, pw: Float32) -> Float32:
    """scikit-learn `_mean_tweedie_deviance`'s per-row term. The caller has
    validated the domain (mu > 0; y >= 0 for 1 <= power < 2; y > 0 for
    power >= 2), so no branch below divides by zero or logs a non-positive."""
    if pw == Float32(0):
        var d = ftz(y - mu)
        return identical_mul(d, d)
    if pw == Float32(1):
        # 2 * (xlogy(y, y / mu) - y + mu); xlogy(0, .) = 0
        var xl = Float32(0)
        if y != Float32(0):
            xl = identical_mul(y, identical_log(identical_div(y, mu)))
        return identical_mul(Float32(2), ftz(ftz(xl - y) + mu))
    if pw == Float32(2):
        var a = identical_log(identical_div(mu, y))
        return identical_mul(Float32(2), ftz(ftz(a + identical_div(y, mu)) - Float32(1)))
    var one_m = ftz(Float32(1) - pw)
    var two_m = ftz(Float32(2) - pw)
    var yp = y if y > Float32(0) else Float32(0)
    var t1 = identical_div(identical_pow(yp, two_m), identical_mul(one_m, two_m))
    var t2 = identical_div(identical_mul(y, identical_pow(mu, one_m)), one_m)
    var t3 = identical_div(identical_pow(mu, two_m), two_m)
    return identical_mul(Float32(2), ftz(ftz(t1 - t2) + t3))


def reg_term_unit(t: Int, f: FP, q: IP):
    """q = [Y, P, OUT, D, kind, S, PB]; t = r*D + c over n*D elements.
    P is per element, or per column when PB = 1 (a broadcast mean or
    quantile). S is the arena slot of the kind's scalar (alpha, power)."""
    var Y = p(q, 0)
    var P = p(q, 1)
    var OUT = p(q, 2)
    var D = p(q, 3)
    var kind = p(q, 4)
    var S = p(q, 5)
    var PB = p(q, 6)
    var c = t % D
    var y = ld(f, Y + t)
    var pr = ld(f, P + (c if PB == 1 else t))
    var v: Float32
    if kind == TERM_SQ:
        var d = ftz(y - pr)
        v = identical_mul(d, d)
    elif kind == TERM_ABS:
        v = abs(ftz(y - pr))
    elif kind == TERM_SQLOG:
        var d = ftz(identical_log1p(y) - identical_log1p(pr))
        v = identical_mul(d, d)
    elif kind == TERM_APE:
        var ay = abs(y)
        v = identical_div(abs(ftz(pr - y)), ay if ay > EPS64 else EPS64)
    elif kind == TERM_PINBALL:
        var a = ld(f, S)
        var d = ftz(y - pr)
        if d >= Float32(0):
            v = identical_mul(a, d)
        else:
            v = identical_mul(ftz(a - Float32(1)), d)
    elif kind == TERM_TWEEDIE:
        v = _tweedie(y, pr, ld(f, S))
    else:
        v = ftz(y - pr)
    st(f, OUT + t, v)


@always_inline
def _before(f: FP, V: Int, D: Int, c: Int, i: Int, j: Int) -> Bool:
    """Row i sorts before row j in column c: by `key`, ties by row index."""
    var ki = key(ld(f, V + i * D + c))
    var kj = key(ld(f, V + j * D + c))
    return ki < kj or (ki == kj and i < j)


def col_sort_unit(t: Int, f: FP, q: IP):
    """q = [V, n, D, ORD]; t = column. ORD[t*n .. t*n+n) = the rows of column
    t in ascending (key, row) order, by heapsort (DEVIATION 6101: a strict
    total order, so every correct sort gives these indices)."""
    var V = p(q, 0)
    var n = p(q, 1)
    var D = p(q, 2)
    var O = p(q, 3) + t * n
    for i in range(n):
        sti(f, O + i, i)
    if n < 2:
        return
    var start = n // 2 - 1
    while start >= 0:
        _sift(f, V, D, t, O, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = ldi(f, O)
        sti(f, O, ldi(f, O + end))
        sti(f, O + end, tmp)
        _sift(f, V, D, t, O, 0, end)
        end -= 1


def _sift(f: FP, V: Int, D: Int, c: Int, O: Int, start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and _before(f, V, D, c, ldi(f, O + child), ldi(f, O + child + 1)):
            child += 1
        if _before(f, V, D, c, ldi(f, O + root), ldi(f, O + child)):
            var tmp = ldi(f, O + root)
            sti(f, O + root, ldi(f, O + child))
            sti(f, O + child, tmp)
            root = child
        else:
            return


def wpercentile_unit(t: Int, f: FP, q: IP):
    """q = [V, n, D, ORD, W, R, avg, OUT, CDF]; t = column. scikit-learn 1.9
    `_weighted_percentile(array, sample_weight, percentile_rank, average)`
    over ORD's sorted column: the weight CDF is a SEQUENTIAL ascending Float32
    prefix (DEVIATION 6107; unit weights count exactly), the rank R is in
    percent. Where scikit-learn nudges a zero rank with `nextafter` to the
    smallest SUBNORMAL, this nudges to the smallest NORMAL (a subnormal is
    flushed to zero by DEVIATION 6104); both select the first positive-weight
    row. OUT[t] is written; an all-zero weight column writes the flag -1 at
    CDF[t*n] for the caller to answer NaN by name."""
    var V = p(q, 0)
    var n = p(q, 1)
    var D = p(q, 2)
    var O = p(q, 3) + t * n
    var W = p(q, 4)
    var rank = ld(f, p(q, 5))
    var avg = p(q, 6)
    var OUT = p(q, 7)
    var C = p(q, 8) + t * n
    var acc = Float32(0)
    for i in range(n):
        var r = ldi(f, O + i)
        var w = Float32(1) if W < 0 else ld(f, W + r)
        acc = fadd(acc, w)
        st(f, C + i, acc)
    wpct_select(t, f, q)


def wpct_select(t: Int, f: FP, q: IP):
    """`wpercentile_unit` after its CDF: the search and the average. The
    parallel plan (x_metrics/plan.mojo) writes the CDF with
    x_metrics/par.mojo's gather and prefix units, then runs this."""
    var V = p(q, 0)
    var n = p(q, 1)
    var D = p(q, 2)
    var O = p(q, 3) + t * n
    var W = p(q, 4)
    var rank = ld(f, p(q, 5))
    var avg = p(q, 6)
    var OUT = p(q, 7)
    var C = p(q, 8) + t * n
    var total = ld(f, C + n - 1)
    if total == Float32(0):
        st(f, OUT + t, Float32(0))
        sti(f, C, -1)
        return
    var adj = identical_mul(identical_div(rank, Float32(100)), total)
    if adj == Float32(0):
        adj = Float32(1.1754943508222875e-38)
    # searchsorted(cdf, adj, side='left'): the first i with cdf[i] >= adj
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(f, C + mid) < adj:
            lo = mid + 1
        else:
            hi = mid
    var max_idx = n - 1
    var idx = lo if lo < max_idx else max_idx
    var r0 = ldi(f, O + idx)
    var v0 = ld(f, V + r0 * D + t)
    if avg == 0:
        st(f, OUT + t, v0)
        return
    var above = ftz(ld(f, C + idx) - adj)
    if above > Float32(1.1920929e-07):
        st(f, OUT + t, v0)
        return
    var nxt = idx + 1 if idx + 1 < max_idx else max_idx
    var r1 = ldi(f, O + nxt)
    var w1 = Float32(1) if W < 0 else ld(f, W + r1)
    if w1 == Float32(0):
        # searchsorted(cdf, cdf[idx], side='right'): the first i with cdf[i] > cdf[idx]
        var cv = ld(f, C + idx)
        var a = 0
        var b = n
        while a < b:
            var mid = (a + b) // 2
            if ld(f, C + mid) <= cv:
                a = mid + 1
            else:
                b = mid
        var j = a if a <= max_idx else idx
        r1 = ldi(f, O + j)
    var v1 = ld(f, V + r1 * D + t)
    st(f, OUT + t, identical_div(ftz(v0 + v1), Float32(2)))


def col_max_unit(t: Int, f: FP, q: IP):
    """q = [V, n, D, OUT]; t = column: the largest value, rows ascending."""
    var V = p(q, 0)
    var n = p(q, 1)
    var D = p(q, 2)
    var m = ld(f, V + t)
    for r in range(1, n):
        var v = ld(f, V + r * D + t)
        if v > m:
            m = v
    st(f, p(q, 3) + t, m)


# C09 requested SQ/ABS term pair: one target/prediction load, independent
# outputs consumed by the unchanged PairSum streams and soft-f64 tails.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def reg_pair_unit(t: Int, f: FP, q: IP):
    # q=[Y,P,SQ,ABS,D,PB]. Planner admits identical inputs only.
    var D = p(q, 4)
    var y = ld(f, p(q, 0)+t)
    var pred = ld(f, p(q, 1)+(t % D if p(q, 5) == 1 else t))
    var delta = ftz(y-pred)
    st(f, p(q, 2)+t, identical_mul(delta, delta))
    st(f, p(q, 3)+t, abs(delta))
