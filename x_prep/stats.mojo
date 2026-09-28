# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Univariate feature scores and their p-values, in the prep lane's program
model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/feature_selection/_univariate_selection.py`
(`f_oneway` / `f_classif`, `chi2`, `r_regression` / `f_regression` with
force_finite). The p-values are scipy's `fdtrc` and `chdtrc`, here the
regularized incomplete beta (continued fraction, Numerical Recipes 3rd ed.
6.4 `betacf`) and incomplete gamma (6.2 `gser` / `gcf`) in float32, with
lgamma by recurrence plus Stirling's series. The ANOVA sums are the two-pass
spelling (within-class squares about each class mean, between-class squares
of the means), not the reference's sum-of-squares-minus-square-of-sums,
which cancels catastrophically in float32. A score the reference returns as
NaN comes out NaN (the one canonical quiet NaN word, never a vendor's 0/0)
and an infinite one +inf, as the reference: f_classif of a constant feature
(or of a single class) is NaN with p-value NaN, of a feature constant within
every class but not across them +inf with p-value 0; chi2 of an all-zero
feature is NaN with p-value NaN. f_regression / r_regression apply the
reference's force_finite values (0 and p-value 1 for NaN, float32 max and
p-value 0 for an infinite F) only when asked.
"""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, st, canonical_nan, RUN, run_block
from checks.numerics import ftz
from x_prep.prims import add, sub, mul, div, logf, expf

comptime F32_MAX = Float32(3.4028235e38)


@always_inline
def pos_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))
comptime SF_ITERS = 300
comptime SF_EPS = Float32(3.0e-7)
comptime SF_TINY = Float32(1.0e-30)


def lgammaf(x_in: Float32) -> Float32:
    """log Gamma(x) for x > 0: shift to x >= 8 by the recurrence, then
    Stirling's series."""
    var x = x_in
    var shift = Float32(0)
    while x < Float32(8):
        shift = add(shift, logf(x))
        x = add(x, Float32(1))
    var inv = div(Float32(1), x)
    var inv2 = mul(inv, inv)
    var series = mul(inv, sub(Float32(0.083333333), mul(inv2, sub(Float32(0.0027777778), mul(inv2, Float32(0.00079365079))))))
    var v = add(sub(mul(sub(x, Float32(0.5)), logf(x)), x), Float32(0.91893853))
    return sub(add(v, series), shift)


def _betacf(a: Float32, b: Float32, x: Float32) -> Float32:
    var qab = add(a, b)
    var qap = add(a, Float32(1))
    var qam = sub(a, Float32(1))
    var c = Float32(1)
    var d = sub(Float32(1), div(mul(qab, x), qap))
    if abs(d) < SF_TINY:
        d = SF_TINY
    d = div(Float32(1), d)
    var h = d
    for m in range(1, SF_ITERS + 1):
        var fm = Float32(m)
        var m2 = mul(Float32(2), fm)
        var aa = div(mul(mul(fm, sub(b, fm)), x), mul(add(qam, m2), add(a, m2)))
        d = add(Float32(1), mul(aa, d))
        if abs(d) < SF_TINY:
            d = SF_TINY
        c = add(Float32(1), div(aa, c))
        if abs(c) < SF_TINY:
            c = SF_TINY
        d = div(Float32(1), d)
        h = mul(h, mul(d, c))
        aa = div(sub(Float32(0), mul(mul(add(a, fm), add(qab, fm)), x)), mul(add(a, m2), add(qap, m2)))
        d = add(Float32(1), mul(aa, d))
        if abs(d) < SF_TINY:
            d = SF_TINY
        c = add(Float32(1), div(aa, c))
        if abs(c) < SF_TINY:
            c = SF_TINY
        d = div(Float32(1), d)
        var del_ = mul(d, c)
        h = mul(h, del_)
        if abs(sub(del_, Float32(1))) < SF_EPS:
            break
    return h


def betainc(a: Float32, b: Float32, x: Float32) -> Float32:
    """The regularized incomplete beta I_x(a, b), 0 <= x <= 1."""
    if x <= Float32(0):
        return Float32(0)
    if x >= Float32(1):
        return Float32(1)
    var lbt = add(sub(sub(lgammaf(add(a, b)), lgammaf(a)), lgammaf(b)),
                  add(mul(a, logf(x)), mul(b, logf(sub(Float32(1), x)))))
    var bt = expf(lbt)
    if x < div(add(a, Float32(1)), add(add(a, b), Float32(2))):
        return div(mul(bt, _betacf(a, b, x)), a)
    return sub(Float32(1), div(mul(bt, _betacf(b, a, sub(Float32(1), x))), b))


def gammaincc(a: Float32, x: Float32) -> Float32:
    """The regularized upper incomplete gamma Q(a, x)."""
    if x <= Float32(0):
        return Float32(1)
    var gln = lgammaf(a)
    if x < add(a, Float32(1)):
        var ap = a
        var s = div(Float32(1), a)
        var del_ = s
        for _ in range(SF_ITERS):
            ap = add(ap, Float32(1))
            del_ = mul(del_, div(x, ap))
            s = add(s, del_)
            if abs(del_) < mul(abs(s), SF_EPS):
                break
        return sub(Float32(1), mul(s, expf(sub(sub(mul(a, logf(x)), x), gln))))
    var b = add(sub(x, a), Float32(1))
    var c = div(Float32(1), SF_TINY)
    var d = div(Float32(1), b)
    var h = d
    for i in range(1, SF_ITERS + 1):
        var an = mul(sub(Float32(0), Float32(i)), sub(Float32(i), a))
        b = add(b, Float32(2))
        d = add(mul(an, d), b)
        if abs(d) < SF_TINY:
            d = SF_TINY
        c = add(b, div(an, c))
        if abs(c) < SF_TINY:
            c = SF_TINY
        d = div(Float32(1), d)
        var del_ = mul(d, c)
        h = mul(h, del_)
        if abs(sub(del_, Float32(1))) < SF_EPS:
            break
    return mul(expf(sub(sub(mul(a, logf(x)), x), gln)), h)


def f_sf(dfn: Float32, dfd: Float32, fv: Float32) -> Float32:
    """scipy.special.fdtrc: P(F > fv)."""
    if fv <= Float32(0):
        return Float32(1)
    return betainc(mul(Float32(0.5), dfd), mul(Float32(0.5), dfn), div(dfd, add(dfd, mul(dfn, fv))))


def f_classif_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, CNT, MEAN, SCORES, PV]; t = feature. One-way ANOVA
    F over the class codes Y (CNT, MEAN from the class_stats unit)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 4)
    var c = t
    var tot = Float32(0)
    for k in range(K):
        tot = add(tot, mul(ld(f, p(q, 5) + k), ld(f, p(q, 6) + k * d + c)))
    var gm = div(tot, Float32(n))
    var ssb = Float32(0)
    for k in range(K):
        var e = sub(ld(f, p(q, 6) + k * d + c), gm)
        ssb = add(ssb, mul(ld(f, p(q, 5) + k), mul(e, e)))
    var ssw = Float32(0)
    var X = p(q, 0)
    var Y = p(q, 3)
    var M = p(q, 6)
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var by = run_block[RUN](f, Y + i0, 1)
        var bx = run_block[RUN](f, X + i0 * d + c, d)
        comptime for u in range(RUN):
            var e = sub(ftz(bx[u]), ld(f, M + Int(ftz(by[u])) * d + c))
            ssw = add(ssw, mul(e, e))
    for i in range(full, n):
        var k = Int(ld(f, Y + i))
        var e = sub(ld(f, X + i * d + c), ld(f, M + k * d + c))
        ssw = add(ssw, mul(e, e))
    var dfb = Float32(K - 1)
    var dfw = Float32(n - K)
    var score = canonical_nan()
    var pv = canonical_nan()
    if K < 2:
        pass
    elif ssw > Float32(0):
        score = div(div(ssb, dfb), div(ssw, dfw))
        pv = f_sf(dfb, dfw, score)
    elif ssb > Float32(0):
        score = pos_inf()
        pv = Float32(0)
    st(f, p(q, 7) + c, score)
    st(f, p(q, 8) + c, pv)


def f_regression_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, CENTER, SCORES, PV, CORR, FORCE_FINITE]; t = feature.
    Pearson r of column t with Y (centred unless CENTER == 0), then F = r^2 /
    (1 - r^2) * dof, dof = n - 2 (n - 1 uncentred). At the edges, with
    FORCE_FINITE != 0: r = 0, F = 0, p = 1 for a constant column or target,
    F = float32 max, p = 0 for |r| = 1; with FORCE_FINITE == 0 the
    reference's raw values: r, F and p NaN, and F = +inf, p = 0."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t
    var mx = Float32(0)
    var my = Float32(0)
    var X = p(q, 0)
    var Y = p(q, 3)
    var full = n - n % RUN
    if p(q, 4) != 0:
        var sx = Float32(0)
        var sy = Float32(0)
        for i0 in range(0, full, RUN):
            var bx = run_block[RUN](f, X + i0 * d + c, d)
            var by = run_block[RUN](f, Y + i0, 1)
            comptime for u in range(RUN):
                sx = add(sx, ftz(bx[u]))
                sy = add(sy, ftz(by[u]))
        for i in range(full, n):
            sx = add(sx, ld(f, X + i * d + c))
            sy = add(sy, ld(f, Y + i))
        mx = div(sx, Float32(n))
        my = div(sy, Float32(n))
    var sxy = Float32(0)
    var sxx = Float32(0)
    var syy = Float32(0)
    for i0 in range(0, full, RUN):
        var bx = run_block[RUN](f, X + i0 * d + c, d)
        var by = run_block[RUN](f, Y + i0, 1)
        comptime for u in range(RUN):
            var ex = sub(ftz(bx[u]), mx)
            var ey = sub(ftz(by[u]), my)
            sxy = add(sxy, mul(ex, ey))
            sxx = add(sxx, mul(ex, ex))
            syy = add(syy, mul(ey, ey))
    for i in range(full, n):
        var ex = sub(ld(f, X + i * d + c), mx)
        var ey = sub(ld(f, Y + i), my)
        sxy = add(sxy, mul(ex, ey))
        sxx = add(sxx, mul(ex, ex))
        syy = add(syy, mul(ey, ey))
    var dof = Float32(n - 2) if p(q, 4) != 0 else Float32(n - 1)
    var ff = p(q, 8) != 0
    var score = Float32(0) if ff else canonical_nan()
    var pv = Float32(1) if ff else canonical_nan()
    var r = Float32(0) if ff else canonical_nan()
    if sxx > Float32(0) and syy > Float32(0):
        r = div(sxy, mul(sqrtf_(sxx), sqrtf_(syy)))
        var r2 = mul(r, r)
        if r2 >= Float32(1):
            score = F32_MAX if ff else pos_inf()
            pv = Float32(0)
        else:
            score = mul(div(r2, sub(Float32(1), r2)), dof)
            pv = f_sf(Float32(1), dof, score)
    st(f, p(q, 5) + c, score)
    st(f, p(q, 6) + c, pv)
    if p(q, 7) >= 0:
        st(f, p(q, 7) + c, r)


@always_inline
def sqrtf_(x: Float32) -> Float32:
    from x_prep.prims import sqrtf
    return sqrtf(x)


def chi2_unit(t: Int, f: FP, q: IP):
    """q = [OBS, K, d, CNT, n, SCORES, PV]; t = feature. OBS = the class-by-
    feature sums (class_stats SUM); expected = class share * feature total;
    chi2 = sum_k (obs - exp)^2 / exp, p = Q((K - 1) / 2, chi2 / 2); an
    all-zero feature (every expected count 0) is NaN with p-value NaN, as the
    reference's 0 / 0."""
    var K = p(q, 1)
    var d = p(q, 2)
    var c = t
    var total = Float32(0)
    for k in range(K):
        total = add(total, ld(f, p(q, 0) + k * d + c))
    var chi = Float32(0)
    for k in range(K):
        var ex = mul(div(ld(f, p(q, 3) + k), Float32(p(q, 4))), total)
        if ex > Float32(0):
            var e = sub(ld(f, p(q, 0) + k * d + c), ex)
            chi = add(chi, div(mul(e, e), ex))
    if not total > Float32(0):
        st(f, p(q, 5) + c, canonical_nan())
        st(f, p(q, 6) + c, canonical_nan())
        return
    st(f, p(q, 5) + c, chi)
    st(f, p(q, 6) + c, gammaincc(mul(Float32(0.5), Float32(K - 1)), mul(Float32(0.5), chi)))
