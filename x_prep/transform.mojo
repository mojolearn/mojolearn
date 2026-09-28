# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Elementwise transformer units (QuantileTransformer, PowerTransformer,
Normalizer, PolynomialFeatures, SplineTransformer), in the prep lane's
program model (x_prep/common.mojo).

References: scikit-learn 1.9 `sklearn/preprocessing/_data.py`
(QuantileTransformer `_transform_col`, BOUNDS_THRESHOLD = 1e-7;
PowerTransformer `_yeo_johnson_transform`, `_box_cox_optimize`,
`_yeo_johnson_optimize`; `normalize`), `_polynomial.py`
(PolynomialFeatures `_combinations`, SplineTransformer), numpy
`np.interp` (`numpy/_core/src/multiarray/compiled_base.c`, arr_interp), and
P. J. Acklam's inverse normal CDF (the rational approximation, one Halley
refinement) for `scipy.stats.norm.ppf`.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_log1p, identical_erf, _cephes_erfcf_ge1
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, is_nan, canonical_nan, RUN, run_block
from x_prep.prims import add, acc_add, sub, mul, div, logf, expf, sqrtf, zero_to_one

#: norm.ppf(1e-7 - eps) and its mirror: QuantileTransformer's normal clip.
comptime QT_CLIP = Float32(5.1993375)
comptime QT_BOUND = Float32(1.0e-7)


@always_inline
def _xp(f: FP, Q: Int, nq: Int, k: Int, rev: Bool) -> Float32:
    if rev:
        return sub(Float32(0), ld(f, Q + nq - 1 - k))
    return ld(f, Q + k)


def interp(f: FP, x: Float32, Q: Int, R: Int, nq: Int, rev: Bool) -> Float32:
    """numpy.interp(x, xp, fp) with xp = Q (or -Q reversed) and fp = R (or
    -R reversed), xp nondecreasing; x not NaN."""
    var x0 = _xp(f, Q, nq, 0, rev)
    var xl = _xp(f, Q, nq, nq - 1, rev)
    if x < x0:
        return _xp(f, R, nq, 0, rev)
    if x > xl:
        return _xp(f, R, nq, nq - 1, rev)
    if x == xl:
        return _xp(f, R, nq, nq - 1, rev)
    # largest j with xp[j] <= x (then x < xp[j+1])
    var lo = 0
    var hi = nq - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if _xp(f, Q, nq, mid, rev) <= x:
            lo = mid
        else:
            hi = mid
    var xa = _xp(f, Q, nq, lo, rev)
    var xb = _xp(f, Q, nq, lo + 1, rev)
    var ya = _xp(f, R, nq, lo, rev)
    var yb = _xp(f, R, nq, lo + 1, rev)
    var slope = div(sub(yb, ya), sub(xb, xa))
    return add(mul(slope, sub(x, xa)), ya)


def norm_ppf(pv: Float32) -> Float32:
    """Acklam's inverse normal CDF for 0 < p < 1, float32."""
    var a1 = Float32(-3.969683028665376e+01)
    var a2 = Float32(2.209460984245205e+02)
    var a3 = Float32(-2.759285104469687e+02)
    var a4 = Float32(1.383577518672690e+02)
    var a5 = Float32(-3.066479806614716e+01)
    var a6 = Float32(2.506628277459239e+00)
    var b1 = Float32(-5.447609879822406e+01)
    var b2 = Float32(1.615858368580409e+02)
    var b3 = Float32(-1.556989798598866e+02)
    var b4 = Float32(6.680131188771972e+01)
    var b5 = Float32(-1.328068155288572e+01)
    var c1 = Float32(-7.784894002430293e-03)
    var c2 = Float32(-3.223964580411365e-01)
    var c3 = Float32(-2.400758277161838e+00)
    var c4 = Float32(-2.549732539343734e+00)
    var c5 = Float32(4.374664141464968e+00)
    var c6 = Float32(2.938163982698783e+00)
    var d1 = Float32(7.784695709041462e-03)
    var d2 = Float32(3.224671290700398e-01)
    var d3 = Float32(2.445134137142996e+00)
    var d4 = Float32(3.754408661907416e+00)
    var plow = Float32(0.02425)
    if pv < plow or pv > sub(Float32(1), plow):
        var qq = pv if pv < plow else sub(Float32(1), pv)
        var r = sqrtf(mul(Float32(-2), logf(qq)))
        var num = add(mul(add(mul(add(mul(add(mul(add(mul(c1, r), c2), r), c3), r), c4), r), c5), r), c6)
        var den = add(mul(add(mul(add(mul(add(mul(d1, r), d2), r), d3), r), d4), r), Float32(1))
        var v = div(num, den)
        return v if pv < plow else sub(Float32(0), v)
    var qq = sub(pv, Float32(0.5))
    var r = mul(qq, qq)
    var num = mul(add(mul(add(mul(add(mul(add(mul(add(mul(a1, r), a2), r), a3), r), a4), r), a5), r), a6), qq)
    var den = add(mul(add(mul(add(mul(add(mul(add(mul(b1, r), b2), r), b3), r), b4), r), b5), r), Float32(1))
    return div(num, den)


def qt_apply_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Q, nq, REFS, dist, OUT]; t = element. Q is column-major
    (column c's nq quantiles at Q + c*nq). dist 0 uniform, 1 normal. A NaN
    input is copied bit for bit."""
    var d = p(q, 2)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 7) + t, x)
        return
    x = ftz(x)
    var nq = p(q, 4)
    var Q = p(q, 3) + c * nq
    var R = p(q, 5)
    var lb = ld(f, Q)
    var ub = ld(f, Q + nq - 1)
    var normal = p(q, 6) == 1
    var at_lo: Bool
    var at_hi: Bool
    if normal:
        at_lo = sub(x, QT_BOUND) < lb
        at_hi = add(x, QT_BOUND) > ub
    else:
        at_lo = x == lb
        at_hi = x == ub
    var y: Float32
    if at_lo:
        y = sub(Float32(0), QT_CLIP) if normal else Float32(0)
    elif at_hi:
        y = QT_CLIP if normal else Float32(1)
    else:
        var u = mul(Float32(0.5), sub(interp(f, x, Q, R, nq, False),
                                      interp(f, sub(Float32(0), x), Q, R, nq, True)))
        y = u
        if normal:
            if u <= Float32(0):
                y = sub(Float32(0), QT_CLIP)
            elif u >= Float32(1):
                y = QT_CLIP
            else:
                y = norm_ppf(u)
                if y < sub(Float32(0), QT_CLIP):
                    y = sub(Float32(0), QT_CLIP)
                if y > QT_CLIP:
                    y = QT_CLIP
    st(f, p(q, 7) + t, y)


def norm_cdf(x: Float32) -> Float32:
    """scipy.stats.norm.cdf (Cephes `ndtr`: 0.5 erfc(-x / sqrt 2), with
    Cephes `erfcf` = 1 - erf below 1 and the continued-fraction arm above),
    float32; x not NaN."""
    var v = mul(sub(Float32(0), x), Float32(0.70710677))
    var e: Float32
    if abs(v) < Float32(1):
        e = sub(Float32(1), ftz(identical_erf(v)))
    else:
        e = ftz(_cephes_erfcf_ge1(v))
    return mul(Float32(0.5), e)


def qt_inverse_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Q, nq, REFS, dist, OUT]; t = element: the reference's
    `_transform_col(inverse=True)`. dist 1 (normal) maps x through norm.cdf
    first; then x at or below 0 is Q[0], at or above 1 is Q[nq-1] (within
    BOUNDS_THRESHOLD for normal, exactly for uniform), otherwise
    np.interp(x, REFS, Q[:, c]). A NaN input is copied bit for bit."""
    var d = p(q, 2)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 7) + t, x)
        return
    x = ftz(x)
    var nq = p(q, 4)
    var Q = p(q, 3) + c * nq
    var normal = p(q, 6) == 1
    if normal:
        x = norm_cdf(x)
    var at_lo: Bool
    var at_hi: Bool
    if normal:
        at_lo = sub(x, QT_BOUND) < Float32(0)
        at_hi = add(x, QT_BOUND) > Float32(1)
    else:
        at_lo = x == Float32(0)
        at_hi = x == Float32(1)
    var y: Float32
    if at_lo:
        y = ld(f, Q)
    elif at_hi:
        y = ld(f, Q + nq - 1)
    else:
        y = interp(f, x, p(q, 5), Q, nq, False)
    st(f, p(q, 7) + t, y)


@always_inline
def log1pf(x: Float32) -> Float32:
    return ftz(identical_log1p(ftz(x)))


@always_inline
def power_log(x: Float32, method: Int) -> Float32:
    """The lambda-free logarithm the power transform of x is built on: log x
    (box-cox, METHOD 1), log1p x (yeo-johnson, x >= 0), log1p(-x) (x < 0).
    It is also the Jacobian term J(x) up to its sign (`_neg_llf`)."""
    if method == 1:
        return logf(x)
    if x >= Float32(0):
        return log1pf(x)
    return log1pf(sub(Float32(0), x))


@always_inline
def power_from_log(lg: Float32, nonneg: Bool, lam: Float32, method: Int) -> Float32:
    """The power transform of x from `power_log(x, method)` and x >= 0:
    scipy.special.boxcox (METHOD 1, x > 0) or scipy.stats.yeojohnson's
    transform. `power` is this of `power_log`, so a caller that keeps the
    logarithm across lambdas (x_prep/host/power.mojo) computes the same words."""
    if method == 1 or nonneg:
        if lam == Float32(0):
            return lg
        return div(sub(expf(mul(lam, lg)), Float32(1)), lam)
    var l2 = sub(Float32(2), lam)
    if l2 == Float32(0):
        return sub(Float32(0), lg)
    return sub(Float32(0), div(sub(expf(mul(l2, lg)), Float32(1)), l2))


def yeo_johnson(x: Float32, lam: Float32) -> Float32:
    """scipy.stats.yeojohnson's transform of one value (x not NaN)."""
    return power_from_log(power_log(x, 0), x >= Float32(0), lam, 0)


def box_cox(x: Float32, lam: Float32) -> Float32:
    """scipy.special.boxcox of one value (x > 0)."""
    return power_from_log(power_log(x, 1), True, lam, 1)


def power(x: Float32, lam: Float32, method: Int) -> Float32:
    return power_from_log(power_log(x, method), x >= Float32(0), lam, method)


def _neg_llf(f: FP, X: Int, n: Int, d: Int, c: Int, lam: Float32, method: Int) -> Float32:
    """Minus scipy's yeojohnson_llf / boxcox_llf over the non-NaN entries:
    n/2 log var(T(x)) - (lam - 1) sum J(x), J = sign(x) log1p|x| or log x."""
    var cnt = 0
    var s = Float32(0)
    var sj = Float32(0)
    for i in range(n):
        var x = ld(f, X + i * d + c)
        if is_nan(x):
            continue
        s = add(s, power(x, lam, method))
        if method == 1:
            sj = add(sj, logf(x))
        elif x >= Float32(0):
            sj = add(sj, log1pf(x))
        else:
            sj = sub(sj, log1pf(sub(Float32(0), x)))
        cnt += 1
    if cnt == 0:
        return Float32(0)
    var mean = div(s, Float32(cnt))
    var ss = Float32(0)
    for i in range(n):
        var x = ld(f, X + i * d + c)
        if is_nan(x):
            continue
        var e = sub(power(x, lam, method), mean)
        ss = add(ss, mul(e, e))
    var var_ = div(ss, Float32(cnt))
    return sub(mul(mul(Float32(0.5), Float32(cnt)), logf(var_)), mul(sub(lam, Float32(1)), sj))


comptime PT_LO = Float32(-8)
comptime PT_HI = Float32(8)
comptime PT_ITERS = 48
#: (3 - sqrt 5) / 2
comptime GOLDEN = Float32(0.381966)


def pt_fit_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, ST, LAMBDA]; t = column. The lambda minimising
    the negative log-likelihood by a golden-section search over [-8, 8] with
    a fixed step count (the reference runs scipy's Brent from the bracket
    (-2, 2)); a constant column is lambda 1 for yeo-johnson (METHOD 0)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var method = p(q, 3)
    var c = t
    if method == 0 and ld(f, p(q, 4) + 2 * d + c) == Float32(0):
        st(f, p(q, 5) + c, Float32(1))
        return
    var a = PT_LO
    var b = PT_HI
    var x1 = add(a, mul(GOLDEN, sub(b, a)))
    var x2 = sub(b, mul(GOLDEN, sub(b, a)))
    var f1 = _neg_llf(f, p(q, 0), n, d, c, x1, method)
    var f2 = _neg_llf(f, p(q, 0), n, d, c, x2, method)
    for _ in range(PT_ITERS):
        if f1 <= f2 or f2 != f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = add(a, mul(GOLDEN, sub(b, a)))
            f1 = _neg_llf(f, p(q, 0), n, d, c, x1, method)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = sub(b, mul(GOLDEN, sub(b, a)))
            f2 = _neg_llf(f, p(q, 0), n, d, c, x2, method)
    st(f, p(q, 5) + c, mul(add(a, b), Float32(0.5)))


# ------------------------------------------------ pt_fit as stages (lane prep-apple)
# pt_fit_unit runs the whole golden-section search on one thread per column:
# 50 log-likelihood evaluations, each two passes over the column with an exp
# and a log1p per row (13.8 s of 14.0 s in PowerTransformer.fit at 100k rows
# x 16 columns on the M4 Pro). The same search as stages: `pt_init` sets up
# each column's bracket, then per evaluation `pt_map` computes the transform
# T(x; lambda) of every element at once (one thread per element) and
# `pt_fold` folds the column's T in ascending row order into the negative
# log-likelihood exactly as `_neg_llf` does, and takes the search's step.
# Every value is the one `_neg_llf` computes (T is `power` of the same
# flushed x, stored and reloaded bit for bit; the folds add in the same
# order), so lambda is the same word.

#: evaluations: the two starting points, then one per iteration
comptime PT_EVALS = PT_ITERS + 2
#: per-column search state: a, b, x1, x2, f1, f2, side, skip, sum J, and the
#: non-NaN row count folded at K = 0 (an int word)
comptime PT_STATE = 10


def pt_init_unit(t: Int, f: FP, q: IP):
    """q = [METHOD, ST, d, LAMBDA, STATE, LEVAL]; t = column. A constant
    column (yeo-johnson) is lambda 1 and skipped, as `pt_fit_unit`; otherwise
    the bracket [PT_LO, PT_HI], its two golden points, and x1 to evaluate."""
    var d = p(q, 2)
    var c = t
    var S = p(q, 4) + c * PT_STATE
    var a = PT_LO
    var b = PT_HI
    var x1 = add(a, mul(GOLDEN, sub(b, a)))
    var x2 = sub(b, mul(GOLDEN, sub(b, a)))
    var skip = p(q, 0) == 0 and ld(f, p(q, 1) + 2 * d + c) == Float32(0)
    f.unsafe_store(S + 0, a)
    f.unsafe_store(S + 1, b)
    f.unsafe_store(S + 2, x1)
    f.unsafe_store(S + 3, x2)
    f.unsafe_store(S + 4, Float32(0))
    f.unsafe_store(S + 5, Float32(0))
    f.unsafe_store(S + 6, Float32(0))
    f.unsafe_store(S + 7, Float32(1) if skip else Float32(0))
    f.unsafe_store(S + 8, Float32(0))
    f.unsafe_store(S + 9, Float32(0))
    f.unsafe_store(p(q, 5) + c, x1)
    if skip:
        st(f, p(q, 3) + c, Float32(1))


def pt_map_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, LEVAL, T]; t = element i*d + c. T[c*n + i] =
    power(x, LEVAL[c]) of the flushed x (`_neg_llf`'s value), COLUMN MAJOR so
    the fold reads each column contiguously; a NaN element writes the
    canonical NaN (the fold skips it by that word: `power` of a non-NaN x is
    never NaN for yeo-johnson, and box-cox input with x <= 0, the only NaN
    source, is refused after the fit, its values discarded)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var c = t % d
    var i = t // d
    var x = ld(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 5) + c * n + i, canonical_nan())
        return
    f.unsafe_store(p(q, 5) + c * n + i, power(x, raw(f, p(q, 4) + c), p(q, 3)))


@always_inline
def _pt_take1(x: Float32, tv: Float32, method: Int, first: Bool, mut cnt: Int, mut s: Float32,
              mut sj: Float32):
    """One row of `_neg_llf`'s first pass (x flushed, tv its transform)."""
    if is_nan(x):
        return
    s = acc_add(s, tv)
    if first:
        if method == 1:
            sj = add(sj, logf(x))
        elif x >= Float32(0):
            sj = add(sj, log1pf(x))
        else:
            sj = sub(sj, log1pf(sub(Float32(0), x)))
    cnt += 1


@always_inline
def _pt_take2(x: Float32, tv: Float32, mean: Float32, mut ss: Float32):
    """One row of `_neg_llf`'s second pass."""
    if is_nan(x):
        return
    var e = sub(tv, mean)
    ss = acc_add(ss, mul(e, e))


def pt_fold_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, T, K, STATE, LEVAL, LAMBDA]; t = column. The
    negative log-likelihood of evaluation K (at LEVAL[c]) folded as `_neg_llf`
    (sum J is folded once, at K = 0: every evaluation folds the same values in
    the same order, so it is the same word), then `pt_fit_unit`'s step: K = 0
    is f1, K = 1 is f2, a later K is the point the previous step chose; after
    the last iteration LAMBDA = (a + b) / 2."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var method = p(q, 3)
    var T = p(q, 4)
    var K = p(q, 5)
    var c = t
    var S = p(q, 6) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var first = K == 0
    var cnt = 0
    var sm = Float32(0)
    var sj = raw(f, S + 8)
    if first:
        sj = Float32(0)
    var Tc = T + c * n
    var full = n - n % RUN
    if first:
        # sum J needs x: the one evaluation that reads X
        for i0 in range(0, full, RUN):
            var bx = run_block[RUN](f, X + i0 * d + c, d)
            var bt = run_block[RUN](f, Tc + i0, 1)
            comptime for u in range(RUN):
                _pt_take1(ftz(bx[u]), bt[u], method, True, cnt, sm, sj)
        for i in range(full, n):
            _pt_take1(ld(f, X + i * d + c), raw(f, Tc + i), method, True, cnt, sm, sj)
        f.unsafe_store(S + 8, sj)
        sti(f, S + 9, cnt)
    elif ldi(f, S + 9) == n:
        # no NaN row in this column (K = 0 counted n): the same adds, no test
        cnt = n
        for i0 in range(0, full, RUN):
            var bt = run_block[RUN](f, Tc + i0, 1)
            comptime for u in range(RUN):
                sm = acc_add(sm, bt[u])
        for i in range(full, n):
            sm = acc_add(sm, raw(f, Tc + i))
    else:
        # a row is NaN exactly where its T word is (pt_map_unit)
        for i0 in range(0, full, RUN):
            var bt = run_block[RUN](f, Tc + i0, 1)
            comptime for u in range(RUN):
                _pt_take1(bt[u], bt[u], method, False, cnt, sm, sj)
        for i in range(full, n):
            var tv = raw(f, Tc + i)
            _pt_take1(tv, tv, method, False, cnt, sm, sj)
    var ss = Float32(0)
    if cnt == n and n > 0:
        var mean = div(sm, Float32(cnt))
        for i0 in range(0, full, RUN):
            var bt = run_block[RUN](f, Tc + i0, 1)
            comptime for u in range(RUN):
                var e = sub(bt[u], mean)
                ss = acc_add(ss, mul(e, e))
        for i in range(full, n):
            var e = sub(raw(f, Tc + i), mean)
            ss = acc_add(ss, mul(e, e))
    elif cnt > 0:
        var mean = div(sm, Float32(cnt))
        for i0 in range(0, full, RUN):
            var bt = run_block[RUN](f, Tc + i0, 1)
            comptime for u in range(RUN):
                _pt_take2(bt[u], bt[u], mean, ss)
        for i in range(full, n):
            var tv = raw(f, Tc + i)
            _pt_take2(tv, tv, mean, ss)
    pt_finish(t, f, q, cnt, sj, ss)


def pt_finish(t: Int, f: FP, q: IP, cnt: Int, sj: Float32, ss: Float32):
    """pt_fold's end for column t, from its folded row count, sum J and
    squared deviations: the negative log-likelihood of evaluation K, then the
    golden-section step (also the FAST device fold's end, x_prep/fastred.mojo)."""
    var K = p(q, 5)
    var c = t
    var S = p(q, 6) + c * PT_STATE
    var lam = raw(f, p(q, 7) + c)
    var val = Float32(0)
    if cnt > 0:
        var var_ = div(ss, Float32(cnt))
        val = sub(mul(mul(Float32(0.5), Float32(cnt)), logf(var_)), mul(sub(lam, Float32(1)), sj))
    var a = raw(f, S + 0)
    var b = raw(f, S + 1)
    var x1 = raw(f, S + 2)
    var x2 = raw(f, S + 3)
    var f1 = raw(f, S + 4)
    var f2 = raw(f, S + 5)
    if K == 0:
        f1 = val
        f.unsafe_store(p(q, 7) + c, x2)
    else:
        if K == 1 or raw(f, S + 6) == Float32(2):
            f2 = val
        else:
            f1 = val
        if K - 1 < PT_ITERS:
            if f1 <= f2 or f2 != f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = add(a, mul(GOLDEN, sub(b, a)))
                f.unsafe_store(S + 6, Float32(1))
                f.unsafe_store(p(q, 7) + c, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = sub(b, mul(GOLDEN, sub(b, a)))
                f.unsafe_store(S + 6, Float32(2))
                f.unsafe_store(p(q, 7) + c, x2)
        else:
            st(f, p(q, 8) + c, mul(add(a, b), Float32(0.5)))
    f.unsafe_store(S + 0, a)
    f.unsafe_store(S + 1, b)
    f.unsafe_store(S + 2, x1)
    f.unsafe_store(S + 3, x2)
    f.unsafe_store(S + 4, f1)
    f.unsafe_store(S + 5, f2)


def pt_apply_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, LAMBDA, METHOD, MEAN, SCALE, OUT]; t = element. The power
    transform, then (MEAN >= 0) standardisation; NaN is copied."""
    var d = p(q, 2)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 7) + t, x)
        return
    var v = power(ftz(x), ld(f, p(q, 3) + c), p(q, 4))
    if p(q, 5) >= 0:
        v = div(sub(v, ld(f, p(q, 5) + c)), ld(f, p(q, 6) + c))
    st(f, p(q, 7) + t, v)


#: np.spacing(1.0), the reference's "lambda is zero" test in the inverse.
comptime F64_SPACING1 = Float32(2.220446049250313e-16)


def _inv_pow_m1(v: Float32, lam: Float32) -> Float32:
    """(v * lam + 1) ** (1 / lam) - 1 as exp(log1p(v * lam) / lam) - 1."""
    return sub(expf(div(log1pf(mul(v, lam)), lam)), Float32(1))


def pt_inverse_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, LAMBDA, METHOD, MEAN, SCALE, OUT]; t = element: the
    reference's inverse_transform. With MEAN >= 0 first x * SCALE + MEAN
    (StandardScaler.inverse_transform), then scipy's inv_boxcox (METHOD 1:
    exp(x) at lambda 0, else exp(log1p(lambda x) / lambda)) or sklearn's
    `_yeo_johnson_inverse_transform` (METHOD 0). An input NaN is copied; a
    NaN the map makes is the canonical quiet NaN."""
    var d = p(q, 2)
    var c = t % d
    var x = raw(f, p(q, 0) + t)
    if is_nan(x):
        f.unsafe_store(p(q, 7) + t, x)
        return
    var v = ftz(x)
    if p(q, 5) >= 0:
        v = add(mul(v, ld(f, p(q, 6) + c)), ld(f, p(q, 5) + c))
    var lam = ld(f, p(q, 3) + c)
    var y: Float32
    if p(q, 4) == 1:
        if lam == Float32(0):
            y = expf(v)
        else:
            y = expf(div(log1pf(mul(lam, v)), lam))
    elif v >= Float32(0):
        if abs(lam) < F64_SPACING1:
            y = sub(expf(v), Float32(1))
        else:
            y = _inv_pow_m1(v, lam)
    else:
        var l2 = sub(Float32(2), lam)
        if abs(l2) > F64_SPACING1:
            y = sub(Float32(0), _inv_pow_m1(sub(Float32(0), v), l2))
        else:
            y = sub(Float32(1), expf(sub(Float32(0), v)))
    if y != y:
        y = canonical_nan()  # a base below 0 (numpy's NaN power): one NaN word on every column
    st(f, p(q, 7) + t, y)


def std_params_unit(t: Int, f: FP, q: IP):
    """q = [ST, d, MEAN, SCALE]; t = column: StandardScaler's mean and
    scale (population std, `_handle_zeros_in_scale`) from col_stats rows."""
    var d = p(q, 1)
    st(f, p(q, 2) + t, ld(f, p(q, 0) + d + t))
    st(f, p(q, 3) + t, zero_to_one(sqrtf(ld(f, p(q, 0) + 2 * d + t))))


def normalize_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, NORM, OUT]; t = row. NORM 0 l1, 1 l2, 2 max (sklearn
    `normalize`, columns ascending); a zero norm leaves the row unchanged."""
    var d = p(q, 2)
    var X = p(q, 0) + t * d
    var kind = p(q, 3)
    var s = Float32(0)
    for c in range(d):
        var v = ld(f, X + c)
        if kind == 0:
            s = add(s, abs(v))
        elif kind == 1:
            s = add(s, mul(v, v))
        elif abs(v) > s:
            s = abs(v)
    if kind == 1:
        s = sqrtf(s)
    if s == Float32(0):
        s = Float32(1)
    for c in range(d):
        st(f, p(q, 4) + t * d + c, div(ld(f, X + c), s))


def poly_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, IDX, START, nout, OUT]; t = i*nout + o. Output column o is
    the product of the input columns IDX[START[o] : START[o+1]], left to right
    (an empty product, the bias, is 1)."""
    var nout = p(q, 5)
    var i = t // nout
    var o = t % nout
    var a = Int(ld(f, p(q, 4) + o))
    var b = Int(ld(f, p(q, 4) + o + 1))
    var v = Float32(1)
    for k in range(a, b):
        v = mul(v, ld(f, p(q, 0) + i * p(q, 2) + Int(ld(f, p(q, 3) + k))))
    st(f, p(q, 6) + t, v)


def robust_uv_unit(t: Int, f: FP, q: IP):
    """q = [SCALE, QF]; t = column: RobustScaler(unit_variance=True),
    SCALE /= norm.ppf(QF[2]) - norm.ppf(QF[0]) (the quantile fractions)."""
    var adjust = sub(norm_ppf(ld(f, p(q, 1) + 2)), norm_ppf(ld(f, p(q, 1))))
    st(f, p(q, 0) + t, div(ld(f, p(q, 0) + t), adjust))
