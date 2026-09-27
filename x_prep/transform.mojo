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
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, raw, st, is_nan
from x_prep.prims import add, sub, mul, div, logf, expf, sqrtf

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
