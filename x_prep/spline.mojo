# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SplineTransformer units, in the prep lane's program model.

Reference: scikit-learn 1.9 `sklearn/preprocessing/_polynomial.py`
(SplineTransformer `_get_base_knot_positions`, `fit` knot extension and
periodic wrap, `transform` with extrapolation 'constant' / 'continue' /
'linear' / 'periodic'), and the
B-spline basis by the Cox-de Boor triangle (de Boor, A Practical Guide to
Splines, BSPLVB), which is what scipy's BSpline evaluates.
"""
from std.memory import bitcast
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st, is_nan, canonical_nan
from x_prep.prims import add, sub, mul, div

comptime MAX_DEGREE = 7


def spline_knots_unit(t: Int, f: FP, q: IP):
    """q = [BASE, nk, d, degree, KNOTS, UNIFORM, ST, PERIODIC]; t = column.
    With UNIFORM the base knots are first written as numpy's linspace(min,
    max, nk) from the col_stats rows ST (start + i * step, the last one max);
    otherwise BASE[c*nk :] holds them already. KNOTS[c*(nk + 2*degree) :] is
    the extended knot vector: degree knots below at the first spacing and
    degree above at the last; with PERIODIC, the reference's wrap instead
    (base[-(degree+1):-1] - period below, base[1:degree+1] + period above,
    period = base[-1] - base[0])."""
    var nk = p(q, 1)
    var d = p(q, 2)
    var k = p(q, 3)
    var B = p(q, 0) + t * nk
    if p(q, 5) != 0:
        var lo = ld(f, p(q, 6) + 3 * d + t)
        var hi = ld(f, p(q, 6) + 4 * d + t)
        var step = div(sub(hi, lo), Float32(nk - 1))
        for i in range(nk - 1):
            st(f, B + i, add(mul(Float32(i), step), lo))
        st(f, B + nk - 1, hi)
    var K = p(q, 4) + t * (nk + 2 * k)
    if p(q, 7) != 0:
        var period = sub(ld(f, B + nk - 1), ld(f, B))
        for i in range(k):
            st(f, K + i, sub(ld(f, B + nk - 1 - k + i), period))
        for i in range(nk):
            st(f, K + k + i, ld(f, B + i))
        for i in range(k):
            st(f, K + k + nk + i, add(ld(f, B + 1 + i), period))
        return
    var dmin = sub(ld(f, B + 1), ld(f, B))
    var dmax = sub(ld(f, B + nk - 1), ld(f, B + nk - 2))
    for i in range(k):
        st(f, K + i, sub(ld(f, B), mul(Float32(k - i), dmin)))
    for i in range(nk):
        st(f, K + k + i, ld(f, B + i))
    for i in range(k):
        st(f, K + k + nk + i, add(ld(f, B + nk - 1), mul(Float32(i + 1), dmax)))


@always_inline
def _span(f: FP, K: Int, k: Int, nspl: Int, x: Float32) -> Int:
    """The knot span i (t_i <= x < t_{i+1}) in [k, nspl - 1]; outside the
    base interval, the boundary span (scipy's extrapolate=True)."""
    var i = k
    while i + 1 < nspl and ld(f, K + i + 1) <= x:
        i += 1
    return i


def _bsplvb(f: FP, K: Int, i: Int, k: Int, x: Float32, mut out: InlineArray[Float32, MAX_DEGREE + 1]):
    """The k+1 nonzero degree-k basis values B_{i-k} .. B_i at x on span i
    into out[0..k] (de Boor's BSPLVB)."""
    out[0] = Float32(1)
    var left = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    var right = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    for j in range(1, k + 1):
        left[j] = sub(x, ld(f, K + i + 1 - j))
        right[j] = sub(ld(f, K + i + j), x)
        var saved = Float32(0)
        for r in range(j):
            var den = add(right[r + 1], left[j - r])
            # a zero-length knot span (a constant column): 0/0 is taken as 0,
            # de Boor's convention, never a NaN
            var temp = div(out[r], den) if den != Float32(0) else Float32(0)
            out[r] = add(saved, mul(right[r + 1], temp))
            saved = mul(left[j - r], temp)
        out[j] = saved


def _basis(f: FP, K: Int, k: Int, nspl: Int, x: Float32, mut out: InlineArray[Float32, MAX_DEGREE + 1]) -> Int:
    """The k+1 nonzero basis values at x into out[0..k]; returns the span i
    (they are B_{i-k} .. B_i). x outside the base interval uses the boundary
    span's polynomials (scipy's extrapolate=True)."""
    var i = _span(f, K, k, nspl, x)
    _bsplvb(f, K, i, k, x, out)
    return i


def _deriv(f: FP, K: Int, i: Int, k: Int, x: Float32, mut out: InlineArray[Float32, MAX_DEGREE + 1]):
    """The first derivatives of B_{i-k} .. B_i at x on span i into out[0..k]:
    B'_j = k * (N_j / (t_{j+k} - t_j) - N_{j+1} / (t_{j+k+1} - t_{j+1})) with
    N the degree k-1 basis (a zero-length span's term is 0); degree 0: 0."""
    for r in range(k + 1):
        out[r] = Float32(0)
    if k == 0:
        return
    var lo = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    _bsplvb(f, K, i, k - 1, x, lo)
    for r in range(k + 1):
        var j = i - k + r
        var a = Float32(0)
        var b = Float32(0)
        if r >= 1:
            var den = sub(ld(f, K + j + k), ld(f, K + j))
            if den != Float32(0):
                a = div(lo[r - 1], den)
        if r <= k - 1:
            var den = sub(ld(f, K + j + k + 1), ld(f, K + j + 1))
            if den != Float32(0):
                b = div(lo[r], den)
        out[r] = mul(Float32(k), sub(a, b))


@always_inline
def _mod_exact(a: Float32, b: Float32) -> Float32:
    """fmod(a, b) for b > 0, exact (C fmod: the sign of a), by integer long
    division of the significands; a NaN or infinite a gives NaN."""
    var ua = bitcast[DType.uint32](a)
    var ub = bitcast[DType.uint32](b)
    var ea = Int((ua >> 23) & 0xFF)
    var eb = Int((ub >> 23) & 0xFF)
    if ea == 0xFF:
        return canonical_nan()
    if abs(a) < b:
        return a
    var ma = Int(ua & 0x7FFFFF)
    var mb = Int(ub & 0x7FFFFF)
    if ea == 0:
        ea = 1
    else:
        ma |= 0x800000
    if eb == 0:
        eb = 1
    else:
        mb |= 0x800000
    var r = ma % mb
    for _ in range(ea - eb):
        r = (2 * r) % mb
    # r * 2^(eb - 150) (eb biased, 23 fraction bits), exactly representable
    var v = Float32(r)
    var e = eb - 150
    while e > 0:
        v = v * Float32(2)
        e -= 1
    while e < 0:
        v = v * Float32(0.5)
        e += 1
    # a subnormal result is flushed, as every unit operand is (DEVIATION 5408)
    return ftz(-v if a < Float32(0) else v)


def spline_apply_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, KNOTS, nk, degree, EXTRAP, W, BIAS, OUT]; t = i*d + c.
    Writes feature c's block of OUT row i (nspl = nk + degree - 1 columns, or
    nk - 1 periodic; the last dropped unless BIAS). EXTRAP 0 constant (the
    reference's: the first `degree` columns take the lower boundary values,
    the last `degree` the upper), 1 continue (the boundary span's
    polynomials), 2 linear (the first / last max(degree, 2) columns continue
    as value + (x - edge) * derivative at the edge), 3 periodic (x wrapped
    into the base interval by numpy's remainder, the wrapped basis functions
    added to the first degree columns). A NaN x (handle_missing='zeros')
    leaves its block 0. OUT arrives zeroed."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var nk = p(q, 4)
    var k = p(q, 5)
    var extrap = p(q, 6)
    var nb = nk + k - 1
    var nspl = nk - 1 if extrap == 3 else nb
    var K = p(q, 3) + c * (nk + 2 * k)
    var W = p(q, 7)
    var width = nspl if p(q, 8) != 0 else nspl - 1
    var O = p(q, 9) + i * W + c * width
    var x = ld(f, p(q, 0) + t)
    if is_nan(x):
        return
    var xmin = ld(f, K + k)
    var xmax = ld(f, K + k + nk - 1)
    var b = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    if extrap == 3:
        var per = sub(xmax, xmin)
        var xr = Float32(0)
        if per > Float32(0):
            var m = _mod_exact(sub(x, xmin), per)
            if m != Float32(0) and m < Float32(0):
                m = add(m, per)
            xr = add(xmin, m)
        var span = _basis(f, K, k, nb, xr, b)
        for r in range(k + 1):
            var col = span - k + r
            if col >= nspl:
                col -= nspl
            if col < width:
                st(f, O + col, add(ld(f, O + col), b[r]))
        return
    if extrap == 1 or (x >= xmin and x <= xmax):
        var span = _basis(f, K, k, nb, x, b)
        for r in range(k + 1):
            var col = span - k + r
            if col < width:
                st(f, O + col, b[r])
        return
    var below = x < xmin
    var edge = xmin if below else xmax
    var span = _basis(f, K, k, nb, edge, b)
    if extrap == 0:
        for r in range(k + 1):
            var col = span - k + r
            var keep = (col < k) if below else (col >= nspl - k)
            if keep and col < width:
                st(f, O + col, b[r])
        return
    var db = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    _deriv(f, K, span, k, edge, db)
    var nlin = k if k > 1 else k + 1
    var dx = sub(x, edge)
    for r in range(k + 1):
        var col = span - k + r
        var keep = (col < nlin) if below else (col >= nspl - nlin)
        if keep and col < width:
            st(f, O + col, add(b[r], mul(dx, db[r])))
