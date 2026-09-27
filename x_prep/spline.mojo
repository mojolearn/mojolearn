# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SplineTransformer units, in the prep lane's program model.

Reference: scikit-learn 1.9 `sklearn/preprocessing/_polynomial.py`
(SplineTransformer `_get_base_knot_positions`, `fit` knot extension,
`transform` with extrapolation 'constant' / 'continue'), and the
B-spline basis by the Cox-de Boor triangle (de Boor, A Practical Guide to
Splines, BSPLVB), which is what scipy's BSpline evaluates.
"""
from x_prep.common import FP, IP, p, ld, st, is_nan
from x_prep.prims import add, sub, mul, div

comptime MAX_DEGREE = 7


def spline_knots_unit(t: Int, f: FP, q: IP):
    """q = [BASE, nk, d, degree, KNOTS, UNIFORM, ST]; t = column. With UNIFORM
    the base knots are first written as numpy's linspace(min, max, nk) from
    the col_stats rows ST (start + i * step, the last one max); otherwise
    BASE[c*nk :] holds them already. KNOTS[c*(nk + 2*degree) :] is the
    extended knot vector: degree knots below at the first spacing and degree
    above at the last."""
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
    var dmin = sub(ld(f, B + 1), ld(f, B))
    var dmax = sub(ld(f, B + nk - 1), ld(f, B + nk - 2))
    for i in range(k):
        st(f, K + i, sub(ld(f, B), mul(Float32(k - i), dmin)))
    for i in range(nk):
        st(f, K + k + i, ld(f, B + i))
    for i in range(k):
        st(f, K + k + nk + i, add(ld(f, B + nk - 1), mul(Float32(i + 1), dmax)))


def _basis(f: FP, K: Int, k: Int, nspl: Int, x: Float32, mut out: InlineArray[Float32, MAX_DEGREE + 1]) -> Int:
    """The k+1 nonzero basis values at x into out[0..k]; returns the span i
    (they are B_{i-k} .. B_i). x outside the base interval uses the boundary
    span's polynomials (scipy's extrapolate=True)."""
    var i = k
    while i + 1 < nspl and ld(f, K + i + 1) <= x:
        i += 1
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
    return i


def spline_apply_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, KNOTS, nk, degree, EXTRAP, W, BIAS, OUT]; t = i*d + c.
    Writes feature c's block of OUT row i (nspl = nk + degree - 1 columns, the
    last dropped unless BIAS). EXTRAP 0 constant (the boundary value), 1
    continue (the boundary span's polynomials). OUT arrives zeroed."""
    var d = p(q, 2)
    var i = t // d
    var c = t % d
    var nk = p(q, 4)
    var k = p(q, 5)
    var nspl = nk + k - 1
    var K = p(q, 3) + c * (nk + 2 * k)
    var W = p(q, 7)
    var width = nspl if p(q, 8) != 0 else nspl - 1
    var O = p(q, 9) + i * W + c * width
    var x = ld(f, p(q, 0) + t)
    var xmin = ld(f, K + k)
    var xmax = ld(f, K + k + nk - 1)
    var extrap = p(q, 6)
    var b = InlineArray[Float32, MAX_DEGREE + 1](fill=Float32(0))
    if extrap == 1 or (x >= xmin and x <= xmax):
        var span = _basis(f, K, k, nspl, x, b)
        for r in range(k + 1):
            var col = span - k + r
            if col < width:
                st(f, O + col, b[r])
        return
    var edge = xmin if x < xmin else xmax
    var span = _basis(f, K, k, nspl, edge, b)
    for r in range(k + 1):
        var col = span - k + r
        if col < width:
            st(f, O + col, b[r])
