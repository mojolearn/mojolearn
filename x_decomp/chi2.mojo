# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The chi-squared CDF and quantile of MinCovDet, and LDA's host bound terms, in Mojo (lane py-runtime-b,
2026-10-05): `_expansion_decomp._chi2_cdf` and `_chi2_quantile`, parameter
math only (dof and a probability, no data). The same steps as the Python:
the prefactor x^a e^-x / Gamma(a + 1) through the cells' logs, lgamma and exp
on float32 words (`ew_cell`, the cell every executor applies), the series
in float64, the 80 bisections in float64 and the result rounded once to
float32. So the same words on every column."""
from x_decomp.cells import F32Ptr, ew_cell

comptime _OP_ADD = 0
comptime _OP_SUB = 1
comptime _OP_MUL = 2
comptime _OP_MAX = 30
comptime _OP_EXP = 9
comptime _OP_LOGS = 10
comptime _OP_LGAMMA = 37
#: the series's term bound and the bisection count of `_chi2_cdf` / `_chi2_quantile`
comptime CHI2_TERMS = 4000
comptime CHI2_BISECT = 80


#: the continued fraction's term bound and its floor (Numerical Recipes `gcf`)
comptime CHI2_CF_TERMS = 4000
comptime CHI2_CF_FPMIN = 1e-300


def chi2_series(a: Float64, x: Float64) -> Float64:
    """The series sum of P(a, x) = pref * sum, pref = x^a e^-x / Gamma(a + 1)."""
    var term = 1.0
    var tot = 1.0
    for n in range(1, CHI2_TERMS):  # small-loop(CHI2_TERMS: series terms): float64 scalar series, no data
        term *= x / (a + Float64(n))
        tot += term
        if term < 1e-17 * tot:
            break
    return tot


def chi2_cf(a: Float64, x: Float64) -> Float64:
    """The modified Lentz continued fraction of Q(a, x) = pref * cf,
    pref = x^a e^-x / Gamma(a) (Numerical Recipes `gcf`), for x >= a + 1."""
    var b = x + 1.0 - a
    var c = 1.0 / CHI2_CF_FPMIN
    var d = 1.0 / b
    var h = d
    for i in range(1, CHI2_CF_TERMS):  # small-loop(CHI2_CF_TERMS: fraction terms): float64 scalar fraction, no data
        var fi = Float64(i)
        var an = -fi * (fi - a)
        b += 2.0
        d = an * d + b
        if abs(d) < CHI2_CF_FPMIN:
            d = CHI2_CF_FPMIN
        c = b + an / c
        if abs(c) < CHI2_CF_FPMIN:
            c = CHI2_CF_FPMIN
        d = 1.0 / d
        var dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-15:
            break
    return h


def chi2_cdf(dof: Float64, m: Float64) -> Float64:
    """P(chi2_dof <= m): the regularized lower incomplete gamma P(dof/2, m/2).
    The standard split (lane apple-holes, 2026-10-12): the series for P below
    x = a + 1, and 1 - Q by the continued fraction from x = a + 1 up. The
    series's prefactor e^(a ln x - x - lgamma(a + 1)) underflows the float32
    exp cell far above the mean (dof 220 at the bisection's upper bracket:
    exponent about -196), so P read 0 over the upper range and MinCovDet's
    consistency factor divided by 0; Q's prefactor underflows to 0 there,
    where 1 - Q = 1 is the right value."""
    var a = dof / 2.0
    var x = m / 2.0
    if x <= 0:
        return 0.0
    var z = Float32(0)
    var lx = Float64(ew_cell(_OP_LOGS, Float32(x), z, z, Float32(1e-30)))
    if x < a + 1.0:
        var lg = Float64(ew_cell(_OP_LGAMMA, Float32(a + 1), z, z, z))
        var pref = Float64(ew_cell(_OP_EXP, Float32(a * lx - x - lg), z, z, z))
        return pref * chi2_series(a, x)
    var lga = Float64(ew_cell(_OP_LGAMMA, Float32(a), z, z, z))
    var prefq = Float64(ew_cell(_OP_EXP, Float32(a * lx - x - lga), z, z, z))
    return 1.0 - prefq * chi2_cf(a, x)


def chi2_quantile(dof: Float64, upper: Float64) -> Float64:
    """The point m with P(chi2_dof > m) = upper, by 80 bisections, rounded
    once to float32."""
    var target = 1.0 - upper
    var lo = 0.0
    var hi = 4.0 * dof + 40.0
    if not (hi > 1.0):
        hi = 1.0
    for _ in range(CHI2_BISECT):  # small-loop(CHI2_BISECT: bisection steps): float64 scalar search, no data
        var mid = 0.5 * (lo + hi)
        if chi2_cdf(dof, mid) < target:
            lo = mid
        else:
            hi = mid
    return Float64(Float32(0.5 * (lo + hi)))


def consistency_factor(p: Float64, alpha: Float64) raises -> Float64:
    """sklearn `_robust_covariance.py::_consistency_factor` (Pison 2002):
    alpha / chi2.cdf(chi2.ppf(alpha, p), p + 2), rounded once to float32
    (lane apple-holes: the division was Python's)."""
    var q = chi2_quantile(p, 1.0 - alpha)
    var c = chi2_cdf(p + 2.0, q)
    if not (c > 0.0):
        raise Error("x_decomp: chi2 consistency factor: the cdf is 0")
    return Float64(Float32(alpha / c))


def lda_bound_host(px: F32Ptr, pd: F32Ptr, pc: F32Ptr, po: F32Ptr, n: Int, kk: Int, vv: Int, floor: Float32):
    """The bound's term matrix x * (lse_t + max) per (i, w) (see below)."""
    var z = Float32(0)
    for c in range(n * vv):  # host column only (registered in the host binding)
        var i = c // vv
        var w = c - i * vv
        var mx = Float32(0)
        for t in range(kk):
            var term = ew_cell(_OP_ADD, ew_cell(_OP_ADD, z, pd.unsafe_load(i * kk + t), z, z), pc.unsafe_load(t * vv + w), z, z)
            mx = term if t == 0 else ew_cell(_OP_MAX, mx, term, z, z)
        var acc = Float32(0)
        for t in range(kk):
            var term = ew_cell(_OP_ADD, ew_cell(_OP_ADD, z, pd.unsafe_load(i * kk + t), z, z), pc.unsafe_load(t * vv + w), z, z)
            acc = ew_cell(_OP_ADD, acc, ew_cell(_OP_EXP, ew_cell(_OP_SUB, term, mx, z, z), z, z, z), z, z)
        var lse = ew_cell(_OP_ADD, ew_cell(_OP_LOGS, acc, z, z, floor), mx, z, z)
        po.unsafe_store(c, ew_cell(_OP_MUL, px.unsafe_load(c), lse, z, z))


def consistency_factor_run(p: Float64, num: Float64, dn: Float64) raises -> Float64:
    """`consistency_factor_py`'s typed body: the den check, then
    `consistency_factor(p, num / den)`."""
    if not (dn > 0.0):
        raise Error("x_decomp: chi2 consistency factor: den must be > 0")
    return consistency_factor(p, num / dn)


def lda_bound_host_run(
    px: F32Ptr, pd: F32Ptr, pc: F32Ptr, po: F32Ptr, n: Int, kk: Int, vv: Int, f: Float64
) raises -> Int:
    """`lda_bound_host_py`'s typed body: the shape check, the term matrix,
    the term count n * v."""
    if n < 0 or kk < 1 or vv < 0 or n * vv > 2147483647:
        raise Error("x_decomp: lda_bound shape out of range")
    var floor = Float32(f)
    lda_bound_host(px, pd, pc, po, n, kk, vv, floor)
    return n * vv
