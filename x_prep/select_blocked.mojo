# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BLOCKED ROW FOLDS of the univariate scores (lane fam-prep-metrics,
2026-10-04), IDENTICAL on every vendor and the host column.

`f_classif_unit` and `f_regression_unit` (x_prep/stats.mojo) fold all n rows
of a feature on ONE thread: at the board's 1M rows that is d threads busy
and the rest of the GPU idle. Here the rows are cut into x_prep/blocked.mojo's
XB-row blocks: one unit per (block, feature) folds its block ascending from
zero, and one unit per feature folds the block partials ascending from zero
and finishes the score exactly as the serial unit does. The same units run
on the host (the program model, x_prep/common.mojo), so the host column gets
the same order.

BITS. The within-class squares, the column / target sums and the centred
cross products take the blocked order: a NEW order for n > XB (one block at
n <= XB: the serial unit's words). Every other operation is the serial
unit's. A blocked sum is never less accurate than the row-order one.

-D MOJOLEARN_IDN_SELECT_BLOCKED_OFF restores the serial units (the switch is
bit 4 of `x_prep_idn_fam`; python/mojolearn/_expansion_prep.py stages these
ops only when it is set).
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, st, canonical_nan
from x_prep.prims import add, sub, mul, div
from x_prep.stats import F32_MAX, pos_inf, f_sf, sqrtf_
from x_prep.blocked import _span

comptime IDN_SELECT_BLOCKED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_SELECT_BLOCKED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


# ---------------------------------------------------------------- f_classif
def fcb_part_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, MEAN, PS, nb]; t = b*d + c. Over the block's rows
    ascending from zero: PS[t] = the sum of (x - MEAN[y, c])^2, f_classif's
    within-class squares."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var M = p(q, 4)
    var c = t % d
    var r = _span(t // d, n)
    var ssw = Float32(0)
    for i in range(r[0], r[1]):
        var k = Int(ld(f, Y + i))
        var e = sub(ld(f, X + i * d + c), ld(f, M + k * d + c))
        ssw = add(ssw, mul(e, e))
    st(f, p(q, 5) + t, ssw)


def fcb_fin_unit(t: Int, f: FP, q: IP):
    """q = [PS, n, d, nb, K, CNT, MEAN, SCORES, PV]; t = feature.
    `f_classif_unit` with the within-class squares folded from fcb_part's
    partials, blocks ascending from zero."""
    var PS = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var nb = p(q, 3)
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
    for b in range(nb):
        ssw = add(ssw, ld(f, PS + b * d + c))
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


# ---------------------------------------------------------------- f_regression
def frb_part1_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, PSX, PSY, nb]; t = b*d + c. Over the block's rows
    ascending from zero: PSX[t] = the sum of column c and, for c == 0,
    PSY[b] = the sum of the target."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var c = t % d
    var b = t // d
    var r = _span(b, n)
    var sx = Float32(0)
    for i in range(r[0], r[1]):
        sx = add(sx, ld(f, X + i * d + c))
    st(f, p(q, 4) + t, sx)
    if c == 0:
        var sy = Float32(0)
        for i in range(r[0], r[1]):
            sy = add(sy, ld(f, Y + i))
        st(f, p(q, 5) + b, sy)


def frb_mean_unit(t: Int, f: FP, q: IP):
    """q = [PSX, PSY, nb, d, n, MX, MY]; t = feature. MX[t] = the block sums
    folded ascending from zero, over n; for t == 0 also MY[0] likewise."""
    var nb = p(q, 2)
    var d = p(q, 3)
    var nf = Float32(p(q, 4))
    var sx = Float32(0)
    for b in range(nb):
        sx = add(sx, ld(f, p(q, 0) + b * d + t))
    st(f, p(q, 5) + t, div(sx, nf))
    if t == 0:
        var sy = Float32(0)
        for b in range(nb):
            sy = add(sy, ld(f, p(q, 1) + b))
        st(f, p(q, 6), div(sy, nf))


def frb_part2_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, MX, MY, PXY, PXX, PYY, nb]; t = b*d + c. Over the
    block's rows ascending from zero, with ex = x - MX[c] and ey = y - MY[0]
    (MX < 0: uncentred, both means zero): PXY[t] = sum ex ey, PXX[t] = sum
    ex^2 and, for c == 0, PYY[b] = sum ey^2."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var c = t % d
    var b = t // d
    var r = _span(b, n)
    var mx = Float32(0)
    var my = Float32(0)
    if p(q, 4) >= 0:
        mx = ld(f, p(q, 4) + c)
        my = ld(f, p(q, 5))
    var sxy = Float32(0)
    var sxx = Float32(0)
    var syy = Float32(0)
    for i in range(r[0], r[1]):
        var ex = sub(ld(f, X + i * d + c), mx)
        var ey = sub(ld(f, Y + i), my)
        sxy = add(sxy, mul(ex, ey))
        sxx = add(sxx, mul(ex, ex))
        syy = add(syy, mul(ey, ey))
    st(f, p(q, 6) + t, sxy)
    st(f, p(q, 7) + t, sxx)
    if c == 0:
        st(f, p(q, 8) + b, syy)


def frb_fin_unit(t: Int, f: FP, q: IP):
    """q = [PXY, PXX, PYY, nb, d, n, CENTER, SCORES, PV, CORR, FORCE_FINITE];
    t = feature. `f_regression_unit`'s score from frb_part2's partials,
    blocks ascending from zero."""
    var nb = p(q, 3)
    var d = p(q, 4)
    var n = p(q, 5)
    var c = t
    var sxy = Float32(0)
    var sxx = Float32(0)
    var syy = Float32(0)
    for b in range(nb):
        sxy = add(sxy, ld(f, p(q, 0) + b * d + c))
        sxx = add(sxx, ld(f, p(q, 1) + b * d + c))
        syy = add(syy, ld(f, p(q, 2) + b))
    var dof = Float32(n - 2) if p(q, 6) != 0 else Float32(n - 1)
    var ff = p(q, 10) != 0
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
    st(f, p(q, 7) + c, score)
    st(f, p(q, 8) + c, pv)
    if p(q, 9) >= 0:
        st(f, p(q, 9) + c, r)
