# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IterativeImputer units with its default estimator, BayesianRidge, in the
prep lane's program model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/impute/_iterative.py` (`fit_transform`
round-robin over the features in the imputation order, `_impute_one_feature`
on the rows where the feature is observed, the inf-norm stop against
tol * max|X_observed|) and `sklearn/linear_model/_bayes.py` (BayesianRidge
`fit`, `_update_coef_`). The reference's SVD of the centred training block is
the eigendecomposition of its Gram matrix (x_prep/eigh.mojo), and its sse
||y - X coef||^2 is spelled from the Gram matrix, y'y - 2 coef'X'y +
coef'X'X coef (clamped at 0), so one BayesianRidge fit costs O(d^2) per
iteration after one O(n d^2) Gram pass. A FLAG slot, set once the imputer
has converged, turns every later stage of the program into a no-op, so a
whole max_iter program runs in one binding call.
"""
from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div

comptime BR_MAX_ITER = 300
comptime BR_TOL = Float32(1.0e-3)
comptime BR_PRIOR = Float32(1.0e-6)
comptime F64_EPS = Float32(2.220446e-16)


@always_inline
def _done(f: FP, q: IP, k: Int) -> Bool:
    return ld(f, p(q, k)) != Float32(0)


def ii_mean_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, MASK, j, MEANS, CNT, FLAG]; t = column a. The mean of
    column a over the rows where feature j is observed (MASK == 0); a == 0
    also writes that row count to CNT."""
    if _done(f, q, 7):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var j = p(q, 4)
    var s = Float32(0)
    var cnt = 0
    for i in range(n):
        if ld(f, p(q, 3) + i * d + j) != Float32(0):
            continue
        s = add(s, ld(f, p(q, 0) + i * d + t))
        cnt += 1
    st(f, p(q, 5) + t, div(s, Float32(cnt)) if cnt > 0 else Float32(0))
    if t == 0:
        st(f, p(q, 6), Float32(cnt))


def ii_gram_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, MASK, j, MEANS, G, FLAG]; t = a*d + b. The centred cross
    product of columns a and b over feature j's observed rows."""
    if _done(f, q, 7):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var j = p(q, 4)
    var a = t // d
    var b = t % d
    var ma = ld(f, p(q, 5) + a)
    var mb = ld(f, p(q, 5) + b)
    var s = Float32(0)
    for i in range(n):
        if ld(f, p(q, 3) + i * d + j) != Float32(0):
            continue
        s = add(s, mul(sub(ld(f, p(q, 0) + i * d + a), ma), sub(ld(f, p(q, 0) + i * d + b), mb)))
    st(f, p(q, 6) + t, s)


def ii_sub_unit(t: Int, f: FP, q: IP):
    """q = [G, d, j, GS, FLAG]; t = 0: GS = G without row and column j."""
    if _done(f, q, 4):
        return
    var d = p(q, 1)
    var j = p(q, 2)
    var r = 0
    for a in range(d):
        if a == j:
            continue
        var c = 0
        for b in range(d):
            if b == j:
                continue
            st(f, p(q, 3) + r * (d - 1) + c, ld(f, p(q, 0) + a * d + b))
            c += 1
        r += 1


def ii_br_unit(t: Int, f: FP, q: IP):
    """q = [G, d, j, EIG, V, MEANS, CNT, COEF, INTER, FLAG, W]; t = 0.
    BayesianRidge on the centred block: X'X = G without j (eigenpairs EIG, V
    of size p = d - 1), X'y = G[:, j], y'y = G[j, j]. COEF[a] (0 at j) and
    INTER = mean_j - sum_a mean_a COEF[a]. W is p floats of scratch."""
    if _done(f, q, 9):
        return
    var d = p(q, 1)
    var j = p(q, 2)
    var G = p(q, 0)
    var EIG = p(q, 3)
    var V = p(q, 4)
    var W = p(q, 10)
    var pp = d - 1
    var ntr = ld(f, p(q, 6))
    var yty = ld(f, G + j * d + j)
    # W = V' X'y
    for r in range(pp):
        var s = Float32(0)
        var a2 = 0
        for a in range(d):
            if a == j:
                continue
            s = add(s, mul(ld(f, V + a2 * pp + r), ld(f, G + a * d + j)))
            a2 += 1
        st(f, W + r, s)
    var yvar = div(yty, ntr) if ntr > Float32(0) else Float32(0)
    var alpha = div(Float32(1), add(yvar, F64_EPS))
    var lam = Float32(1)
    var C = p(q, 7)
    for a in range(d):
        st(f, C + a, Float32(0))
    for it in range(BR_MAX_ITER + 1):
        var ratio = div(lam, alpha)
        var change = Float32(0)
        var a2 = 0
        var csq = Float32(0)
        for a in range(d):
            if a == j:
                continue
            var s = Float32(0)
            for r in range(pp):
                var e = ld(f, EIG + r)
                if e < Float32(0):
                    e = Float32(0)
                s = add(s, div(mul(ld(f, V + a2 * pp + r), ld(f, W + r)), add(e, ratio)))
            change = add(change, abs(sub(s, ld(f, C + a))))
            st(f, C + a, s)
            csq = add(csq, mul(s, s))
            a2 += 1
        if it == BR_MAX_ITER:
            break          # the final coef, from the last alpha and lambda
        # sse = y'y - 2 coef'X'y + coef'X'X coef
        var cxy = Float32(0)
        var quad = Float32(0)
        for a in range(d):
            if a == j:
                continue
            cxy = add(cxy, mul(ld(f, C + a), ld(f, G + a * d + j)))
            var row = Float32(0)
            for b in range(d):
                if b == j:
                    continue
                row = add(row, mul(ld(f, G + a * d + b), ld(f, C + b)))
            quad = add(quad, mul(ld(f, C + a), row))
        var sse = add(sub(yty, mul(Float32(2), cxy)), quad)
        if sse < Float32(0):
            sse = Float32(0)
        var gamma = Float32(0)
        for r in range(pp):
            var e = ld(f, EIG + r)
            if e < Float32(0):
                e = Float32(0)
            var ae = mul(alpha, e)
            gamma = add(gamma, div(ae, add(lam, ae)))
        lam = div(add(gamma, mul(Float32(2), BR_PRIOR)), add(csq, mul(Float32(2), BR_PRIOR)))
        alpha = div(add(sub(ntr, gamma), mul(Float32(2), BR_PRIOR)), add(sse, mul(Float32(2), BR_PRIOR)))
        if it != 0 and change < BR_TOL:
            # one more pass computes the final coef from the updated alpha, lambda
            var ratio2 = div(lam, alpha)
            var b2 = 0
            for a in range(d):
                if a == j:
                    continue
                var s = Float32(0)
                for r in range(pp):
                    var e = ld(f, EIG + r)
                    if e < Float32(0):
                        e = Float32(0)
                    s = add(s, div(mul(ld(f, V + b2 * pp + r), ld(f, W + r)), add(e, ratio2)))
                st(f, C + a, s)
                b2 += 1
            break
    var inter = ld(f, p(q, 5) + j)
    for a in range(d):
        if a == j:
            continue
        inter = sub(inter, mul(ld(f, p(q, 5) + a), ld(f, C + a)))
    st(f, p(q, 8), inter)


def ii_predict_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, MASK, j, COEF, INTER, BOUNDS, FLAG]; t = row. Where
    feature j is missing, X[i, j] = clip(INTER + sum_a COEF[a] X[i, a],
    BOUNDS[2j], BOUNDS[2j+1]), a ascending."""
    if p(q, 8) >= 0 and _done(f, q, 8):
        return
    var d = p(q, 2)
    var j = p(q, 4)
    if ld(f, p(q, 3) + t * d + j) == Float32(0):
        return
    var v = ld(f, p(q, 6))
    for a in range(d):
        if a == j:
            continue
        v = add(v, mul(ld(f, p(q, 5) + a), ld(f, p(q, 0) + t * d + a)))
    var lo = ld(f, p(q, 7) + 2 * j)
    var hi = ld(f, p(q, 7) + 2 * j + 1)
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    st(f, p(q, 0) + t * d + j, v)


def ii_snapshot_unit(t: Int, f: FP, q: IP):
    """q = [X, PREV, FLAG]; t = element: PREV = X."""
    if _done(f, q, 2):
        return
    st(f, p(q, 1) + t, ld(f, p(q, 0) + t))


def ii_conv_unit(t: Int, f: FP, q: IP):
    """q = [X, PREV, count, TOL, FLAG, NITER]; t = 0. One more round counted;
    FLAG = 1 when max |X - PREV| < TOL (the reference's inf-norm stop)."""
    if _done(f, q, 4):
        return
    st(f, p(q, 5), add(ld(f, p(q, 5)), Float32(1)))
    var m = Float32(0)
    for i in range(p(q, 2)):
        var e = abs(sub(ld(f, p(q, 0) + i), ld(f, p(q, 1) + i)))
        if e > m:
            m = e
    if m < ld(f, p(q, 3)):
        st(f, p(q, 4), Float32(1))


def nan_mask_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, KEEP, dout, OUT]; t = i*dout + jj: 1 where input column
    KEEP[jj] of row i is NaN, else 0."""
    var dout = p(q, 4)
    var i = t // dout
    var c = Int(ld(f, p(q, 3) + t % dout))
    var x = f.unsafe_load(p(q, 0) + i * p(q, 2) + c)
    st(f, p(q, 5) + t, Float32(1) if x != x else Float32(0))
