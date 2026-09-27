# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lars and LassoLars (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_least_angle.py`,
`_lars_path_solver` with a precomputed Gram (the `while True:` loop at its
line ~637): the most correlated inactive feature joins with the sign of its
correlation, the equiangular direction is `L L' ls = sign_A` scaled by
AA = 1/sqrt(sum ls*sign), the step is min(g1, g2, C/AA) over the inactive
correlations, the lasso drop at z_pos = min_pos(-coef_A / ls) (the lar
method flips the sign and adds nothing next step), the early stop at
alpha <= alpha_min with linear interpolation, the degenerate-regressor skip
(pivot < 1e-7) and the lasso stop when alpha grows. Named differences:
  * the correlations are recomputed as X'y - G coef every step instead of
    their incremental `Cov -= gamma * corr_eq_dir` (the same quantity), and
    their `np.around(corr_eq_dir, cov_precision)` is not applied;
  * the Cholesky of G_AA is refactored each step (x_linear/ops.mojo) instead
    of updated/downdated; ties in argmax|Cov| go to the lowest feature index
    (theirs: the lowest position in their permuted Cov array);
  * float32 throughout; equality tolerance float32 eps, tiny32 as theirs.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmin, fsign, ld, st, ldi, sti, i2f,
    fill, copy, cholesky, chol_solve, centered_gram,
)

comptime BIG = Float32(3.0e38)
comptime TINY32 = Float32(1.1754944e-38)
comptime EQ_TOL = Float32(1.1920929e-07)


def lars_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, lasso]; fp: [alpha_min].
    res: coef d, intercept, n_iter, alpha, n_active, active d.
    fw: xm d | G d*d | xty d | prev d | cov d | L d*d | ls d | sgn d | corr d.
    iw: state d (0 inactive, 1 active, 2 degenerate) | active list d."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var lasso = ldi(ip, 2) != 0
    var alpha_min = ld(fp, 0)
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var prev = xty + d
    var cov = prev + d
    var ll = cov + d
    var ls = ll + d * d
    var sgn = ls + d
    var corr = sgn + d
    var state = 0
    var act = d
    var ym = Float32(0)
    if fi:
        for j in range(d):
            var acc = Float32(0)
            for i in range(n):
                acc = fa(acc, ld(x, i * d + j))
            st(fw, xm + j, fd(acc, i2f(n)))
        var acc = Float32(0)
        for i in range(n):
            acc = fa(acc, ld(y, i))
        ym = fd(acc, i2f(n))
    else:
        fill(fw, xm, d, Float32(0))
    centered_gram(x, n, d, fw, xm, fw, gg)
    for j in range(d):
        var acc = Float32(0)
        var mj = ld(fw, xm + j)
        for i in range(n):
            acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(y, i), ym), acc)
        st(fw, xty + j, acc)
    fill(res, 0, d, Float32(0))
    fill(fw, prev, d, Float32(0))
    for j in range(d):
        sti(iw, state + j, 0)
    var k = 0
    var n_iter = 0
    var drop = False
    var alpha = Float32(0)
    var prev_alpha = Float32(0)
    var guard = 0
    while guard < 4 * d + 4 * max_iter + 8:
        guard += 1
        # correlations with the current residual
        for j in range(d):
            var acc = ld(fw, xty + j)
            for l in range(d):
                acc = fs(acc, fm(ld(fw, gg + j * d + l), ld(res, l)))
            st(fw, cov + j, acc)
        var c_idx = -1
        var cbig = Float32(0)
        for j in range(d):
            if ldi(iw, state + j) == 0:
                var a = fabs(ld(fw, cov + j))
                if c_idx < 0 or a > cbig:
                    c_idx = j
                    cbig = a
        alpha = fd(cbig, i2f(n))
        if alpha <= fa(alpha_min, EQ_TOL):
            if fabs(fs(alpha, alpha_min)) > EQ_TOL:
                if n_iter > 0:
                    var ss = fd(fs(prev_alpha, alpha_min), fs(prev_alpha, alpha))
                    for j in range(d):
                        var pj = ld(fw, prev + j)
                        st(res, j, fmad(ss, fs(ld(res, j), pj), pj))
                alpha = alpha_min
            break
        if n_iter >= max_iter or k >= d:
            break
        if not drop:
            if c_idx < 0:
                break
            sti(iw, act + k, c_idx)
            st(fw, sgn + k, fsign(ld(fw, cov + c_idx)))
            # the new pivot of the Cholesky of G_AA
            for a in range(k + 1):
                for b in range(k + 1):
                    st(fw, ll + a * (k + 1) + b, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + b)))
            var okc = cholesky(fw, ll, k + 1)
            if not okc or ld(fw, ll + k * (k + 1) + k) < Float32(1e-7):
                sti(iw, state + c_idx, 2)  # their degenerate-regressor skip
                continue
            sti(iw, state + c_idx, 1)
            k += 1
        if lasso and n_iter > 0 and prev_alpha < alpha:
            break
        # equiangular direction over the active set
        for a in range(k):
            for b in range(k):
                st(fw, ll + a * k + b, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + b)))
        _ = cholesky(fw, ll, k)
        for a in range(k):
            st(fw, ls + a, ld(fw, sgn + a))
        chol_solve(fw, ll, k, fw, ls)
        var aa: Float32
        if k == 1 and ld(fw, ls) == 0:
            st(fw, ls, Float32(1))
            aa = Float32(1)
        else:
            var sm = Float32(0)
            for a in range(k):
                sm = fmad(ld(fw, ls + a), ld(fw, sgn + a), sm)
            aa = fd(Float32(1), fsqrt(sm))
            for a in range(k):
                st(fw, ls + a, fm(ld(fw, ls + a), aa))
        var gamma = fd(cbig, aa)
        for j in range(d):
            if ldi(iw, state + j) != 0:
                continue
            var cj = Float32(0)
            for a in range(k):
                cj = fmad(ld(fw, gg + j * d + ldi(iw, act + a)), ld(fw, ls + a), cj)
            var cv = ld(fw, cov + j)
            var g1 = fd(fs(cbig, cv), fa(fs(aa, cj), TINY32))
            if g1 > 0 and g1 < gamma:
                gamma = g1
            var g2 = fd(fa(cbig, cv), fa(fa(aa, cj), TINY32))
            if g2 > 0 and g2 < gamma:
                gamma = g2
        drop = False
        var z_pos = BIG
        for a in range(k):
            var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
            if z > 0 and z < z_pos:
                z_pos = z
        if z_pos < gamma:
            for a in range(k):
                var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                if z == z_pos:
                    st(fw, sgn + a, -ld(fw, sgn + a))
            if lasso:
                gamma = z_pos
            drop = True
        n_iter += 1
        copy(fw, prev, res, 0, d)
        prev_alpha = alpha
        for a in range(k):
            var j = ldi(iw, act + a)
            st(res, j, fmad(gamma, ld(fw, ls + a), ld(res, j)))
        if drop and lasso:
            var w = 0
            for a in range(k):
                var j = ldi(iw, act + a)
                var z = fd(-ld(fw, prev + j), fa(ld(fw, ls + a), TINY32))
                if z == z_pos:
                    sti(iw, state + j, 0)
                    st(res, j, Float32(0))
                else:
                    sti(iw, act + w, j)
                    st(fw, sgn + w, ld(fw, sgn + a))
                    w += 1
            k = w
    var intercept = Float32(0)
    if fi:
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(res, j), acc)
        intercept = fs(ym, acc)
    st(res, d, intercept)
    st(res, d + 1, i2f(n_iter))
    st(res, d + 2, alpha)
    st(res, d + 3, i2f(k))
    for a in range(d):
        st(res, d + 4 + a, i2f(ldi(iw, act + a)) if a < k else Float32(-1))
