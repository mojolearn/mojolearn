# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lars and LassoLars (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_least_angle.py`,
`_lars_path_solver` with a precomputed Gram (the `while True:` loop at its
line ~637): the most correlated inactive feature joins with the sign of its
correlation, the equiangular direction is `L L' ls = sign_A` scaled by
AA = 1/sqrt(sum ls*sign), the step is min(g1, g2, C/AA) over the inactive
correlations, the lasso drop at z_pos = min_pos(-coef_A / ls), the early stop at
alpha <= alpha_min with linear interpolation, the degenerate-regressor skip
(pivot < 1e-7) and the lasso stop when alpha grows. Named differences:
  * the correlations are recomputed as X'y - G coef every step instead of
    their incremental `Cov -= gamma * corr_eq_dir` (the same quantity), and
    their `np.around(corr_eq_dir, cov_precision)` is not applied;
  * the Cholesky of G_AA is refactored each step (x_linear/ops.mojo) instead
    of updated/downdated; ties in argmax|Cov| go to the lowest feature index
    (theirs: the lowest position in their permuted Cov array);
  * float32 throughout; equality tolerance float32 eps, tiny32 as theirs;
  * DEVIATION 5010: the lar method lets an active coefficient cross zero
    (plain LAR); theirs flips its sign and adds nothing next step, which
    diverges (see the comment at the step).
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmin, fsign, ld, st, ldi, sti, i2f,
    fill, copy, cholesky, chol_solve, centered_gram, centered_xty, add_acc,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.tops import t_col_means, t_mean, t_centered_gram, t_centered_xty, t_cholesky
from std.sys.compile import is_defined
from core.classical_centered import centered_gram_v1_cell, centered_cross_v1_cell, bm_column_mean_cell

# lane/lars-grid-gram (2026-10-11), PROMOTED to the default (lane
# i17-decide-nbr-lin-prep, 2026-10-11); `-D MOJOLEARN_IDN_LARS_GRID_GRAM_OFF`
# (or MOJOLEARN_IDN_ALL_OFF) restores main's moments grid. A/B one run per arm,
# main@b2657cc89 (runs v0045/v0046, a1628/a1629), NV L40S / AMD MI325X ms:
#   lars taxi          59.1 -> 3.1 (0.05)    / 126.1 -> 3.3 (0.03)
#   lasso-lars taxi    58.9 -> 3.0 (0.05)    / 125.7 -> 3.3 (0.03)
#   lasso-lars istella 112.8 -> 83.4 (0.74)  / 191.6 -> 74.6 (0.39)
#   lars istella       165.6 -> 340.7 (2.06) / 390.2 -> 1155.2 (2.96)  SLOWER
# Digests NV == AMD on both arms; quality equal or better (lars istella r2
# 0.3090 -> 0.3287, the rest equal to 1e-4). OPEN FOLLOW-UP: the lars istella
# regression (one cell of four; the net is a large win). Find its structural
# cause and fix it by size/cost reasoning that covers neighboring shapes; no
# dataset- or dimension-keyed dodge (CLAUDE.md, No dimension targeting).
# The route takes LARS / LassoLars' moments from the
# row-parallel blocked moments (core/blocked_moments.mojo: leaf column sums +
# binary-counter fold for the means of X and y, `bm_centered_gram_panels` and
# `bm_centered_cross_panels` at leaves of `contract_leaf_size(n)` rows) instead
# of main's moments grid (x_linear/moments_grid.mojo: one block per 16-column
# tile pair, one serial chain over every row per cell). Structural reason: at
# taxi's 16 features main's grid is 2 tiles -> 3 blocks, each cell one
# 1M-row dependent chain; the blocked form puts one block on every leaf of
# rows (hundreds of blocks) and folds the leaves in log2 levels. Expected
# lars taxi 60.8 -> ~10 ms NV (ideas-dense-linalg-20261011 F5). Bits change
# (the fold order), on NVIDIA, AMD and this host column together: the host
# below runs the device's reference cells (`centered_gram_v1_cell`,
# `centered_cross_v1_cell`, `bm_column_mean_cell`). Apple runs the same
# blocked fold (lane/apple-lars, 2026-10-12; x_linear/device.mojo
# `XL_IDN_LARS_GRID_GRAM`), so its column matches NVIDIA == AMD == host.
comptime LARS_GRID_GRAM = not is_defined["MOJOLEARN_IDN_LARS_GRID_GRAM_OFF"]() and not is_defined[
    "MOJOLEARN_IDN_ALL_OFF"
]()


def _lars_host_blocked_moments(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, gg: Int, xty: Int) -> Float32:
    """The host column of MOJOLEARN_IDN_LARS_GRID_GRAM: the means (zeros
    without an intercept), the centered Gram (upper cells, mirrored: the
    product a*b is commutative, so the mirror is the same word) and the
    centered X'y, each by the blocked fold the device runs. Returns y's mean."""
    var ym = Float32(0)
    fill(fw, xm, d, Float32(0))
    if fi and n > 0:
        for j in range(d):
            st(fw, xm + j, bm_column_mean_cell(x, n, d, j))
        ym = bm_column_mean_cell(y, n, 1, 0)
    var mp = fw + xm
    for i in range(d):
        for j in range(i, d):
            var v = centered_gram_v1_cell(x, mp, n, d, i, j)
            st(fw, gg + i * d + j, v)
            st(fw, gg + j * d + i, v)
    for j in range(d):
        st(fw, xty + j, centered_cross_v1_cell(x, mp, y, ym, n, d, j))
    return ym

comptime BIG = Float32(3.0e38)
comptime TINY32 = Float32(1.1754944e-38)
comptime EQ_TOL = Float32(1.1920929e-07)


def lars_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, lasso, positive]; fp: [alpha_min].
    res: coef d, intercept, n_iter, alpha, n_active, active d.
    fw: xm d | G d*d | xty d | prev d | cov d | L d*d | ls d | sgn d | corr d.
    iw: state d (0 inactive, 1 active, 2 degenerate) | active list d."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var lasso = ldi(ip, 2) != 0
    var alpha_min = ld(fp, 0)
    var positive = ldi(ip, 3) != 0
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
    # Team form: the row passes (means, Gram, X'y) across the team
    # (x_linear/tops.mojo); the path itself (d x d) on the lead.
    var ym = Float32(0)
    comptime if is_gpu():
        # the device (x_linear/device.mojo `lars_path_kernel`): the moments
        # grid already wrote the means, the Gram and X'y into fw, and y's
        # mean waits in fw[prev] (read here, then the slot zeroed as the
        # path expects); no row pass runs on the team
        ym = ld(fw, prev)
        t.sync()
        if t.lead():
            st(fw, prev, Float32(0))
        t.sync()
    else:
        # the host: one row pass per statistic, vector accumulators (lane linear-cpu)
        comptime if LARS_GRID_GRAM:
            ym = _lars_host_blocked_moments(x, y, n, d, fi, fw, xm, gg, xty)
        else:
            if fi:
                fill(fw, xm, d, Float32(0))
                for i in range(n):
                    add_acc(fw, xm, x, i * d, d)
                for j in range(d):
                    st(fw, xm + j, fd(ld(fw, xm + j), i2f(n)))
                var acc = Float32(0)
                for i in range(n):
                    acc = fa(acc, ld(y, i))
                ym = fd(acc, i2f(n))
            else:
                fill(fw, xm, d, Float32(0))
            centered_gram(x, n, d, fw, xm, fw, gg)
            centered_xty(x, y, n, d, fw, xm, ym, fw, xty)
    comptime if is_gpu():
        if t.nt > 1:
            _lars_path_team(t, n, d, max_iter, lasso, positive, alpha_min, fi, ym, fw, iw, res,
                            xm, gg, xty, prev, cov, ll, ls, sgn, corr, state, act)
            return
    if not t.lead():
        return
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
                # their positive=True takes argmax(Cov), not argmax |Cov|
                var a = ld(fw, cov + j) if positive else fabs(ld(fw, cov + j))
                # DEVIATION 5005 (IDENTITY_PATHS row 105): strict >, the
                # lowest index wins an exact tie
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
            st(fw, sgn + k, Float32(1) if positive else fsign(ld(fw, cov + c_idx)))
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
            if not positive:
                var g2 = fd(fa(cbig, cv), fa(fa(aa, cj), TINY32))
                if g2 > 0 and g2 < gamma:
                    gamma = g2
        drop = False
        var z_pos = BIG
        # DEVIATION 5010 (x_linear/README.md): the zero crossing is the lasso
        # modification only. Their lar method also flips sign_active and
        # skips the next addition when an active coefficient crosses zero,
        # which breaks the equiangular invariant (an active correlation keeps
        # its sign in LAR; only the coefficient changes sign) and diverges:
        # their own Lars, float64, ends far from least squares even with
        # every feature active (python -m mojolearn.tests.test_x_linear_sanity
        # lars-crossing). Plain LAR (Efron et al. 2004) lets it cross.
        if lasso:
            for a in range(k):
                var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                if z > 0 and z < z_pos:
                    z_pos = z
            if z_pos < gamma:
                for a in range(k):
                    var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                    if z == z_pos:
                        st(fw, sgn + a, -ld(fw, sgn + a))
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


def _lars_path_team(
    t: Team, n: Int, d: Int, max_iter: Int, lasso: Bool, positive: Bool, alpha_min: Float32, fi: Bool, ym: Float32,
    fw: FP, iw: IP, res: FP, xm: Int, gg: Int, xty: Int, prev: Int, cov: Int, ll: Int, ls: Int, sgn: Int, corr: Int,
    state: Int, act: Int,
):
    """lars_fit's path on the whole team (lane/neural-pass90, 2026-10-01).
    Every thread runs the loop and computes its scalar decisions from the
    same words (read after a barrier), so the control flow is uniform; the
    lead alone writes the shared state. The O(d^2) and O(k^3) parts are split
    across the team, each value by the lead loop's own statements: the
    correlations (one j a thread, its chain over l ascending), the two
    Cholesky factors a step (`t_cholesky`) and the d x k products of the step
    search (cj, one j a thread); the minimum over j stays the lead's scan.
    The device ran the whole path on the lead thread: Lars on istella 100k x
    220 took 9.4 s on the M4's GPU. `-D MOJOLEARN_X_LINEAR_LARS_LEAD=1`
    restores the lead-only loop."""
    if t.lead():
        fill(res, 0, d, Float32(0))
        fill(fw, prev, d, Float32(0))
        for j in range(d):
            sti(iw, state + j, 0)
    t.sync()
    var k = 0
    var n_iter = 0
    var drop = False
    var alpha = Float32(0)
    var prev_alpha = Float32(0)
    var guard = 0
    while guard < 4 * d + 4 * max_iter + 8:
        guard += 1
        for j in range(t.tid, d, t.nt):
            var acc = ld(fw, xty + j)
            for l in range(d):
                acc = fs(acc, fm(ld(fw, gg + j * d + l), ld(res, l)))
            st(fw, cov + j, acc)
        t.sync()
        var c_idx = -1
        var cbig = Float32(0)
        for j in range(d):
            if ldi(iw, state + j) == 0:
                var a = ld(fw, cov + j) if positive else fabs(ld(fw, cov + j))
                if c_idx < 0 or a > cbig:
                    c_idx = j
                    cbig = a
        alpha = fd(cbig, i2f(n))
        if alpha <= fa(alpha_min, EQ_TOL):
            if fabs(fs(alpha, alpha_min)) > EQ_TOL:
                if n_iter > 0 and t.lead():
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
            if t.lead():
                sti(iw, act + k, c_idx)
                st(fw, sgn + k, Float32(1) if positive else fsign(ld(fw, cov + c_idx)))
            t.sync()
            for a in range(t.tid, k + 1, t.nt):
                for b in range(k + 1):
                    st(fw, ll + a * (k + 1) + b, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + b)))
            t.sync()
            var okc = t_cholesky(t, fw, ll, k + 1)
            t.sync()
            if not okc or ld(fw, ll + k * (k + 1) + k) < Float32(1e-7):
                if t.lead():
                    sti(iw, state + c_idx, 2)
                t.sync()
                continue
            if t.lead():
                sti(iw, state + c_idx, 1)
            k += 1
            t.sync()
        if lasso and n_iter > 0 and prev_alpha < alpha:
            break
        for a in range(t.tid, k, t.nt):
            for b in range(k):
                st(fw, ll + a * k + b, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + b)))
        t.sync()
        _ = t_cholesky(t, fw, ll, k)
        var aa = Float32(0)
        if t.lead():
            for a in range(k):
                st(fw, ls + a, ld(fw, sgn + a))
            chol_solve(fw, ll, k, fw, ls)
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
        aa = t.bcast(aa, 2)
        # cj for every inactive j, one j a thread (the lead loop's chain)
        for j in range(t.tid, d, t.nt):
            if ldi(iw, state + j) != 0:
                continue
            var cj = Float32(0)
            for a in range(k):
                cj = fmad(ld(fw, gg + j * d + ldi(iw, act + a)), ld(fw, ls + a), cj)
            st(fw, corr + j, cj)
        t.sync()
        var gamma = fd(cbig, aa)
        var z_pos = BIG
        drop = False
        var kk = k
        if t.lead():
            for j in range(d):
                if ldi(iw, state + j) != 0:
                    continue
                var cj = ld(fw, corr + j)
                var cv = ld(fw, cov + j)
                var g1 = fd(fs(cbig, cv), fa(fs(aa, cj), TINY32))
                if g1 > 0 and g1 < gamma:
                    gamma = g1
                if not positive:
                    var g2 = fd(fa(cbig, cv), fa(fa(aa, cj), TINY32))
                    if g2 > 0 and g2 < gamma:
                        gamma = g2
            if lasso:
                for a in range(k):
                    var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                    if z > 0 and z < z_pos:
                        z_pos = z
                if z_pos < gamma:
                    for a in range(k):
                        var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                        if z == z_pos:
                            st(fw, sgn + a, -ld(fw, sgn + a))
                    gamma = z_pos
                    drop = True
            copy(fw, prev, res, 0, d)
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
                kk = w
        drop = t.bcast_int(1 if drop else 0, 1) == 1
        k = t.bcast_int(kk, 3)
        n_iter += 1
        prev_alpha = alpha
    if not t.lead():
        return
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
