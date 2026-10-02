# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BayesianRidge and ARDRegression (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_bayes.py`:
  * `BayesianRidge.fit` / `_update_coef_`: the evidence iteration
    gamma = sum(alpha ev / (lambda + alpha ev)), lambda = (gamma + 2 l1) /
    (|w|^2 + 2 l2), alpha = (n - gamma + 2 a1) / (sse + 2 a2), stopping on
    sum|w_old - w| < tol, then one last coefficient update. Theirs takes the
    SVD of X; here the eigenpairs of the centered Gram X'X come from cyclic
    Jacobi (x_linear/ops.mojo `jacobi_eig`), ev = S^2, and
    w = V diag(1 / (ev + lambda/alpha)) V' X'y.
  * `ARDRegression.fit` / `_update_sigma`: sigma = inv(diag(lambda_keep) +
    alpha X_keep'X_keep) by Cholesky (theirs pinvh; the Woodbury branch for
    n < d is the same matrix), coef_keep = alpha sigma X_keep'y, pruning at
    lambda >= threshold_lambda, and the final sigma/coef update.
Centering (fit_intercept) is their `_preprocess_data`: column means and the
target mean, rows ascending. float32 throughout.
"""
from std.sys.compile import is_defined
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fabs, fmax, ld, st, ldi, sti, i2f, fill, copy,
    cholesky, chol_solve, jacobi_eig, centered_gram, centered_xty, mean_of,
    add_acc, axpy_acc, axpy_centered, par_rows,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.tops import t_cholesky, upper_cell, t_col_means, t_centered_gram, t_centered_xty, t_sum, t_mean, fold_sq, chain_cfmad, t_jacobi_eig


def _center(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, iw: IP) -> Float32:
    if fi:
        fill(fw, xm, d, Float32(0))
        for i in range(n):
            add_acc(fw, xm, x, i * d, d)
        for j in range(d):
            st(fw, xm + j, fd(ld(fw, xm + j), i2f(n)))
        return mean_of(y, n)
    fill(fw, xm, d, Float32(0))
    return Float32(0)


def _resid_rows(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP):
    """sc[i] = (y_i - ym) - sum_j (x_ij - xm_j) coef_j, j ascending: the map
    half of the sse (lane linear-cpu)."""

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm d, imm fw, imm xm, imm ym, imm coef, imm coff, imm sc}:
        for i in range(lo, hi):
            var p = Float32(0)
            for j in range(d):
                p = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(coef, coff + j), p)
            st(sc, i, fs(fs(ld(y, i), ym), p))

    par_rows(rows_map, n)


def _sse(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP) -> Float32:
    _resid_rows(x, y, n, d, fw, xm, ym, coef, coff, sc)
    var acc = Float32(0)
    for i in range(n):
        var r = ld(sc, i)
        acc = fmad(r, r, acc)
    return acc


def _intercept(d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int) -> Float32:
    var acc = Float32(0)
    for j in range(d):
        acc = fmad(ld(fw, xm + j), ld(coef, coff + j), acc)
    return fs(ym, acc)


def _wmean_center(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, wsum: Float32) -> Float32:
    """Weighted means (their _preprocess_data with sample_weight); w at y + n."""
    if not fi:
        fill(fw, xm, d, Float32(0))
        return Float32(0)
    fill(fw, xm, d, Float32(0))
    for i in range(n):
        axpy_acc(fw, xm, ld(y, n + i), x, i * d, d)
    for j in range(d):
        st(fw, xm + j, fd(ld(fw, xm + j), wsum))
    var acc = Float32(0)
    for i in range(n):
        acc = fmad(ld(y, n + i), ld(y, i), acc)
    return fd(acc, wsum)


def _wsse(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP) -> Float32:
    """sum_i w_i r_i^2: their sse on the sqrt(w)-rescaled data."""
    _resid_rows(x, y, n, d, fw, xm, ym, coef, coff, sc)
    var acc = Float32(0)
    for i in range(n):
        var r = ld(sc, i)
        acc = fmad(fm(ld(y, n + i), r), r, acc)
    return acc


def _t_sse(t: Team, x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int,
           sw: Bool, sc: FP) -> Float32:
    """`_sse` (or `_wsse` with sw) on the team: each row's residual across the
    team (row buffer 0), the lead's fold in ascending row order, broadcast.
    The host maps the residuals into `sc` and folds them (lane linear-cpu)."""
    comptime if not is_gpu():
        return _wsse(x, y, n, d, fw, xm, ym, coef, coff, sc) if sw else _sse(x, y, n, d, fw, xm, ym, coef, coff, sc)
    var rb = t.row(0)
    for i in range(t.tid, n, t.nt):
        var p = Float32(0)
        for j in range(d):
            p = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(coef, coff + j), p)
        st(rb, i, fs(fs(ld(y, i), ym), p))
    t.sync()
    var acc = Float32(0)
    if t.lead():
        if sw:
            for i in range(n):
                var r = ld(rb, i)
                acc = fmad(fm(ld(y, n + i), r), r, acc)
        else:
            acc = fold_sq(rb, 0, n)
    return t.bcast(acc)


def _var(y: FP, n: Int) -> Float32:
    var m = mean_of(y, n)
    var acc = Float32(0)
    for i in range(n):
        var r = fs(ld(y, i), m)
        acc = fmad(r, r, acc)
    return fd(acc, i2f(n))


def bayes_ridge_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [tol, alpha_1, alpha_2, lambda_1,
    lambda_2, alpha_init (<0: none), lambda_init (<0: none)].
    With sample_weight, y = targets n | weights n: weighted centering, the
    sqrt(w)-rescaled Gram, X'y and sse, the weighted variance and sum(w) in
    the alpha update (theirs, sw_sum).
    res: coef d, intercept, alpha_, lambda_, n_iter.
    fw: xm d | G d*d | xty d | V d*d | vty d | old d | tmp d, then at
    3d^2 + 5d the host's sse scratch n (lane linear-cpu).
    Team form: the row passes (means, Gram, X'y, sse) across the team
    (x_linear/tops.mojo), the d x d algebra and the scalar updates on the lead."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var tol = ld(fp, 0)
    var a1 = ld(fp, 1)
    var a2 = ld(fp, 2)
    var l1 = ld(fp, 3)
    var l2 = ld(fp, 4)
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var vv = xty + d
    var vty = vv + d * d
    var old = vty + d
    var tmp = old + d
    var sw = ldi(ip, 2) != 0
    var wsum = i2f(n)
    var ym = Float32(0)
    comptime if is_gpu():
        if sw:
            wsum = t_sum(t, y + n, n)
            # weighted means (their _preprocess_data with sample_weight)
            if fi:
                for j in range(t.tid, d, t.nt):
                    var acc = Float32(0)
                    for i in range(n):
                        acc = fmad(ld(y, n + i), ld(x, i * d + j), acc)
                    st(fw, xm + j, fd(acc, wsum))
                if t.lead():
                    var acc = Float32(0)
                    for i in range(n):
                        acc = fmad(ld(y, n + i), ld(y, i), acc)
                    ym = fd(acc, wsum)
                ym = t.bcast(ym, 1)
            else:
                if t.lead():
                    fill(fw, xm, d, Float32(0))
                t.sync()
            var cells = d * (d + 1) // 2
            for c in range(t.tid, cells, t.nt):
                var jk = upper_cell(c, d)
                var j = jk[0]
                var k = jk[1]
                var acc = Float32(0)
                for i in range(n):
                    acc = fmad(fm(ld(y, n + i), fs(ld(x, i * d + j), ld(fw, xm + j))), fs(ld(x, i * d + k), ld(fw, xm + k)), acc)
                st(fw, gg + j * d + k, acc)
                st(fw, gg + k * d + j, acc)
            t.sync()
        else:
            if fi:
                t_col_means(t, x, n, d, fw, xm)
                ym = t_mean(t, y, n, 1)
            else:
                if t.lead():
                    fill(fw, xm, d, Float32(0))
                t.sync()
            # ip[4] (device only, lane/neural-pass87): 1 when x_linear/device.mojo's
            # grid kernels already wrote this centered Gram into fw[gg, gg + d*d)
            if ldi(ip, 4) == 0:
                t_centered_gram(t, x, n, d, fw, xm, fw, gg)
        var yc = ym
        # X'y on centered data
        for j in range(t.tid, d, t.nt):
            var acc = Float32(0)
            var mj = ld(fw, xm + j)
            if sw:
                for i in range(n):
                    var xc = fs(ld(x, i * d + j), mj)
                    xc = fm(ld(y, n + i), xc)
                    acc = fmad(xc, fs(ld(y, i), yc), acc)
            else:
                acc = chain_cfmad(x, j, d, mj, y, 0, 1, yc, n)
            st(fw, xty + j, acc)
        t.sync()
    else:
        # the host's row passes: one pass per statistic, vector accumulators
        if sw:
            wsum = Float32(0)
            for i in range(n):
                wsum = fa(wsum, ld(y, n + i))
            ym = _wmean_center(x, y, n, d, fi, fw, xm, wsum)
            for j in range(d):
                fill(fw, gg + j * d + j, d - j, Float32(0))
            for i in range(n):
                var wi = ld(y, n + i)
                for j in range(d):
                    var a = fm(wi, fs(ld(x, i * d + j), ld(fw, xm + j)))
                    axpy_centered(fw, gg + j * d + j, a, x, i * d + j, fw, xm + j, d - j)
            for j in range(d):
                for k in range(j + 1, d):
                    st(fw, gg + k * d + j, ld(fw, gg + j * d + k))
        else:
            ym = _center(x, y, n, d, fi, fw, xm, iw)
            centered_gram(x, n, d, fw, xm, fw, gg)
        var yc = ym
        # X'y on centered data
        fill(fw, xty, d, Float32(0))
        for i in range(n):
            var b = fs(ld(y, i), yc)
            if sw:
                axpy_centered[True](fw, xty, b, x, i * d, fw, xm, d, ld(y, n + i))
            else:
                axpy_centered(fw, xty, b, x, i * d, fw, xm, d)
    var alpha = ld(fp, 5)
    t_jacobi_eig(t, fw, gg, fw, vv, d, 60)
    if t.lead():
        for j in range(d):
            var ev = ld(fw, gg + j * d + j)
            st(fw, tmp + j, fmax(Float32(0), ev))
            var acc = Float32(0)
            for k in range(d):
                acc = fmad(ld(fw, vv + k * d + j), ld(fw, xty + k), acc)
            st(fw, vty + j, acc)
        if alpha < 0:
            var yvar = _var(y, n)
            if sw:
                # np.average((y - y_mean) ** 2, weights=sample_weight)
                var m = Float32(0)
                for i in range(n):
                    m = fmad(ld(y, n + i), ld(y, i), m)
                m = fd(m, wsum)
                var acc = Float32(0)
                for i in range(n):
                    var r = fs(ld(y, i), m)
                    acc = fmad(fm(ld(y, n + i), r), r, acc)
                yvar = fd(acc, wsum)
            alpha = fd(Float32(1), fa(yvar, Float32(1.1920929e-07)))
    alpha = t.bcast(alpha, 2)
    var lam = ld(fp, 6)
    if lam < 0:
        lam = Float32(1)
    var iters = 0
    for it in range(max_iter + 1):
        # coef = V diag(1/(ev + lam/alpha)) V' X'y
        if t.lead():
            var ratio = fd(lam, alpha)
            for j in range(d):
                var acc = Float32(0)
                for k in range(d):
                    acc = fmad(ld(fw, vv + j * d + k), fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio)), acc)
                st(res, j, acc)
        t.sync()
        if it == max_iter:
            break  # the last update after the loop
        iters = it + 1
        var sse = _t_sse(t, x, y, n, d, fw, xm, ym, res, 0, sw, fw + 3 * d * d + 5 * d)
        var stop = 0
        if t.lead():
            var gamma = Float32(0)
            for k in range(d):
                var aev = fm(alpha, ld(fw, tmp + k))
                gamma = fa(gamma, fd(aev, fa(lam, aev)))
            var wn = Float32(0)
            for j in range(d):
                wn = fmad(ld(res, j), ld(res, j), wn)
            lam = fd(fa(gamma, fm(Float32(2), l1)), fa(wn, fm(Float32(2), l2)))
            alpha = fd(fa(fs(wsum, gamma), fm(Float32(2), a1)), fa(sse, fm(Float32(2), a2)))
            if it != 0:
                var delta = Float32(0)
                for j in range(d):
                    delta = fa(delta, fabs(fs(ld(fw, old + j), ld(res, j))))
                if delta < tol:
                    # their loop breaks here and the update below the loop runs
                    var ratio2 = fd(lam, alpha)
                    for j in range(d):
                        var acc = Float32(0)
                        for k in range(d):
                            acc = fmad(ld(fw, vv + j * d + k), fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio2)), acc)
                        st(res, j, acc)
                    stop = 1
            if stop == 0:
                copy(fw, old, res, 0, d)
        lam = t.bcast(lam, 1)
        alpha = t.bcast(alpha, 2)
        if t.bcast_int(stop, 3) == 1:
            break
    if t.lead():
        st(res, d, _intercept(d, fw, xm, ym, res, 0) if fi else Float32(0))
        st(res, d + 1, alpha)
        st(res, d + 2, lam)
        st(res, d + 3, i2f(iters))


def _ard_sigma(d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int) -> Int:
    """sigma (dk x dk, over the kept features in ascending order) = inv(diag(lambda) + alpha G). Returns dk."""
    var dk = 0
    for j in range(d):
        if ldi(iw, keep + j) != 0:
            sti(iw, keep + d + dk, j)
            dk += 1
    for a in range(dk):
        var ja = ldi(iw, keep + d + a)
        for b in range(dk):
            var jb = ldi(iw, keep + d + b)
            var v = fm(alpha, ld(fw, gg + ja * d + jb))
            if a == b:
                v = fa(v, ld(fw, lamo + ja))
            st(fw, aa + a * dk + b, v)
    _ = cholesky(fw, aa, dk)
    for c in range(dk):
        for r in range(dk):
            st(fw, sg + c * dk + r, Float32(1) if r == c else Float32(0))
        chol_solve(fw, aa, dk, fw, sg + c * dk)
    return dk


comptime ARD_TEAM_MIN = 32


def _t_ard_sigma(t: Team, d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int) -> Int:
    """`_ard_sigma` on the team (lane/neural-pass86, 2026-10-01): the kept
    list on the lead, the rows of diag(lambda) + alpha G split across the
    team, the Cholesky by `t_cholesky`, then the dk columns of the inverse,
    each its own `chol_solve` reading only the factor, split across the team.
    Every word is `_ard_sigma`'s. On the device the whole of it ran on the
    lead thread every iteration (ARD at 220 features: about 57 ms an
    iteration on the M4's GPU). `-D MOJOLEARN_X_LINEAR_ARD_LEAD=1` restores
    the lead-only call."""
    comptime if is_defined["MOJOLEARN_X_LINEAR_ARD_LEAD"]():
        var dk0 = 0
        if t.lead():
            dk0 = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        t.sync()
        return dk0
    if t.nt <= 1:
        return _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
    # below ARD_TEAM_MIN features the column barriers of `t_cholesky` cost
    # more than the lead's serial factor (ARD taxi, 11 features: 32.6 to
    # 36.7 ms on the L40S); the lead runs `_ard_sigma`, the same words
    if d < ARD_TEAM_MIN:
        var dk1 = 0
        if t.lead():
            dk1 = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        return t.bcast_int(dk1, 0)
    var dk = 0
    if t.lead():
        for j in range(d):
            if ldi(iw, keep + j) != 0:
                sti(iw, keep + d + dk, j)
                dk += 1
    dk = t.bcast_int(dk, 0)
    for a in range(t.tid, dk, t.nt):
        var ja = ldi(iw, keep + d + a)
        for b in range(dk):
            var jb = ldi(iw, keep + d + b)
            var v = fm(alpha, ld(fw, gg + ja * d + jb))
            if a == b:
                v = fa(v, ld(fw, lamo + ja))
            st(fw, aa + a * dk + b, v)
    t.sync()
    _ = t_cholesky(t, fw, aa, dk)
    for c in range(t.tid, dk, t.nt):
        for r in range(dk):
            st(fw, sg + c * dk + r, Float32(1) if r == c else Float32(0))
        chol_solve(fw, aa, dk, fw, sg + c * dk)
    t.sync()
    return dk


def _t_ard_coef(t: Team, d: Int, dk: Int, fw: FP, sg: Int, xty: Int, alpha: Float32, iw: IP, keep: Int, res: FP):
    """`_ard_coef` with its dk outputs split across the team (each its own
    chain over b ascending); the same words."""
    if t.nt <= 1:
        _ard_coef(d, dk, fw, sg, xty, alpha, iw, keep, res)
        return
    for j in range(t.tid, d, t.nt):
        st(res, j, Float32(0))
    t.sync()
    for a in range(t.tid, dk, t.nt):
        var acc = Float32(0)
        for b in range(dk):
            acc = fmad(ld(fw, sg + b * dk + a), ld(fw, xty + ldi(iw, keep + d + b)), acc)
        st(res, ldi(iw, keep + d + a), fm(alpha, acc))
    t.sync()


def _ard_coef(d: Int, dk: Int, fw: FP, sg: Int, xty: Int, alpha: Float32, iw: IP, keep: Int, res: FP):
    fill(res, 0, d, Float32(0))
    for a in range(dk):
        var acc = Float32(0)
        for b in range(dk):
            acc = fmad(ld(fw, sg + b * dk + a), ld(fw, xty + ldi(iw, keep + d + b)), acc)
        st(res, ldi(iw, keep + d + a), fm(alpha, acc))


def ard_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept]; fp: [tol, alpha_1, alpha_2, lambda_1,
    lambda_2, threshold_lambda].
    res: coef d, intercept, alpha_, lambda_ d, n_iter.
    fw: xm d | G d*d | xty d | A d*d | sigma d*d | lambda d | old d | sse scratch n (the host's).
    iw: keep d | kept index d.
    Team form: the row passes across the team, sigma, the coefficients and
    the pruning on the lead."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var tol = ld(fp, 0)
    var a1 = ld(fp, 1)
    var a2 = ld(fp, 2)
    var l1 = ld(fp, 3)
    var l2 = ld(fp, 4)
    var thr = ld(fp, 5)
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var aa = xty + d
    var sg = aa + d * d
    var lamo = sg + d * d
    var old = lamo + d
    var keep = 0
    var ym = Float32(0)
    comptime if is_gpu():
        if fi:
            t_col_means(t, x, n, d, fw, xm)
            ym = t_mean(t, y, n, 1)
        else:
            if t.lead():
                fill(fw, xm, d, Float32(0))
            t.sync()
        # ip[4] (device only, lane/neural-pass87): the grid's Gram is already in fw
        if ldi(ip, 4) == 0:
            t_centered_gram(t, x, n, d, fw, xm, fw, gg)
        t_centered_xty(t, x, y, n, d, fw, xm, ym, fw, xty)
    else:
        ym = _center(x, y, n, d, fi, fw, xm, iw)
        centered_gram(x, n, d, fw, xm, fw, gg)
        centered_xty(x, y, n, d, fw, xm, ym, fw, xty)
    var alpha = Float32(0)
    if t.lead():
        alpha = fd(Float32(1), fa(_var(y, n), Float32(1.1920929e-07)))
        fill(fw, lamo, d, Float32(1))
        fill(res, 0, d, Float32(0))
        for j in range(d):
            sti(iw, keep + j, 1)
    alpha = t.bcast(alpha, 2)
    var iters = 0
    var any_kept = True
    for it in range(max_iter):
        iters = it + 1
        var dk = _t_ard_sigma(t, d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _t_ard_coef(t, d, dk, fw, sg, xty, alpha, iw, keep, res)
        var sse = _t_sse(t, x, y, n, d, fw, xm, ym, res, 0, False, fw + 3 * d * d + 4 * d)
        var stop = 0
        if t.lead():
            var gsum = Float32(0)
            for a in range(dk):
                var j = ldi(iw, keep + d + a)
                var gam = fs(Float32(1), fm(ld(fw, lamo + j), ld(fw, sg + a * dk + a)))
                gsum = fa(gsum, gam)
                var cj = ld(res, j)
                st(fw, lamo + j, fd(fa(gam, fm(Float32(2), l1)), fa(fm(cj, cj), fm(Float32(2), l2))))
            alpha = fd(fa(fs(i2f(n), gsum), fm(Float32(2), a1)), fa(sse, fm(Float32(2), a2)))
            any_kept = False
            for j in range(d):
                var k = 1 if ld(fw, lamo + j) < thr else 0
                sti(iw, keep + j, k)
                if k == 0:
                    st(res, j, Float32(0))
                else:
                    any_kept = True
            if it > 0:
                var delta = Float32(0)
                for j in range(d):
                    delta = fa(delta, fabs(fs(ld(fw, old + j), ld(res, j))))
                if delta < tol:
                    stop = 1
            if stop == 0:
                copy(fw, old, res, 0, d)
                if not any_kept:
                    stop = 1
        alpha = t.bcast(alpha, 2)
        if t.bcast_int(stop, 3) == 1:
            break
    # the lead's any_kept (the other threads never computed it)
    any_kept = t.bcast_int(1 if any_kept else 0, 0) == 1
    if any_kept:
        var dk = _t_ard_sigma(t, d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _t_ard_coef(t, d, dk, fw, sg, xty, alpha, iw, keep, res)
    if not t.lead():
        return
    if not any_kept:
        fill(res, 0, d, Float32(0))
    st(res, d, _intercept(d, fw, xm, ym, res, 0) if fi else Float32(0))
    st(res, d + 1, alpha)
    copy(res, d + 2, fw, lamo, d)
    st(res, d + 2 + d, i2f(iters))
