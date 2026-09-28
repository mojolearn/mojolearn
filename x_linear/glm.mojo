# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GENERALIZED LINEAR MODELS: PoissonRegressor, GammaRegressor,
TweedieRegressor (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_glm/glm.py`
(`_GeneralizedLinearRegressor.fit`: objective
`1/n * sum(loss_i) + 1/2 * alpha * ||w||^2`, the intercept unpenalized and
started at `link(mean(y))`) and the half-Tweedie losses of
`sklearn/_loss/loss.py` (`HalfPoissonLoss`, `HalfGammaLoss`,
`HalfTweedieLoss`, `HalfTweedieLossIdentity`). Their default solver is
L-BFGS; this is their `newton-cholesky` form instead (`_glm/_newton_solver.py`
`NewtonCholeskySolver`): the exact Hessian, a Cholesky solve (x_linear/ops.mojo)
and an Armijo backtracking line search that halves the step. The loss
constants that do not depend on the coefficients are dropped (the
minimizer is the same). float32 throughout.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fexp, flog, fabs, fmax, ld, st, ldi, i2f,
    fill, copy, row_dot, cholesky, chol_solve, mean_of, axpy_acc, par_rows, row_dots,
)

comptime GLM_LINK_IDENTITY = 0
#: rows per block of the gradient/Hessian fold (a block of X stays in cache
#: while every unit folds it)
comptime GLM_ROW_BLOCK = 8192
comptime GLM_LINK_LOG = 1


@always_inline
def _unit(power: Float32, link: Int, y: Float32, eta: Float32, what: Int) -> Float32:
    """what 0: loss, 1: d/deta, 2: d2/deta2 (clamped at zero by the caller)."""
    if link == GLM_LINK_IDENTITY:
        var r = fs(eta, y)
        if what == 0:
            return fm(Float32(0.5), fm(r, r))
        return r if what == 1 else Float32(1)
    if power == 0:
        var mu = fexp(eta)
        var r = fs(mu, y)
        if what == 0:
            return fm(Float32(0.5), fm(r, r))
        if what == 1:
            return fm(r, mu)
        return fm(mu, fs(fm(Float32(2), mu), y))
    if power == 1:
        var mu = fexp(eta)
        if what == 0:
            return fs(mu, fm(y, eta))
        return fs(mu, y) if what == 1 else mu
    if power == 2:
        var e = fexp(-eta)
        if what == 0:
            return fa(eta, fm(y, e))
        return fs(Float32(1), fm(y, e)) if what == 1 else fm(y, e)
    var a2 = fexp(fm(fs(Float32(2), power), eta))
    var a1 = fexp(fm(fs(Float32(1), power), eta))
    if what == 0:
        return fs(fd(a2, fs(Float32(2), power)), fd(fm(y, a1), fs(Float32(1), power)))
    if what == 1:
        return fs(a2, fm(y, a1))
    return fs(fm(fs(Float32(2), power), a2), fm(fm(fs(Float32(1), power), y), a1))


def _objective(x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
               theta: FP, toff: Int, eta: FP, sw: Bool, den: Float32, ls: FP) -> Float32:
    """Map (eta and each row's loss term into `ls`), then fold rows ascending."""
    var b = ld(theta, toff + d) if fi else Float32(0)

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm power, imm link, imm theta,
                                     imm toff, imm eta, imm sw, imm ls, imm b}:
        row_dots(x, lo, hi, d, theta, toff, eta)
        for i in range(lo, hi):
            var e = fa(ld(eta, i), b)
            st(eta, i, e)
            var l = _unit(power, link, ld(y, i), e, 0)
            if sw:
                l = fm(ld(y, n + i), l)
            st(ls, i, l)

    par_rows(rows_map, n)
    var acc = Float32(0)
    for i in range(n):
        acc = fa(acc, ld(ls, i))
    var reg = Float32(0)
    for j in range(d):
        var w = ld(theta, toff + j)
        reg = fmad(w, w, reg)
    return fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))


def glm_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, link, sample_weight]; fp: [power, alpha, tol].
    With sample_weight, y = targets n | weights n and the objective is
    (1 / sum w) sum w_i loss_i + alpha/2 |w|^2 (theirs, glm.py).
    res: coef d, intercept 1, n_iter 1, converged 1.
    fw: eta n | grad m | H m*m | step m | trial m | s1 n | s2 n (m = d + 1)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var link = ldi(ip, 2)
    var power = ld(fp, 0)
    var alpha = ld(fp, 1)
    var tol = ld(fp, 2)
    var sw = ldi(ip, 3) != 0
    var den = i2f(n)
    if sw:
        den = Float32(0)
        for i in range(n):
            den = fa(den, ld(y, n + i))
    var m = d + 1 if fi else d
    var eta = fw
    var g = fw + n
    var h = g + m
    var step = h + m * m
    var trial = step + m
    var s1 = trial + m
    var s2 = s1 + n
    fill(res, 0, d + 3, Float32(0))
    if fi:
        var ym = mean_of(y, n)
        if sw:
            var acc = Float32(0)
            for i in range(n):
                acc = fmad(ld(y, n + i), ld(y, i), acc)
            ym = fd(acc, den)
        st(res, d, flog(ym) if link == GLM_LINK_LOG else ym)
    var iters = 0
    var converged = False
    var f = _objective(x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1)
    for it in range(max_iter):
        # gradient and Hessian at res (eta holds the current linear predictor)
        fill(g, 0, m, Float32(0))
        fill(h, 0, m * m, Float32(0))
        var gp = g
        var hp = h

        def rows_gh(lo: Int, hi: Int) {imm y, imm n, imm power, imm link, imm eta, imm sw, imm s1, imm s2}:
            for i in range(lo, hi):
                var e = ld(eta, i)
                var gi = _unit(power, link, ld(y, i), e, 1)
                var hi_ = fmax(Float32(0), _unit(power, link, ld(y, i), e, 2))
                if sw:
                    gi = fm(ld(y, n + i), gi)
                    hi_ = fm(ld(y, n + i), hi_)
                st(s1, i, gi)
                st(s2, i, hi_)

        par_rows(rows_gh, n)
        # every entry of g and of H's lower triangle is its own accumulator.
        # Units: 0..d-1 the Hessian rows, d the intercept's row, d+1 the
        # gradient; a unit owns its accumulators and folds rows ascending,
        # so units may run at once, one row block at a time (lane linear-cpu)
        var units = d + 2
        var b0 = 0
        while b0 < n:
            var b1 = b0 + GLM_ROW_BLOCK
            if b1 > n:
                b1 = n

            def units_fold(lo: Int, hi: Int) {imm x, imm d, imm m, imm fi, imm gp, imm hp, imm s1, imm s2,
                                              imm b0, imm b1}:
                for u in range(lo, hi):
                    if u < d:
                        for i in range(b0, b1):
                            axpy_acc(hp, u * m, fm(ld(s2, i), ld(x, i * d + u)), x, i * d, u + 1)
                    elif u == d:
                        if fi:
                            for i in range(b0, b1):
                                var hi_ = ld(s2, i)
                                axpy_acc(hp, d * m, hi_, x, i * d, d)
                                st(hp, d * m + d, fa(ld(hp, d * m + d), hi_))
                    else:
                        for i in range(b0, b1):
                            var gi = ld(s1, i)
                            axpy_acc(gp, 0, gi, x, i * d, d)
                            if fi:
                                st(gp, d, fa(ld(gp, d), gi))

            par_rows(units_fold, units, 1)
            b0 = b1
        var inv_n = fd(Float32(1), den)
        var gmax = Float32(0)
        for j in range(m):
            var gj = fm(ld(g, j), inv_n)
            if j < d:
                gj = fmad(alpha, ld(res, j), gj)
            st(g, j, gj)
            gmax = fmax(gmax, fabs(gj))
        if gmax <= tol:
            converged = True
            break
        iters = it + 1
        for j in range(m):
            for k in range(j + 1):
                var v = fm(ld(h, j * m + k), inv_n)
                if j == k and j < d:
                    v = fa(v, alpha)
                st(h, j * m + k, v)
                st(h, k * m + j, v)
        for j in range(m):
            st(step, j, -ld(g, j))
        var ok = cholesky(h, 0, m)
        if ok:
            chol_solve(h, 0, m, step, 0)
        var slope = Float32(0)
        for j in range(m):
            slope = fmad(ld(g, j), ld(step, j), slope)
        if not (slope < 0):
            break
        var t = Float32(1)
        var accepted = False
        for _ in range(40):
            for j in range(m):
                st(trial, j, fmad(t, ld(step, j), ld(res, j)))
            var ft = _objective(x, y, n, d, fi, power, link, alpha, trial, 0, eta, sw, den, s1)
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), t), slope)):
                copy(res, 0, trial, 0, m)
                f = ft
                accepted = True
                break
            t = fm(t, Float32(0.5))
        if not accepted:
            # no decrease at float32 resolution: the fit has converged as far as it can
            f = _objective(x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1)
            break
    if not fi:
        st(res, d, Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
