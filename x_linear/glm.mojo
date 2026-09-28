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
    fill, copy, row_dot, cholesky, chol_solve, mean_of,
)
from x_linear.team import Team

comptime GLM_LINK_IDENTITY = 0
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


def _objective(t: Team, x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
               theta: FP, toff: Int, eta: FP, sw: Bool, den: Float32) -> Float32:
    """Rows dealt across the team (eta and each row's loss term), then the
    lead folds the terms in ascending row order: the one-thread sequence."""
    var b = ld(theta, toff + d) if fi else Float32(0)
    var lt = t.row(0)
    for i in range(t.tid, n, t.nt):
        var e = fa(row_dot(x, i, d, theta, toff), b)
        st(eta, i, e)
        var l = _unit(power, link, ld(y, i), e, 0)
        if sw:
            l = fm(ld(y, n + i), l)
        st(lt, i, l)
    t.sync()
    var f = Float32(0)
    if t.lead():
        var acc = Float32(0)
        for i in range(n):
            acc = fa(acc, ld(lt, i))
        var reg = Float32(0)
        for j in range(d):
            var w = ld(theta, toff + j)
            reg = fmad(w, w, reg)
        f = fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))
    return t.bcast(f)


def glm_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, link, sample_weight]; fp: [power, alpha, tol].
    With sample_weight, y = targets n | weights n and the objective is
    (1 / sum w) sum w_i loss_i + alpha/2 |w|^2 (theirs, glm.py).
    res: coef d, intercept 1, n_iter 1, converged 1.
    fw: eta n | grad m | H m*m | step m | trial m (m = d + 1).
    Team rows: 0 loss terms, 1 d/deta, 2 d2/deta2 (x_linear/team.mojo)."""
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
        if t.lead():
            for i in range(n):
                den = fa(den, ld(y, n + i))
        den = t.bcast(den)
    var m = d + 1 if fi else d
    var eta = fw
    var g = fw + n
    var h = g + m
    var step = h + m * m
    var trial = step + m
    var gr = t.row(1)
    var hr = t.row(2)
    if t.lead():
        fill(res, 0, d + 3, Float32(0))
        if fi:
            var ym = mean_of(y, n)
            if sw:
                var acc = Float32(0)
                for i in range(n):
                    acc = fmad(ld(y, n + i), ld(y, i), acc)
                ym = fd(acc, den)
            st(res, d, flog(ym) if link == GLM_LINK_LOG else ym)
    t.sync()
    var iters = 0
    var converged = False
    var f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den)
    for it in range(max_iter):
        # gradient and Hessian at res (eta holds the current linear predictor):
        # each row's two derivatives across the team, then one thread per
        # gradient cell and per lower-triangle Hessian cell, rows ascending
        for i in range(t.tid, n, t.nt):
            var e = ld(eta, i)
            var gi = _unit(power, link, ld(y, i), e, 1)
            var hi = fmax(Float32(0), _unit(power, link, ld(y, i), e, 2))
            if sw:
                gi = fm(ld(y, n + i), gi)
                hi = fm(ld(y, n + i), hi)
            st(gr, i, gi)
            st(hr, i, hi)
        t.sync()
        var cells = m + m * (m + 1) // 2
        for c in range(t.tid, cells, t.nt):
            if c < m:
                var acc = Float32(0)
                if c < d:
                    for i in range(n):
                        acc = fmad(ld(gr, i), ld(x, i * d + c), acc)
                else:
                    for i in range(n):
                        acc = fa(acc, ld(gr, i))
                st(g, c, acc)
                continue
            # lower-triangle cell (j, k), k <= j, row-major over j
            var q = c - m
            var j = 0
            while (j + 1) * (j + 2) // 2 <= q:
                j += 1
            var k = q - j * (j + 1) // 2
            var acc = Float32(0)
            if j < d:
                for i in range(n):
                    var hx = fm(ld(hr, i), ld(x, i * d + j))
                    acc = fmad(hx, ld(x, i * d + k), acc)
            elif k < d:
                for i in range(n):
                    acc = fmad(ld(hr, i), ld(x, i * d + k), acc)
            else:
                for i in range(n):
                    acc = fa(acc, ld(hr, i))
            st(h, j * m + k, acc)
        t.sync()
        # the small dense step (m x m) on the lead thread
        var flag = 0  # 0 continue, 1 converged, 2 stop (no descent)
        var slope = Float32(0)
        if t.lead():
            var inv_n = fd(Float32(1), den)
            var gmax = Float32(0)
            for j in range(m):
                var gj = fm(ld(g, j), inv_n)
                if j < d:
                    gj = fmad(alpha, ld(res, j), gj)
                st(g, j, gj)
                gmax = fmax(gmax, fabs(gj))
            if gmax <= tol:
                flag = 1
            else:
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
                for j in range(m):
                    slope = fmad(ld(g, j), ld(step, j), slope)
                if not (slope < 0):
                    flag = 2
        flag = t.bcast_int(flag, 1)
        if flag == 1:
            converged = True
            break
        iters = it + 1
        if flag == 2:
            break
        slope = t.bcast(slope, 2)
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            if t.lead():
                for j in range(m):
                    st(trial, j, fmad(tt, ld(step, j), ld(res, j)))
            t.sync()
            var ft = _objective(t, x, y, n, d, fi, power, link, alpha, trial, 0, eta, sw, den)
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                if t.lead():
                    copy(res, 0, trial, 0, m)
                t.sync()
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            # no decrease at float32 resolution: the fit has converged as far as it can
            f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den)
            break
    if t.lead():
        if not fi:
            st(res, d, Float32(0))
        st(res, d + 1, i2f(iters))
        st(res, d + 2, Float32(1) if converged else Float32(0))
