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
from std.sys.info import is_gpu
from std.gpu import WARP_SIZE
from std.sys.compile import is_defined
from x_linear.team import Team
from x_linear.tops import (
    fold_fa, chain_fmad, chain_fmad_scaled, t_fold_fa_staged, fold_fa_blocked, chain_fmad_blocked,
    chain_fmad_scaled_blocked, t_fold_fa_blocked, fold_parts, fold_blocks, FOLD_BLOCK, X_LINEAR_SERIAL_FOLDS,
)

comptime GLM_LINK_IDENTITY = 0
comptime GLM_LINK_LOG = 1
#: rows per block of the host gradient/Hessian fold (a block of X stays in
#: cache while every unit folds it; lane linear-cpu)
comptime GLM_ROW_BLOCK = 8192
comptime GLM_STALL_ITERS = 3


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


def _objective_team(t: Team, x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
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
    # the blocked order (lane/neural-pass97); row 1 (d/deta) is free between Hessians
    var acc0 = t_fold_fa_blocked(t, lt, n, t.row(1))
    if t.lead():
        var acc = acc0
        var reg = Float32(0)
        for j in range(d):
            var w = ld(theta, toff + j)
            reg = fmad(w, w, reg)
        f = fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))
    return t.bcast(f)


def _objective_host(x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
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
    var acc = fold_fa_blocked(ls, 0, 1, n)
    var reg = Float32(0)
    for j in range(d):
        var w = ld(theta, toff + j)
        reg = fmad(w, w, reg)
    return fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))


def _objective(t: Team, x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
               theta: FP, toff: Int, eta: FP, sw: Bool, den: Float32, ls: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `ls` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _objective_team(t, x, y, n, d, fi, power, link, alpha, theta, toff, eta, sw, den)
    else:
        return _objective_host(x, y, n, d, fi, power, link, alpha, theta, toff, eta, sw, den, ls)


@always_inline
def _glm_deriv_row(y: FP, n: Int, i: Int, power: Float32, link: Int, eta: FP, sw: Bool, gr: FP, hr: FP):
    """Row i's d/deta and d2/deta2 (weighted) into gr[i], hr[i]."""
    var e = ld(eta, i)
    var gi = _unit(power, link, ld(y, i), e, 1)
    var hi = fmax(Float32(0), _unit(power, link, ld(y, i), e, 2))
    if sw:
        gi = fm(ld(y, n + i), gi)
        hi = fm(ld(y, n + i), hi)
    st(gr, i, gi)
    st(hr, i, hi)


@always_inline
def _glm_cell(c: Int, x: FP, gr: FP, hr: FP, n: Int, d: Int, m: Int, g: FP, h: FP):
    """Cell c's chain over the rows ascending: c < m the gradient (g[c]),
    else the lower-triangle Hessian cell (j, k) (h[j*m + k])."""
    if c < m:
        var acc: Float32
        if c < d:
            acc = chain_fmad_blocked(gr, 0, 1, x, c, d, n)
        else:
            acc = fold_fa_blocked(gr, 0, 1, n)
        st(g, c, acc)
        return
    # lower-triangle cell (j, k), k <= j, row-major over j
    var q = c - m
    var j = 0
    while (j + 1) * (j + 2) // 2 <= q:
        j += 1
    var k = q - j * (j + 1) // 2
    var acc: Float32
    if j < d:
        acc = chain_fmad_scaled_blocked(hr, x, j, k, d, n)
    elif k < d:
        acc = chain_fmad_blocked(hr, 0, 1, x, k, d, n)
    else:
        acc = fold_fa_blocked(hr, 0, 1, n)
    st(h, j * m + k, acc)



@always_inline
def _glm_cell_rows(c: Int, x: FP, gr: FP, hr: FP, lo: Int, cnt: Int, d: Int, m: Int, g: FP, h: FP):
    """`_glm_cell`'s chain over rows [lo, lo + cnt) only, continuing from the
    cell's stored value when lo > 0: an acc is always flushed, so the resumed
    chain is the whole chain's words (lane/neural-pass89, the Apple slices).
    The serial order only: the grid driver runs it under
    X_LINEAR_SERIAL_FOLDS, where `_glm_cell`'s blocked chains are these."""
    var xs = x + lo * d
    var grs = gr + lo
    var hrs = hr + lo
    if c < m:
        var init = ld(g, c) if lo > 0 else Float32(0)
        var acc: Float32
        if c < d:
            acc = chain_fmad(grs, 0, 1, xs, c, d, cnt, init)
        else:
            acc = fold_fa(grs, 0, 1, cnt, init)
        st(g, c, acc)
        return
    # lower-triangle cell (j, k), k <= j, row-major over j
    var q = c - m
    var j = 0
    while (j + 1) * (j + 2) // 2 <= q:
        j += 1
    var k = q - j * (j + 1) // 2
    var init = ld(h, j * m + k) if lo > 0 else Float32(0)
    var acc: Float32
    if j < d:
        acc = chain_fmad_scaled(hrs, xs, j, k, d, cnt, init)
    elif k < d:
        acc = chain_fmad(hrs, 0, 1, xs, k, d, cnt, init)
    else:
        acc = fold_fa(hrs, 0, 1, cnt, init)
    st(h, j * m + k, acc)


@always_inline
def _glm_cell_jk(c: Int, m: Int) -> Tuple[Int, Int]:
    """Cell c >= m: its lower-triangle (j, k)."""
    var q = c - m
    var j = 0
    while (j + 1) * (j + 2) // 2 <= q:
        j += 1
    return (j, q - j * (j + 1) // 2)


@always_inline
def _glm_cell_part(c: Int, x: FP, gr: FP, hr: FP, d: Int, m: Int, lo: Int, cnt: Int) -> Float32:
    """Cell c's chain over the block of rows [lo, lo + cnt), from zero: the
    partial `_glm_cell`'s blocked folds combine (lane/neural-pass97)."""
    if c < m:
        if c < d:
            return chain_fmad(gr, lo, 1, x, lo * d + c, d, cnt)
        return fold_fa(gr, lo, 1, cnt)
    var jk = _glm_cell_jk(c, m)
    var j = jk[0]
    var k = jk[1]
    if j < d:
        return chain_fmad_scaled(hr + lo, x + lo * d, j, k, d, cnt)
    if k < d:
        return chain_fmad(hr, lo, 1, x, lo * d + k, d, cnt)
    return fold_fa(hr, lo, 1, cnt)


@always_inline
def _glm_cell_store(c: Int, v: Float32, g: FP, h: FP, m: Int):
    if c < m:
        st(g, c, v)
        return
    var jk = _glm_cell_jk(c, m)
    st(h, jk[0] * m + jk[1], v)


@always_inline
def _glm_slot_count(d: Int, m: Int) -> Int:
    """Slots of the warp-uniform cell layout (see glm_fit)."""
    var n_hd = d * (d + 1) // 2
    var n_row = m * (m + 1) // 2 - n_hd
    var g0 = ((n_hd + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    var r0 = g0 + ((m + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    return r0 + n_row


@always_inline
def _glm_slot_cell(sl: Int, d: Int, m: Int) -> Int:
    """The cell a slot runs, -1 for a pad slot: [Hessian rows j < d][pad]
    [gradient][pad][Hessian row d]."""
    var n_hd = d * (d + 1) // 2
    var g0 = ((n_hd + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    var r0 = g0 + ((m + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    if sl < n_hd:
        return m + sl
    if sl < g0:
        return -1
    if sl < g0 + m:
        return sl - g0
    if sl < r0:
        return -1
    return m + n_hd + (sl - r0)


def _glm_step(g: FP, h: FP, step: FP, res: FP, m: Int, d: Int, alpha: Float32, den: Float32, tol: Float32,
              it: Int, f: Float32) -> Tuple[Int, Float32]:
    """The small dense Newton step (m x m) on one thread: (flag, slope), flag
    0 continue, 1 converged, 2 stop (no descent). glm_fit's lead block,
    shared with the device's host-driven form (lane/neural-pass89)."""
    var flag = 0
    var slope = Float32(0)
    var inv_n = fd(Float32(1), den)
    var gmax = Float32(0)
    for j in range(m):
        var gj = fm(ld(g, j), inv_n)
        if j < d:
            gj = fmad(alpha, ld(res, j), gj)
        st(g, j, gj)
        gmax = fmax(gmax, fabs(gj))
    comptime if is_defined["MOJOLEARN_GLM_TRACE"]() and not is_gpu():
        print("GLM_TRACE it", it, "f", f, "gmax", gmax, "tol", tol)
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
        comptime if is_defined["MOJOLEARN_GLM_TRACE"]() and not is_gpu():
            print("GLM_TRACE it", it, "slope", slope, "chol_ok", ok, "step0", ld(step, 0), "stepd", ld(step, m - 1))
        if not (slope < 0):
            flag = 2
    return (flag, slope)


def glm_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, link, sample_weight]; fp: [power, alpha, tol].
    With sample_weight, y = targets n | weights n and the objective is
    (1 / sum w) sum w_i loss_i + alpha/2 |w|^2 (theirs, glm.py).
    res: coef d, intercept 1, n_iter 1, converged 1.
    fw: eta n | grad m | H m*m | step m | trial m | s1 n | s2 n (m = d + 1;
    s1, s2 the host's map, lane linear-cpu).
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
    var s1 = trial + m
    var s2 = s1 + n
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
    var f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1)
    # lane/neural-pass68 (2026-10-01): iterations whose accepted step left the
    # objective unchanged at float32 resolution, in a row. In float32 the mean
    # gradient of a million rows keeps a noise floor above `tol` (taxi fares:
    # gmax stalls at 6e-4 against 1e-4 while the step is 2e-6), so the
    # gradient test never fires and the fit ran every one of its 100
    # iterations, each a line search of a dozen objective passes that moved
    # the objective by nothing (78 s on the MI325X). Theirs converges in
    # float64 by the gradient test; this stops once GLM_STALL_ITERS such
    # iterations have passed: the coefficients of the 100-iteration fit to
    # within the objective's resolution (taxi: iteration 10-11 of 100, the
    # same held-out R2 to 1e-6; a stop at 5 by the Newton decrement, theirs'
    # second criterion, read 0.035736 against 0.035965 and was dropped).
    var stall = 0
    for it in range(max_iter):
        comptime if is_gpu():
            # gradient and Hessian at res (eta holds the current linear predictor):
            # each row's two derivatives across the team, then one thread per
            # gradient cell and per lower-triangle Hessian cell, rows ascending
            for i in range(t.tid, n, t.nt):
                _glm_deriv_row(y, n, i, power, link, eta, sw, gr, hr)
            t.sync()
            var cells = m + m * (m + 1) // 2
            # lane/neural-pass88 (2026-10-01): cells laid out so a warp holds
            # ONE kind of chain. In cell order a warp mixed the gradient
            # chains, the intercept fold and the scaled Hessian chains, and a
            # warp runs its divergent branches one after another: three
            # million-row loops for the warp that held all three (the taxi
            # block's 90 cells: 107 ms a Newton step on the M4's GPU). The
            # slots are [Hessian rows j < d][pad][gradient][pad][Hessian row
            # d]; a slot maps to its cell and runs that cell's chain: the
            # same words. `-D MOJOLEARN_GLM_CELLS_MIXED=1` restores cell order.
            var slots = _glm_slot_count(d, m)
            comptime if is_defined["MOJOLEARN_GLM_CELLS_MIXED"]():
                slots = cells
            for sl in range(t.tid, slots, t.nt):
                var c = sl
                comptime if not is_defined["MOJOLEARN_GLM_CELLS_MIXED"]():
                    c = _glm_slot_cell(sl, d, m)
                    if c < 0:
                        continue
                _glm_cell(c, x, gr, hr, n, d, m, g, h)
            t.sync()
        else:
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
            comptime if X_LINEAR_SERIAL_FOLDS:
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
            else:
                # lane/neural-pass97: the blocked order. Every accumulator of
                # FOLD_BLOCK rows from zero (the units' statements over the
                # block's rows, blocks at once), then each accumulator's
                # partials folded blocks ascending.
                var nb = fold_blocks(n)
                var words = m + m * m
                var pl = List[Float32](length=max(nb * words, 1), fill=Float32(0))
                var pp = FP(unsafe_from_address=Int(pl.unsafe_ptr()))

                def blocks_fold(lo: Int, hi: Int) {imm x, imm d, imm m, imm fi, imm s1, imm s2, imm n, imm pp, imm words}:
                    for bk in range(lo, hi):
                        var gpb = pp + bk * words
                        var hpb = gpb + m
                        var b0 = bk * FOLD_BLOCK
                        var b1 = min(n, b0 + FOLD_BLOCK)
                        for u in range(d + 2):
                            if u < d:
                                for i in range(b0, b1):
                                    axpy_acc(hpb, u * m, fm(ld(s2, i), ld(x, i * d + u)), x, i * d, u + 1)
                            elif u == d:
                                if fi:
                                    for i in range(b0, b1):
                                        var hi_ = ld(s2, i)
                                        axpy_acc(hpb, d * m, hi_, x, i * d, d)
                                        st(hpb, d * m + d, fa(ld(hpb, d * m + d), hi_))
                            else:
                                for i in range(b0, b1):
                                    var gi = ld(s1, i)
                                    axpy_acc(gpb, 0, gi, x, i * d, d)
                                    if fi:
                                        st(gpb, d, fa(ld(gpb, d), gi))

                par_rows(blocks_fold, nb, 1)
                for q in range(words):
                    var acc = Float32(0)
                    for bk in range(nb):
                        acc = fa(acc, ld(pp, bk * words + q))
                    if q < m:
                        st(g, q, acc)
                    else:
                        st(h, q - m, acc)
                _ = pl^
        # the small dense step (m x m) on the lead thread
        var flag = 0  # 0 continue, 1 converged, 2 stop (no descent)
        var slope = Float32(0)
        if t.lead():
            var fs_ = _glm_step(g, h, step, res, m, d, alpha, den, tol, it, f)
            flag = fs_[0]
            slope = fs_[1]
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
            var ft = _objective(t, x, y, n, d, fi, power, link, alpha, trial, 0, eta, sw, den, s1)
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                if t.lead():
                    copy(res, 0, trial, 0, m)
                t.sync()
                if ft == f:
                    stall += 1
                else:
                    stall = 0
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        comptime if is_defined["MOJOLEARN_GLM_TRACE"]() and not is_gpu():
            if t.lead():
                print("GLM_TRACE it", it, "accepted", accepted, "tt", tt, "f", f)
        if not accepted:
            # no decrease at float32 resolution: the fit has converged as far as it can
            f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1)
            break
        if stall >= GLM_STALL_ITERS:
            converged = True
            break
    if t.lead():
        if not fi:
            st(res, d, Float32(0))
        st(res, d + 1, i2f(iters))
        st(res, d + 2, Float32(1) if converged else Float32(0))
