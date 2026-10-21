# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TweedieRegressor's (and the Poisson and Gamma regressors') fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/glm.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/rows_host.mojo) and no
GPU-reachable module runs host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `_glm_fit_grid`); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from experiments.classical_identical_ideas.linear_controls import C16_GLM_FUSED
from x_linear.tops import FOLD_BLOCK, fold_blocks, fold_fa_blocked
from x_linear.rows_host import par_rows
from x_linear.ops import (
    FP,
    IP,
    axpy_acc,
    copy,
    fa,
    fd,
    fill,
    fm,
    fmad,
    fmax,
    i2f,
    ld,
    ldi,
    row_dots,
    seq_rows,
    st,
)
from x_linear.team import Team
from std.sys.compile import is_defined
from std.sys.info import is_gpu
from x_linear.glm import (
    GLM_STALL_ITERS,
    _glm_cell,
    _glm_deriv_row,
    _glm_slot_cell,
    _glm_slot_count,
    _glm_step,
    _objective_team,
    _unit,
    _unit_all,
    glm_den,
    glm_start,
)


def _objective_host(x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
               theta: FP, toff: Int, eta: FP, sw: Bool, den: Float32, ls: FP, hs: FP) -> Float32:
    """Map (eta and each row's loss term into `ls`), then fold rows ascending."""
    var b = ld(theta, toff + d) if fi else Float32(0)
    var loss_words = List[Float32](length=n if C16_GLM_FUSED else 1, fill=Float32(0))
    var losses = FP(unsafe_from_address=Int(loss_words.unsafe_ptr())) if C16_GLM_FUSED else ls

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm power, imm link, imm theta,
                                     imm toff, imm eta, imm sw, imm ls, imm hs, imm losses, imm b}:
        row_dots(x, lo, hi, d, theta, toff, eta)
        for i in range(lo, hi):
            var e = fa(ld(eta, i), b)
            st(eta, i, e)
            var l = Float32(0)
            comptime if C16_GLM_FUSED:
                var values = _unit_all(power, link, ld(y, i), e)
                l = values[0]
                var gi = values[1]
                var hi_ = fmax(Float32(0), values[2])
                if sw:
                    gi = fm(ld(y, n + i), gi)
                    hi_ = fm(ld(y, n + i), hi_)
                st(ls, i, gi)
                st(hs, i, hi_)
            else:
                l = _unit(power, link, ld(y, i), e, 0)
            if sw:
                l = fm(ld(y, n + i), l)
            st(losses, i, l)

    par_rows(rows_map, n)
    var acc = fold_fa_blocked(losses, 0, 1, n)
    _ = loss_words^
    var reg = Float32(0)
    for j in range(d):
        var w = ld(theta, toff + j)
        reg = fmad(w, w, reg)
    return fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))


def _objective(t: Team, x: FP, y: FP, n: Int, d: Int, fi: Bool, power: Float32, link: Int, alpha: Float32,
               theta: FP, toff: Int, eta: FP, sw: Bool, den: Float32, ls: FP, hs: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `ls` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _objective_team(t, x, y, n, d, fi, power, link, alpha, theta, toff, eta, sw, den)
    else:
        return _objective_host(x, y, n, d, fi, power, link, alpha, theta, toff, eta, sw, den, ls, hs)


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
            den = glm_den(y, n, sw)
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
            st(res, d, glm_start(y, n, den, link, sw))
    t.sync()
    var iters = 0
    var converged = False
    var f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1, s2)
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

            comptime if not C16_GLM_FUSED:
                par_rows(rows_gh, n)
            # every entry of g and of H's lower triangle is its own accumulator.
            # Units: 0..d-1 the Hessian rows, d the intercept's row, d+1 the
            # gradient; a unit owns its accumulators and folds rows ascending,
            # so units may run at once, one row block at a time (lane linear-cpu)
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

            seq_rows(blocks_fold, nb, 1)
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
            var ft = _objective(t, x, y, n, d, fi, power, link, alpha, trial, 0, eta, sw, den, s1, s2)
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
            f = _objective(t, x, y, n, d, fi, power, link, alpha, res, 0, eta, sw, den, s1, s2)
            break
        if stall >= GLM_STALL_ITERS:
            converged = True
            break
    if t.lead():
        if not fi:
            st(res, d, Float32(0))
        st(res, d + 1, i2f(iters))
        st(res, d + 2, Float32(1) if converged else Float32(0))
