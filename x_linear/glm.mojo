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
from std.sys.compile import is_defined
from x_linear.team import Team
from x_linear.tops import fold_fa, chain_fmad, chain_fmad_scaled, t_fold_fa_staged, _acc_fa, _acc_fmad, _fm
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.info import is_nvidia_gpu, is_amd_gpu, is_apple_gpu

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
    var acc0 = t_fold_fa_staged(t, lt, 0, n)
    if t.lead():
        var acc = acc0
        var reg = Float32(0)
        for j in range(d):
            var w = ld(theta, toff + j)
            reg = fmad(w, w, reg)
        f = fa(fd(acc, den), fm(fm(Float32(0.5), alpha), reg))
    return t.bcast(f)


#: lane/neural-pass88 (2026-10-01): the gradient and Hessian cells from
#: threadgroup memory. Each cell is ONE thread's chain over the rows
#: ascending (its bits); a device thread fetched its rows' x words from
#: global memory at a stride of d words, so the chains waited on memory:
#: 107 ms a Newton step for the 90 cells of the taxi block (1,000,000 x 11)
#: on the M4's GPU, the GLM fit's largest cost. Here the team stages chunks
#: of rr rows (x rows, d/deta, d2/deta2: contiguous, coalesced, double
#: buffered) and every cell thread folds the chunk from threadgroup memory
#: with its chain's own step, carrying its accumulator in g / h between
#: chunks. Same steps, same order, same words.
#: `-D MOJOLEARN_GLM_CELLS_UNSTAGED=1` restores the global-memory chains.
comptime GLM_STAGE_WORDS = 2048  # a buffer; two buffers = 16 KB (Apple's limit is 32 KB)


@always_inline
def _glm_cell_kind(c: Int, d: Int, m: Int) -> Tuple[Int, Int, Int, Int]:
    """(kind, j, k, out index): kind 0 grad x_c, 1 grad intercept, 2 hess
    (j, k) j < d, 3 hess (d, k < d), 4 hess (d, d)."""
    if c < m:
        return (0 if c < d else 1, c, 0, c)
    var q = c - m
    var j = 0
    while (j + 1) * (j + 2) // 2 <= q:
        j += 1
    var k = q - j * (j + 1) // 2
    var kind = 2 if j < d else (3 if k < d else 4)
    return (kind, j, k, m + j * m + k)


def _glm_cells_staged(t: Team, x: FP, gr: FP, hr: FP, n: Int, d: Int, m: Int, rr: Int, g: FP, h: FP):
    comptime if is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu():
        var buf = stack_allocation[2 * GLM_STAGE_WORDS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var cells = m + m * (m + 1) // 2
        # g and h hold the accumulators between chunks (h is g + m in fw)
        for c in range(t.tid, cells, t.nt):
            var kc = _glm_cell_kind(c, d, m)
            if kc[0] <= 1:
                st(g, kc[3], Float32(0))
            else:
                st(h, kc[1] * m + kc[2], Float32(0))
        var chunks = (n + rr - 1) // rr
        var stride = rr * (d + 2)

        @always_inline
        def stage(ci: Int, dstoff: Int) {imm t, imm x, imm gr, imm hr, imm n, imm d, imm rr, imm buf}:
            var i0 = ci * rr
            var cnt = min(rr, n - i0)
            for u in range(t.tid, cnt * d, t.nt):
                buf[dstoff + u] = ld(x, i0 * d + u)
            for u in range(t.tid, cnt, t.nt):
                buf[dstoff + rr * d + u] = ld(gr, i0 + u)
                buf[dstoff + rr * d + rr + u] = ld(hr, i0 + u)

        if chunks > 0:
            stage(0, 0)
        t.sync()  # also orders the zeroed accumulators
        for ci in range(chunks):
            var cur = (ci & 1) * stride
            if ci + 1 < chunks:
                stage(ci + 1, ((ci + 1) & 1) * stride)
            var cnt = min(rr, n - ci * rr)
            var xs = cur
            var gs = cur + rr * d
            var hs = gs + rr
            for c in range(t.tid, cells, t.nt):
                var kc = _glm_cell_kind(c, d, m)
                var kind = kc[0]
                var j = kc[1]
                var k = kc[2]
                var acc = ld(g, j) if kind <= 1 else ld(h, j * m + k)
                var r = 0
                # 16 rows' words into registers ahead of the chain (a thread
                # waits once per block, not once per step), then the steps
                while r + 16 <= cnt:
                    var pa = SIMD[DType.float32, 16]()
                    var pb = SIMD[DType.float32, 16]()
                    comptime for u in range(16):
                        if kind == 0:
                            pa[u] = buf[gs + r + u]
                            pb[u] = buf[xs + (r + u) * d + j]
                        elif kind == 1:
                            pa[u] = buf[gs + r + u]
                        elif kind == 2:
                            pa[u] = buf[xs + (r + u) * d + j]
                            pb[u] = buf[xs + (r + u) * d + k]
                        else:
                            pa[u] = buf[hs + r + u]
                            if kind == 3:
                                pb[u] = buf[xs + (r + u) * d + k]
                    if kind == 2:
                        var ph = SIMD[DType.float32, 16]()
                        comptime for u in range(16):
                            ph[u] = buf[hs + r + u]
                        comptime for u in range(16):
                            acc = _acc_fmad(_fm(ph[u], pa[u]), pb[u], acc)
                    elif kind == 0 or kind == 3:
                        comptime for u in range(16):
                            acc = _acc_fmad(pa[u], pb[u], acc)
                    else:
                        comptime for u in range(16):
                            acc = _acc_fa(acc, pa[u])
                    r += 16
                while r < cnt:
                    if kind == 0:
                        acc = _acc_fmad(buf[gs + r], buf[xs + r * d + j], acc)
                    elif kind == 1:
                        acc = _acc_fa(acc, buf[gs + r])
                    elif kind == 2:
                        acc = _acc_fmad(_fm(buf[hs + r], buf[xs + r * d + j]), buf[xs + r * d + k], acc)
                    elif kind == 3:
                        acc = _acc_fmad(buf[hs + r], buf[xs + r * d + k], acc)
                    else:
                        acc = _acc_fa(acc, buf[hs + r])
                    r += 1
                if kind <= 1:
                    st(g, j, acc)
                else:
                    st(h, j * m + k, acc)
            # the staged words are threadgroup memory and every g / h word
            # is its own thread's: a threadgroup barrier is enough between chunks
            barrier()
        t.sync()


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
    var acc = Float32(0)
    for i in range(n):
        acc = fa(acc, ld(ls, i))
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
            var rr = GLM_STAGE_WORDS // (d + 2)
            comptime if not is_defined["MOJOLEARN_GLM_CELLS_UNSTAGED"]():
                if rr >= 1:
                    _glm_cells_staged(t, x, gr, hr, n, d, m, rr, g, h)
            if rr < 1 or is_defined["MOJOLEARN_GLM_CELLS_UNSTAGED"]():
                for c in range(t.tid, cells, t.nt):
                    if c < m:
                        var acc: Float32
                        if c < d:
                            acc = chain_fmad(gr, 0, 1, x, c, d, n)
                        else:
                            acc = fold_fa(gr, 0, 1, n)
                        st(g, c, acc)
                        continue
                    # lower-triangle cell (j, k), k <= j, row-major over j
                    var q = c - m
                    var j = 0
                    while (j + 1) * (j + 2) // 2 <= q:
                        j += 1
                    var k = q - j * (j + 1) // 2
                    var acc: Float32
                    if j < d:
                        acc = chain_fmad_scaled(hr, x, j, k, d, n)
                    elif k < d:
                        acc = chain_fmad(hr, 0, 1, x, k, d, n)
                    else:
                        acc = fold_fa(hr, 0, 1, n)
                    st(h, j * m + k, acc)
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
