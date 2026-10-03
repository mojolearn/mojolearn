# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HuberRegressor (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_huber.py`,
`_huber_loss_and_gradient` (Owen's joint objective in (w, b, sigma):
n*sigma + sum_inliers r^2/sigma + sum_outliers (2 eps |r| - sigma eps^2)
+ alpha ||w||^2, outliers |r| > eps*sigma) and `HuberRegressor.fit`
(w = 0, b = 0, sigma = 1 at the start). Theirs is L-BFGS-B with the bound
sigma >= 10 * float64 eps; here sigma = exp(s) and s is free, minimized by
x_linear/lbfgs.mojo (the same minimizer; a different path).
"""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fexp, fabs, ld, st, ldi, i2f, fill, row_dot, axpy_acc, par_rows, row_dots
from std.sys.info import is_gpu
from x_linear.lbfgs import lbfgs, lbfgs_work, Objective
from x_linear.team import Team, team_at
from x_linear.vfold import vdot
from x_linear.tops import chain_fmad, fold_fa, fold_parts, fold_blocks, FOLD_BLOCK
from std.memory import bitcast
from std.gpu import WARP_SIZE


comptime HUBER_U = 16


def _fold_inliers(rr: FP, y: FP, n: Int, thr: Float32, sw: Bool) -> Float32:
    """The lead's `sq` over rows ascending (inliers |r| <= thr only), its
    loads issued HUBER_U rows ahead."""
    var sq = Float32(0)
    var i = 0
    while i < n:
        var m = min(HUBER_U, n - i)
        var rv = InlineArray[Float32, HUBER_U](fill=Float32(0))
        var wv = InlineArray[Float32, HUBER_U](fill=Float32(0))
        comptime for u in range(HUBER_U):
            if u < m:
                rv[u] = ld(rr, i + u)
                if sw:
                    wv[u] = ld(y, n + i + u)
        comptime for u in range(HUBER_U):
            if u < m:
                var r = rv[u]
                if not (fabs(r) > thr):
                    if sw:
                        sq = fmad(fm(wv[u], r), r, sq)
                    else:
                        sq = fmad(r, r, sq)
        i += HUBER_U
    return sq


def _fold_outliers(rr: FP, y: FP, n: Int, thr: Float32, sw: Bool) -> Tuple[Float32, Int, Float32]:
    """The lead's `out_abs`, `n_out` and `w_out` over rows ascending
    (outliers |r| > thr only), loads issued HUBER_U rows ahead."""
    var out_abs = Float32(0)
    var n_out = 0
    var w_out = Float32(0)
    var i = 0
    while i < n:
        var m = min(HUBER_U, n - i)
        var rv = InlineArray[Float32, HUBER_U](fill=Float32(0))
        var wv = InlineArray[Float32, HUBER_U](fill=Float32(0))
        comptime for u in range(HUBER_U):
            if u < m:
                rv[u] = ld(rr, i + u)
                if sw:
                    wv[u] = ld(y, n + i + u)
        comptime for u in range(HUBER_U):
            if u < m:
                var ar = fabs(rv[u])
                if ar > thr:
                    if sw:
                        w_out = fa(w_out, wv[u])
                        out_abs = fmad(wv[u], ar, out_abs)
                    else:
                        n_out += 1
                        out_abs = fa(out_abs, ar)
        i += HUBER_U
    return (out_abs, n_out, w_out)


# ------------------------------------------------ the blocked order (lane/neural-pass97 order)
@always_inline
def _huber_part(o: Int, cells: Int, d: Int, x: FP, y: FP, n: Int, cr: FP, rr: FP, thr: Float32, sw: Bool,
                lo: Int, cnt: Int) -> Float32:
    """Task o of the block [lo, lo + cnt) from zero: o < cells a gradient cell
    (o == d the intercept), then the inlier sum, the outlier |r| sum, the
    outlier weight, the total weight and the outlier count (its bits)."""
    if o < cells:
        if o < d:
            return chain_fmad(cr, lo, 1, x, lo * d + o, d, cnt)
        return fold_fa(cr, lo, 1, cnt)
    var q = o - cells
    var acc = Float32(0)
    var k = 0
    for i in range(lo, lo + cnt):
        var r = ld(rr, i)
        var ar = fabs(r)
        var wi = ld(y, n + i) if sw else Float32(1)
        if q == 0:
            if not (ar > thr):
                acc = fmad(fm(wi, r), r, acc) if sw else fmad(r, r, acc)
        elif q == 1:
            if ar > thr:
                acc = fmad(wi, ar, acc) if sw else fa(acc, ar)
        elif q == 2:
            if sw and ar > thr:
                acc = fa(acc, wi)
        elif q == 3:
            if sw:
                acc = fa(acc, wi)
        else:
            if not sw and ar > thr:
                k += 1
    if q == 4:
        return bitcast[DType.float32](Int32(k))
    return acc



@always_inline
def huber_map_row(i: Int, x: FP, y: FP, n: Int, d: Int, th: FP, toff: Int, b: Float32, thr: Float32,
                  two_over_sigma: Float32, two_eps: Float32, sw: Bool, cr: FP, rr: FP):
    """Row i's residual (rr) and gradient coefficient (cr): the team map's
    statements (lane/neural-pass129: the grid objective runs them too)."""
    var r = fs(fs(ld(y, i), row_dot(x, i, d, th, toff)), b)
    var ar = fabs(r)
    var coefv: Float32
    if sw:
        var wi = ld(y, n + i)
        if ar > thr:
            coefv = fm(wi, -two_eps if r > 0 else two_eps)
        else:
            coefv = fm(-two_over_sigma, fm(wi, r))
    elif ar > thr:
        coefv = -two_eps if r > 0 else two_eps
    else:
        coefv = fm(-two_over_sigma, r)
    st(rr, i, r)
    st(cr, i, coefv)


def huber_finish_t(v: Team, g: FP, goff: Int, th: FP, toff: Int, d: Int, p: Int, nrows: Float32, eps: Float32,
                   alpha: Float32, sigma: Float32, two_eps: Float32, sw: Bool, sq: Float32, out_abs: Float32,
                   n_out: Int, w_out: Float32, w_all: Float32, parts: FP) -> Float32:
    """The objective's last statements on the folded sums, on a team (the
    device L-BFGS's finish block) or a team of one: ||w||^2 in the vfold
    order (lane cgr4-device-optim; it was one ascending chain), the penalty
    gradient a thread a weight, the sigma cell by the lead. Every thread
    returns f. nrows: the row count as float32 (i2f)."""
    var wn = vdot(v, th, toff, th, toff, d, parts)
    for j in range(v.tid, d, v.nt):
        var w = ld(th, toff + j)
        st(g, goff + j, fmad(fm(Float32(2), alpha), w, ld(g, goff + j)))
    var squared_loss = fd(sq, sigma)
    var eps2 = fm(eps, eps)
    var cnt_out = w_out if sw else i2f(n_out)
    var cnt = w_all if sw else nrows
    var outlier_loss = fs(fm(two_eps, out_abs), fm(fm(sigma, cnt_out), eps2))
    var gsigma = fs(fs(cnt, fm(cnt_out, eps2)), fd(squared_loss, sigma))
    if v.lead():
        st(g, goff + p - 1, fm(gsigma, sigma))
    v.sync()
    return fa(fa(fa(fm(cnt, sigma), squared_loss), outlier_loss), fm(alpha, wn))


def huber_finish(g: FP, goff: Int, th: FP, toff: Int, d: Int, p: Int, n: Int, eps: Float32, alpha: Float32,
                 sigma: Float32, two_eps: Float32, sw: Bool, sq: Float32, out_abs: Float32, n_out: Int,
                 w_out: Float32, w_all: Float32) -> Float32:
    """`huber_finish_t` on one thread."""
    return huber_finish_t(team_at(0, 1, g, 0, 0, 0), g, goff, th, toff, d, p, i2f(n), eps, alpha, sigma, two_eps, sw,
                          sq, out_abs, n_out, w_out, w_all, g)


def _huber_objective_team(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) -> Float32:
    """Team form: each row's residual and gradient coefficient across the
    team; the lead folds the loss sums in ascending row order; one thread
    per gradient cell folds its rows ascending. The one-thread sequence."""
    var fi = ldi(ip, 1) != 0
    var eps = ld(fp, 0)
    var alpha = ld(fp, 1)
    var p = d + 2 if fi else d + 1
    var s = ld(th, toff + p - 1)
    var sigma = fexp(s)
    var b = ld(th, toff + d) if fi else Float32(0)
    var sw = ldi(ip, 2) != 0
    var thr = fm(eps, sigma)
    var two_over_sigma = fd(Float32(2), sigma)
    var two_eps = fm(Float32(2), eps)
    var cr = t.row(0)
    var rr = t.row(1)
    for i in range(t.tid, n, t.nt):
        huber_map_row(i, x, y, n, d, th, toff, b, thr, two_over_sigma, two_eps, sw, cr, rr)
    t.sync()
    var cells = d + 1 if fi else d
    # lane/linear-apple2: the loss sums the lead folded over the rows are
    # split by accumulator, each still one thread's ascending fold with the
    # lead's expressions: thread `cells` the inlier squares, `cells + 1` the
    # outlier terms and count, `cells + 2` the weight total (sample_weight
    # only); team slots 8..12 carry them to the lead.
    var sl = t.slot_at.unsafe_origin_cast[MutAnyOrigin]()
    # the fold threads lead warps of their own (a fold sharing a warp with
    # the chains would run after them, not beside them)
    var base = ((cells + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    var roles = t.nt >= base + 3 * WARP_SIZE
    var nbk = fold_blocks(n)
    var ntask = cells + 5
    var scr = t.row(2)
    var fits = ntask * nbk <= n
    # the blocked order (lane/neural-pass97): (task, block) work across the
    # team into team row 2, then each gradient cell's partials folded
    if fits:
        for q in range(t.tid, ntask * nbk, t.nt):
            var bk = q // ntask
            var o = q - bk * ntask
            var lo = bk * FOLD_BLOCK
            st(scr, o * nbk + bk, _huber_part(o, cells, d, x, y, n, cr, rr, thr, sw, lo, min(FOLD_BLOCK, n - lo)))
        t.sync()
        for o in range(t.tid, cells, t.nt):
            st(g, goff + o, fold_parts(scr, o * nbk, nbk))
    else:
        for o in range(t.tid, cells, t.nt):
            var accb = Float32(0)
            for bk in range(nbk):
                var lo = bk * FOLD_BLOCK
                accb = fa(accb, _huber_part(o, cells, d, x, y, n, cr, rr, thr, sw, lo, min(FOLD_BLOCK, n - lo)))
            st(g, goff + o, accb)
    t.sync()
    var out = Float32(0)
    if t.lead():
        var sq: Float32
        var out_abs: Float32
        var n_out: Int
        var w_out: Float32
        var w_all = Float32(0)
        var p0 = Float32(0)
        var p1 = Float32(0)
        var p2 = Float32(0)
        var p3 = Float32(0)
        var kn = 0
        for bk in range(nbk):
            var lo = bk * FOLD_BLOCK
            var cnt = min(FOLD_BLOCK, n - lo)
            if fits:
                p0 = fa(p0, ld(scr, cells * nbk + bk))
                p1 = fa(p1, ld(scr, (cells + 1) * nbk + bk))
                p2 = fa(p2, ld(scr, (cells + 2) * nbk + bk))
                p3 = fa(p3, ld(scr, (cells + 3) * nbk + bk))
                kn += Int(bitcast[DType.int32](ld(scr, (cells + 4) * nbk + bk)))
            else:
                p0 = fa(p0, _huber_part(cells, cells, d, x, y, n, cr, rr, thr, sw, lo, cnt))
                p1 = fa(p1, _huber_part(cells + 1, cells, d, x, y, n, cr, rr, thr, sw, lo, cnt))
                p2 = fa(p2, _huber_part(cells + 2, cells, d, x, y, n, cr, rr, thr, sw, lo, cnt))
                p3 = fa(p3, _huber_part(cells + 3, cells, d, x, y, n, cr, rr, thr, sw, lo, cnt))
                kn += Int(bitcast[DType.int32](_huber_part(cells + 4, cells, d, x, y, n, cr, rr, thr, sw, lo, cnt)))
        sq = p0
        out_abs = p1
        w_out = p2
        if sw:
            w_all = p3
        n_out = kn
        out = huber_finish(g, goff, th, toff, d, p, n, eps, alpha, sigma, two_eps, sw, sq, out_abs, n_out, w_out, w_all)
    return t.bcast(out)


def _huber_objective_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """Map (each row's residual into `sc`), then fold rows ascending."""
    var fi = ldi(ip, 1) != 0
    var eps = ld(fp, 0)
    var alpha = ld(fp, 1)
    var p = d + 2 if fi else d + 1
    var s = ld(th, toff + p - 1)
    var sigma = fexp(s)
    var b = ld(th, toff + d) if fi else Float32(0)
    fill(g, goff, p, Float32(0))
    var sq = Float32(0)
    var out_abs = Float32(0)
    var n_out = 0
    var sw = ldi(ip, 2) != 0
    var w_out = Float32(0)
    var w_all = Float32(0)
    var thr = fm(eps, sigma)
    var two_over_sigma = fd(Float32(2), sigma)
    var two_eps = fm(Float32(2), eps)

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm d, imm th, imm toff, imm b, imm sc}:
        row_dots(x, lo, hi, d, th, toff, sc)
        for i in range(lo, hi):
            st(sc, i, fs(fs(ld(y, i), ld(sc, i)), b))

    par_rows(rows_map, n)
    # the blocked order (lane/neural-pass97): every sum from zero over
    # FOLD_BLOCK rows, folded into its total at the block's end
    var cells = d + 1 if fi else d
    var pl = List[Float32](length=max(cells, 1), fill=Float32(0))
    var pg = FP(unsafe_from_address=Int(pl.unsafe_ptr()))
    var psq = Float32(0)
    var pout = Float32(0)
    var pwo = Float32(0)
    var pwa = Float32(0)
    for i in range(n):
        if i > 0 and i % FOLD_BLOCK == 0:
            for o in range(cells):
                st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
                st(pg, o, Float32(0))
            sq = fa(sq, psq)
            out_abs = fa(out_abs, pout)
            w_out = fa(w_out, pwo)
            w_all = fa(w_all, pwa)
            psq = Float32(0)
            pout = Float32(0)
            pwo = Float32(0)
            pwa = Float32(0)
        var r = ld(sc, i)
        var ar = fabs(r)
        var coefv: Float32
        if sw:
            var wi = ld(y, n + i)
            pwa = fa(pwa, wi)
            if ar > thr:
                pwo = fa(pwo, wi)
                pout = fmad(wi, ar, pout)
                coefv = fm(wi, -two_eps if r > 0 else two_eps)
            else:
                psq = fmad(fm(wi, r), r, psq)
                coefv = fm(-two_over_sigma, fm(wi, r))
        elif ar > thr:
            n_out += 1
            pout = fa(pout, ar)
            coefv = -two_eps if r > 0 else two_eps
        else:
            psq = fmad(r, r, psq)
            coefv = fm(-two_over_sigma, r)
        axpy_acc(pg, 0, coefv, x, i * d, d)
        if fi:
            st(pg, d, fa(ld(pg, d), coefv))
    if n > 0:
        for o in range(cells):
            st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
        sq = fa(sq, psq)
        out_abs = fa(out_abs, pout)
        w_out = fa(w_out, pwo)
        w_all = fa(w_all, pwa)
    _ = pl^
    return huber_finish(g, goff, th, toff, d, p, n, eps, alpha, sigma, two_eps, sw, sq, out_abs, n_out, w_out, w_all)


def huber_objective(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `sc` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _huber_objective_team(t, x, y, n, d, ip, fp, th, toff, g, goff)
    else:
        return _huber_objective_host(x, y, n, d, ip, fp, th, toff, g, goff, sc)


def huber_fit[obj: Objective = huber_objective](t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [epsilon, alpha, tol].
    With sample_weight, y = targets n | weights n (their weighted objective).
    res: coef d, intercept 1, scale 1, n_iter 1 | theta scratch (P).
    fw: lbfgs_work(P) | objective scratch n (the host's map)."""
    var fi = ldi(ip, 1) != 0
    var p = d + 2 if fi else d + 1
    var th = d + 4
    if t.lead():
        fill(res, th, p, Float32(0))
    t.sync()
    var it = lbfgs[obj](t, x, y, n, d, ip, fp, res, th, p, ldi(ip, 0), ld(fp, 2), fw, 0, fw + lbfgs_work(p))
    if not t.lead():
        return
    for j in range(d):
        st(res, j, ld(res, th + j))
    st(res, d, ld(res, th + d) if fi else Float32(0))
    st(res, d + 1, fexp(ld(res, th + p - 1)))
    st(res, d + 2, i2f(it if it >= 0 else -it))
