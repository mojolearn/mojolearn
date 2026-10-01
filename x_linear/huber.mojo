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
from std.sys.compile import is_defined
from x_linear.lbfgs import lbfgs, lbfgs_work
from x_linear.team import Team
from x_linear.tops import chain_fmad, fold_fa
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
    for c in range(t.tid, cells, t.nt):
        var acc: Float32
        if c < d:
            acc = chain_fmad(cr, 0, 1, x, c, d, n)
        else:
            acc = fold_fa(cr, 0, 1, n)
        st(g, goff + c, acc)
    if roles and t.tid >= base and (t.tid - base) % WARP_SIZE == 0:
        var role = (t.tid - base) // WARP_SIZE
        if role == 0:
            st(sl, 8, _fold_inliers(rr, y, n, thr, sw))
        elif role == 1:
            var o = _fold_outliers(rr, y, n, thr, sw)
            st(sl, 9, o[0])
            st(sl, 10, bitcast[DType.float32](Int32(o[1])))
            st(sl, 11, o[2])
        elif role == 2 and sw:
            var w_all = Float32(0)
            for i in range(n):
                w_all = fa(w_all, ld(y, n + i))
            st(sl, 12, w_all)
    t.sync()
    var out = Float32(0)
    if t.lead():
        var sq: Float32
        var out_abs: Float32
        var n_out: Int
        var w_out: Float32
        var w_all = Float32(0)
        if roles:
            sq = ld(sl, 8)
            out_abs = ld(sl, 9)
            n_out = Int(bitcast[DType.int32](ld(sl, 10)))
            w_out = ld(sl, 11)
            if sw:
                w_all = ld(sl, 12)
        else:
            sq = _fold_inliers(rr, y, n, thr, sw)
            var o = _fold_outliers(rr, y, n, thr, sw)
            out_abs = o[0]
            n_out = o[1]
            w_out = o[2]
            if sw:
                for i in range(n):
                    w_all = fa(w_all, ld(y, n + i))
        var wn = Float32(0)
        for j in range(d):
            var w = ld(th, toff + j)
            wn = fmad(w, w, wn)
            st(g, goff + j, fmad(fm(Float32(2), alpha), w, ld(g, goff + j)))
        var squared_loss = fd(sq, sigma)
        var eps2 = fm(eps, eps)
        var cnt_out = w_out if sw else i2f(n_out)
        var cnt = w_all if sw else i2f(n)
        var outlier_loss = fs(fm(two_eps, out_abs), fm(fm(sigma, cnt_out), eps2))
        var gsigma = fs(fs(cnt, fm(cnt_out, eps2)), fd(squared_loss, sigma))
        st(g, goff + p - 1, fm(gsigma, sigma))
        out = fa(fa(fa(fm(cnt, sigma), squared_loss), outlier_loss), fm(alpha, wn))
    return t.bcast(out)


@always_inline
def _huber_coef(r: Float32, sw: Bool, wi: Float32, thr: Float32, two_eps: Float32, two_over_sigma: Float32) -> Float32:
    """The gradient coefficient of a row with residual r: the one-pass
    fold's statements (lane/neural-pass84)."""
    if sw:
        if fabs(r) > thr:
            return fm(wi, -two_eps if r > 0 else two_eps)
        return fm(-two_over_sigma, fm(wi, r))
    if fabs(r) > thr:
        return -two_eps if r > 0 else two_eps
    return fm(-two_over_sigma, r)


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
    comptime if is_defined["MOJOLEARN_HUBER_HOST_SERIAL_FOLD"]():
        for i in range(n):
            var r = ld(sc, i)
            var ar = fabs(r)
            var coefv: Float32
            if sw:
                # their weighted form: each term times w_i, n becomes sum w
                var wi = ld(y, n + i)
                w_all = fa(w_all, wi)
                if ar > thr:
                    w_out = fa(w_out, wi)
                    out_abs = fmad(wi, ar, out_abs)
                    coefv = fm(wi, -two_eps if r > 0 else two_eps)
                else:
                    sq = fmad(fm(wi, r), r, sq)
                    coefv = fm(-two_over_sigma, fm(wi, r))
            elif ar > thr:
                n_out += 1
                out_abs = fa(out_abs, ar)
                coefv = -two_eps if r > 0 else two_eps
            else:
                sq = fmad(r, r, sq)
                coefv = fm(-two_over_sigma, r)
            axpy_acc(g, goff, coefv, x, i * d, d)
            if fi:
                st(g, goff + d, fa(ld(g, goff + d), coefv))
    else:
        # lane/neural-pass84 (2026-10-01): the fold as units on the host
        # pool. Unit u < nb owns the gradient columns of band u and folds
        # every row into them ascending (`axpy_acc`, one fmad per column,
        # the coefficient recomputed from the row's residual by the same
        # statements); the last unit owns the scalar sums and the intercept.
        # Every accumulator's chain is the one-pass loop's, so the bits are
        # too. `-D MOJOLEARN_HUBER_HOST_SERIAL_FOLD=1` restores the one pass.
        var bw = 1 if d <= 32 else 16
        var nb = (d + bw - 1) // bw
        var sums = List[Float32](length=4, fill=Float32(0))
        var cnts = List[Int](length=1, fill=0)
        var sp = sums.unsafe_ptr()
        var cp = cnts.unsafe_ptr()

        def units(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm sc, imm g, imm goff, imm sw, imm fi, imm thr,
                                      imm two_eps, imm two_over_sigma, imm bw, imm nb, imm sp, imm cp}:
            for u in range(lo, hi):
                if u < nb:
                    var j0 = u * bw
                    var cw = min(d, j0 + bw) - j0
                    for i in range(n):
                        var cv = _huber_coef(ld(sc, i), sw, ld(y, n + i) if sw else Float32(1), thr, two_eps, two_over_sigma)
                        axpy_acc(g, goff + j0, cv, x, i * d + j0, cw)
                    continue
                var sq_ = Float32(0)
                var out_abs_ = Float32(0)
                var w_out_ = Float32(0)
                var w_all_ = Float32(0)
                var n_out_ = 0
                for i in range(n):
                    var r = ld(sc, i)
                    var ar = fabs(r)
                    var wi = ld(y, n + i) if sw else Float32(1)
                    if sw:
                        w_all_ = fa(w_all_, wi)
                        if ar > thr:
                            w_out_ = fa(w_out_, wi)
                            out_abs_ = fmad(wi, ar, out_abs_)
                        else:
                            sq_ = fmad(fm(wi, r), r, sq_)
                    elif ar > thr:
                        n_out_ += 1
                        out_abs_ = fa(out_abs_, ar)
                    else:
                        sq_ = fmad(r, r, sq_)
                    if fi:
                        var cv = _huber_coef(r, sw, wi, thr, two_eps, two_over_sigma)
                        st(g, goff + d, fa(ld(g, goff + d), cv))
                sp[0] = sq_
                sp[1] = out_abs_
                sp[2] = w_out_
                sp[3] = w_all_
                cp[0] = n_out_

        par_rows(units, nb + 1, 1)
        sq = sums[0]
        out_abs = sums[1]
        w_out = sums[2]
        w_all = sums[3]
        n_out = cnts[0]
    var wn = Float32(0)
    for j in range(d):
        var w = ld(th, toff + j)
        wn = fmad(w, w, wn)
        st(g, goff + j, fmad(fm(Float32(2), alpha), w, ld(g, goff + j)))
    var squared_loss = fd(sq, sigma)
    var eps2 = fm(eps, eps)
    var cnt_out = w_out if sw else i2f(n_out)
    var cnt = w_all if sw else i2f(n)
    var outlier_loss = fs(fm(two_eps, out_abs), fm(fm(sigma, cnt_out), eps2))
    var gsigma = fs(fs(cnt, fm(cnt_out, eps2)), fd(squared_loss, sigma))
    st(g, goff + p - 1, fm(gsigma, sigma))
    return fa(fa(fa(fm(cnt, sigma), squared_loss), outlier_loss), fm(alpha, wn))


def huber_objective(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `sc` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _huber_objective_team(t, x, y, n, d, ip, fp, th, toff, g, goff)
    else:
        return _huber_objective_host(x, y, n, d, ip, fp, th, toff, g, goff, sc)


def huber_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
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
    var it = lbfgs[huber_objective](t, x, y, n, d, ip, fp, res, th, p, ldi(ip, 0), ld(fp, 2), fw, 0, fw + lbfgs_work(p))
    if not t.lead():
        return
    for j in range(d):
        st(res, j, ld(res, th + j))
    st(res, d, ld(res, th + d) if fi else Float32(0))
    st(res, d + 1, fexp(ld(res, th + p - 1)))
    st(res, d + 2, i2f(it if it >= 0 else -it))
