# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LassoCV and ElasticNetCV: a warm-started coordinate-descent path per fold
(lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_coordinate_descent.py`
(`LinearModelCV.fit`, `_alpha_grid`, `_path_residuals`) and
`sklearn/linear_model/_cd_fast.pyx` (`enet_coordinate_descent_gram`,
`gap_enet_gram`, `dual_gap_formulation_A`):
  * the grid: alpha_max = max|Xc'yc| / (n l1_ratio) on the FULL data,
    geometric down to alpha_max * eps (their np.geomspace), one grid per
    l1_ratio, computed once and shared by every fold;
  * each fold (their KFold, the fold ids come from Python) centers its own
    training rows, runs the path from w = 0 with warm starts, and scores the
    held-out rows with intercept = y_mean - x_mean'w;
  * the Gram CD: l1 = alpha l1_ratio n, l2 = alpha (1 - l1_ratio) n,
    cyclic coordinates, w_j = sign(t) max(|t| - l1, 0) / (Q_jj + l2) with
    t = q_j - (Qw)_j + w_j Q_jj, and the duality-gap check (tol * |y|^2)
    whenever max|dw| / max|w| <= tol or at the last sweep;
  * the mean over folds picks (l1_ratio, alpha), the first minimum winning,
    and the refit on all rows starts from w = 0.
Named differences: no gap-safe screening (theirs `do_screening`; it only
skips coordinates), and the refit uses the Gram form (theirs sets
precompute=False for the refit; the same minimizer). float32 throughout.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fabs, fmax, fexp, flog, fsign, ld, st, ldi, i2f, fill, copy,
    add_acc, axpy_centered,
)
from experiments.classical_identical_ideas.linear_controls import C18_RESIDUAL_NEXT, ENETCV_FOLD_BLOCKS, ENETCV_SCORE_BLOCKS
from x_linear.enetcv_blocks import fb_host_stats, fb_host_prep
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.tops import upper_cell, fold_fa_ix, fold_sq_ix, chain_cfmad_ix


def alpha_grid_value(amax: Float32, eps: Float32, k: Int, a_n: Int) -> Float32:
    """DEVIATION 5009: alpha_k = alpha_max * exp((k / (A - 1)) * log(eps)),
    the portable exp and log (their np.geomspace is 10 ** linspace of
    log10s; eps ** (k / (A - 1)) is another legal spelling with other bits)."""
    var frac = fd(i2f(k), i2f(a_n - 1)) if a_n > 1 else Float32(0)
    return fm(amax, fexp(fm(frac, flog(eps))))


def enet_gram_cd(fw: FP, gg: Int, q: Int, qw: Int, w: Int, d: Int, ynorm2: Float32,
                 l1: Float32, l2: Float32, max_iter: Int, tol: Float32, positive: Bool = False) -> Int:
    """Their enet_coordinate_descent_gram without screening; w warm, Qw is
    recomputed from w at entry. Returns the sweeps run."""
    for j in range(d):
        var acc = Float32(0)
        for k in range(d):
            acc = fmad(ld(fw, gg + j * d + k), ld(fw, w + k), acc)
        st(fw, qw + j, acc)
    var tol_s = fm(tol, ynorm2)
    if _gap(fw, q, qw, w, d, ynorm2, l1, l2, positive) <= tol_s:
        return 0
    for it in range(max_iter):
        var w_max = Float32(0)
        var dw_max = Float32(0)
        var next_qw = ld(fw, qw) if d > 0 else Float32(0)
        for j in range(d):
            var qjj = ld(fw, gg + j * d + j)
            if qjj == 0:
                if j + 1 < d:
                    next_qw = ld(fw, qw + j + 1)
                continue
            var wj = ld(fw, w + j)
            var residual = ld(fw, qw + j)
            comptime if C18_RESIDUAL_NEXT:
                residual = next_qw
            var t = fa(fs(ld(fw, q + j), residual), fm(wj, qjj))
            var nw = fd(fm(fsign(t), fmax(fs(fabs(t), l1), Float32(0))), fa(qjj, l2))
            if positive and t < 0:
                nw = Float32(0)  # theirs: positive and tmp < 0 -> w_j = 0
            st(fw, w + j, nw)
            if nw != wj:
                var delta = fs(nw, wj)
                for k in range(d):
                    var updated = fmad(delta, ld(fw, gg + j * d + k), ld(fw, qw + k))
                    st(fw, qw + k, updated)
                    # C18: forward the next ordered consumer directly from
                    # the write; the coordinate order and FMA are unchanged.
                    comptime if C18_RESIDUAL_NEXT:
                        if k == j + 1:
                            next_qw = updated
            elif j + 1 < d:
                next_qw = ld(fw, qw + j + 1)
            var dw = fabs(fs(nw, wj))
            if dw > dw_max:
                dw_max = dw
            if fabs(nw) > w_max:
                w_max = fabs(nw)
        if w_max == 0 or fd(dw_max, w_max) <= tol or it == max_iter - 1:
            if _gap(fw, q, qw, w, d, ynorm2, l1, l2, positive) <= tol_s:
                return it + 1
    return max_iter


def _gap(fw: FP, q: Int, qw: Int, w: Int, d: Int, ynorm2: Float32, l1: Float32, l2: Float32,
         positive: Bool) -> Float32:
    """gap_enet_gram, formulation A (l1 > 0), or B when l1 == 0 < l2, or
    the gradient norm when both are zero."""
    var wl2 = Float32(0)
    var qdw = Float32(0)
    var wqw = Float32(0)
    var wl1 = Float32(0)
    for j in range(d):
        var wj = ld(fw, w + j)
        wl2 = fmad(wj, wj, wl2)
        qdw = fmad(wj, ld(fw, q + j), qdw)
        wqw = fmad(wj, ld(fw, qw + j), wqw)
        wl1 = fa(wl1, fabs(wj))
    var r2 = fs(fa(ynorm2, wqw), fm(Float32(2), qdw))
    var ry = fs(ynorm2, qdw)
    var dual_norm = Float32(0)
    if l1 == 0:
        for j in range(d):
            var a = fs(ld(fw, q + j), ld(fw, qw + j))
            dual_norm = fmad(a, a, dual_norm)
        if l2 == 0:
            return dual_norm
        var g = fs(fa(r2, fm(fm(Float32(0.5), l2), wl2)), ry)
        return fa(g, fm(fd(Float32(1), fm(Float32(2), l2)), dual_norm))
    for j in range(d):
        var a = fs(fs(ld(fw, q + j), ld(fw, qw + j)), fm(l2, ld(fw, w + j)))
        dual_norm = fmax(dual_norm, a if positive else fabs(a))
    var base = fa(r2, fm(l2, wl2))
    var primal = fa(fm(Float32(0.5), base), fm(l1, wl1))
    var scale = fd(l1, dual_norm) if dual_norm > l1 else Float32(1)
    var dual = fa(fm(fm(Float32(-0.5), fm(scale, scale)), base), fm(scale, ry))
    return fs(primal, dual)


def _prep_team(t: Team, x: FP, y: FP, n: Int, d: Int, fid: FP, fold: Int, fi: Bool,
          fw: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """Centers the rows whose fold id != `fold` (all rows when fold < 0):
    x means, Gram, X'y; fw[sc:sc+3] = (y mean, |yc|^2, rows).
    Team form: the lead lists the training rows (ascending, team row 1) and
    the held-out rows (team row 2); one thread per mean, Gram cell and X'y
    cell folds its rows ascending (x_linear/tops.mojo chains); the y folds
    on the lead."""
    var tr = t.row(1).bitcast[Int32]()
    var te = t.row(2).bitcast[Int32]()
    var ym = Float32(0)
    var rows = 0
    if t.lead():
        var nte = 0
        for i in range(n):
            if fold < 0 or Int(ld(fid, i)) != fold:
                tr.unsafe_store(rows, Int32(i))
                rows += 1
            else:
                te.unsafe_store(nte, Int32(i))
                nte += 1
        if fi:
            ym = fd(fold_fa_ix(y, tr, rows), i2f(rows))
        var yn = Float32(0)
        for k in range(rows):
            var r = fs(ld(y, Int(tr.unsafe_load(k))), ym)
            yn = fmad(r, r, yn)
        st(fw, sc, ym)
        st(fw, sc + 1, yn)
        st(fw, sc + 2, i2f(rows))
    rows = t.bcast_int(rows, 1)
    ym = t.bcast(ym, 2)
    for j in range(t.tid, d, t.nt):
        var acc = Float32(0)
        if fi:
            acc = fd(fold_fa_ix(x, tr, rows, j, d), i2f(rows))
        st(fw, xm + j, acc)
    t.sync()
    var cells = d * (d + 1) // 2
    for c in range(t.tid, cells + d, t.nt):
        if c < cells:
            var jk = upper_cell(c, d)
            var j = jk[0]
            var k = jk[1]
            var acc = chain_cfmad_ix(x, j, d, ld(fw, xm + j), x, k, d, ld(fw, xm + k), tr, rows)
            st(fw, gg + j * d + k, acc)
            st(fw, gg + k * d + j, acc)
        else:
            var j = c - cells
            st(fw, q + j, chain_cfmad_ix(x, j, d, ld(fw, xm + j), y, 0, 1, ym, tr, rows))
    t.sync()


def _prep_host(x: FP, y: FP, n: Int, d: Int, fid: FP, fold: Int, fi: Bool,
          fw: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """Centers the rows whose fold id != `fold` (all rows when fold < 0):
    x means, Gram, X'y; fw[sc:sc+3] = (y mean, |yc|^2, rows). Each mean,
    Gram entry and X'y entry is its own accumulator folded rows ascending,
    walked as one row pass per stage (lane linear-cpu)."""
    var rows = 0
    var ym = Float32(0)
    var yacc = Float32(0)
    fill(fw, xm, d, Float32(0))
    for i in range(n):
        if fold < 0 or Int(ld(fid, i)) != fold:
            rows += 1
            if fi:
                add_acc(fw, xm, x, i * d, d)
                yacc = fa(yacc, ld(y, i))
    if fi:
        for j in range(d):
            st(fw, xm + j, fd(ld(fw, xm + j), i2f(rows)))
        ym = fd(yacc, i2f(rows))
    var yn = Float32(0)
    for i in range(n):
        if fold < 0 or Int(ld(fid, i)) != fold:
            var r = fs(ld(y, i), ym)
            yn = fmad(r, r, yn)
    for j in range(d):
        fill(fw, gg + j * d + j, d - j, Float32(0))
    fill(fw, q, d, Float32(0))
    for i in range(n):
        if fold < 0 or Int(ld(fid, i)) != fold:
            for j in range(d):
                var a = fs(ld(x, i * d + j), ld(fw, xm + j))
                axpy_centered(fw, gg + j * d + j, a, x, i * d + j, fw, xm + j, d - j)
            axpy_centered(fw, q, fs(ld(y, i), ym), x, i * d, fw, xm, d)
    for j in range(d):
        for k in range(j + 1, d):
            st(fw, gg + k * d + j, ld(fw, gg + j * d + k))
    st(fw, sc, ym)
    st(fw, sc + 1, yn)
    st(fw, sc + 2, i2f(rows))


def _prep(t: Team, x: FP, y: FP, n: Int, d: Int, fid: FP, fold: Int, fi: Bool,
          fw: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """The device runs the team schedule (which also lists the fold's rows in
    team rows 1 and 2), the host one row pass per stage (lane linear-cpu)."""
    comptime if is_gpu():
        _prep_team(t, x, y, n, d, fid, fold, fi, fw, xm, gg, q, sc)
    else:
        _prep_host(x, y, n, d, fid, fold, fi, fw, xm, gg, q, sc)


def _prep_cv(t: Team, x: FP, y: FP, n: Int, d: Int, fid: FP, fold: Int, fi: Bool,
             fw: FP, xm: Int, gg: Int, q: Int, sc: Int, wk: FP, f_n: Int):
    """`_prep`, or on the host column under ENETCV_FOLD_BLOCKS the device
    grid's fold-block statistics (x_linear/enetcv_blocks.mojo, same cells;
    wk: the host binding's extra words, `fb_host_words`)."""
    comptime if ENETCV_FOLD_BLOCKS and not is_gpu():
        fb_host_prep(wk, d, f_n, fold, fi, fw, xm, gg, q, sc)
    else:
        _prep(t, x, y, n, d, fid, fold, fi, fw, xm, gg, q, sc)


def enetcv_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, n_alphas A, n_folds F, n_l1 L, explicit_alphas, positive].
    fp: [eps, tol, l1_ratios (L), explicit alphas (A, descending) if given].
    y: targets n | fold ids n (as float32).
    res: coef d | intercept | alpha_ | l1_ratio_ | n_iter | alphas L*A | mse L*A*F.
    fw: xm d | G d*d | q d | Qw d | w d | scalars 3 | path A*(d+1) | path errors A
    (the path words: the host schedule's, lane linear-cpu).
    Team form: the row passes (`_prep`, the held-out residuals) across the
    team; the coordinate descent (d x d) and every fold of the scores on the lead."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var a_n = ldi(ip, 2)
    var f_n = ldi(ip, 3)
    var l_n = ldi(ip, 4)
    var explicit = ldi(ip, 5) != 0
    var positive = ldi(ip, 6) != 0
    var eps = ld(fp, 0)
    var tol = ld(fp, 1)
    var fid = y + n
    var xm = 0
    var gg = d
    var q = gg + d * d
    var qw = q + d
    var w = qw + d
    var sc = w + d
    var pb = sc + 3
    var pacc = pb + a_n * (d + 1)
    var alphas = d + 4
    var mse = alphas + l_n * a_n
    var rr = t.row(0)
    # the host column's fold-block statistics past the fit's own words (the
    # host binding sizes them under ENETCV_FOLD_BLOCKS)
    var wk = fw + d * d + 4 * d + 3 + a_n * (d + 2)
    comptime if ENETCV_FOLD_BLOCKS and not is_gpu():
        fb_host_stats(x, y, n, d, f_n, wk)
    # the grids, on all rows
    _prep_cv(t, x, y, n, d, fid, -1, fi, fw, xm, gg, q, sc, wk, f_n)
    if t.lead():
        ecv_alphas(res, alphas, fp, l_n, a_n, explicit, eps, fw, q, d, n)
    t.sync()
    # the path on each fold
    for f in range(f_n):
        _prep_cv(t, x, y, n, d, fid, f, fi, fw, xm, gg, q, sc, wk, f_n)
        var ym = ld(fw, sc)
        var yn = ld(fw, sc + 1)
        var rows = Int(ld(fw, sc + 2))
        var n_te = n - rows
        for l in range(l_n):
            var l1r = ld(fp, 2 + l)
            comptime if is_gpu():
                if t.lead():
                    fill(fw, w, d, Float32(0))
                for k in range(a_n):
                    var b = ym
                    if t.lead():
                        var alpha = ld(res, alphas + l * a_n + k)
                        var l1 = fm(fm(alpha, l1r), i2f(rows))
                        var l2 = fm(fm(alpha, fs(Float32(1), l1r)), i2f(rows))
                        _ = enet_gram_cd(fw, gg, q, qw, w, d, yn, l1, l2, max_iter, tol, positive)
                        for j in range(d):
                            b = fs(b, fm(ld(fw, xm + j), ld(fw, w + j)))
                    b = t.bcast(b, 4)
                    var te = t.row(2).bitcast[Int32]()
                    for qq in range(t.tid, n_te, t.nt):
                        var i = Int(te.unsafe_load(qq))
                        var p = b
                        for j in range(d):
                            p = fmad(ld(x, i * d + j), ld(fw, w + j), p)
                        st(rr, i, fs(p, ld(y, i)))
                    t.sync()
                    if t.lead():
                        var acc = fold_sq_ix(rr, te, n_te)
                        st(res, mse + (l * a_n + k) * f_n + f, fd(acc, i2f(n_te)) if n_te > 0 else Float32(0))
                    t.sync()
            else:
                # the host: the whole path, then one pass over the held-out rows
                fill(fw, w, d, Float32(0))
                for k in range(a_n):
                    var alpha = ld(res, alphas + l * a_n + k)
                    var l1 = fm(fm(alpha, l1r), i2f(rows))
                    var l2 = fm(fm(alpha, fs(Float32(1), l1r)), i2f(rows))
                    _ = enet_gram_cd(fw, gg, q, qw, w, d, yn, l1, l2, max_iter, tol, positive)
                    var b = ym
                    for j in range(d):
                        b = fs(b, fm(ld(fw, xm + j), ld(fw, w + j)))
                    copy(fw, pb + k * (d + 1), fw, w, d)
                    st(fw, pb + k * (d + 1) + d, b)
                # the held-out squared errors of the whole path, one row pass:
                # alpha k's sum is its own accumulator, rows ascending
                comptime if ENETCV_SCORE_BLOCKS > 0:
                    # lane/classical-cv-folds: the device's (path, alpha, row
                    # block) partials folded blocks ascending (`esb_host_score`)
                    esb_host_score(x, y, fid, n, d, f, a_n, fw, pb, pacc)
                else:
                    fill(fw, pacc, a_n, Float32(0))
                    for i in range(n):
                        if Int(ld(fid, i)) == f:
                            for k in range(a_n):
                                var o = pb + k * (d + 1)
                                var p = ld(fw, o + d)
                                for j in range(d):
                                    p = fmad(ld(x, i * d + j), ld(fw, o + j), p)
                                var r = fs(p, ld(y, i))
                                st(fw, pacc + k, fmad(r, r, ld(fw, pacc + k)))
                for k in range(a_n):
                    st(res, mse + (l * a_n + k) * f_n + f, fd(ld(fw, pacc + k), i2f(n_te)) if n_te > 0 else Float32(0))
    if t.lead():
        ecv_choose(res, fp, d, l_n, a_n, f_n)
    t.sync()
    # the refit on all rows, from zero
    _prep_cv(t, x, y, n, d, fid, -1, fi, fw, xm, gg, q, sc, wk, f_n)
    if not t.lead():
        return
    var l1r = ld(res, d + 2)
    var alpha = ld(res, d + 1)
    fill(fw, w, d, Float32(0))
    var iters = enet_gram_cd(fw, gg, q, qw, w, d, ld(fw, sc + 1), fm(fm(alpha, l1r), i2f(n)),
                             fm(fm(alpha, fs(Float32(1), l1r)), i2f(n)), max_iter, tol, positive)
    ecv_finish(res, fw, xm, w, sc, d, fi, alpha, l1r, iters)


def ecv_alphas(res: FP, alphas: Int, fp: FP, l_n: Int, a_n: Int, explicit: Bool, eps: Float32,
               fw: FP, q: Int, d: Int, n: Int):
    """The grid of each l1_ratio from the full data's X'y at fw[q:q+d]
    (their `_alpha_grid`), or the explicit alphas, into res[alphas:]."""
    for l in range(l_n):
        var l1r = ld(fp, 2 + l)
        if explicit:
            for k in range(a_n):
                st(res, alphas + l * a_n + k, ld(fp, 2 + l_n + k))
            continue
        var qmax = Float32(0)
        for j in range(d):
            qmax = fmax(qmax, fabs(ld(fw, q + j)))
        var amax = fd(qmax, fm(i2f(n), l1r))
        if amax <= Float32(1e-6):
            for k in range(a_n):
                st(res, alphas + l * a_n + k, Float32(1e-6))
            continue
        for k in range(a_n):
            st(res, alphas + l * a_n + k, alpha_grid_value(amax, eps, k, a_n))


def ecv_alpha_cell(res: FP, alphas: Int, fp: FP, l_n: Int, a_n: Int, explicit: Bool, eps: Float32,
                   fw: FP, q: Int, d: Int, n: Int, l: Int, k: Int):
    """Cell (l, k) of `ecv_alphas`, with its statements: the device runs one
    cell per thread, the host loops them (same values, same chains)."""
    if explicit:
        st(res, alphas + l * a_n + k, ld(fp, 2 + l_n + k))
        return
    var l1r = ld(fp, 2 + l)
    var qmax = Float32(0)
    for j in range(d):
        qmax = fmax(qmax, fabs(ld(fw, q + j)))
    var amax = fd(qmax, fm(i2f(n), l1r))
    if amax <= Float32(1e-6):
        st(res, alphas + l * a_n + k, Float32(1e-6))
        return
    st(res, alphas + l * a_n + k, alpha_grid_value(amax, eps, k, a_n))


def ecv_choose(res: FP, fp: FP, d: Int, l_n: Int, a_n: Int, f_n: Int):
    """The choice: the smallest mean over folds, first on a tie; alpha_
    and l1_ratio_ into res[d + 1], res[d + 2]."""
    var alphas = d + 4
    var mse = alphas + l_n * a_n
    var best_l = 0
    var best_k = 0
    var best = Float32(0)
    for l in range(l_n):
        for k in range(a_n):
            var acc = Float32(0)
            for f in range(f_n):
                acc = fa(acc, ld(res, mse + (l * a_n + k) * f_n + f))
            var m = fd(acc, i2f(f_n))
            if (l == 0 and k == 0) or m < best:  # DEVIATION 5005: the first minimum
                best = m
                best_l = l
                best_k = k
    st(res, d + 1, ld(res, alphas + best_l * a_n + best_k))
    st(res, d + 2, ld(fp, 2 + best_l))


def ecv_finish(res: FP, fw: FP, xm: Int, w: Int, sc: Int, d: Int, fi: Bool, alpha: Float32,
               l1r: Float32, iters: Int):
    """The refit's coef, intercept, alpha_, l1_ratio_, n_iter into res."""
    copy(res, 0, fw, w, d)
    var b = ld(fw, sc)
    for j in range(d):
        b = fs(b, fm(ld(fw, xm + j), ld(fw, w + j)))
    st(res, d, b if fi else Float32(0))
    st(res, d + 1, alpha)
    st(res, d + 2, l1r)
    st(res, d + 3, i2f(iters))


# ------------------------------------------------ the grid form (lane/neural-pass109)
# x_linear/cd_grid.mojo runs LassoCV / ElasticNetCV as launches over the
# whole device: every fold's means, Gram and X'y at once (one thread per
# value), one block per (fold, l1_ratio) path, one thread per held-out sum.
# The helpers below are the host schedule's statements with the fold's rows
# chosen by their fold id (rows ascending), so every word is the host's.


@always_inline
def _ecv_row(fid: FP, i: Int, fold: Int, held: Bool) -> Bool:
    """Row i is in the fold's held-out rows (held) or its training rows
    (all rows when fold < 0)."""
    if held:
        return Int(ld(fid, i)) == fold
    return fold < 0 or Int(ld(fid, i)) != fold


def ecv_rows(fid: FP, n: Int, fold: Int) -> Int:
    var rows = 0
    for i in range(n):
        if _ecv_row(fid, i, fold, False):
            rows += 1
    return rows


# Rows whose loads a grid chain issues before folding them (scheduling
# only: the folds still take the rows one at a time, ascending).
comptime ECV_U = 32
comptime ECV_UH = 8


def ecv_fold_fa(v: FP, off: Int, step: Int, fid: FP, n: Int, fold: Int) -> Float32:
    """acc = fa(acc, v[off + i*step]) over the fold's training rows ascending
    (`add_acc` per column, the y sum of `_prep_host`)."""
    var acc = Float32(0)
    var i0 = 0
    while i0 + ECV_U <= n:
        var bv = SIMD[DType.float32, ECV_U]()
        var bf = SIMD[DType.float32, ECV_U]()
        comptime for u in range(ECV_U):
            bv[u] = ld(v, off + (i0 + u) * step)
            bf[u] = ld(fid, i0 + u)
        comptime for u in range(ECV_U):
            if fold < 0 or Int(bf[u]) != fold:
                acc = fa(acc, bv[u])
        i0 += ECV_U
    for i in range(i0, n):
        if _ecv_row(fid, i, fold, False):
            acc = fa(acc, ld(v, off + i * step))
    return acc


def ecv_cfmad(a: FP, aoff: Int, astep: Int, ma: Float32, b: FP, boff: Int, bstep: Int, mb: Float32,
              fid: FP, n: Int, fold: Int) -> Float32:
    """acc = fmad(a_i - ma, b_i - mb, acc) over the fold's training rows
    ascending: a Gram, X'y or |yc|^2 entry of `_prep_host`."""
    var acc = Float32(0)
    var i0 = 0
    while i0 + ECV_U <= n:
        var ba = SIMD[DType.float32, ECV_U]()
        var bb = SIMD[DType.float32, ECV_U]()
        var bf = SIMD[DType.float32, ECV_U]()
        comptime for u in range(ECV_U):
            ba[u] = ld(a, aoff + (i0 + u) * astep)
            bb[u] = ld(b, boff + (i0 + u) * bstep)
            bf[u] = ld(fid, i0 + u)
        comptime for u in range(ECV_U):
            if fold < 0 or Int(bf[u]) != fold:
                acc = fmad(fs(ba[u], ma), fs(bb[u], mb), acc)
        i0 += ECV_U
    for i in range(i0, n):
        if _ecv_row(fid, i, fold, False):
            acc = fmad(fs(ld(a, aoff + i * astep), ma), fs(ld(b, boff + i * bstep), mb), acc)
    return acc


def ecv_held_sse(x: FP, y: FP, fid: FP, n: Int, d: Int, fold: Int, wb: FP, o: Int) -> Float32:
    """The held-out squared errors of the path point at wb[o:o+d+1] (coef,
    intercept), rows ascending: the host schedule's per-alpha accumulator.
    ECV_UH rows' predictions run side by side (each its own chain over j),
    then fold in row order; a block with no held-out row is skipped."""
    var acc = Float32(0)
    var b0 = ld(wb, o + d)
    var i0 = 0
    while i0 + ECV_UH <= n:
        var held = SIMD[DType.bool, ECV_UH](fill=False)
        var any = False
        comptime for u in range(ECV_UH):
            held[u] = Int(ld(fid, i0 + u)) == fold
            any = any or held[u]
        if any:
            var pv = SIMD[DType.float32, ECV_UH](b0)
            for j in range(d):
                var wj = ld(wb, o + j)
                comptime for u in range(ECV_UH):
                    pv[u] = fmad(ld(x, (i0 + u) * d + j), wj, pv[u])
            comptime for u in range(ECV_UH):
                if held[u]:
                    var r = fs(pv[u], ld(y, i0 + u))
                    acc = fmad(r, r, acc)
        i0 += ECV_UH
    for i in range(i0, n):
        if _ecv_row(fid, i, fold, True):
            var p = b0
            for j in range(d):
                p = fmad(ld(x, i * d + j), ld(wb, o + j), p)
            var r = fs(p, ld(y, i))
            acc = fmad(r, r, acc)
    return acc


# ------------------------------------------------ ENETCV_SCORE_BLOCKS (lane/classical-cv-folds, 2026-10-07)
# `-D MOJOLEARN_CLASSICAL_ENETCV_SCORE_BLOCKS=1024|4096` (rows per block, a
# fixed count, not a data shape; default off, NOT MEASURED): the held-out
# squared errors of a path point as block partials. The fold's held-out rows
# lie in its span [lo, lo + span) (KFold: exactly them; a row of another
# fold inside the span is skipped by its id). Block b of the span folds the
# path point's squared errors over its held-out rows ascending from zero
# (`ecv_held_sse`'s statements per row); the block partials are folded
# blocks ascending (fa from zero, the kf_sq / kf_score shape). Changes
# bits (the fold order; the host column and both GPU vendors together).
# Cost reasoning (no shape rule): the incumbent runs one block per (fold,
# l1_ratio) path over every held-out row for every alpha (n/k * A * d
# multiply-adds on one block); here span / ESB_ROWS blocks per path share
# that work, and the fold adds A * span / ESB_ROWS additions per path.
comptime ESB_ROWS = ENETCV_SCORE_BLOCKS if ENETCV_SCORE_BLOCKS > 0 else 1
"""Rows of a fold's span one score block owns (1 is a placeholder while the control is off)."""


@always_inline
def esb_blocks(span: Int) -> Int:
    return (span + ESB_ROWS - 1) // ESB_ROWS


def esb_part(x: FP, y: FP, fid: FP, d: Int, f: Int, lo: Int, span: Int, b: Int, wb: FP, o: Int) -> Float32:
    """Block b of fold f's span: the squared errors of the path point
    wb[o:o+d+1] (coef, intercept) over the block's held-out rows ascending,
    from zero: the row's prediction a chain over j from the intercept, the
    residual squared into the accumulator (fmad)."""
    var b0 = ld(wb, o + d)
    var acc = Float32(0)
    var r_lo = lo + b * ESB_ROWS
    var r_hi = min(r_lo + ESB_ROWS, lo + span)
    for i in range(r_lo, r_hi):
        if Int(ld(fid, i)) == f:
            var p = b0
            for j in range(d):
                p = fmad(ld(x, i * d + j), ld(wb, o + j), p)
            var r = fs(p, ld(y, i))
            acc = fmad(r, r, acc)
    return acc


def esb_host_score(x: FP, y: FP, fid: FP, n: Int, d: Int, f: Int, a_n: Int, fw: FP, pb: Int, pacc: Int):
    """The host column: fold f's span from a walk over the fold ids (the
    device's `ecv_span_*` words: (lo, hi - lo), (0, 0) without a row), then
    alpha k's block partials folded blocks ascending into fw[pacc + k]
    (`esb_score_kernel` and `esb_fold_kernel`'s words)."""
    var lo = n
    var hi = 0
    for i in range(n):
        if Int(ld(fid, i)) == f:
            if i < lo:
                lo = i
            hi = i + 1
    var span = 0
    if hi > 0:
        span = hi - lo
    else:
        lo = 0
    fill(fw, pacc, a_n, Float32(0))
    for b in range(esb_blocks(span)):
        for k in range(a_n):
            st(fw, pacc + k, fa(ld(fw, pacc + k), esb_part(x, y, fid, d, f, lo, span, b, fw, pb + k * (d + 1))))


def t_enet_gram_cd(t: Team, fw: FP, gg: Int, q: Int, qw: Int, w: Int, d: Int, ynorm2: Float32,
                   l1: Float32, l2: Float32, max_iter: Int, tol: Float32, positive: Bool = False) -> Int:
    """`enet_gram_cd` on the team. Coordinate j is updated by thread
    j % nt, which owns w[j] and Qw[k] for k = tid, tid + nt, ...; it
    publishes (new w_j, old w_j) through a broadcast slot (double buffered
    by the barrier count) and every thread applies the Qw update to the
    entries it owns. The statements per word are the serial loop's; the
    convergence test reads the same broadcast words in every thread, so the
    sweep count is uniform. The gap runs on the lead after a barrier. A team
    of one runs `enet_gram_cd`."""
    if t.nt <= 1:
        return enet_gram_cd(fw, gg, q, qw, w, d, ynorm2, l1, l2, max_iter, tol, positive)
    var slot = t.slot_at.unsafe_origin_cast[MutAnyOrigin]()
    for j in range(t.tid, d, t.nt):
        var acc = Float32(0)
        for k in range(d):
            acc = fmad(ld(fw, gg + j * d + k), ld(fw, w + k), acc)
        st(fw, qw + j, acc)
    t.sync()
    var tol_s = fm(tol, ynorm2)
    var g0 = Float32(0)
    if t.lead():
        g0 = _gap(fw, q, qw, w, d, ynorm2, l1, l2, positive)
    if t.bcast(g0, 6) <= tol_s:
        return 0
    var nb = 0
    for it in range(max_iter):
        var w_max = Float32(0)
        var dw_max = Float32(0)
        var next_qw = ld(fw, qw + t.tid) if t.tid < d else Float32(0)
        for j in range(d):
            var qjj = ld(fw, gg + j * d + j)
            if qjj == 0:
                if j + 1 < d and (j + 1) % t.nt == t.tid:
                    next_qw = ld(fw, qw + j + 1)
                continue
            var s = 8 + (nb & 1) * 2
            nb += 1
            if j % t.nt == t.tid:
                var wj0 = ld(fw, w + j)
                var residual = ld(fw, qw + j)
                comptime if C18_RESIDUAL_NEXT:
                    residual = next_qw
                var tt = fa(fs(ld(fw, q + j), residual), fm(wj0, qjj))
                var nw0 = fd(fm(fsign(tt), fmax(fs(fabs(tt), l1), Float32(0))), fa(qjj, l2))
                if positive and tt < 0:
                    nw0 = Float32(0)
                st(fw, w + j, nw0)
                slot.unsafe_store(s, nw0)
                slot.unsafe_store(s + 1, wj0)
            t.sync()
            var nw = slot.unsafe_load(s)
            var wj = slot.unsafe_load(s + 1)
            if nw != wj:
                var delta = fs(nw, wj)
                for k in range(t.tid, d, t.nt):
                    var updated = fmad(delta, ld(fw, gg + j * d + k), ld(fw, qw + k))
                    st(fw, qw + k, updated)
                    comptime if C18_RESIDUAL_NEXT:
                        if k == j + 1:
                            next_qw = updated
            elif j + 1 < d and (j + 1) % t.nt == t.tid:
                next_qw = ld(fw, qw + j + 1)
            var dw = fabs(fs(nw, wj))
            if dw > dw_max:
                dw_max = dw
            if fabs(nw) > w_max:
                w_max = fabs(nw)
        if w_max == 0 or fd(dw_max, w_max) <= tol or it == max_iter - 1:
            t.sync()
            var g = Float32(0)
            if t.lead():
                g = _gap(fw, q, qw, w, d, ynorm2, l1, l2, positive)
            if t.bcast(g, 6) <= tol_s:
                return it + 1
    return max_iter
