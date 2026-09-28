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
)
from x_linear.team import Team
from x_linear.tops import upper_cell


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
        for j in range(d):
            var qjj = ld(fw, gg + j * d + j)
            if qjj == 0:
                continue
            var wj = ld(fw, w + j)
            var t = fa(fs(ld(fw, q + j), ld(fw, qw + j)), fm(wj, qjj))
            var nw = fd(fm(fsign(t), fmax(fs(fabs(t), l1), Float32(0))), fa(qjj, l2))
            if positive and t < 0:
                nw = Float32(0)  # theirs: positive and tmp < 0 -> w_j = 0
            st(fw, w + j, nw)
            if nw != wj:
                var delta = fs(nw, wj)
                for k in range(d):
                    st(fw, qw + k, fmad(delta, ld(fw, gg + j * d + k), ld(fw, qw + k)))
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


def _prep(t: Team, x: FP, y: FP, n: Int, d: Int, fid: FP, fold: Int, fi: Bool,
          fw: FP, xm: Int, gg: Int, q: Int, sc: Int):
    """Centers the rows whose fold id != `fold` (all rows when fold < 0):
    x means, Gram, X'y; fw[sc:sc+3] = (y mean, |yc|^2, rows).
    Team form: one thread per mean, Gram cell and X'y cell, the y folds on
    the lead."""
    var ym = Float32(0)
    var rows = 0
    if t.lead():
        for i in range(n):
            if fold < 0 or Int(ld(fid, i)) != fold:
                rows += 1
        if fi:
            var acc = Float32(0)
            for i in range(n):
                if fold < 0 or Int(ld(fid, i)) != fold:
                    acc = fa(acc, ld(y, i))
            ym = fd(acc, i2f(rows))
        var yn = Float32(0)
        for i in range(n):
            if fold < 0 or Int(ld(fid, i)) != fold:
                var r = fs(ld(y, i), ym)
                yn = fmad(r, r, yn)
        st(fw, sc, ym)
        st(fw, sc + 1, yn)
        st(fw, sc + 2, i2f(rows))
    rows = t.bcast_int(rows, 1)
    ym = t.bcast(ym, 2)
    for j in range(t.tid, d, t.nt):
        var acc = Float32(0)
        if fi:
            for i in range(n):
                if fold < 0 or Int(ld(fid, i)) != fold:
                    acc = fa(acc, ld(x, i * d + j))
            acc = fd(acc, i2f(rows))
        st(fw, xm + j, acc)
    t.sync()
    var cells = d * (d + 1) // 2
    for c in range(t.tid, cells + d, t.nt):
        if c < cells:
            var jk = upper_cell(c, d)
            var j = jk[0]
            var k = jk[1]
            var mj = ld(fw, xm + j)
            var mk = ld(fw, xm + k)
            var acc = Float32(0)
            for i in range(n):
                if fold < 0 or Int(ld(fid, i)) != fold:
                    acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(x, i * d + k), mk), acc)
            st(fw, gg + j * d + k, acc)
            st(fw, gg + k * d + j, acc)
        else:
            var j = c - cells
            var mj = ld(fw, xm + j)
            var acc = Float32(0)
            for i in range(n):
                if fold < 0 or Int(ld(fid, i)) != fold:
                    acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(y, i), ym), acc)
            st(fw, q + j, acc)
    t.sync()


def enetcv_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, n_alphas A, n_folds F, n_l1 L, explicit_alphas, positive].
    fp: [eps, tol, l1_ratios (L), explicit alphas (A, descending) if given].
    y: targets n | fold ids n (as float32).
    res: coef d | intercept | alpha_ | l1_ratio_ | n_iter | alphas L*A | mse L*A*F.
    fw: xm d | G d*d | q d | Qw d | w d | scalars 3.
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
    var alphas = d + 4
    var mse = alphas + l_n * a_n
    var rr = t.row(0)
    # the grids, on all rows
    _prep(t, x, y, n, d, fid, -1, fi, fw, xm, gg, q, sc)
    if t.lead():
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
    t.sync()
    # the path on each fold
    for f in range(f_n):
        _prep(t, x, y, n, d, fid, f, fi, fw, xm, gg, q, sc)
        var ym = ld(fw, sc)
        var yn = ld(fw, sc + 1)
        var rows = Int(ld(fw, sc + 2))
        var n_te = n - rows
        for l in range(l_n):
            var l1r = ld(fp, 2 + l)
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
                for i in range(t.tid, n, t.nt):
                    if Int(ld(fid, i)) == f:
                        var p = b
                        for j in range(d):
                            p = fmad(ld(x, i * d + j), ld(fw, w + j), p)
                        st(rr, i, fs(p, ld(y, i)))
                t.sync()
                if t.lead():
                    var acc = Float32(0)
                    for i in range(n):
                        if Int(ld(fid, i)) == f:
                            var r = ld(rr, i)
                            acc = fmad(r, r, acc)
                    st(res, mse + (l * a_n + k) * f_n + f, fd(acc, i2f(n_te)) if n_te > 0 else Float32(0))
                t.sync()
    if t.lead():
        # the choice: the smallest mean over folds, first on a tie
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
    t.sync()
    # the refit on all rows, from zero
    _prep(t, x, y, n, d, fid, -1, fi, fw, xm, gg, q, sc)
    if not t.lead():
        return
    var l1r = ld(res, d + 2)
    var alpha = ld(res, d + 1)
    fill(fw, w, d, Float32(0))
    var iters = enet_gram_cd(fw, gg, q, qw, w, d, ld(fw, sc + 1), fm(fm(alpha, l1r), i2f(n)),
                             fm(fm(alpha, fs(Float32(1), l1r)), i2f(n)), max_iter, tol, positive)
    copy(res, 0, fw, w, d)
    var b = ld(fw, sc)
    for j in range(d):
        b = fs(b, fm(ld(fw, xm + j), ld(fw, w + j)))
    st(res, d, b if fi else Float32(0))
    st(res, d + 1, alpha)
    st(res, d + 2, l1r)
    st(res, d + 3, i2f(iters))
