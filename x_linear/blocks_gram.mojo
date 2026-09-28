# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST ON APPLE, PAST ONE BLOCK: the fits that work from the centered Gram
(lane/linear-apple3, 2026-09-28; see x_linear/blocks.mojo).

LassoCV / ElasticNetCV, RidgeClassifier / RidgeCV, BayesianRidge,
ARDRegression, Lars / LassoLars. Each one's passes over the rows (the
means, the centered Gram and X'y, a residual sum, the held-out errors of a
path, the leave-one-out residuals) are ONE launch of n / XB_ROWS blocks
whose block partials the host sums in float64; the d x d algebra (the
Gram coordinate descent, Cholesky, Jacobi, the LARS path) is the fit's own
code, run on the host. The per-row expressions are the one-block fits'.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, fmin, fsign, ld, st, ldi, sti, i2f,
    fill, copy, cholesky, chol_solve, jacobi_eig,
)
from x_linear.team import team_barrier
from x_linear.tops import upper_cell
from x_linear.blocks import XB, Part, XB_TPB, XB_ROWS, _host_fp, _zeros, xb_sumsq
from x_linear.cd import enet_gram_cd, alpha_grid_value
from x_linear.ridge import _ridge_solve_best
from x_linear.bayes import _ard_sigma, _ard_coef, _intercept
from x_linear.lars import BIG, TINY32, EQ_TOL

comptime XG_BAYES = 4
comptime XG_ARD = 5
comptime XG_LARS = 6
comptime XG_RIDGE = 8
comptime XG_ENETCV = 9


def gram_handles(algo: Int) -> Bool:
    return algo == XG_BAYES or algo == XG_ARD or algo == XG_LARS or algo == XG_RIDGE or algo == XG_ENETCV


# ---------------------------------------------------------------- kernels

def xb_moments_kernel(
    x: FP, y: FP, th: FP, part: FP, n_in: Int32, d_in: Int32, tn_in: Int32, woff_in: Int32,
    foff_in: Int32, fold_in: Int32, what_in: Int32,
):
    """The rows: all of them, or with foff >= 0 those whose fold id
    y[foff + i] is not `fold`. With woff >= 0 row i weighs y[woff + i].
    y holds the targets n x T row-major. th: xm d | ym T | yc T.
    what 0: d + T outputs, sum w x_j and sum w y_t.
    what 1: the centered Gram's upper triangle (x_linear/tops.mojo
    `upper_cell` order), then the d * T cells of X'y (t * d + j), then the
    T sums w (y_t - yc_t)^2."""
    var n = Int(n_in)
    var d = Int(d_in)
    var t_n = Int(tn_in)
    var woff = Int(woff_in)
    var foff = Int(foff_in)
    var fold = Int(fold_in)
    var what = Int(what_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var gc = d * (d + 1) // 2
    var cells = d + t_n
    if what == 1:
        cells = gc + d * t_n + t_n
    var c = tid
    while c < cells:
        var acc = Float32(0)
        if what == 0:
            for i in range(r0, r1):
                if foff >= 0 and Int(ld(y, foff + i)) == fold:
                    continue
                var v = ld(x, i * d + c) if c < d else ld(y, i * t_n + (c - d))
                if woff >= 0:
                    acc = fmad(ld(y, woff + i), v, acc)
                else:
                    acc = fa(acc, v)
        elif c < gc:
            var jk = upper_cell(c, d)
            var j = jk[0]
            var k = jk[1]
            var mj = ld(th, j)
            var mk = ld(th, k)
            for i in range(r0, r1):
                if foff >= 0 and Int(ld(y, foff + i)) == fold:
                    continue
                var a = fs(ld(x, i * d + j), mj)
                if woff >= 0:
                    a = fm(ld(y, woff + i), a)
                acc = fmad(a, fs(ld(x, i * d + k), mk), acc)
        elif c < gc + d * t_n:
            var o = c - gc
            var tt = o // d
            var j = o - tt * d
            var mj = ld(th, j)
            var ymt = ld(th, d + tt)
            for i in range(r0, r1):
                if foff >= 0 and Int(ld(y, foff + i)) == fold:
                    continue
                var a = fs(ld(x, i * d + j), mj)
                if woff >= 0:
                    a = fm(ld(y, woff + i), a)
                acc = fmad(a, fs(ld(y, i * t_n + tt), ymt), acc)
        else:
            var tt = c - gc - d * t_n
            var yct = ld(th, d + t_n + tt)
            for i in range(r0, r1):
                if foff >= 0 and Int(ld(y, foff + i)) == fold:
                    continue
                var r = fs(ld(y, i * t_n + tt), yct)
                if woff >= 0:
                    acc = fmad(fm(ld(y, woff + i), r), r, acc)
                else:
                    acc = fmad(r, r, acc)
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xb_sse_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, woff_in: Int32,
):
    """th: xm d | ym | coef d. One output: sum w r^2 with
    r = (y - ym) - sum_j (x_j - xm_j) coef_j (x_linear/bayes.mojo `_t_sse`)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var woff = Int(woff_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var ym = ld(th, d)
    var i = r0 + tid
    while i < r1:
        var p = Float32(0)
        for j in range(d):
            p = fmad(fs(ld(x, i * d + j), ld(th, j)), ld(th, d + 1 + j), p)
        st(rw, i, fs(fs(ld(y, i), ym), p))
        i += XB_TPB
    team_barrier()
    if tid == 0:
        var acc = Float32(0)
        if woff >= 0:
            for q in range(r0, r1):
                var r = ld(rw, q)
                acc = fmad(fm(ld(y, woff + q), r), r, acc)
        else:
            acc = xb_sumsq(rw, r0, r1 - r0)
        st(part, blk, acc)


def xb_path_kernel(
    x: FP, y: FP, th: FP, part: FP, n_in: Int32, d_in: Int32, an_in: Int32, foff_in: Int32, fold_in: Int32,
):
    """th: A models of (w d, b). Output k: the squared error of model k
    summed over the rows whose fold id y[foff + i] IS `fold`
    (x_linear/cd.mojo: p = b, then fmad over j ascending)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var a_n = Int(an_in)
    var foff = Int(foff_in)
    var fold = Int(fold_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var k = tid
    while k < a_n:
        var o = k * (d + 1)
        var bk = ld(th, o + d)
        var acc = Float32(0)
        for i in range(r0, r1):
            if Int(ld(y, foff + i)) != fold:
                continue
            var p = bk
            for j in range(d):
                p = fmad(ld(x, i * d + j), ld(th, o + j), p)
            var r = fs(p, ld(y, i))
            acc = fmad(r, r, acc)
        st(part, blk * a_n + k, acc)
        k += XB_TPB


def xb_loo_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, flags_in: Int32, woff_in: Int32,
):
    """RidgeCV's leave-one-out rows (x_linear/ridge.mojo `_loo_rows`).
    th: xm d | ym | rhs d | sum w | the Cholesky factor d * d. Each thread
    solves its rows into its own d words of rw and sums their terms; one
    output per thread."""
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var woff = Int(woff_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var z = rw + (blk * XB_TPB + tid) * d
    var ym = ld(th, d)
    var wsum = ld(th, 2 * d + 1)
    var mm = 2 * d + 2
    var acc = Float32(0)
    var i = r0 + tid
    while i < r1:
        var e = fs(ld(y, i), ym)
        for j in range(d):
            var xc = fs(ld(x, i * d + j), ld(th, j))
            st(z, j, xc)
            e = fs(e, fm(xc, ld(th, d + 1 + j)))
        chol_solve(th, mm, d, z, 0)
        if sw:
            var wi = ld(y, woff + i)
            var q = Float32(0)
            for j in range(d):
                q = fmad(fs(ld(x, i * d + j), ld(th, j)), ld(z, j), q)
            var h = fm(wi, q)
            if fi:
                h = fa(fd(wi, wsum), h)
            var loo = fd(e, fs(Float32(1), h))
            acc = fmad(fm(wi, loo), loo, acc)
        else:
            var h = fd(Float32(1), i2f(n)) if fi else Float32(0)
            for j in range(d):
                h = fmad(fs(ld(x, i * d + j), ld(th, j)), ld(z, j), h)
            var loo = fd(e, fs(Float32(1), h))
            acc = fmad(loo, loo, acc)
        i += XB_TPB
    st(part, blk * XB_TPB + tid, acc)


# ------------------------------------------------------------ host passes

def _moments(
    mut b: XB, mut ps: Part, mut pm: Part, ctx: DeviceContext, fw: FP, xm: Int, ym: Int, gg: Int,
    xty: Int, yy: Int, t_n: Int, woff: Int, foff: Int, fold: Int, fi: Bool, den: Float32,
    true_mean: Bool,
) raises:
    """fw[xm:] the column means and fw[ym:] the target means (zeros without
    fit_intercept), fw[gg:] the centered Gram (both triangles), fw[xty:]
    X'y (t * d + j), fw[yy:] the T sums of (y - yc)^2: yc the target mean
    with `true_mean`, else fw[ym]."""
    var d = b.d
    ctx.enqueue_function[xb_moments_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), ps.dev.unsafe_ptr(),
        Int32(b.n), Int32(d), Int32(t_n), Int32(woff), Int32(foff), Int32(fold), Int32(0),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    ps.fetch(ctx)
    for j in range(d):
        var v = fd(ps.total(j), den) if fi else Float32(0)
        st(fw, xm + j, v)
        b.param(j, v)
    for tt in range(t_n):
        var m = fd(ps.total(d + tt), den)
        var v = m if fi else Float32(0)
        st(fw, ym + tt, v)
        b.param(d + tt, v)
        b.param(d + t_n + tt, m if true_mean else v)
    b.push(ctx)
    ctx.enqueue_function[xb_moments_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), pm.dev.unsafe_ptr(),
        Int32(b.n), Int32(d), Int32(t_n), Int32(woff), Int32(foff), Int32(fold), Int32(1),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pm.fetch(ctx)
    var gc = d * (d + 1) // 2
    for c in range(gc):
        var jk = upper_cell(c, d)
        var acc = pm.total(c)
        st(fw, gg + jk[0] * d + jk[1], acc)
        st(fw, gg + jk[1] * d + jk[0], acc)
    for o in range(d * t_n):
        st(fw, xty + o, pm.total(gc + o))
    for tt in range(t_n):
        st(fw, yy + tt, pm.total(gc + d * t_n + tt))


def _sse(
    mut b: XB, mut pe: Part, ctx: DeviceContext, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int,
    woff: Int,
) raises -> Float32:
    var d = b.d
    for j in range(d):
        b.param(j, ld(fw, xm + j))
        b.param(d + 1 + j, ld(coef, coff + j))
    b.param(d, ym)
    b.push(ctx)
    ctx.enqueue_function[xb_sse_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pe.dev.unsafe_ptr(),
        Int32(b.n), Int32(d), Int32(woff),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pe.fetch(ctx)
    return pe.total(0)


def _weight_sum(y: FP, woff: Int, n: Int) -> Float32:
    var acc = Float32(0)
    for i in range(n):
        acc = fa(acc, ld(y, woff + i))
    return acc


# ----------------------------------------------------- LassoCV / ElasticNetCV

def enetcv_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/cd.mojo `enetcv_fit` (the host schedule: a fold's whole
    path, then its held-out errors in one pass).
    res: coef d | intercept | alpha_ | l1_ratio_ | n_iter | alphas L*A | mse L*A*F.
    Host work: xm d | ym | yy | G d*d | q d | Qw d | w d."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var a_n = Int(ip[2])
    var f_n = Int(ip[3])
    var l_n = Int(ip[4])
    var explicit = Int(ip[5]) != 0
    var positive = Int(ip[6]) != 0
    var eps = fp[0]
    var tol = fp[1]
    var xm = 0
    var ym = d
    var yy = d + 1
    var gg = d + 2
    var q = gg + d * d
    var qw = q + d
    var w = qw + d
    var alphas = d + 4
    var mse = alphas + l_n * a_n
    var gc = d * (d + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, max(d + 2, a_n * (d + 1)), 1)
    var ps = Part(ctx, b.nb, d + 1)
    var pm = Part(ctx, b.nb, gc + d + 1)
    var pp = Part(ctx, b.nb, a_n)
    var hw = _zeros(w + d)
    var fw = _host_fp(hw)
    # the grids, on all rows
    _moments(b, ps, pm, ctx, fw, xm, ym, gg, q, yy, 1, -1, -1, -1, fi, i2f(n), False)
    for l in range(l_n):
        var l1r = fp[2 + l]
        if explicit:
            for k in range(a_n):
                st(res, alphas + l * a_n + k, fp[2 + l_n + k])
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
    # the path on each fold
    for f in range(f_n):
        var rows = 0
        for i in range(n):
            if Int(ld(y, n + i)) != f:
                rows += 1
        var n_te = n - rows
        _moments(b, ps, pm, ctx, fw, xm, ym, gg, q, yy, 1, -1, n, f, fi, i2f(rows), False)
        var ymv = ld(fw, ym)
        var yn = ld(fw, yy)
        for l in range(l_n):
            var l1r = fp[2 + l]
            fill(fw, w, d, Float32(0))
            for k in range(a_n):
                var alpha = ld(res, alphas + l * a_n + k)
                var l1 = fm(fm(alpha, l1r), i2f(rows))
                var l2 = fm(fm(alpha, fs(Float32(1), l1r)), i2f(rows))
                _ = enet_gram_cd(fw, gg, q, qw, w, d, yn, l1, l2, max_iter, tol, positive)
                var bk = ymv
                for j in range(d):
                    bk = fs(bk, fm(ld(fw, xm + j), ld(fw, w + j)))
                    b.param(k * (d + 1) + j, ld(fw, w + j))
                b.param(k * (d + 1) + d, bk)
            b.push(ctx)
            ctx.enqueue_function[xb_path_kernel](
                b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), pp.dev.unsafe_ptr(),
                Int32(n), Int32(d), Int32(a_n), Int32(n), Int32(f),
                grid_dim=b.nb, block_dim=XB_TPB,
            )
            pp.fetch(ctx)
            for k in range(a_n):
                st(res, mse + (l * a_n + k) * f_n + f, fd(pp.total(k), i2f(n_te)) if n_te > 0 else Float32(0))
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
    var alpha = ld(res, alphas + best_l * a_n + best_k)
    var l1r = fp[2 + best_l]
    # the refit on all rows, from zero
    _moments(b, ps, pm, ctx, fw, xm, ym, gg, q, yy, 1, -1, -1, -1, fi, i2f(n), False)
    fill(fw, w, d, Float32(0))
    var iters = enet_gram_cd(fw, gg, q, qw, w, d, ld(fw, yy), fm(fm(alpha, l1r), i2f(n)),
                             fm(fm(alpha, fs(Float32(1), l1r)), i2f(n)), max_iter, tol, positive)
    copy(res, 0, fw, w, d)
    var bz = ld(fw, ym)
    for j in range(d):
        bz = fs(bz, fm(ld(fw, xm + j), ld(fw, w + j)))
    st(res, d, bz if fi else Float32(0))
    st(res, d + 1, alpha)
    st(res, d + 2, l1r)
    st(res, d + 3, i2f(iters))
    ctx.synchronize()
    _ = hw^
    _ = ps^
    _ = pm^
    _ = pp^
    _ = b^


# ---------------------------------------------- RidgeClassifier / RidgeCV

def ridge_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/ridge.mojo `ridge_fit`.
    res: coef T*d | intercept T | alpha | best_score | A mean squared LOO errors.
    Host work (the layout `_ridge_solve_best` reads): xm d | G d*d | M d*d |
    rhs d | ym T | xty d*T, then yy T."""
    var t_n = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var a_n = Int(ip[2])
    var sw = Int(ip[3]) != 0
    var xm = 0
    var gg = d
    var mm = gg + d * d
    var rhs = mm + d * d
    var ym = rhs + d
    var xty = ym + t_n
    var yy = xty + d * t_n
    var wo = n * t_n
    var woff = wo if sw else -1
    var gc = d * (d + 1) // 2
    # the leave-one-out threads' own d words each: blocks * XB_TPB * d
    var own = (((n + XB_ROWS - 1) // XB_ROWS) * XB_TPB * d) // n + 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, max(d + 2 * t_n, 2 * d + 2 + d * d), own)
    b.fi = fi
    b.sw = sw
    var ps = Part(ctx, b.nb, d + t_n)
    var pm = Part(ctx, b.nb, gc + d * t_n + t_n)
    var pl = Part(ctx, b.nb, XB_TPB)
    var hw = _zeros(yy + t_n)
    var fw = _host_fp(hw)
    var hfp = fp.copy()
    var fpp = _host_fp(hfp)
    var wsum = Float32(0)
    if sw:
        wsum = _weight_sum(y, wo, n)
    _moments(b, ps, pm, ctx, fw, xm, ym, gg, xty, yy, t_n, woff, -1, -1, fi, wsum if sw else i2f(n), False)
    var best = 0
    var best_err = Float32(0)
    if a_n > 1:
        for a in range(a_n):
            var alpha = fp[a]
            copy(fw, mm, fw, gg, d * d)
            for j in range(d):
                st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
            _ = cholesky(fw, mm, d)
            copy(fw, rhs, fw, xty, d)
            chol_solve(fw, mm, d, fw, rhs)
            for j in range(d):
                b.param(j, ld(fw, xm + j))
                b.param(d + 1 + j, ld(fw, rhs + j))
            b.param(d, ld(fw, ym))
            b.param(2 * d + 1, wsum)
            for o in range(d * d):
                b.param(2 * d + 2 + o, ld(fw, mm + o))
            b.push(ctx)
            ctx.enqueue_function[xb_loo_kernel](
                b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(),
                pl.dev.unsafe_ptr(),
                Int32(n), Int32(d), Int32(b.flags()), Int32(woff),
                grid_dim=b.nb, block_dim=XB_TPB,
            )
            pl.fetch(ctx)
            var e64 = Float64(0)
            for c in range(XB_TPB):
                e64 += pl.total64(c)
            var err = fd(Float32(e64), i2f(n))
            st(res, t_n * d + t_n + 2 + a, err)
            if a == 0 or err < best_err:  # DEVIATION 5005: the first minimum
                best = a
                best_err = err
    _ridge_solve_best(fpp, res, fw, d, t_n, fi, best, best_err)
    ctx.synchronize()
    _ = hw^
    _ = hfp^
    _ = ps^
    _ = pm^
    _ = pl^
    _ = b^


# ------------------------------------------------------------ BayesianRidge

def bayes_ridge_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/bayes.mojo `bayes_ridge_fit`.
    res: coef d, intercept, alpha_, lambda_, n_iter.
    Host work: xm d | G d*d | xty d | V d*d | vty d | old d | tmp d | ym | yy."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var sw = Int(ip[2]) != 0
    var tol = fp[0]
    var a1 = fp[1]
    var a2 = fp[2]
    var l1 = fp[3]
    var l2 = fp[4]
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var vv = xty + d
    var vty = vv + d * d
    var old = vty + d
    var tmp = old + d
    var ymo = tmp + d
    var yy = ymo + 1
    var woff = n if sw else -1
    var gc = d * (d + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, 2 * d + 3, 1)
    var ps = Part(ctx, b.nb, d + 1)
    var pm = Part(ctx, b.nb, gc + d + 1)
    var pe = Part(ctx, b.nb, 1)
    var hw = _zeros(yy + 1)
    var fw = _host_fp(hw)
    var wsum = i2f(n)
    if sw:
        wsum = _weight_sum(y, n, n)
    _moments(b, ps, pm, ctx, fw, xm, ymo, gg, xty, yy, 1, woff, -1, -1, fi, wsum, True)
    var ym = ld(fw, ymo)
    var alpha = fp[5]
    jacobi_eig(fw, gg, fw, vv, d, 60)
    for j in range(d):
        var ev = ld(fw, gg + j * d + j)
        st(fw, tmp + j, fmax(Float32(0), ev))
        var acc = Float32(0)
        for k in range(d):
            acc = fmad(ld(fw, vv + k * d + j), ld(fw, xty + k), acc)
        st(fw, vty + j, acc)
    if alpha < 0:
        # the (weighted) variance of y about its mean
        var yvar = fd(ld(fw, yy), wsum)
        alpha = fd(Float32(1), fa(yvar, Float32(1.1920929e-07)))
    var lam = fp[6]
    if lam < 0:
        lam = Float32(1)
    var iters = 0
    for it in range(max_iter + 1):
        var ratio = fd(lam, alpha)
        for j in range(d):
            var acc = Float32(0)
            for k in range(d):
                acc = fmad(ld(fw, vv + j * d + k), fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio)), acc)
            st(res, j, acc)
        if it == max_iter:
            break  # the last update after the loop
        iters = it + 1
        var sse = _sse(b, pe, ctx, fw, xm, ym, res, 0, woff)
        var stop = 0
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
        if stop == 1:
            break
    st(res, d, _intercept(d, fw, xm, ym, res, 0) if fi else Float32(0))
    st(res, d + 1, alpha)
    st(res, d + 2, lam)
    st(res, d + 3, i2f(iters))
    ctx.synchronize()
    _ = hw^
    _ = ps^
    _ = pm^
    _ = pe^
    _ = b^


# ------------------------------------------------------------ ARDRegression

def ard_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/bayes.mojo `ard_fit`.
    res: coef d, intercept, alpha_, lambda_ d, n_iter.
    Host work: xm d | G d*d | xty d | A d*d | sigma d*d | lambda d | old d | ym | yy;
    integers: keep d | kept index d."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var tol = fp[0]
    var a1 = fp[1]
    var a2 = fp[2]
    var l1 = fp[3]
    var l2 = fp[4]
    var thr = fp[5]
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var aa = xty + d
    var sg = aa + d * d
    var lamo = sg + d * d
    var old = lamo + d
    var ymo = old + d
    var yy = ymo + 1
    var keep = 0
    var gc = d * (d + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, 2 * d + 3, 1)
    var ps = Part(ctx, b.nb, d + 1)
    var pm = Part(ctx, b.nb, gc + d + 1)
    var pe = Part(ctx, b.nb, 1)
    var hw = _zeros(yy + 1)
    var fw = _host_fp(hw)
    var hiw = List[Int32](capacity=2 * d + 1)
    for _ in range(2 * d + 1):
        hiw.append(Int32(0))
    var iw = IP(unsafe_from_address=Int(hiw.unsafe_ptr()))
    _moments(b, ps, pm, ctx, fw, xm, ymo, gg, xty, yy, 1, -1, -1, -1, fi, i2f(n), True)
    var ym = ld(fw, ymo)
    var alpha = fd(Float32(1), fa(fd(ld(fw, yy), i2f(n)), Float32(1.1920929e-07)))
    fill(fw, lamo, d, Float32(1))
    fill(res, 0, d, Float32(0))
    for j in range(d):
        sti(iw, keep + j, 1)
    var iters = 0
    var any_kept = True
    for it in range(max_iter):
        iters = it + 1
        var dk = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _ard_coef(d, dk, fw, sg, xty, alpha, iw, keep, res)
        var sse = _sse(b, pe, ctx, fw, xm, ym, res, 0, -1)
        var stop = 0
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
            var k = 0
            if ld(fw, lamo + j) < thr:
                k = 1
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
        if stop == 1:
            break
    if any_kept:
        var dk = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _ard_coef(d, dk, fw, sg, xty, alpha, iw, keep, res)
    else:
        fill(res, 0, d, Float32(0))
    st(res, d, _intercept(d, fw, xm, ym, res, 0) if fi else Float32(0))
    st(res, d + 1, alpha)
    copy(res, d + 2, fw, lamo, d)
    st(res, d + 2 + d, i2f(iters))
    ctx.synchronize()
    _ = hw^
    _ = hiw^
    _ = ps^
    _ = pm^
    _ = pe^
    _ = b^


# ------------------------------------------------------- Lars / LassoLars

def lars_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/lars.mojo `lars_fit`: the means, the Gram and X'y in
    blocks, then the path, statement for statement, on the host.
    res: coef d, intercept, n_iter, alpha, n_active, active d.
    Host work: xm d | G d*d | xty d | prev d | cov d | L d*d | ls d | sgn d | ym | yy;
    integers: state d | active list d."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var lasso = Int(ip[2]) != 0
    var positive = Int(ip[3]) != 0
    var alpha_min = fp[0]
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var prev = xty + d
    var cov = prev + d
    var ll = cov + d
    var ls = ll + d * d
    var sgn = ls + d
    var ymo = sgn + d
    var yy = ymo + 1
    var state = 0
    var act = d
    var gc = d * (d + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, d + 2, 1)
    var ps = Part(ctx, b.nb, d + 1)
    var pm = Part(ctx, b.nb, gc + d + 1)
    var hw = _zeros(yy + 1)
    var fw = _host_fp(hw)
    var hiw = List[Int32](capacity=2 * d + 1)
    for _ in range(2 * d + 1):
        hiw.append(Int32(0))
    var iw = IP(unsafe_from_address=Int(hiw.unsafe_ptr()))
    _moments(b, ps, pm, ctx, fw, xm, ymo, gg, xty, yy, 1, -1, -1, -1, fi, i2f(n), False)
    var ym = ld(fw, ymo)
    fill(res, 0, d, Float32(0))
    fill(fw, prev, d, Float32(0))
    for j in range(d):
        sti(iw, state + j, 0)
    var k = 0
    var n_iter = 0
    var drop = False
    var alpha = Float32(0)
    var prev_alpha = Float32(0)
    var guard = 0
    while guard < 4 * d + 4 * max_iter + 8:
        guard += 1
        # correlations with the current residual
        for j in range(d):
            var acc = ld(fw, xty + j)
            for l in range(d):
                acc = fs(acc, fm(ld(fw, gg + j * d + l), ld(res, l)))
            st(fw, cov + j, acc)
        var c_idx = -1
        var cbig = Float32(0)
        for j in range(d):
            if ldi(iw, state + j) == 0:
                var a = ld(fw, cov + j) if positive else fabs(ld(fw, cov + j))
                # DEVIATION 5005 (IDENTITY_PATHS row 105): strict >, the
                # lowest index wins an exact tie
                if c_idx < 0 or a > cbig:
                    c_idx = j
                    cbig = a
        alpha = fd(cbig, i2f(n))
        if alpha <= fa(alpha_min, EQ_TOL):
            if fabs(fs(alpha, alpha_min)) > EQ_TOL:
                if n_iter > 0:
                    var ss = fd(fs(prev_alpha, alpha_min), fs(prev_alpha, alpha))
                    for j in range(d):
                        var pj = ld(fw, prev + j)
                        st(res, j, fmad(ss, fs(ld(res, j), pj), pj))
                alpha = alpha_min
            break
        if n_iter >= max_iter or k >= d:
            break
        if not drop:
            if c_idx < 0:
                break
            sti(iw, act + k, c_idx)
            st(fw, sgn + k, Float32(1) if positive else fsign(ld(fw, cov + c_idx)))
            # the new pivot of the Cholesky of G_AA
            for a in range(k + 1):
                for bb in range(k + 1):
                    st(fw, ll + a * (k + 1) + bb, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + bb)))
            var okc = cholesky(fw, ll, k + 1)
            if not okc or ld(fw, ll + k * (k + 1) + k) < Float32(1e-7):
                sti(iw, state + c_idx, 2)  # their degenerate-regressor skip
                continue
            sti(iw, state + c_idx, 1)
            k += 1
        if lasso and n_iter > 0 and prev_alpha < alpha:
            break
        # equiangular direction over the active set
        for a in range(k):
            for bb in range(k):
                st(fw, ll + a * k + bb, ld(fw, gg + ldi(iw, act + a) * d + ldi(iw, act + bb)))
        _ = cholesky(fw, ll, k)
        for a in range(k):
            st(fw, ls + a, ld(fw, sgn + a))
        chol_solve(fw, ll, k, fw, ls)
        var aa: Float32
        if k == 1 and ld(fw, ls) == 0:
            st(fw, ls, Float32(1))
            aa = Float32(1)
        else:
            var sm = Float32(0)
            for a in range(k):
                sm = fmad(ld(fw, ls + a), ld(fw, sgn + a), sm)
            aa = fd(Float32(1), fsqrt(sm))
            for a in range(k):
                st(fw, ls + a, fm(ld(fw, ls + a), aa))
        var gamma = fd(cbig, aa)
        for j in range(d):
            if ldi(iw, state + j) != 0:
                continue
            var cj = Float32(0)
            for a in range(k):
                cj = fmad(ld(fw, gg + j * d + ldi(iw, act + a)), ld(fw, ls + a), cj)
            var cv = ld(fw, cov + j)
            var g1 = fd(fs(cbig, cv), fa(fs(aa, cj), TINY32))
            if g1 > 0 and g1 < gamma:
                gamma = g1
            if not positive:
                var g2 = fd(fa(cbig, cv), fa(fa(aa, cj), TINY32))
                if g2 > 0 and g2 < gamma:
                    gamma = g2
        drop = False
        var z_pos = BIG
        for a in range(k):
            var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
            if z > 0 and z < z_pos:
                z_pos = z
        if z_pos < gamma:
            for a in range(k):
                var z = fd(-ld(res, ldi(iw, act + a)), fa(ld(fw, ls + a), TINY32))
                if z == z_pos:
                    st(fw, sgn + a, -ld(fw, sgn + a))
            if lasso:
                gamma = z_pos
            drop = True
        n_iter += 1
        copy(fw, prev, res, 0, d)
        prev_alpha = alpha
        for a in range(k):
            var j = ldi(iw, act + a)
            st(res, j, fmad(gamma, ld(fw, ls + a), ld(res, j)))
        if drop and lasso:
            var wq = 0
            for a in range(k):
                var j = ldi(iw, act + a)
                var z = fd(-ld(fw, prev + j), fa(ld(fw, ls + a), TINY32))
                if z == z_pos:
                    sti(iw, state + j, 0)
                    st(res, j, Float32(0))
                else:
                    sti(iw, act + wq, j)
                    st(fw, sgn + wq, ld(fw, sgn + a))
                    wq += 1
            k = wq
    var intercept = Float32(0)
    if fi:
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(res, j), acc)
        intercept = fs(ym, acc)
    st(res, d, intercept)
    st(res, d + 1, i2f(n_iter))
    st(res, d + 2, alpha)
    st(res, d + 3, i2f(k))
    for a in range(d):
        st(res, d + 4 + a, i2f(ldi(iw, act + a)) if a < k else Float32(-1))
    ctx.synchronize()
    _ = hw^
    _ = hiw^
    _ = ps^
    _ = pm^
    _ = b^


# ------------------------------------------------------------------ entry

def gram_fit(
    ctx: DeviceContext, algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """The fit `algo` (one `gram_handles` names), its result in res."""
    fill(res, 0, n_out, Float32(0))
    if algo == XG_ENETCV:
        enetcv_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XG_RIDGE:
        ridge_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XG_BAYES:
        bayes_ridge_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XG_ARD:
        ard_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XG_LARS:
        lars_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
