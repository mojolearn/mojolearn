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
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fexp, fabs, ld, st, ldi, i2f, fill, row_dot
from x_linear.lbfgs import lbfgs, lbfgs_work


def huber_objective(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) -> Float32:
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
    for i in range(n):
        var r = fs(fs(ld(y, i), row_dot(x, i, d, th, toff)), b)
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
        for j in range(d):
            st(g, goff + j, fmad(coefv, ld(x, i * d + j), ld(g, goff + j)))
        if fi:
            st(g, goff + d, fa(ld(g, goff + d), coefv))
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


def huber_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [epsilon, alpha, tol].
    With sample_weight, y = targets n | weights n (their weighted objective).
    res: coef d, intercept 1, scale 1, n_iter 1 | theta scratch (P).
    fw: lbfgs_work(P)."""
    var fi = ldi(ip, 1) != 0
    var p = d + 2 if fi else d + 1
    var th = d + 4
    fill(res, th, p, Float32(0))
    var it = lbfgs[huber_objective](x, y, n, d, ip, fp, res, th, p, ldi(ip, 0), ld(fp, 2), fw, 0)
    for j in range(d):
        st(res, j, ld(res, th + j))
    st(res, d, ld(res, th + d) if fi else Float32(0))
    st(res, d + 1, fexp(ld(res, th + p - 1)))
    st(res, d + 2, i2f(it if it >= 0 else -it))
