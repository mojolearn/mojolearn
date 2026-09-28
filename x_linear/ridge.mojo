# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ridge for RidgeClassifier (several targets) and RidgeCV (leave-one-out
over a grid of alphas) (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_ridge.py`:
  * `_solve_cholesky` (the dense 'auto' solver): (X'X + alpha I) W = X'Y on
    centered data (`_preprocess_data`), intercept = y_mean - x_mean' w;
  * `_RidgeGCV` (RidgeCV's default cv=None): the efficient leave-one-out
    error e_i / (1 - h_ii) with h_ii = 1/n + xc_i' (X'X + alpha I)^-1 xc_i
    when there is an intercept (their unpenalized sqrt(sw) column is
    orthogonal to the centered X, so this is the same hat diagonal); the
    alpha with the smallest mean squared LOO error wins, the first on a tie,
    and `best_score_` is minus that mean.
Theirs takes an SVD/eigendecomposition of X; here each alpha is one
Cholesky (x_linear/ops.mojo). float32, rows ascending.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, ld, st, ldi, i2f, fill, copy, cholesky, chol_solve, centered_gram,
    axpy_acc, add_acc, axpy_centered,
)


def ridge_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_targets T, fit_intercept, n_alphas A, sample_weight]; fp: alphas (A).
    With sample_weight, y = targets n*T | weights n (weighted means, the
    weighted Gram, and their weighted GCV errors w_i e_i^2 / (1 - h_i)^2).
    y: n x T row-major. A == 1: fit; A > 1 (T == 1): leave-one-out choice.
    res: coef T*d | intercept T | alpha | best_score | A mean squared LOO errors.
    fw: xm d | G d*d | M d*d | rhs d | ym T | xty d*T | z d."""
    var t_n = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var a_n = ldi(ip, 2)
    var xm = 0
    var gg = d
    var mm = gg + d * d
    var rhs = mm + d * d
    var ym = rhs + d
    var xty = ym + t_n
    var zz = xty + d * t_n
    var sw = ldi(ip, 3) != 0
    var wo = n * t_n
    var wsum = Float32(0)
    if sw:
        for i in range(n):
            wsum = fa(wsum, ld(y, wo + i))
    # one pass over the rows per block; every mean, Gram entry and X'y entry
    # is its own accumulator, rows ascending (lane linear-cpu)
    fill(fw, xm, d, Float32(0))
    fill(fw, ym, t_n, Float32(0))
    if fi:
        for i in range(n):
            if sw:
                var wi = ld(y, wo + i)
                axpy_acc(fw, xm, wi, x, i * d, d)
                axpy_acc(fw, ym, wi, y, i * t_n, t_n)
            else:
                add_acc(fw, xm, x, i * d, d)
                add_acc(fw, ym, y, i * t_n, t_n)
        var den = wsum if sw else i2f(n)
        for j in range(d):
            st(fw, xm + j, fd(ld(fw, xm + j), den))
        for t in range(t_n):
            st(fw, ym + t, fd(ld(fw, ym + t), den))
    if sw:
        # sum_i w_i xc_i xc_i' (theirs: the sqrt(w) rescale of _rescale_data)
        for j in range(d):
            fill(fw, gg + j * d + j, d - j, Float32(0))
        for i in range(n):
            var wi = ld(y, wo + i)
            for j in range(d):
                var a = fm(wi, fs(ld(x, i * d + j), ld(fw, xm + j)))
                axpy_centered(fw, gg + j * d + j, a, x, i * d + j, fw, xm + j, d - j)
        for j in range(d):
            for k in range(j + 1, d):
                st(fw, gg + k * d + j, ld(fw, gg + j * d + k))
    else:
        centered_gram(x, n, d, fw, xm, fw, gg)
    fill(fw, xty, d * t_n, Float32(0))
    for i in range(n):
        var wi = ld(y, wo + i) if sw else Float32(1)
        for t in range(t_n):
            var b = fs(ld(y, i * t_n + t), ld(fw, ym + t))
            if sw:
                axpy_centered[True](fw, xty + t * d, b, x, i * d, fw, xm, d, wi)
            else:
                axpy_centered(fw, xty + t * d, b, x, i * d, fw, xm, d)
    var best = 0
    var best_err = Float32(0)
    if a_n > 1:
        for a in range(a_n):
            var alpha = ld(fp, a)
            copy(fw, mm, fw, gg, d * d)
            for j in range(d):
                st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
            _ = cholesky(fw, mm, d)
            copy(fw, rhs, fw, xty, d)
            chol_solve(fw, mm, d, fw, rhs)
            var err = Float32(0)
            for i in range(n):
                var e = fs(ld(y, i), ld(fw, ym))
                for j in range(d):
                    var xc = fs(ld(x, i * d + j), ld(fw, xm + j))
                    st(fw, zz + j, xc)
                    e = fs(e, fm(xc, ld(fw, rhs + j)))
                chol_solve(fw, mm, d, fw, zz)
                if sw:
                    # their GCV on the sqrt(w)-rescaled problem
                    var wi = ld(y, wo + i)
                    var q = Float32(0)
                    for j in range(d):
                        q = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(fw, zz + j), q)
                    var h = fm(wi, q)
                    if fi:
                        h = fa(fd(wi, wsum), h)
                    var loo = fd(e, fs(Float32(1), h))
                    err = fmad(fm(wi, loo), loo, err)
                else:
                    var h = fd(Float32(1), i2f(n)) if fi else Float32(0)
                    for j in range(d):
                        h = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(fw, zz + j), h)
                    var loo = fd(e, fs(Float32(1), h))
                    err = fmad(loo, loo, err)
            err = fd(err, i2f(n))
            st(res, t_n * d + t_n + 2 + a, err)
            if a == 0 or err < best_err:  # DEVIATION 5005: the first minimum
                best = a
                best_err = err
    var alpha = ld(fp, best)
    copy(fw, mm, fw, gg, d * d)
    for j in range(d):
        st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
    _ = cholesky(fw, mm, d)
    for t in range(t_n):
        copy(fw, rhs, fw, xty + t * d, d)
        chol_solve(fw, mm, d, fw, rhs)
        copy(res, t * d, fw, rhs, d)
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(fw, rhs + j), acc)
        st(res, t_n * d + t, fs(ld(fw, ym + t), acc) if fi else Float32(0))
    st(res, t_n * d + t_n, alpha)
    st(res, t_n * d + t_n + 1, -best_err)
