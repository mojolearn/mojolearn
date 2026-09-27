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
)


def ridge_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_targets T, fit_intercept, n_alphas A]; fp: alphas (A).
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
    for j in range(d):
        var acc = Float32(0)
        if fi:
            for i in range(n):
                acc = fa(acc, ld(x, i * d + j))
            acc = fd(acc, i2f(n))
        st(fw, xm + j, acc)
    for t in range(t_n):
        var acc = Float32(0)
        if fi:
            for i in range(n):
                acc = fa(acc, ld(y, i * t_n + t))
            acc = fd(acc, i2f(n))
        st(fw, ym + t, acc)
    centered_gram(x, n, d, fw, xm, fw, gg)
    for t in range(t_n):
        var ymt = ld(fw, ym + t)
        for j in range(d):
            var acc = Float32(0)
            var mj = ld(fw, xm + j)
            for i in range(n):
                acc = fmad(fs(ld(x, i * d + j), mj), fs(ld(y, i * t_n + t), ymt), acc)
            st(fw, xty + t * d + j, acc)
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
                var h = fd(Float32(1), i2f(n)) if fi else Float32(0)
                for j in range(d):
                    h = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(fw, zz + j), h)
                var loo = fd(e, fs(Float32(1), h))
                err = fmad(loo, loo, err)
            err = fd(err, i2f(n))
            st(res, t_n * d + t_n + 2 + a, err)
            if a == 0 or err < best_err:
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
