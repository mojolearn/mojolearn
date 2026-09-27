# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QuantileRegressor by ADMM (lane/algos-linear, 2026-09-27).

Reference problem: scikit-learn `sklearn/linear_model/_quantile.py`,
min (1/n) sum_i rho_q(y_i - x_i'w - b) + alpha ||w||_1 (intercept
unpenalized), which they solve as a linear program (HiGHS). THIS IS NOT
THEIR SOLVER: it is the scaled-form ADMM of Boyd, Parikh, Chu, Peleato &
Eckstein, "Distributed Optimization and Statistical Learning via the
Alternating Direction Method of Multipliers" (2011), with the splitting
r = y - A beta (A = [X, 1]) and z = w:
    beta <- (A'A + E)^-1 (A'(y - r - u) + E'(z - v))   (one Cholesky, reused)
    r    <- prox of (1/(n rho)) rho_q at y - A beta - u  (section 6.4's pinball prox)
    z    <- soft(w + v, alpha / rho)                    (section 6.4 / 4.4.3)
    u    <- u + A beta + r - y ;  v <- v + w - z
with residual balancing every 10 iterations (section 3.4.1, mu = 10,
tau = 2) and the stopping rule of section 3.3.1 (eps_abs, eps_rel). Every
iteration runs in the same fixed order; the answer is w = z (exact zeros)
and b from beta. It reaches the LP's optimum to ADMM precision, not to a
vertex, so coefficients agree with theirs to a tolerance, not exactly.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, ld, st, ldi, i2f, fill, copy,
    cholesky, chol_solve, row_dot, mean_of,
)


def _soft(a: Float32, t: Float32) -> Float32:
    if a > t:
        return fs(a, t)
    if a < -t:
        return fa(a, t)
    return Float32(0)


def quantile_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept]; fp: [quantile, alpha, eps_abs, eps_rel].
    res: coef d, intercept, n_iter, converged.
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var q = ld(fp, 0)
    var alpha = ld(fp, 1)
    var eps_abs = ld(fp, 2)
    var eps_rel = ld(fp, 3)
    var m = d + 1 if fi else d
    var mm = 0
    var beta = m * m
    var rhs = beta + m
    var r = rhs + m
    var u = r + n
    var ab = u + n
    var tmp = ab + n
    var z = tmp + n
    var v = z + d
    # M = A'A + E, rows ascending
    for j in range(m):
        for k in range(j + 1):
            var acc = Float32(0)
            for i in range(n):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                var ak = ld(x, i * d + k) if k < d else Float32(1)
                acc = fmad(aj, ak, acc)
            if j == k and j < d:
                acc = fa(acc, Float32(1))
            st(fw, mm + j * m + k, acc)
            st(fw, mm + k * m + j, acc)
    _ = cholesky(fw, mm, m)
    fill(fw, beta, m, Float32(0))
    fill(fw, r, n, Float32(0))
    fill(fw, u, n, Float32(0))
    fill(fw, z, d, Float32(0))
    fill(fw, v, d, Float32(0))
    var ym = mean_of(y, n)
    var spread = Float32(0)
    for i in range(n):
        spread = fa(spread, fabs(fs(ld(y, i), ym)))
    spread = fmax(fd(spread, i2f(n)), Float32(1e-6))
    var rho = fd(Float32(1), fm(i2f(n), spread))
    var ynorm = Float32(0)
    for i in range(n):
        ynorm = fmad(ld(y, i), ld(y, i), ynorm)
    ynorm = fsqrt(ynorm)
    var iters = 0
    var converged = False
    for it in range(max_iter):
        iters = it + 1
        # beta-update
        for j in range(m):
            var acc = Float32(0)
            for i in range(n):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                acc = fmad(aj, fs(fs(ld(y, i), ld(fw, r + i)), ld(fw, u + i)), acc)
            if j < d:
                acc = fa(acc, fs(ld(fw, z + j), ld(fw, v + j)))
            st(fw, rhs + j, acc)
        chol_solve(fw, mm, m, fw, rhs)
        copy(fw, beta, fw, rhs, m)
        var b = ld(fw, beta + d) if fi else Float32(0)
        # r-update (keep the old r in tmp for the dual residual)
        var kq = fd(Float32(1), fm(i2f(n), rho))
        var up = fm(q, kq)
        var lo = fm(fs(Float32(1), q), kq)
        var abn = Float32(0)
        var rn = Float32(0)
        for i in range(n):
            var abi = fa(row_dot(x, i, d, fw, beta), b)
            st(fw, ab + i, abi)
            abn = fmad(abi, abi, abn)
            var vv = fs(fs(ld(y, i), abi), ld(fw, u + i))
            var nr: Float32
            if vv > up:
                nr = fs(vv, up)
            elif vv < -lo:
                nr = fa(vv, lo)
            else:
                nr = Float32(0)
            st(fw, tmp + i, fs(nr, ld(fw, r + i)))
            st(fw, r + i, nr)
            rn = fmad(nr, nr, rn)
        # z-update
        var zdiff = Float32(0)
        var wn = Float32(0)
        var zn = Float32(0)
        var t = fd(alpha, rho)
        for j in range(d):
            var wj = ld(fw, beta + j)
            wn = fmad(wj, wj, wn)
            var nz = _soft(fa(wj, ld(fw, v + j)), t)
            var dz = fs(nz, ld(fw, z + j))
            zdiff = fmad(dz, dz, zdiff)
            st(fw, z + j, nz)
            zn = fmad(nz, nz, zn)
        # dual updates and the primal residual
        var prim = Float32(0)
        for i in range(n):
            var pr = fs(fa(ld(fw, ab + i), ld(fw, r + i)), ld(y, i))
            prim = fmad(pr, pr, prim)
            st(fw, u + i, fa(ld(fw, u + i), pr))
        for j in range(d):
            var pr = fs(ld(fw, beta + j), ld(fw, z + j))
            prim = fmad(pr, pr, prim)
            st(fw, v + j, fa(ld(fw, v + j), pr))
        # dual residual rho * || [A' dr ; dz] ||
        var dual = zdiff
        for j in range(m):
            var acc = Float32(0)
            for i in range(n):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                acc = fmad(aj, ld(fw, tmp + i), acc)
            dual = fmad(acc, acc, dual)
        var prim_n = fsqrt(prim)
        var dual_n = fm(rho, fsqrt(dual))
        var scale_p = fmax(fmax(fsqrt(abn), fsqrt(rn)), fmax(ynorm, fmax(fsqrt(wn), fsqrt(zn))))
        var eps_p = fa(fm(eps_abs, fsqrt(i2f(n + d))), fm(eps_rel, scale_p))
        var un = Float32(0)
        for i in range(n):
            un = fmad(ld(fw, u + i), ld(fw, u + i), un)
        for j in range(d):
            un = fmad(ld(fw, v + j), ld(fw, v + j), un)
        var eps_d = fa(fm(eps_abs, fsqrt(i2f(m))), fm(fm(eps_rel, rho), fsqrt(un)))
        if prim_n <= eps_p and dual_n <= eps_d:
            converged = True
            break
        if (it + 1) % 10 == 0:
            var factor = Float32(0)
            if prim_n > fm(Float32(10), dual_n):
                factor = Float32(2)
            elif dual_n > fm(Float32(10), prim_n):
                factor = Float32(0.5)
            if factor != 0:
                rho = fm(rho, factor)
                var inv = fd(Float32(1), factor)
                for i in range(n):
                    st(fw, u + i, fm(ld(fw, u + i), inv))
                for j in range(d):
                    st(fw, v + j, fm(ld(fw, v + j), inv))
    copy(res, 0, fw, z, d)
    st(res, d, ld(fw, beta + d) if fi else Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
