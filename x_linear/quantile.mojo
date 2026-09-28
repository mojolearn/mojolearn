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
from x_linear.team import Team
from x_linear.tops import t_sum


def _soft(a: Float32, t: Float32) -> Float32:
    if a > t:
        return fs(a, t)
    if a < -t:
        return fa(a, t)
    return Float32(0)


def quantile_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    With sample_weight, y = targets n | weights n and the loss is
    (1/sum w) sum w_i rho_q(r_i) (theirs: sum w rho + alpha sum(w) |w|_1).
    res: coef d, intercept, n_iter, converged.
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d.
    Team form: the per-row updates across the team, one thread per cell of
    A'A and of each A' product; every norm fold, the solve and the z/v
    updates on the lead (team row buffer 0 holds the primal residuals)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var q = ld(fp, 0)
    var alpha = ld(fp, 1)
    var eps_abs = ld(fp, 2)
    var eps_rel = ld(fp, 3)
    var sw = ldi(ip, 2) != 0
    var den = i2f(n)
    if sw:
        den = t_sum(t, y + n, n)
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
    var prb = t.row(0)
    # M = A'A + E, rows ascending
    var cells = m * (m + 1) // 2
    for c in range(t.tid, cells, t.nt):
        var j = 0
        var qq = c
        while qq >= j + 1:
            qq -= j + 1
            j += 1
        var k = qq
        var acc = Float32(0)
        for i in range(n):
            var aj = ld(x, i * d + j) if j < d else Float32(1)
            var ak = ld(x, i * d + k) if k < d else Float32(1)
            acc = fmad(aj, ak, acc)
        if j == k and j < d:
            acc = fa(acc, Float32(1))
        st(fw, mm + j * m + k, acc)
        st(fw, mm + k * m + j, acc)
    for i in range(t.tid, n, t.nt):
        st(fw, r + i, Float32(0))
        st(fw, u + i, Float32(0))
    t.sync()
    var rho = Float32(0)
    var ynorm = Float32(0)
    if t.lead():
        _ = cholesky(fw, mm, m)
        fill(fw, beta, m, Float32(0))
        fill(fw, z, d, Float32(0))
        fill(fw, v, d, Float32(0))
        var ym = mean_of(y, n)
        var spread = Float32(0)
        for i in range(n):
            spread = fa(spread, fabs(fs(ld(y, i), ym)))
        spread = fmax(fd(spread, i2f(n)), Float32(1e-6))
        rho = fd(Float32(1), fm(i2f(n), spread))
        for i in range(n):
            ynorm = fmad(ld(y, i), ld(y, i), ynorm)
        ynorm = fsqrt(ynorm)
    rho = t.bcast(rho, 1)
    ynorm = t.bcast(ynorm, 2)
    var iters = 0
    var converged = False
    for it in range(max_iter):
        iters = it + 1
        # beta-update
        for j in range(t.tid, m, t.nt):
            var acc = Float32(0)
            for i in range(n):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                acc = fmad(aj, fs(fs(ld(y, i), ld(fw, r + i)), ld(fw, u + i)), acc)
            if j < d:
                acc = fa(acc, fs(ld(fw, z + j), ld(fw, v + j)))
            st(fw, rhs + j, acc)
        t.sync()
        if t.lead():
            chol_solve(fw, mm, m, fw, rhs)
            copy(fw, beta, fw, rhs, m)
        t.sync()
        var b = ld(fw, beta + d) if fi else Float32(0)
        # r-update (keep the old r in tmp for the dual residual)
        var kq = fd(Float32(1), fm(den, rho))
        for i in range(t.tid, n, t.nt):
            var up = fm(q, kq)
            var lo = fm(fs(Float32(1), q), kq)
            var abi = fa(row_dot(x, i, d, fw, beta), b)
            st(fw, ab + i, abi)
            var vv = fs(fs(ld(y, i), abi), ld(fw, u + i))
            var nr: Float32
            if sw:
                var ki = fm(ld(y, n + i), kq)
                up = fm(q, ki)
                lo = fm(fs(Float32(1), q), ki)
            if vv > up:
                nr = fs(vv, up)
            elif vv < -lo:
                nr = fa(vv, lo)
            else:
                nr = Float32(0)
            st(fw, tmp + i, fs(nr, ld(fw, r + i)))
            st(fw, r + i, nr)
            # dual update and the primal residual of this row
            var pr = fs(fa(abi, nr), ld(y, i))
            st(prb, i, pr)
            st(fw, u + i, fa(ld(fw, u + i), pr))
        t.sync()
        # the dual residual's A' dr, one thread per cell (into rhs, free now)
        for j in range(t.tid, m, t.nt):
            var acc = Float32(0)
            for i in range(n):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                acc = fmad(aj, ld(fw, tmp + i), acc)
            st(fw, rhs + j, acc)
        t.sync()
        var flag = 0  # bit 0: converged, bit 1: rescaled
        var inv = Float32(1)
        if t.lead():
            var abn = Float32(0)
            var rn = Float32(0)
            for i in range(n):
                var abi = ld(fw, ab + i)
                abn = fmad(abi, abi, abn)
                var nr = ld(fw, r + i)
                rn = fmad(nr, nr, rn)
            # z-update
            var zdiff = Float32(0)
            var wn = Float32(0)
            var zn = Float32(0)
            var tt = fd(alpha, rho)
            for j in range(d):
                var wj = ld(fw, beta + j)
                wn = fmad(wj, wj, wn)
                var nz = _soft(fa(wj, ld(fw, v + j)), tt)
                var dz = fs(nz, ld(fw, z + j))
                zdiff = fmad(dz, dz, zdiff)
                st(fw, z + j, nz)
                zn = fmad(nz, nz, zn)
            var prim = Float32(0)
            for i in range(n):
                var pr = ld(prb, i)
                prim = fmad(pr, pr, prim)
            for j in range(d):
                var pr = fs(ld(fw, beta + j), ld(fw, z + j))
                prim = fmad(pr, pr, prim)
                st(fw, v + j, fa(ld(fw, v + j), pr))
            # dual residual rho * || [A' dr ; dz] ||
            var dual = zdiff
            for j in range(m):
                var acc = ld(fw, rhs + j)
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
                flag = 1
            elif (it + 1) % 10 == 0:
                var factor = Float32(0)
                if prim_n > fm(Float32(10), dual_n):
                    factor = Float32(2)
                elif dual_n > fm(Float32(10), prim_n):
                    factor = Float32(0.5)
                if factor != 0:
                    rho = fm(rho, factor)
                    inv = fd(Float32(1), factor)
                    for j in range(d):
                        st(fw, v + j, fm(ld(fw, v + j), inv))
                    flag = 2
        flag = t.bcast_int(flag, 3)
        if flag == 1:
            converged = True
            break
        if flag == 2:
            rho = t.bcast(rho, 1)
            inv = t.bcast(inv, 2)
            for i in range(t.tid, n, t.nt):
                st(fw, u + i, fm(ld(fw, u + i), inv))
            t.sync()
    if not t.lead():
        return
    copy(res, 0, fw, z, d)
    st(res, d, ld(fw, beta + d) if fi else Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
