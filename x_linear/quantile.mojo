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

COLUMN EQUILIBRATION (lane/gap-board-refusals, 2026-10-02): the w = z
split is D w = D z with D = diag(||x_j||) (1 for a zero column), the
ADMM above on the column-scaled problem A D^-1 mapped back: E = D^2 in the
beta system, rhs gets D^2 (z - v), z <- soft(w + v, alpha / (rho d_j^2)),
and the residuals and norms are the scaled problem's (D (w - z), D w, D z,
D v, D dz and the d feature rows of A' dr divided by d_j). With E = I the
float32 system A'A + I lost the 1 on large or collinear columns (Istella:
a non-positive Cholesky pivot, NaN through every iteration); A'A + D^2 is
A'A + diag(A'A) on the weights: column-scaled, the columns' cosine matrix
plus I, every eigenvalue at least 1 there.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, ld, st, ldi, i2f, fill, copy,
    cholesky, chol_solve, row_dot, mean_of, axpy_acc, par_rows, row_dots,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from std.gpu import WARP_SIZE
from x_linear.tops import t_sum, fold_sq, chain_fmad, fold_one_fmad


def _soft(a: Float32, t: Float32) -> Float32:
    if a > t:
        return fs(a, t)
    if a < -t:
        return fa(a, t)
    return Float32(0)


def _quantile_fit_team(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """The team schedule (see quantile_fit).
    ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    With sample_weight, y = targets n | weights n and the loss is
    (1/sum w) sum w_i rho_q(r_i) (theirs: sum w rho + alpha sum(w) |w|_1).
    res: coef d, intercept, n_iter, converged.
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d |
    (next rhs m, the host's) | dsq d (d_j^2, the column equilibration).
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
    var dq = v + d + m
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
            var dj = acc if acc > Float32(0) else Float32(1)
            st(fw, dq + j, dj)
            acc = fa(acc, dj)
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
    # lane/linear-apple2: warp roles in the A' dr pass. Chains A' dr on
    # threads [0, m); the NEXT iteration's A'(y - r - u) chains on threads
    # [base1, base1 + m) (its y - r - u written by the r-update pass: the
    # words the next iteration would write, unless the residual balancing
    # rescales u, and then that iteration recomputes them); the lead's four
    # row folds on the first thread of the warps from base2. Every chain and
    # fold is still one thread's ascending loop. With too few threads the
    # lead folds and every iteration computes its own A'(y - r - u).
    var wm = ((m + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    var base1 = wm
    var base2 = 2 * wm
    var roles = t.nt >= base2 + 4 * WARP_SIZE
    var have_next = False
    var nx = t.row(2)
    for it in range(max_iter):
        iters = it + 1
        # beta-update: each row's y - r - u across the team (team row 1),
        # then one thread per cell of A'(y - r - u)
        var vb = t.row(1)
        if have_next:
            for j in range(t.tid, m, t.nt):
                var acc = ld(nx, j)
                if j < d:
                    acc = fa(acc, fm(ld(fw, dq + j), fs(ld(fw, z + j), ld(fw, v + j))))
                st(fw, rhs + j, acc)
            t.sync()
        else:
            for i in range(t.tid, n, t.nt):
                st(vb, i, fs(fs(ld(y, i), ld(fw, r + i)), ld(fw, u + i)))
            t.sync()
            for j in range(t.tid, m, t.nt):
                var acc = chain_fmad(x, j, d, vb, 0, 1, n) if j < d else fold_one_fmad(vb, 0, n)
                if j < d:
                    acc = fa(acc, fm(ld(fw, dq + j), fs(ld(fw, z + j), ld(fw, v + j))))
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
            var un_i = fa(ld(fw, u + i), pr)
            st(fw, u + i, un_i)
            if roles:
                st(vb, i, fs(fs(ld(y, i), nr), un_i))
        t.sync()
        # the dual residual's A' dr, one thread per cell (into rhs, free now);
        # lane/linear-apple2: four more threads fold ||ab||^2, ||r||^2, the
        # primal residuals' and ||u||^2 over the rows (each one thread's
        # ascending fold, the lead's former sequence) into team slots 8..11,
        # so the lead no longer runs four n-row folds one after another
        var sl = t.slot_at.unsafe_origin_cast[MutAnyOrigin]()
        # each fold thread leads a warp of its own (see huber.mojo)
        for j in range(t.tid, m, t.nt):
            var acc = chain_fmad(x, j, d, fw, tmp, 1, n) if j < d else fold_one_fmad(fw, tmp, n)
            st(fw, rhs + j, acc)
        if roles and t.tid >= base1 and t.tid < base1 + m:
            var j = t.tid - base1
            st(nx, j, chain_fmad(x, j, d, vb, 0, 1, n) if j < d else fold_one_fmad(vb, 0, n))
        if roles and t.tid >= base2 and (t.tid - base2) % WARP_SIZE == 0:
            var role = (t.tid - base2) // WARP_SIZE
            if role == 0:
                st(sl, 8, fold_sq(fw, ab, n))
            elif role == 1:
                st(sl, 9, fold_sq(fw, r, n))
            elif role == 2:
                st(sl, 10, fold_sq(prb, 0, n))
            elif role == 3:
                st(sl, 11, fold_sq(fw, u, n))
        t.sync()
        var flag = 0  # bit 0: converged, bit 1: rescaled
        var inv = Float32(1)
        if t.lead():
            var abn = ld(sl, 8) if roles else fold_sq(fw, ab, n)
            var rn = ld(sl, 9) if roles else fold_sq(fw, r, n)
            # z-update
            var zdiff = Float32(0)
            var wn = Float32(0)
            var zn = Float32(0)
            # the column-equilibrated split: threshold alpha / (rho d_j^2), norms of D w, D z, D dz
            for j in range(d):
                var dj = ld(fw, dq + j)
                var wj = ld(fw, beta + j)
                wn = fmad(fm(dj, wj), wj, wn)
                var nz = _soft(fa(wj, ld(fw, v + j)), fd(alpha, fm(rho, dj)))
                var dz = fs(nz, ld(fw, z + j))
                zdiff = fmad(fm(dj, dz), dz, zdiff)
                st(fw, z + j, nz)
                zn = fmad(fm(dj, nz), nz, zn)
            var prim = ld(sl, 10) if roles else fold_sq(prb, 0, n)
            for j in range(d):
                var pr = fs(ld(fw, beta + j), ld(fw, z + j))
                prim = fmad(fm(ld(fw, dq + j), pr), pr, prim)
                st(fw, v + j, fa(ld(fw, v + j), pr))
            # dual residual rho * || [D^-1 A' dr ; D dz] ||
            var dual = zdiff
            for j in range(m):
                var acc = ld(fw, rhs + j)
                if j < d:
                    dual = fa(dual, fd(fm(acc, acc), ld(fw, dq + j)))
                else:
                    dual = fmad(acc, acc, dual)
            var prim_n = fsqrt(prim)
            var dual_n = fm(rho, fsqrt(dual))
            var scale_p = fmax(fmax(fsqrt(abn), fsqrt(rn)), fmax(ynorm, fmax(fsqrt(wn), fsqrt(zn))))
            var eps_p = fa(fm(eps_abs, fsqrt(i2f(n + d))), fm(eps_rel, scale_p))
            var un = ld(sl, 11) if roles else fold_sq(fw, u, n)
            for j in range(d):
                un = fmad(fm(ld(fw, dq + j), ld(fw, v + j)), ld(fw, v + j), un)
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
        have_next = roles and flag == 0
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
def _quantile_fit_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """The host schedule (lane linear-cpu): A'A in one row pass, the
    r-update mapped, then ONE fold pass per iteration for every row
    accumulator (next iteration's A'(y - r - u) folded with it unless u is
    rescaled).
    ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    With sample_weight, y = targets n | weights n and the loss is
    (1/sum w) sum w_i rho_q(r_i) (theirs: sum w rho + alpha sum(w) |w|_1).
    res: coef d, intercept, n_iter, converged.
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d | next rhs m |
    dsq d (d_j^2, the column equilibration)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var q = ld(fp, 0)
    var alpha = ld(fp, 1)
    var eps_abs = ld(fp, 2)
    var eps_rel = ld(fp, 3)
    var sw = ldi(ip, 2) != 0
    var den = i2f(n)
    if sw:
        den = Float32(0)
        for i in range(n):
            den = fa(den, ld(y, n + i))
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
    var nrhs = v + d
    var dq = nrhs + m
    var rhs_ready = False
    # M = A'A + E, rows ascending: each entry of the lower triangle is its
    # own accumulator (lane linear-cpu: one pass over the rows, not one per entry)
    fill(fw, mm, m * m, Float32(0))
    for i in range(n):
        for j in range(d):
            axpy_acc(fw, mm + j * m, ld(x, i * d + j), x, i * d, j + 1)
        if fi:
            axpy_acc(fw, mm + d * m, Float32(1), x, i * d, d)
            st(fw, mm + d * m + d, fmad(Float32(1), Float32(1), ld(fw, mm + d * m + d)))
    for j in range(m):
        for k in range(j + 1):
            var acc = ld(fw, mm + j * m + k)
            if j == k and j < d:
                var dj = acc if acc > Float32(0) else Float32(1)
                st(fw, dq + j, dj)
                acc = fa(acc, dj)
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
        # beta-update: the m sums of A'(y - r - u), rows ascending, one pass
        # (already folded by the previous iteration's pass unless u was rescaled)
        if rhs_ready:
            copy(fw, rhs, fw, nrhs, m)
        else:
            fill(fw, rhs, m, Float32(0))
            for i in range(n):
                var t = fs(fs(ld(y, i), ld(fw, r + i)), ld(fw, u + i))
                axpy_acc(fw, rhs, t, x, i * d, d)
                if fi:
                    st(fw, rhs + d, fmad(Float32(1), t, ld(fw, rhs + d)))
        for j in range(d):
            st(fw, rhs + j, fa(ld(fw, rhs + j), fm(ld(fw, dq + j), fs(ld(fw, z + j), ld(fw, v + j)))))
        chol_solve(fw, mm, m, fw, rhs)
        copy(fw, beta, fw, rhs, m)
        var b = ld(fw, beta + d) if fi else Float32(0)
        # r-update (keep the old r in tmp for the dual residual)
        var kq = fd(Float32(1), fm(den, rho))
        var up = fm(q, kq)
        var lo_ = fm(fs(Float32(1), q), kq)
        var abn = Float32(0)
        var rn = Float32(0)
        var fwp = fw

        def rows_map(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm fwp, imm beta, imm b, imm ab,
                                         imm u, imm r, imm tmp, imm sw, imm q, imm kq, imm up, imm lo_}:
            row_dots(x, lo, hi, d, fwp, beta, fwp + ab)
            for i in range(lo, hi):
                var abi = fa(ld(fwp, ab + i), b)
                st(fwp, ab + i, abi)
                var vv = fs(fs(ld(y, i), abi), ld(fwp, u + i))
                var upi = up
                var loi = lo_
                if sw:
                    var ki = fm(ld(y, n + i), kq)
                    upi = fm(q, ki)
                    loi = fm(fs(Float32(1), q), ki)
                var nr: Float32
                if vv > upi:
                    nr = fs(vv, upi)
                elif vv < -loi:
                    nr = fa(vv, loi)
                else:
                    nr = Float32(0)
                st(fwp, tmp + i, fs(nr, ld(fwp, r + i)))
                st(fwp, r + i, nr)

        par_rows(rows_map, n)
        # ONE fold pass (lane linear-cpu): every accumulator below is its own,
        # rows ascending, as the separate passes had them: |A beta|^2, |r|^2,
        # the primal residual's row part and the dual update u, |u|^2's row
        # part, A' dr (into rhs, free now) and next iteration's A'(y - r - u)
        var prim = Float32(0)
        var un = Float32(0)
        fill(fw, rhs, m, Float32(0))
        fill(fw, nrhs, m, Float32(0))
        for i in range(n):
            var abi = ld(fw, ab + i)
            abn = fmad(abi, abi, abn)
            var nr = ld(fw, r + i)
            rn = fmad(nr, nr, rn)
            var yi = ld(y, i)
            var pr = fs(fa(abi, nr), yi)
            prim = fmad(pr, pr, prim)
            var ui = fa(ld(fw, u + i), pr)
            st(fw, u + i, ui)
            un = fmad(ui, ui, un)
            var ti = ld(fw, tmp + i)
            axpy_acc(fw, rhs, ti, x, i * d, d)
            var t2 = fs(fs(yi, nr), ui)
            axpy_acc(fw, nrhs, t2, x, i * d, d)
            if fi:
                st(fw, rhs + d, fmad(Float32(1), ti, ld(fw, rhs + d)))
                st(fw, nrhs + d, fmad(Float32(1), t2, ld(fw, nrhs + d)))
        rhs_ready = True
        # z-update
        var zdiff = Float32(0)
        var wn = Float32(0)
        var zn = Float32(0)
        # the column-equilibrated split: threshold alpha / (rho d_j^2), norms of D w, D z, D dz
        for j in range(d):
            var dj = ld(fw, dq + j)
            var wj = ld(fw, beta + j)
            wn = fmad(fm(dj, wj), wj, wn)
            var nz = _soft(fa(wj, ld(fw, v + j)), fd(alpha, fm(rho, dj)))
            var dz = fs(nz, ld(fw, z + j))
            zdiff = fmad(fm(dj, dz), dz, zdiff)
            st(fw, z + j, nz)
            zn = fmad(fm(dj, nz), nz, zn)
        # dual updates and the primal residual (the row parts folded above)
        for j in range(d):
            var pr = fs(ld(fw, beta + j), ld(fw, z + j))
            prim = fmad(fm(ld(fw, dq + j), pr), pr, prim)
            st(fw, v + j, fa(ld(fw, v + j), pr))
        # dual residual rho * || [D^-1 A' dr ; D dz] ||
        var dual = zdiff
        # the m sums of A' dr were folded above into rhs
        for j in range(m):
            var acc = ld(fw, rhs + j)
            if j < d:
                dual = fa(dual, fd(fm(acc, acc), ld(fw, dq + j)))
            else:
                dual = fmad(acc, acc, dual)
        var prim_n = fsqrt(prim)
        var dual_n = fm(rho, fsqrt(dual))
        var scale_p = fmax(fmax(fsqrt(abn), fsqrt(rn)), fmax(ynorm, fmax(fsqrt(wn), fsqrt(zn))))
        var eps_p = fa(fm(eps_abs, fsqrt(i2f(n + d))), fm(eps_rel, scale_p))
        for j in range(d):
            un = fmad(fm(ld(fw, dq + j), ld(fw, v + j)), ld(fw, v + j), un)
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
                rhs_ready = False
                for j in range(d):
                    st(fw, v + j, fm(ld(fw, v + j), inv))
    copy(res, 0, fw, z, d)
    st(res, d, ld(fw, beta + d) if fi else Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
def quantile_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d | next rhs m |
    dsq d (next rhs: the host schedule's; dsq: d_j^2, both schedules). The device runs the team schedule, the
    host the map-then-fold schedule; the same bits either way."""
    comptime if is_gpu():
        _quantile_fit_team(t, x, y, n, d, ip, fp, res, fw, iw)
    else:
        _quantile_fit_host(x, y, n, d, ip, fp, res, fw, iw)
