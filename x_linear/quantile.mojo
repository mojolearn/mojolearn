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

THE BLOCKED SCHEDULE (cgr-linear, 2026-10-03): every fold over the rows is
FOLD_BLOCK rows from zero, rows ascending, then the block partials folded
ascending (x_linear/tops.mojo's order), so the device runs every row pass on
the grid (x_linear/quantile_grid.mojo: a thread per (cell, row block), then a
thread per cell) and the host column runs the same folds one block at a
time. The triangular solves are column oriented (`chol_solve_cols`: the
forward solve is the row form's words, the back solve subtracts the solved
entries in descending order), so one block solves them with a barrier per
column. The one-block team schedule is gone.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, ld, st, ldi, i2f, fill, copy,
    cholesky, axpy_acc, add_acc, par_rows, row_dots,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.tops import fold_sq, fold_fa, fold_fa_blocked, FOLD_BLOCK


def _soft(a: Float32, t: Float32) -> Float32:
    if a > t:
        return fs(a, t)
    if a < -t:
        return fa(a, t)
    return Float32(0)


@always_inline
def _balance(prim_n: Float32, dual_n: Float32, eps_p: Float32, eps_d: Float32, rel: Bool) -> Float32:
    """The residual balancing's rho factor (2, 1/2 or 0 = keep). rel (the
    default, fp[4] = 1): the residuals RELATIVE to their stopping
    tolerances, prim/eps_p against dual/eps_d (OSQP's rule); the absolute
    rule (prim against dual, fp[4] = 0, MOJOLEARN_XQ_ABS_BALANCE=1) compares
    two norms in different units and on standardized Istella-S and taxi
    doubled rho to ~4000x its start, where float32 ADMM drifts off the LP
    optimum (Istella-S r2 -1.7e11) and never meets the tolerance (5000 iterations)."""
    var p = fm(prim_n, eps_d) if rel else prim_n
    var dl = fm(dual_n, eps_p) if rel else dual_n
    if p > fm(Float32(10), dl):
        return Float32(2)
    if dl > fm(Float32(10), p):
        return Float32(0.5)
    return Float32(0)



# ------------------------------------------------ shared by both columns
@always_inline
def q_lower_cell(c: Int) -> Tuple[Int, Int]:
    """The c-th cell (j, k), k <= j, of an m x m lower triangle, row by row."""
    var j = 0
    var qq = c
    while qq >= j + 1:
        qq -= j + 1
        j += 1
    return (j, qq)


@always_inline
def q_a(x: FP, i: Int, d: Int, j: Int) -> Float32:
    """Entry (i, j) of A = [X, 1]."""
    return ld(x, i * d + j) if j < d else Float32(1)


@always_inline
def q_gram_part(x: FP, d: Int, j: Int, k: Int, lo: Int, cnt: Int) -> Float32:
    """(A'A)_jk over rows [lo, lo + cnt) from zero, rows ascending."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(q_a(x, i, d, j), q_a(x, i, d, k), acc)
    return acc


@always_inline
def q_gram_cell(fw: FP, mm: Int, m: Int, d: Int, dq: Int, j: Int, k: Int, sum: Float32):
    """Store the folded cell (j, k), k <= j, of M = A'A + D^2 (both halves);
    a weight's diagonal also records d_j^2 (1 for a zero column)."""
    var acc = sum
    if j == k and j < d:
        var dj = acc if acc > Float32(0) else Float32(1)
        st(fw, dq + j, dj)
        acc = fa(acc, dj)
    st(fw, mm + j * m + k, acc)
    st(fw, mm + k * m + j, acc)


@always_inline
def q_spread_part(y: FP, ym: Float32, lo: Int, cnt: Int) -> Float32:
    """sum |y_i - ym| over rows [lo, lo + cnt) from zero, rows ascending."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fa(acc, fabs(fs(ld(y, i), ym)))
    return acc


@always_inline
def q_resid(y: FP, fw: FP, r: Int, u: Int, i: Int) -> Float32:
    """y_i - r_i - u_i, the beta-update's right-hand row."""
    return fs(fs(ld(y, i), ld(fw, r + i)), ld(fw, u + i))


@always_inline
def q_prox(vv: Float32, up: Float32, lo: Float32) -> Float32:
    """The pinball prox of one row."""
    if vv > up:
        return fs(vv, up)
    if vv < -lo:
        return fa(vv, lo)
    return Float32(0)


def chol_solve_cols(l: FP, loff: Int, m: Int, b: FP, boff: Int, w: FP, woff: Int):
    """Solve L L' x = b, column oriented; x lands in b, w is scratch (m).
    Forward: y_k = b_k / L_kk, then b_i -= L_ik y_k for i > k (each b_i
    sees k ascending: the row form's words). Back: x_k = y_k / L_kk, then
    y_i -= L_ki x_k for i < k, k descending. `t_chol_solve_cols` runs the
    same statements with the i of a column across a block."""
    for k in range(m):
        var yk = fd(ld(b, boff + k), ld(l, loff + k * m + k))
        st(w, woff + k, yk)
        for i in range(k + 1, m):
            st(b, boff + i, fs(ld(b, boff + i), fm(ld(l, loff + i * m + k), yk)))
    var k = m - 1
    while k >= 0:
        var xk = fd(ld(w, woff + k), ld(l, loff + k * m + k))
        st(b, boff + k, xk)
        for i in range(k):
            st(w, woff + i, fs(ld(w, woff + i), fm(ld(l, loff + k * m + i), xk)))
        k -= 1


def t_chol_solve_cols(t: Team, l: FP, loff: Int, m: Int, b: FP, boff: Int, w: FP, woff: Int):
    """`chol_solve_cols` on a team: column k's updates across the team, one
    barrier per column; every thread computes the pivot quotient from the
    same words."""
    for k in range(m):
        var yk = fd(ld(b, boff + k), ld(l, loff + k * m + k))
        if t.lead():
            st(w, woff + k, yk)
        for i in range(k + 1 + t.tid, m, t.nt):
            st(b, boff + i, fs(ld(b, boff + i), fm(ld(l, loff + i * m + k), yk)))
        t.sync()
    var k = m - 1
    while k >= 0:
        var xk = fd(ld(w, woff + k), ld(l, loff + k * m + k))
        if t.lead():
            st(b, boff + k, xk)
        for i in range(t.tid, k, t.nt):
            st(w, woff + i, fs(ld(w, woff + i), fm(ld(l, loff + k * m + i), xk)))
        t.sync()
        k -= 1


@always_inline
def q_layout(n: Int, d: Int, m: Int) -> InlineArray[Int, 12]:
    """fw offsets: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d |
    v d | next rhs m | dsq d; [11] the total."""
    var o = InlineArray[Int, 12](fill=0)
    o[0] = 0
    o[1] = m * m
    o[2] = o[1] + m
    o[3] = o[2] + m
    o[4] = o[3] + n
    o[5] = o[4] + n
    o[6] = o[5] + n
    o[7] = o[6] + n
    o[8] = o[7] + d
    o[9] = o[8] + d
    o[10] = o[9] + m
    o[11] = o[10] + d
    return o


def q_start(nf: Float32, sw: Bool, den_sum: Float32, spread_sum: Float32, ysq: Float32) -> InlineArray[Float32, 3]:
    """(den, rho, ||y||) from the folded sums (nf = the row count as a
    float): den = n or sum(w), rho = 1 / (n * max(mean |y - ym|, 1e-6)),
    the target norm."""
    var r = InlineArray[Float32, 3](fill=Float32(0))
    r[0] = den_sum if sw else nf
    var spread = fmax(fd(spread_sum, nf), Float32(1e-6))
    r[1] = fd(Float32(1), fm(nf, spread))
    r[2] = fsqrt(ysq)
    return r


def q_tail(fw: FP, nd: Int, d: Int, m: Int, beta: Int, rhs: Int, z: Int, v: Int, dq: Int,
           abn: Float32, rn: Float32, prim_rows: Float32, un_rows: Float32, ynorm: Float32, rho: Float32,
           alpha: Float32, eps_abs: Float32, eps_rel: Float32, rel_bal: Bool, it: Int) -> Tuple[Int, Float32]:
    """One iteration after its row folds (A' dr folded into rhs): the z- and
    v-updates, the residuals, the stop rule and the balancing. Returns
    (0 go on | 1 converged | 2 rescaled, the rho factor); a rescale has
    already scaled v by 1 / factor (the caller scales u and rho). nd = n + d."""
    var zdiff = Float32(0)
    var wn = Float32(0)
    var zn = Float32(0)
    # the column-equilibrated split: threshold alpha / (rho d_j^2), norms of D w, D z, D dz
    for j in range(d):
        var dj = ld(fw, dq + j)
        var wj = ld(fw, beta + j)
        wn = fmad(fm(dj, wj), wj, wn)
        var nz = _soft(fa(wj, ld(fw, v + j)), fd(alpha, fm(rho, dj)))
        var dzv = fs(nz, ld(fw, z + j))
        zdiff = fmad(fm(dj, dzv), dzv, zdiff)
        st(fw, z + j, nz)
        zn = fmad(fm(dj, nz), nz, zn)
    # dual updates and the primal residual (the row parts folded already)
    var prim = prim_rows
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
    var eps_p = fa(fm(eps_abs, fsqrt(i2f(nd))), fm(eps_rel, scale_p))
    var un = un_rows
    for j in range(d):
        un = fmad(fm(ld(fw, dq + j), ld(fw, v + j)), ld(fw, v + j), un)
    var eps_d = fa(fm(eps_abs, fsqrt(i2f(m))), fm(fm(eps_rel, rho), fsqrt(un)))
    if prim_n <= eps_p and dual_n <= eps_d:
        return (1, Float32(1))
    if (it + 1) % 10 == 0:
        var factor = _balance(prim_n, dual_n, eps_p, eps_d, rel_bal)
        if factor != 0:
            var inv = fd(Float32(1), factor)
            for j in range(d):
                st(fw, v + j, fm(ld(fw, v + j), inv))
            return (2, factor)
    return (0, Float32(1))


def _quantile_fit_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """The host column: the blocked schedule (see the module note) one row
    block at a time, the r-update mapped, then ONE fold pass per iteration
    for every row accumulator (next iteration's A'(y - r - u) folded with it
    unless u is rescaled).
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
    var rel_bal = ld(fp, 4) != Float32(0)
    var sw = ldi(ip, 2) != 0
    var m = d + 1 if fi else d
    var o = q_layout(n, d, m)
    var mm = o[0]
    var beta = o[1]
    var rhs = o[2]
    var r = o[3]
    var u = o[4]
    var ab = o[5]
    var tmp = o[6]
    var z = o[7]
    var v = o[8]
    var nrhs = o[9]
    var dq = o[10]
    var rhs_ready = False
    # block partials: the lower triangle of A'A, then two m-vectors
    var gs = List[Float32](length=max(m * m, 1), fill=Float32(0))
    var ps = List[Float32](length=max(2 * m, 1), fill=Float32(0))
    var gp = FP(unsafe_from_address=Int(gs.unsafe_ptr()))
    var pp = FP(unsafe_from_address=Int(ps.unsafe_ptr()))
    # M = A'A + D^2: each block's lower triangle from zero (each entry its own
    # accumulator, rows ascending), the blocks folded ascending
    fill(fw, mm, m * m, Float32(0))
    var lo = 0
    while lo < n:
        var hi = min(lo + FOLD_BLOCK, n)
        fill(gp, 0, m * m, Float32(0))
        for i in range(lo, hi):
            for j in range(d):
                axpy_acc(gp, j * m, ld(x, i * d + j), x, i * d, j + 1)
            if fi:
                axpy_acc(gp, d * m, Float32(1), x, i * d, d)
                st(gp, d * m + d, fmad(Float32(1), Float32(1), ld(gp, d * m + d)))
        add_acc(fw, mm, gp, 0, m * m)
        lo = hi
    for j in range(m):
        for k in range(j + 1):
            q_gram_cell(fw, mm, m, d, dq, j, k, ld(fw, mm + j * m + k))
    _ = cholesky(fw, mm, m)
    fill(fw, beta, m, Float32(0))
    fill(fw, r, n, Float32(0))
    fill(fw, u, n, Float32(0))
    fill(fw, z, d, Float32(0))
    fill(fw, v, d, Float32(0))
    var den_sum = fold_fa_blocked(y, n, 1, n) if sw else Float32(0)
    var ym = fd(fold_fa_blocked(y, 0, 1, n), i2f(n))
    var spread_sum = Float32(0)
    var ysq = Float32(0)
    lo = 0
    while lo < n:
        var cnt = min(FOLD_BLOCK, n - lo)
        spread_sum = fa(spread_sum, q_spread_part(y, ym, lo, cnt))
        ysq = fa(ysq, fold_sq(y, lo, cnt))
        lo += FOLD_BLOCK
    var s3 = q_start(i2f(n), sw, den_sum, spread_sum, ysq)
    var den = s3[0]
    var rho = s3[1]
    var ynorm = s3[2]
    var iters = 0
    var converged = False
    for it in range(max_iter):
        iters = it + 1
        # beta-update: the m sums of A'(y - r - u), blocked (already folded by
        # the previous iteration's pass unless u was rescaled)
        if rhs_ready:
            copy(fw, rhs, fw, nrhs, m)
        else:
            fill(fw, rhs, m, Float32(0))
            lo = 0
            while lo < n:
                var hi = min(lo + FOLD_BLOCK, n)
                fill(pp, 0, m, Float32(0))
                for i in range(lo, hi):
                    var t = q_resid(y, fw, r, u, i)
                    axpy_acc(pp, 0, t, x, i * d, d)
                    if fi:
                        st(pp, d, fmad(Float32(1), t, ld(pp, d)))
                add_acc(fw, rhs, pp, 0, m)
                lo = hi
        for j in range(d):
            st(fw, rhs + j, fa(ld(fw, rhs + j), fm(ld(fw, dq + j), fs(ld(fw, z + j), ld(fw, v + j)))))
        chol_solve_cols(fw, mm, m, fw, rhs, fw, beta)
        copy(fw, beta, fw, rhs, m)
        var b = ld(fw, beta + d) if fi else Float32(0)
        # r-update (keep the old r in tmp for the dual residual)
        var kq = fd(Float32(1), fm(den, rho))
        var up = fm(q, kq)
        var lo_ = fm(fs(Float32(1), q), kq)
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
                var nr = q_prox(vv, upi, loi)
                st(fwp, tmp + i, fs(nr, ld(fwp, r + i)))
                st(fwp, r + i, nr)

        par_rows(rows_map, n)
        # ONE fold pass, blocked: |A beta|^2, |r|^2, the primal residual's row
        # part and the dual update u, |u|^2's row part, A' dr (into rhs, free
        # now) and next iteration's A'(y - r - u); each block from zero
        var abn = Float32(0)
        var rn = Float32(0)
        var prim = Float32(0)
        var un = Float32(0)
        fill(fw, rhs, m, Float32(0))
        fill(fw, nrhs, m, Float32(0))
        lo = 0
        while lo < n:
            var hi = min(lo + FOLD_BLOCK, n)
            var pa = Float32(0)
            var pr_ = Float32(0)
            var pq = Float32(0)
            var pu = Float32(0)
            fill(pp, 0, 2 * m, Float32(0))
            for i in range(lo, hi):
                var abi = ld(fw, ab + i)
                pa = fmad(abi, abi, pa)
                var nr = ld(fw, r + i)
                pr_ = fmad(nr, nr, pr_)
                var pr = fs(fa(abi, nr), ld(y, i))
                pq = fmad(pr, pr, pq)
                var ui = fa(ld(fw, u + i), pr)
                st(fw, u + i, ui)
                pu = fmad(ui, ui, pu)
                var ti = ld(fw, tmp + i)
                axpy_acc(pp, 0, ti, x, i * d, d)
                var t2 = q_resid(y, fw, r, u, i)
                axpy_acc(pp, m, t2, x, i * d, d)
                if fi:
                    st(pp, d, fmad(Float32(1), ti, ld(pp, d)))
                    st(pp, m + d, fmad(Float32(1), t2, ld(pp, m + d)))
            abn = fa(abn, pa)
            rn = fa(rn, pr_)
            prim = fa(prim, pq)
            un = fa(un, pu)
            add_acc(fw, rhs, pp, 0, m)
            add_acc(fw, nrhs, pp, m, m)
            lo = hi
        rhs_ready = True
        var tl = q_tail(fw, n + d, d, m, beta, rhs, z, v, dq, abn, rn, prim, un, ynorm, rho,
                        alpha, eps_abs, eps_rel, rel_bal, it)
        if tl[0] == 1:
            converged = True
            break
        if tl[0] == 2:
            rho = fm(rho, tl[1])
            var inv = fd(Float32(1), tl[1])
            for i in range(n):
                st(fw, u + i, fm(ld(fw, u + i), inv))
            rhs_ready = False
    copy(res, 0, fw, z, d)
    st(res, d, ld(fw, beta + d) if fi else Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))


def quantile_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d | next rhs m |
    dsq d. The host column's entry; the device binding runs
    x_linear/quantile_grid.mojo `quantile_fit_grid` (the same folds on the grid)."""
    comptime if not is_gpu():
        _quantile_fit_host(x, y, n, d, ip, fp, res, fw, iw)
