# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QuantileRegressor's fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/quantile.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/ops.mojo) and no
GPU-reachable module calls host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `quantile_fit_grid` in x_linear/quantile_grid.mojo); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from x_linear.tops import FOLD_BLOCK, fold_fa_blocked, fold_sq
from x_linear.ops_host import par_rows
from x_linear.ops import (
    FP,
    IP,
    add_acc,
    axpy_acc,
    cholesky,
    copy,
    fa,
    fd,
    fill,
    fm,
    fmad,
    fs,
    i2f,
    ld,
    ldi,
    row_dots,
    st,
)
from x_linear.team import Team
from std.sys.info import is_gpu
from x_linear.quantile import (
    _quantile_polish_host,
    chol_solve_cols,
    q_gram_cell,
    q_layout,
    q_prox,
    q_resid,
    q_spread_part,
    q_start,
    q_tail,
)


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
    # the vertex polish (lane sk-weq-linear): the ADMM answer onto the LP vertex
    # when that does not raise the objective
    _quantile_polish_host(x, y, n, d, fi, sw, q, alpha, den, res)


def quantile_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [quantile, alpha, eps_abs, eps_rel].
    fw: M m*m | beta m | rhs m | r n | u n | ab n | tmp n | z d | v d | next rhs m |
    dsq d. The host column's entry; the device binding runs
    x_linear/quantile_grid.mojo `quantile_fit_grid` (the same folds on the grid)."""
    comptime if not is_gpu():
        _quantile_fit_host(x, y, n, d, ip, fp, res, fw, iw)
