# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ridge's fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/ridge.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/ops.mojo) and no
GPU-reachable module calls host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `_ridge_device` and x_linear/ridge_grid.mojo); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from x_linear.tops import FOLD_BLOCK, fold_fa_blocked
from x_linear.ops import par_rows
from x_linear.ops import (
    FP,
    IP,
    add_acc,
    axpy_acc,
    axpy_centered,
    centered_gram,
    chol_solve,
    cholesky,
    copy,
    fa,
    fd,
    fill,
    fm,
    fs,
    i2f,
    ld,
    ldi,
    st,
)
from x_linear.team import Team
from std.sys.info import is_gpu
from x_linear.ridge import (
    BIG_ERR,
    _chol_trusted,
    _loo_rows,
    _ridge_ff_host,
    _ridge_solve_best,
    ridge_err_part,
)


def _ridge_fit_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_targets T, fit_intercept, n_alphas A, sample_weight]; fp: alphas (A).
    With sample_weight, y = targets n*T | weights n (weighted means, the
    weighted Gram, and their weighted GCV errors w_i e_i^2 / (1 - h_i)^2).
    y: n x T row-major. A == 1: fit; A > 1 (T == 1): leave-one-out choice.
    res: coef T*d | intercept T | alpha | best_score | A mean squared LOO errors.
    fw: see ridge_fit. The host schedule (lane linear-cpu): one row pass per
    block for the means, Gram and X'Y, the leave-one-out rows mapped."""
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
    var la = zz + d
    var lb = la + n
    var sw = ldi(ip, 3) != 0
    var wo = n * t_n
    var wsum = Float32(0)
    if sw:
        wsum = fold_fa_blocked(y, wo, 1, n)
    fill(fw, xm, d, Float32(0))
    fill(fw, ym, t_n, Float32(0))
    fill(fw, xty, d * t_n, Float32(0))
    if sw:
        # cgr-linear: the weighted statistics in the blocked order (see
        # `ridge_w_part`), each block's partials from zero, added ascending
        var scr = List[Float32](length=d * d + d * t_n + d + t_n, fill=Float32(0))
        var sp = FP(unsafe_from_address=Int(scr.unsafe_ptr()))
        if fi:
            var lo = 0
            while lo < n:
                var hi = min(lo + FOLD_BLOCK, n)
                fill(sp, 0, d + t_n, Float32(0))
                for i in range(lo, hi):
                    var wi = ld(y, wo + i)
                    axpy_acc(sp, 0, wi, x, i * d, d)
                    axpy_acc(sp, d, wi, y, i * t_n, t_n)
                add_acc(fw, xm, sp, 0, d)
                add_acc(fw, ym, sp, d, t_n)
                lo = hi
            for j in range(d):
                st(fw, xm + j, fd(ld(fw, xm + j), wsum))
            for tt in range(t_n):
                st(fw, ym + tt, fd(ld(fw, ym + tt), wsum))
        fill(fw, gg, d * d, Float32(0))
        var lo2 = 0
        while lo2 < n:
            var hi = min(lo2 + FOLD_BLOCK, n)
            fill(sp, 0, d * d + d * t_n, Float32(0))
            for i in range(lo2, hi):
                var wi = ld(y, wo + i)
                for j in range(d):
                    var a = fm(wi, fs(ld(x, i * d + j), ld(fw, xm + j)))
                    axpy_centered(sp, j * d + j, a, x, i * d + j, fw, xm + j, d - j)
                for tt in range(t_n):
                    var b = fs(ld(y, i * t_n + tt), ld(fw, ym + tt))
                    axpy_centered[True](sp, d * d + tt * d, b, x, i * d, fw, xm, d, wi)
            add_acc(fw, gg, sp, 0, d * d)
            add_acc(fw, xty, sp, d * d, d * t_n)
            lo2 = hi
        for j in range(d):
            for k in range(j + 1, d):
                st(fw, gg + k * d + j, ld(fw, gg + j * d + k))
        _ = scr^
    else:
        # one pass over the rows; every mean, Gram entry and X'y entry is its
        # own accumulator, rows ascending (lane linear-cpu; the device's
        # moments grid folds the same chains)
        if fi:
            for i in range(n):
                add_acc(fw, xm, x, i * d, d)
                add_acc(fw, ym, y, i * t_n, t_n)
            for j in range(d):
                st(fw, xm + j, fd(ld(fw, xm + j), i2f(n)))
            for tt in range(t_n):
                st(fw, ym + tt, fd(ld(fw, ym + tt), i2f(n)))
        centered_gram(x, n, d, fw, xm, fw, gg)
        for i in range(n):
            for tt in range(t_n):
                var b = fs(ld(y, i * t_n + tt), ld(fw, ym + tt))
                axpy_centered(fw, xty + tt * d, b, x, i * d, fw, xm, d)
    var best = 0
    var best_err = Float32(0)
    if a_n > 1:
        for a in range(a_n):
            var alpha = ld(fp, a)
            copy(fw, mm, fw, gg, d * d)
            for j in range(d):
                st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
            var okf = cholesky(fw, mm, d)
            if not _chol_trusted(okf, fw, mm, gg, d, alpha):
                # (lane/neural-pass93) no LOO error from an untrusted factor
                st(res, t_n * d + t_n + 2 + a, BIG_ERR)
                if a == 0:
                    best_err = BIG_ERR
                continue
            copy(fw, rhs, fw, xty, d)
            chol_solve(fw, mm, d, fw, rhs)
            # map: each row's leave-one-out residual (its own solve), then
            # the fold of err rows ascending (lane linear-cpu)
            var fwp = fw

            def rows_loo(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm fwp, imm ym, imm xm, imm rhs,
                                            imm zz, imm mm, imm sw, imm fi, imm wo, imm wsum, imm la, imm lb}:
                comptime if is_gpu():
                    _loo_rows(x, y, n, d, fwp, ym, xm, rhs, fwp + zz, mm, sw, fi, wo, wsum, fwp + la, fwp + lb, lo, hi)
                else:
                    var zb = List[Float32](length=max(d, 1), fill=Float32(0))
                    _loo_rows(x, y, n, d, fwp, ym, xm, rhs, FP(unsafe_from_address=Int(zb.unsafe_ptr())), mm, sw, fi,
                              wo, wsum, fwp + la, fwp + lb, lo, hi)
                    _ = zb^

            par_rows(rows_loo, n)
            # the error in the blocked order (cgr-linear)
            var err = Float32(0)
            var lo3 = 0
            while lo3 < n:
                err = fa(err, ridge_err_part(fw + la, fw + lb, lo3, min(FOLD_BLOCK, n - lo3)))
                lo3 += FOLD_BLOCK
            err = fd(err, i2f(n))
            st(res, t_n * d + t_n + 2 + a, err)
            if a == 0 or err < best_err:  # DEVIATION 5005: the first minimum
                best = a
                best_err = err
    if not _ridge_solve_best(fp, res, fw, d, t_n, fi, best, best_err, a_n):
        _ridge_ff_host(x, y, n, d, t_n, fi, sw, wo, ld(fp, best), res, a_n)


def ridge_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_targets T, fit_intercept, n_alphas A, sample_weight]; fp: alphas (A).
    With sample_weight, y = targets n*T | weights n (weighted means, the
    weighted Gram, and their weighted GCV errors w_i e_i^2 / (1 - h_i)^2).
    y: n x T row-major. A == 1: fit; A > 1 (T == 1): leave-one-out choice.
    res: coef T*d | intercept T | alpha | best_score | A mean squared LOO errors.
    fw: xm d | G d*d | M d*d | rhs d | ym T | xty d*T | z d | loo terms 2n
    (z and the loo terms: the host schedule's; the team keeps them in its
    own words and row buffer).
    The host column's entry; the device binding runs x_linear/ridge_grid.mojo
    (the statistics, every leave-one-out row and the error folds on the grid,
    the factors on one block team), the same words."""
    comptime if not is_gpu():
        _ridge_fit_host(x, y, n, d, ip, fp, res, fw, iw)
