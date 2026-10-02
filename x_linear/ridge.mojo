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
    axpy_acc, add_acc, axpy_centered, par_rows,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.ff import FF, ff_of, ff_add, ff_add_f, ff_sub, ff_mul, ff_f32, ff_ld, ff_st, ff_cholesky, ff_chol_solve, ff_col_mean, ff_cross
from std.sys.compile import is_defined
from x_linear.tops import upper_cell, t_centered_gram, t_sum, fold_fa, fold_sq, chain_fmad, chain_cfmad


def _loo_rows(x: FP, y: FP, n: Int, d: Int, fw: FP, ym: Int, xm: Int, rhs: Int, zb: FP, mm: Int,
              sw: Bool, fi: Bool, wo: Int, wsum: Float32, la: FP, lb: FP, lo: Int, hi: Int):
    """Rows [lo, hi) of RidgeCV's leave-one-out: the fold term of row i is
    la[i] * lb[i] (w_i loo * loo with sample weights, loo * loo without);
    zb is this block's own solve scratch (d)."""
    for i in range(lo, hi):
        var e = fs(ld(y, i), ld(fw, ym))
        for j in range(d):
            var xc = fs(ld(x, i * d + j), ld(fw, xm + j))
            st(zb, j, xc)
            e = fs(e, fm(xc, ld(fw, rhs + j)))
        chol_solve(fw, mm, d, zb, 0)
        if sw:
            # their GCV on the sqrt(w)-rescaled problem
            var wi = ld(y, wo + i)
            var q = Float32(0)
            for j in range(d):
                q = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(zb, j), q)
            var h = fm(wi, q)
            if fi:
                h = fa(fd(wi, wsum), h)
            var loo = fd(e, fs(Float32(1), h))
            st(la, i, fm(wi, loo))
            st(lb, i, loo)
        else:
            var h = fd(Float32(1), i2f(n)) if fi else Float32(0)
            for j in range(d):
                h = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(zb, j), h)
            var loo = fd(e, fs(Float32(1), h))
            st(la, i, loo)
            st(lb, i, loo)


# ------------------------------------------------ float-float fallback (lane/neural-pass93)
# Andrew, 2026-10-01: Ridge on a Gram float32 cannot factor (istella: the
# centered Gram's eigenvalues span 7.6e17, 21 zero columns) solves in
# float-float (x_linear/ff.mojo, about 48 bits) instead of returning a
# wrong answer: the means, the Gram and X'Y cells and the Cholesky solve,
# each a fixed chain of IDENTICAL float32 operations, the same words on the
# host and every device. The float32 path stays the default; the fit falls
# back when its Cholesky fails or a pivot keeps less than 2^-12 of its
# diagonal (more than 12 bits cancelled). A matrix float-float cannot
# factor either is refused (status 2). `-D MOJOLEARN_RIDGE_FF_ALWAYS=1`
# takes float-float for every fit (the A/B arm for its cost).
comptime BIG_ERR = Float32(3.0e38)
comptime RIDGE_FF_GATE = Float32(0.000244140625)  # 2^-12
comptime RIDGE_FF_ALWAYS = is_defined["MOJOLEARN_RIDGE_FF_ALWAYS"]()


def _chol_trusted(ok: Bool, fw: FP, mm: Int, gg: Int, d: Int, alpha: Float32) -> Bool:
    """Whether a float32 factor of G + alpha I (at mm) may be used."""
    comptime if RIDGE_FF_ALWAYS:
        return False
    if not ok:
        return False
    for j in range(d):
        var l = ld(fw, mm + j * d + j)
        if fm(l, l) < fm(fa(ld(fw, gg + j * d + j), alpha), RIDGE_FF_GATE):
            return False
    return True


def ridge_ff_units(n: Int, d: Int, t_n: Int) -> Int:
    """Units of the float-float statistics: d + t_n means, then the upper
    Gram cells, then d * t_n X'Y cells."""
    return d + t_n + d * (d + 1) // 2 + d * t_n


def ridge_ff_unit(u: Int, x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, wo: Int,
                  st_h: FP, st_l: FP):
    """Unit u of the float-float statistics into st_h / st_l: [xm d | ym T]
    then [G d*d] then [X'Y d*T]. The means units must all be done before
    any cell unit starts (two passes)."""
    var wsum = ff_of(i2f(n))
    if sw:
        wsum = ff_of(Float32(0))
        for i in range(n):
            wsum = ff_add_f(wsum, ld(y, wo + i))
    var gofs = d + t_n
    var xofs = gofs + d * d
    if u < d + t_n:
        var m = ff_of(Float32(0))
        if fi:
            if u < d:
                m = ff_col_mean(x, d, u, n, n, n, y, wo, sw, wsum)
            else:
                m = ff_col_mean(y, t_n, u - d, n, n, n, y, wo, sw, wsum)
        ff_st(st_h, st_l, u, m)
        return
    var c = u - (d + t_n)
    var cells = d * (d + 1) // 2
    if c < cells:
        var jk = upper_cell(c, d)
        var j = jk[0]
        var k = jk[1]
        var v = ff_cross(x, d, j, ff_ld(st_h, st_l, j), x, d, k, ff_ld(st_h, st_l, k), n, n, n, y, wo, sw)
        ff_st(st_h, st_l, gofs + j * d + k, v)
        ff_st(st_h, st_l, gofs + k * d + j, v)
        return
    var q = c - cells
    var j = q // t_n
    var tt = q - j * t_n
    var v = ff_cross(x, d, j, ff_ld(st_h, st_l, j), y, t_n, tt, ff_ld(st_h, st_l, d + tt), n, n, n, y, wo, sw)
    ff_st(st_h, st_l, xofs + j * t_n + tt, v)


def ridge_ff_solve(d: Int, t_n: Int, fi: Bool, alpha: Float32, st_h: FP, st_l: FP, bh: FP, bl: FP,
                   res: FP) -> Bool:
    """(G + alpha I) W = X'Y in float-float from ridge_ff_unit's statistics
    (G factored in place); coef and intercept words into res (T*d | T).
    False when float-float cannot factor it either. bh, bl: d words."""
    var gofs = d + t_n
    var xofs = gofs + d * d
    for j in range(d):
        ff_st(st_h, st_l, gofs + j * d + j, ff_add_f(ff_ld(st_h, st_l, gofs + j * d + j), alpha))
    if not ff_cholesky(st_h + gofs, st_l + gofs, d):
        return False
    for tt in range(t_n):
        for j in range(d):
            ff_st(bh, bl, j, ff_ld(st_h, st_l, xofs + j * t_n + tt))
        ff_chol_solve(st_h + gofs, st_l + gofs, d, bh, bl)
        var acc = ff_of(Float32(0))
        for j in range(d):
            st(res, tt * d + j, ff_f32(ff_ld(bh, bl, j)))
            acc = ff_add(acc, ff_mul(ff_ld(st_h, st_l, j), ff_ld(bh, bl, j)))
        st(res, t_n * d + tt, ff_f32(ff_sub(ff_ld(st_h, st_l, d + tt), acc)) if fi else Float32(0))
    return True


def _ridge_solve_best(fp: FP, res: FP, fw: FP, d: Int, t_n: Int, fi: Bool, best: Int, best_err: Float32,
                      a_n: Int) -> Bool:
    """The fit at the chosen alpha, every target, and the result words: the
    one tail of both schedules below (fw laid out as ridge_fit says).
    False (and status 1 in res[T*d + T + 2 + A]) when the float32 factor is
    not trusted: the caller refits in float-float (lane/neural-pass93)."""
    var xm = 0
    var gg = d
    var mm = gg + d * d
    var rhs = mm + d * d
    var ym = rhs + d
    var xty = ym + t_n
    var alpha = ld(fp, best)
    copy(fw, mm, fw, gg, d * d)
    for j in range(d):
        st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
    var ok = cholesky(fw, mm, d)
    st(res, t_n * d + t_n, alpha)
    st(res, t_n * d + t_n + 1, -best_err)
    if not _chol_trusted(ok, fw, mm, gg, d, alpha):
        st(res, t_n * d + t_n + 2 + a_n, Float32(1))
        return False
    st(res, t_n * d + t_n + 2 + a_n, Float32(0))
    for tt in range(t_n):
        copy(fw, rhs, fw, xty + tt * d, d)
        chol_solve(fw, mm, d, fw, rhs)
        copy(res, tt * d, fw, rhs, d)
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(fw, rhs + j), acc)
        st(res, t_n * d + tt, fs(ld(fw, ym + tt), acc) if fi else Float32(0))
    return True


def _ridge_fit_team(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """The team schedule (see ridge_fit).
    ip: [n_targets T, fit_intercept, n_alphas A, sample_weight]; fp: alphas (A).
    With sample_weight, y = targets n*T | weights n (weighted means, the
    weighted Gram, and their weighted GCV errors w_i e_i^2 / (1 - h_i)^2).
    y: n x T row-major. A == 1: fit; A > 1 (T == 1): leave-one-out choice.
    res: coef T*d | intercept T | alpha | best_score | A mean squared LOO errors.
    fw: xm d | G d*d | M d*d | rhs d | ym T | xty d*T | z d.
    Team form: means, Gram and X'Y one thread per output cell, the
    leave-one-out rows across the team (each thread solves into its own d
    words), the Cholesky solves and every fold of the LOO errors on the lead."""
    var t_n = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var a_n = ldi(ip, 2)
    var xm = 0
    var gg = d
    var mm = gg + d * d
    var rhs = mm + d * d
    var ym = rhs + d
    var xty = ym + t_n
    var sw = ldi(ip, 3) != 0
    var wo = n * t_n
    var wsum = Float32(0)
    if sw:
        wsum = t_sum(t, y + wo, n)
    for j in range(t.tid, d + t_n, t.nt):
        var acc = Float32(0)
        if j < d:
            if fi:
                if sw:
                    acc = fd(chain_fmad(y, wo, 1, x, j, d, n), wsum)
                else:
                    acc = fd(fold_fa(x, j, d, n), i2f(n))
            st(fw, xm + j, acc)
        else:
            var c = j - d
            if fi:
                if sw:
                    acc = fd(chain_fmad(y, wo, 1, y, c, t_n, n), wsum)
                else:
                    acc = fd(fold_fa(y, c, t_n, n), i2f(n))
            st(fw, ym + c, acc)
    t.sync()
    if sw:
        # sum_i w_i xc_i xc_i' (theirs: the sqrt(w) rescale of _rescale_data)
        var cells = d * (d + 1) // 2
        for c in range(t.tid, cells, t.nt):
            var jk = upper_cell(c, d)
            var j = jk[0]
            var k = jk[1]
            var acc = Float32(0)
            var mj = ld(fw, xm + j)
            var mk = ld(fw, xm + k)
            for i in range(n):
                acc = fmad(fm(ld(y, wo + i), fs(ld(x, i * d + j), mj)), fs(ld(x, i * d + k), mk), acc)
            st(fw, gg + j * d + k, acc)
            st(fw, gg + k * d + j, acc)
        t.sync()
    else:
        t_centered_gram(t, x, n, d, fw + xm, 0, fw, gg)
    for c in range(t.tid, t_n * d, t.nt):
        var tt = c // d
        var j = c - tt * d
        var ymt = ld(fw, ym + tt)
        var mj = ld(fw, xm + j)
        var acc = Float32(0)
        if sw:
            for i in range(n):
                var xc = fm(ld(y, wo + i), fs(ld(x, i * d + j), mj))
                acc = fmad(xc, fs(ld(y, i * t_n + tt), ymt), acc)
        else:
            acc = chain_cfmad(x, j, d, mj, y, tt, t_n, ymt, n)
        st(fw, xty + tt * d + j, acc)
    t.sync()
    var best = 0
    var best_err = Float32(0)
    if a_n > 1:
        var lr = t.row(0)
        var zz = t.own()
        for a in range(a_n):
            var alpha = ld(fp, a)
            var trusted = 0
            if t.lead():
                copy(fw, mm, fw, gg, d * d)
                for j in range(d):
                    st(fw, mm + j * d + j, fa(ld(fw, mm + j * d + j), alpha))
                var okf = cholesky(fw, mm, d)
                trusted = 1 if _chol_trusted(okf, fw, mm, gg, d, alpha) else 0
                copy(fw, rhs, fw, xty, d)
                chol_solve(fw, mm, d, fw, rhs)
            if t.bcast_int(trusted, 1) == 0:
                # (lane/neural-pass93) no LOO error from an untrusted factor
                if t.lead():
                    st(res, t_n * d + t_n + 2 + a, BIG_ERR)
                    if a == 0:
                        best_err = BIG_ERR
                t.sync()
                continue
            for i in range(t.tid, n, t.nt):
                var e = fs(ld(y, i), ld(fw, ym))
                for j in range(d):
                    var xc = fs(ld(x, i * d + j), ld(fw, xm + j))
                    st(zz, j, xc)
                    e = fs(e, fm(xc, ld(fw, rhs + j)))
                chol_solve(fw, mm, d, zz, 0)
                if sw:
                    # their GCV on the sqrt(w)-rescaled problem
                    var wi = ld(y, wo + i)
                    var q = Float32(0)
                    for j in range(d):
                        q = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(zz, j), q)
                    var h = fm(wi, q)
                    if fi:
                        h = fa(fd(wi, wsum), h)
                    st(lr, i, fd(e, fs(Float32(1), h)))
                else:
                    var h = fd(Float32(1), i2f(n)) if fi else Float32(0)
                    for j in range(d):
                        h = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(zz, j), h)
                    st(lr, i, fd(e, fs(Float32(1), h)))
            t.sync()
            if t.lead():
                var err = Float32(0)
                if sw:
                    for i in range(n):
                        var loo = ld(lr, i)
                        err = fmad(fm(ld(y, wo + i), loo), loo, err)
                else:
                    err = fold_sq(lr, 0, n)
                err = fd(err, i2f(n))
                st(res, t_n * d + t_n + 2 + a, err)
                if a == 0 or err < best_err:  # DEVIATION 5005: the first minimum
                    best = a
                    best_err = err
            t.sync()
    if not t.lead():
        return
    _ = _ridge_solve_best(fp, res, fw, d, t_n, fi, best, best_err, a_n)


def _ridge_ff_host(x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, wo: Int, alpha: Float32,
                   res: FP, a_n: Int):
    """The float-float refit on the host: the means, then the cells, as
    independent units on the pool; the solve on the calling thread. Status
    2 in res when float-float cannot factor it either."""
    var units = ridge_ff_units(n, d, t_n)
    var nm = d + t_n
    var words = d + t_n + d * d + d * t_n
    var hb = List[Float32](length=2 * words + 2 * d, fill=Float32(0))
    var sh = FP(unsafe_from_address=Int(hb.unsafe_ptr()))
    var sl = sh + words
    var bh = sl + words
    var bl = bh + d

    def means(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm t_n, imm fi, imm sw, imm wo, imm sh, imm sl}:
        for u in range(lo, hi):
            ridge_ff_unit(u, x, y, n, d, t_n, fi, sw, wo, sh, sl)

    def cells(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm t_n, imm fi, imm sw, imm wo, imm sh, imm sl, imm nm}:
        for u in range(lo, hi):
            ridge_ff_unit(nm + u, x, y, n, d, t_n, fi, sw, wo, sh, sl)

    par_rows(means, nm, 1)
    par_rows(cells, units - nm, 1)
    var ok = ridge_ff_solve(d, t_n, fi, alpha, sh, sl, bh, bl, res)
    st(res, t_n * d + t_n + 2 + a_n, Float32(0) if ok else Float32(2))
    _ = hb^


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
        for tt in range(t_n):
            st(fw, ym + tt, fd(ld(fw, ym + tt), den))
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
        for tt in range(t_n):
            var b = fs(ld(y, i * t_n + tt), ld(fw, ym + tt))
            if sw:
                axpy_centered[True](fw, xty + tt * d, b, x, i * d, fw, xm, d, wi)
            else:
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
            var err = Float32(0)
            for i in range(n):
                err = fmad(ld(fw, la + i), ld(fw, lb + i), err)
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
    Team form: means, Gram and X'Y one thread per output cell, the
    leave-one-out rows across the team (each thread solves into its own d
    words), the Cholesky solves and every fold of the LOO errors on the lead."""
    comptime if is_gpu():
        _ridge_fit_team(t, x, y, n, d, ip, fp, res, fw, iw)
    else:
        _ridge_fit_host(x, y, n, d, ip, fp, res, fw, iw)
