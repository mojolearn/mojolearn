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
from experiments.classical_identical_ideas.linear_controls import C14_GROUP_RHS, RIDGECV_FF_BLOCKED
from x_linear.ridge_ff_blocks import rff_stats_host
from x_linear.ops import chol_solve_group
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, ld, st, ldi, i2f, fill, copy, cholesky, chol_solve, centered_gram,
    axpy_acc, add_acc, axpy_centered, seq_rows,
)
from std.sys.info import is_gpu
from x_linear.team import Team
from x_linear.ff import FF, ff_of, ff_add, ff_add_f, ff_sub, ff_mul, ff_f32, ff_ld, ff_st, ff_cholesky, ff_chol_solve, ff_col_mean, ff_cross
from std.sys.compile import is_defined
from x_linear.tops import upper_cell, t_centered_gram, t_sum, fold_fa, fold_sq, chain_fmad, chain_cfmad, t_cholesky, fold_fa_blocked, FOLD_BLOCK


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


def chol_trusted(ok: Bool, l: FP, g: FP, d: Int, alpha: Float32) -> Bool:
    """Whether a float32 factor l of G + alpha I (G at g) may be used: it
    factored and no pivot kept less than RIDGE_FF_GATE of its diagonal."""
    comptime if RIDGE_FF_ALWAYS:
        return False
    if not ok:
        return False
    for j in range(d):
        var lj = ld(l, j * d + j)
        if fm(lj, lj) < fm(fa(ld(g, j * d + j), alpha), RIDGE_FF_GATE):
            return False
    return True


@always_inline
def _chol_trusted(ok: Bool, fw: FP, mm: Int, gg: Int, d: Int, alpha: Float32) -> Bool:
    return chol_trusted(ok, fw + mm, fw + gg, d, alpha)


def ridge_ff_units(n: Int, d: Int, t_n: Int) -> Int:
    """Units of the float-float statistics: d + t_n means, then the upper
    Gram cells, then d * t_n X'Y cells."""
    return d + t_n + d * (d + 1) // 2 + d * t_n


def ridge_ff_unit(u: Int, x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, wo: Int,
                  st_h: FP, st_l: FP, s: Int = 0, e: Int = 0):
    """Unit u of the float-float statistics into st_h / st_l: [xm d | ym T]
    then [G d*d] then [X'Y d*T]. The means units must all be done before
    any cell unit starts (two passes)."""
    # the rows outside [s, e) (k-fold RidgeCV's training rows; all when s == e)
    var wsum = ff_of(i2f(n - (e - s)))
    if sw:
        wsum = ff_of(Float32(0))
        for i in range(n):
            if i >= s and i < e:
                continue
            wsum = ff_add_f(wsum, ld(y, wo + i))
    var gofs = d + t_n
    var xofs = gofs + d * d
    if u < d + t_n:
        var m = ff_of(Float32(0))
        if fi:
            if u < d:
                m = ff_col_mean(x, d, u, n, s, e, y, wo, sw, wsum)
            else:
                m = ff_col_mean(y, t_n, u - d, n, s, e, y, wo, sw, wsum)
        ff_st(st_h, st_l, u, m)
        return
    var c = u - (d + t_n)
    var cells = d * (d + 1) // 2
    if c < cells:
        var jk = upper_cell(c, d)
        var j = jk[0]
        var k = jk[1]
        var v = ff_cross(x, d, j, ff_ld(st_h, st_l, j), x, d, k, ff_ld(st_h, st_l, k), n, s, e, y, wo, sw)
        ff_st(st_h, st_l, gofs + j * d + k, v)
        ff_st(st_h, st_l, gofs + k * d + j, v)
        return
    var q = c - cells
    var j = q // t_n
    var tt = q - j * t_n
    var v = ff_cross(x, d, j, ff_ld(st_h, st_l, j), y, t_n, tt, ff_ld(st_h, st_l, d + tt), n, s, e, y, wo, sw)
    ff_st(st_h, st_l, xofs + j * t_n + tt, v)


def ridge_ff_solve(d: Int, t_n: Int, fi: Bool, alpha: Float32, st_h: FP, st_l: FP, bh: FP, bl: FP,
                   res: FP, fh: FP, fl: FP) -> Bool:
    """(G + alpha I) W = X'Y in float-float from ridge_ff_unit's statistics
    (G + alpha I factored into fh / fl, d*d words each, the statistics kept);
    coef and intercept words into res (T*d | T). False when float-float
    cannot factor it either. bh, bl: d words."""
    var gofs = d + t_n
    var xofs = gofs + d * d
    for q in range(d * d):
        ff_st(fh, fl, q, ff_ld(st_h, st_l, gofs + q))
    for j in range(d):
        ff_st(fh, fl, j * d + j, ff_add_f(ff_ld(fh, fl, j * d + j), alpha))
    if not ff_cholesky(fh, fl, d):
        return False
    for tt in range(t_n):
        for j in range(d):
            ff_st(bh, bl, j, ff_ld(st_h, st_l, xofs + j * t_n + tt))
        ff_chol_solve(fh, fl, d, bh, bl)
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
    comptime if C14_GROUP_RHS:
        copy(res, 0, fw, xty, t_n * d)
        for first in range(0, t_n, 4):
            chol_solve_group(fw, mm, d, res, first, min(4, t_n - first))
        for tt in range(t_n):
            var acc = Float32(0)
            for j in range(d):
                acc = fmad(ld(fw, xm + j), ld(res, tt * d + j), acc)
            st(res, t_n * d + tt, fs(ld(fw, ym + tt), acc) if fi else Float32(0))
        return True
    for tt in range(t_n):
        copy(fw, rhs, fw, xty + tt * d, d)
        chol_solve(fw, mm, d, fw, rhs)
        copy(res, tt * d, fw, rhs, d)
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, xm + j), ld(fw, rhs + j), acc)
        st(res, t_n * d + tt, fs(ld(fw, ym + tt), acc) if fi else Float32(0))
    return True


# ------------------------------------------------ the blocked weighted statistics (cgr-linear)
# With sample weights (w at y + wo, wo = n * T) every statistic folds
# FOLD_BLOCK rows from zero, rows ascending, then the block partials ascending:
# the host below one block at a time, x_linear/ridge_grid.mojo a thread per
# (statistic, block) then a thread per statistic.
@always_inline
def ridge_w_part(x: FP, y: FP, d: Int, t_n: Int, wo: Int, c: Int, lo: Int, cnt: Int) -> Float32:
    """Column c of [X | Y | 1] weighted: sum w_i v_ic from zero (c = d + T:
    sum w_i, `fa` steps)."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        var wi = ld(y, wo + i)
        if c < d:
            acc = fmad(wi, ld(x, i * d + c), acc)
        elif c < d + t_n:
            acc = fmad(wi, ld(y, i * t_n + (c - d)), acc)
        else:
            acc = fa(acc, wi)
    return acc


@always_inline
def ridge_wgram_part(x: FP, y: FP, d: Int, wo: Int, fw: FP, xm: Int, j: Int, k: Int, lo: Int, cnt: Int) -> Float32:
    """sum (w_i (x_ij - xm_j)) (x_ik - xm_k) from zero."""
    var mj = ld(fw, xm + j)
    var mk = ld(fw, xm + k)
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(fm(ld(y, wo + i), fs(ld(x, i * d + j), mj)), fs(ld(x, i * d + k), mk), acc)
    return acc


@always_inline
def ridge_wxty_part(x: FP, y: FP, d: Int, t_n: Int, wo: Int, fw: FP, xm: Int, ym: Int, tt: Int, j: Int,
                    lo: Int, cnt: Int) -> Float32:
    """sum (y_it - ym_t) (w_i (x_ij - xm_j)) from zero."""
    var mj = ld(fw, xm + j)
    var mt = ld(fw, ym + tt)
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(fs(ld(y, i * t_n + tt), mt), fm(ld(y, wo + i), fs(ld(x, i * d + j), mj)), acc)
    return acc


@always_inline
def ridge_err_part(la: FP, lb: FP, lo: Int, cnt: Int) -> Float32:
    """sum la_i lb_i from zero (the leave-one-out error's block)."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(ld(la, i), ld(lb, i), acc)
    return acc


def t_ridge_solve_best(t: Team, fp: FP, res: FP, fw: FP, d: Int, t_n: Int, fi: Bool, best: Int,
                       best_err: Float32, a_n: Int, scr: FP) -> Bool:
    """`_ridge_solve_best` on a team: the factor by `t_cholesky`, then one
    target a thread (each `chol_solve` into its own d words of scr); the
    same statements per value. The lead's verdict to every thread."""
    var gg = d
    var mm = gg + d * d
    var ym = mm + d * d + d
    var xty = ym + t_n
    var alpha = ld(fp, best)
    for c in range(t.tid, d * d, t.nt):
        var v = ld(fw, gg + c)
        if c // d == c % d:
            v = fa(v, alpha)
        st(fw, mm + c, v)
    t.sync()
    var ok = t_cholesky(t, fw, mm, d)
    var trusted = 0
    if t.lead():
        st(res, t_n * d + t_n, alpha)
        st(res, t_n * d + t_n + 1, -best_err)
        trusted = 1 if _chol_trusted(ok, fw, mm, gg, d, alpha) else 0
        st(res, t_n * d + t_n + 2 + a_n, Float32(0) if trusted == 1 else Float32(1))
    if t.bcast_int(trusted, 1) == 0:
        return False
    comptime if C14_GROUP_RHS:
        for tt in range(t.tid, t_n, t.nt):
            copy(res, tt * d, fw, xty + tt * d, d)
        t.sync()
        for group in range(t.tid, (t_n + 3) // 4, t.nt):
            var first = group * 4
            chol_solve_group(fw, mm, d, res, first, min(4, t_n - first))
            for tt in range(first, min(first + 4, t_n)):
                var acc = Float32(0)
                for j in range(d):
                    acc = fmad(ld(fw, j), ld(res, tt * d + j), acc)
                st(res, t_n * d + tt, fs(ld(fw, ym + tt), acc) if fi else Float32(0))
        t.sync()
        return True
    for tt in range(t.tid, t_n, t.nt):
        var z = scr + tt * d
        copy(z, 0, fw, xty + tt * d, d)
        chol_solve(fw, mm, d, z, 0)
        copy(res, tt * d, z, 0, d)
        var acc = Float32(0)
        for j in range(d):
            acc = fmad(ld(fw, j), ld(z, j), acc)
        st(res, t_n * d + tt, fs(ld(fw, ym + tt), acc) if fi else Float32(0))
    t.sync()
    return True


def _ridge_ff_host(x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, wo: Int, alpha: Float32,
                   res: FP, a_n: Int):
    """The float-float refit on the host: the means, then the cells, as
    independent units on the pool; the solve on the calling thread. Status
    2 in res when float-float cannot factor it either."""
    var units = ridge_ff_units(n, d, t_n)
    var nm = d + t_n
    var words = d + t_n + d * d + d * t_n
    var hb = List[Float32](length=2 * words + 2 * d + 2 * d * d, fill=Float32(0))
    var sh = FP(unsafe_from_address=Int(hb.unsafe_ptr()))
    var sl = sh + words
    var bh = sl + words
    var bl = bh + d
    var fh = bl + d
    var fl = fh + d * d

    def means(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm t_n, imm fi, imm sw, imm wo, imm sh, imm sl}:
        for u in range(lo, hi):
            ridge_ff_unit(u, x, y, n, d, t_n, fi, sw, wo, sh, sl)

    def cells(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm t_n, imm fi, imm sw, imm wo, imm sh, imm sl, imm nm}:
        for u in range(lo, hi):
            ridge_ff_unit(nm + u, x, y, n, d, t_n, fi, sw, wo, sh, sl)

    # RIDGECV_FF_BLOCKED (lane/classical-cv-folds): the statistics as the
    # device's block partials folded blocks ascending
    # (x_linear/ridge_ff_blocks.mojo); the solve's words are the team's
    comptime if RIDGECV_FF_BLOCKED:
        rff_stats_host(x, y, n, d, t_n, fi, sw, wo, 0, 0, sh, sl)
    else:
        seq_rows(means, nm, 1)
        seq_rows(cells, units - nm, 1)
    var ok = ridge_ff_solve(d, t_n, fi, alpha, sh, sl, bh, bl, res, fh, fl)
    st(res, t_n * d + t_n + 2 + a_n, Float32(0) if ok else Float32(2))
    _ = hb^