# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BayesianRidge and ARDRegression (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_bayes.py`:
  * `BayesianRidge.fit` / `_update_coef_`: the evidence iteration
    gamma = sum(alpha ev / (lambda + alpha ev)), lambda = (gamma + 2 l1) /
    (|w|^2 + 2 l2), alpha = (n - gamma + 2 a1) / (sse + 2 a2), stopping on
    sum|w_old - w| < tol, then one last coefficient update. Theirs takes the
    SVD of X; here the eigenpairs of the centered Gram X'X come from cyclic
    Jacobi (x_linear/ops.mojo `jacobi_eig`), ev = S^2, and
    w = V diag(1 / (ev + lambda/alpha)) V' X'y.
  * `ARDRegression.fit` / `_update_sigma`: sigma = inv(diag(lambda_keep) +
    alpha X_keep'X_keep) by Cholesky (theirs pinvh; the Woodbury branch for
    n < d is the same matrix), coef_keep = alpha sigma X_keep'y, pruning at
    lambda >= threshold_lambda, and the final sigma/coef update.
Centering (fit_intercept) is their `_preprocess_data`: column means and the
target mean, rows ascending. float32 throughout.
"""
from std.sys.compile import is_defined
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fabs, fmax, fsqrt, ld, st, ldi, sti, i2f, fill, copy,
    cholesky, chol_solve, jacobi_eig, centered_gram, centered_xty, mean_of,
    add_acc, axpy_acc, axpy_centered, par_rows,
)
from std.sys.info import is_gpu, has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_linear.team import Team
from x_linear.tops import fold_parts, fold_blocks, fold_fa_blocked, FOLD_BLOCK, upper_cell, t_col_means, t_centered_gram, t_centered_xty, t_sum, t_mean, fold_sq, chain_cfmad, t_jacobi_eig, t_cholesky


def _center(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, iw: IP) -> Float32:
    if fi:
        fill(fw, xm, d, Float32(0))
        for i in range(n):
            add_acc(fw, xm, x, i * d, d)
        for j in range(d):
            st(fw, xm + j, fd(ld(fw, xm + j), i2f(n)))
        return mean_of(y, n)
    fill(fw, xm, d, Float32(0))
    return Float32(0)


def _resid_rows(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP):
    """sc[i] = (y_i - ym) - sum_j (x_ij - xm_j) coef_j, j ascending: the map
    half of the sse (lane linear-cpu)."""

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm d, imm fw, imm xm, imm ym, imm coef, imm coff, imm sc}:
        for i in range(lo, hi):
            var p = Float32(0)
            for j in range(d):
                p = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(coef, coff + j), p)
            st(sc, i, fs(fs(ld(y, i), ym), p))

    par_rows(rows_map, n)


# ------------------------------------------------ the blocked order (lane/neural-pass97 order)
@always_inline
def _sse_part(rb: FP, y: FP, n: Int, sw: Bool, lo: Int, cnt: Int) -> Float32:
    """The squared-residual sum (weighted with sw) of rows [lo, lo + cnt) from zero."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        var r = ld(rb, i)
        acc = fmad(fm(ld(y, n + i), r), r, acc) if sw else fmad(r, r, acc)
    return acc


def _sse_blocked(rb: FP, y: FP, n: Int, sw: Bool) -> Float32:
    """The squared-residual sum in the blocked order: FOLD_BLOCK rows from
    zero, the partials folded blocks ascending."""
    var acc = Float32(0)
    var lo = 0
    while lo < n:
        acc = fa(acc, _sse_part(rb, y, n, sw, lo, min(FOLD_BLOCK, n - lo)))
        lo += FOLD_BLOCK
    return acc


def _sse(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP) -> Float32:
    _resid_rows(x, y, n, d, fw, xm, ym, coef, coff, sc)
    return _sse_blocked(sc, y, n, False)
    var acc = Float32(0)
    for i in range(n):
        var r = ld(sc, i)
        acc = fmad(r, r, acc)
    return acc


def _intercept(d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int) -> Float32:
    var acc = Float32(0)
    for j in range(d):
        acc = fmad(ld(fw, xm + j), ld(coef, coff + j), acc)
    return fs(ym, acc)


def _wmean_center(x: FP, y: FP, n: Int, d: Int, fi: Bool, fw: FP, xm: Int, wsum: Float32) -> Float32:
    """Weighted means (their _preprocess_data with sample_weight); w at y + n."""
    if not fi:
        fill(fw, xm, d, Float32(0))
        return Float32(0)
    fill(fw, xm, d, Float32(0))
    for i in range(n):
        axpy_acc(fw, xm, ld(y, n + i), x, i * d, d)
    for j in range(d):
        st(fw, xm + j, fd(ld(fw, xm + j), wsum))
    var acc = Float32(0)
    for i in range(n):
        acc = fmad(ld(y, n + i), ld(y, i), acc)
    return fd(acc, wsum)


def _wsse(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP) -> Float32:
    """sum_i w_i r_i^2: their sse on the sqrt(w)-rescaled data."""
    _resid_rows(x, y, n, d, fw, xm, ym, coef, coff, sc)
    return _sse_blocked(sc, y, n, True)
    var acc = Float32(0)
    for i in range(n):
        var r = ld(sc, i)
        acc = fmad(fm(ld(y, n + i), r), r, acc)
    return acc


def _t_sse(t: Team, x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int,
           sw: Bool, sc: FP) -> Float32:
    """`_sse` (or `_wsse` with sw) on the team: each row's residual across the
    team (row buffer 0), the lead's fold in ascending row order, broadcast.
    The host maps the residuals into `sc` and folds them (lane linear-cpu)."""
    comptime if not is_gpu():
        return _wsse(x, y, n, d, fw, xm, ym, coef, coff, sc) if sw else _sse(x, y, n, d, fw, xm, ym, coef, coff, sc)
    var rb = t.row(0)
    for i in range(t.tid, n, t.nt):
        var p = Float32(0)
        for j in range(d):
            p = fmad(fs(ld(x, i * d + j), ld(fw, xm + j)), ld(coef, coff + j), p)
        st(rb, i, fs(fs(ld(y, i), ym), p))
    t.sync()
    var acc = Float32(0)
    # the blocked order (lane/neural-pass97): the block partials across the
    # team into row 1, then the lead folds them
    var nb = fold_blocks(n)
    var pr = t.row(1)
    for bk in range(t.tid, nb, t.nt):
        var lo = bk * FOLD_BLOCK
        st(pr, bk, _sse_part(rb, y, n, sw, lo, min(FOLD_BLOCK, n - lo)))
    t.sync()
    if t.lead():
        acc = fold_parts(pr, 0, nb)
    return t.bcast(acc)
    if t.lead():
        if sw:
            for i in range(n):
                var r = ld(rb, i)
                acc = fmad(fm(ld(y, n + i), r), r, acc)
        else:
            acc = fold_sq(rb, 0, n)
    return t.bcast(acc)


#: FAST on Apple (lane/apple-fast-classical, 2026-10-02): the evidence
#: iteration's sse from the normal equations instead of a pass over the rows,
#: relative to a REFERENCE row pass (lane/apple-fast-bayes): with s0 the sse
#: of a row pass at w0, dw = w - w0, q = Xc'yc and G = Xc'Xc,
#: |yc - Xc w|^2 = s0 + dw'(G (w + w0) - 2 q): O(d) (BayesianRidge, in the
#: eigenbasis) or O(d^2) (ARD) an iteration in place of n * d on ONE block
#: (300 iterations x 1M rows x 220 features of `_t_sse`). The plain form
#: yy - 2 w'q + w'G w cancelled to below zero on istella (220 features,
#: near-null Gram directions with f32 noise eigenvalues, w huge along them):
#: sse clamped to 0, alpha to inf, coef NaN. The delta form carries an error
#: bound (2^-12 relative on every entry of G and q); when it could move sse by
#: more than 2^-8 of itself (or sse is not positive or finite) that iteration
#: makes the row pass and it becomes the new reference. Unweighted fits, n >= d
#: (w0 lives in the host's sse scratch, unused on the device); ip[5] (set by
#: x_linear/device.mojo, `MOJOLEARN_X_LINEAR_GRAM_SSE=0` is the A/B arm) turns
#: it on. IDENTICAL and the other vendors never compile it.
comptime X_LINEAR_GRAM_SSE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
#: |error| <= mag * 2^-12 must stay <= 2^-8 * sse: mag * 2^-4 <= sse
comptime GRAM_SSE_TRUST = Float32(0.0625)


def _t_sse_delta(t: Team, fw: FP, gg: Int, xty: Int, res: FP, w0: Int, d: Int, s0: Float32) -> Float32:
    """s0 + dw'(G (w + w0) - 2 q) on the team (G intact at gg, q at xty, w in
    res, w0 at fw[w0, w0 + d)): thread j's term dw_j ((G (w + w0))_j - 2 q_j)
    in team row 0 and its magnitude |dw_j| (sum_k |G_jk| |w_k + w0_k| + 2 |q_j|)
    in row 1, the lead's sums. Returns -1 when the bound says untrusted."""
    var pr = t.row(0)
    var pm = t.row(1)
    for j in range(t.tid, d, t.nt):
        var dw = fs(ld(res, j), ld(fw, w0 + j))
        var c = Float32(0)
        var m = Float32(0)
        if dw != 0:
            var g = Float32(0)
            var ga = Float32(0)
            for k in range(d):
                var gjk = ld(fw, gg + j * d + k)
                var ws = fa(ld(res, k), ld(fw, w0 + k))
                g = fmad(gjk, ws, g)
                ga = fmad(fabs(gjk), fabs(ws), ga)
            var q = ld(fw, xty + j)
            c = fm(dw, fs(g, fm(Float32(2), q)))
            m = fm(fabs(dw), fa(ga, fm(Float32(2), fabs(q))))
        st(pr, j, c)
        st(pm, j, m)
    t.sync()
    var s = Float32(-1)
    if t.lead():
        var acc = Float32(0)
        var mag = Float32(0)
        for j in range(d):
            acc = fa(acc, ld(pr, j))
            mag = fa(mag, ld(pm, j))
        var v = fa(s0, acc)
        if fm(mag, GRAM_SSE_TRUST) <= v:
            s = v
    return t.bcast(s)


def _var(y: FP, n: Int) -> Float32:
    var m = mean_of(y, n)
    var acc = Float32(0)
    for i in range(n):
        var r = fs(ld(y, i), m)
        acc = fmad(r, r, acc)
    return fd(acc, i2f(n))


def bayes_ymean(y: FP, n: Int, fi: Bool) -> Float32:
    """BayesianRidge's target mean (0 without an intercept) in the blocked
    order: FOLD_BLOCK rows from zero, the partials folded blocks ascending
    (the grid driver folds the same partials, x_linear/device.mojo)."""
    if not fi:
        return Float32(0)
    return fd(fold_fa_blocked(y, 0, 1, n), i2f(n))


# the weighted statistics' row-block partials (cgr-linear): w at y + n,
# each from zero over rows [lo, lo + cnt) ascending
@always_inline
def bayes_wy_part(y: FP, n: Int, lo: Int, cnt: Int) -> Float32:
    """sum w_i y_i."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(ld(y, n + i), ld(y, i), acc)
    return acc


@always_inline
def bayes_wx_part(x: FP, y: FP, n: Int, d: Int, j: Int, lo: Int, cnt: Int) -> Float32:
    """sum w_i x_ij."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(ld(y, n + i), ld(x, i * d + j), acc)
    return acc


@always_inline
def bayes_wgram_part(x: FP, y: FP, n: Int, d: Int, fw: FP, j: Int, k: Int, lo: Int, cnt: Int) -> Float32:
    """sum (w_i (x_ij - xm_j)) (x_ik - xm_k), the means at fw[0, d)."""
    var mj = ld(fw, j)
    var mk = ld(fw, k)
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(fm(ld(y, n + i), fs(ld(x, i * d + j), mj)), fs(ld(x, i * d + k), mk), acc)
    return acc


@always_inline
def bayes_wxty_part(x: FP, y: FP, n: Int, d: Int, fw: FP, j: Int, ym: Float32, lo: Int, cnt: Int) -> Float32:
    """sum (y_i - ym) (w_i (x_ij - xm_j))."""
    var mj = ld(fw, j)
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        acc = fmad(fs(ld(y, i), ym), fm(ld(y, n + i), fs(ld(x, i * d + j), mj)), acc)
    return acc


@always_inline
def bayes_wvar_part(y: FP, n: Int, m: Float32, lo: Int, cnt: Int) -> Float32:
    """sum w_i (y_i - m)^2."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        var r = fs(ld(y, i), m)
        acc = fmad(fm(ld(y, n + i), r), r, acc)
    return acc


@always_inline
def bayes_yvar_part(y: FP, m: Float32, lo: Int, cnt: Int) -> Float32:
    """sum (y_i - m)^2 over rows [lo, lo + cnt) from zero, rows ascending."""
    var acc = Float32(0)
    for i in range(lo, lo + cnt):
        var r = fs(ld(y, i), m)
        acc = fmad(r, r, acc)
    return acc


def bayes_yvar(y: FP, n: Int) -> Float32:
    """`_var` (the starting alpha's target variance) in the blocked order:
    the blocked mean, each block's squared deviations from zero, the
    partials folded blocks ascending."""
    var m = fd(fold_fa_blocked(y, 0, 1, n), i2f(n))
    var acc = Float32(0)
    var lo = 0
    while lo < n:
        acc = fa(acc, bayes_yvar_part(y, m, lo, min(FOLD_BLOCK, n - lo)))
        lo += FOLD_BLOCK
    return fd(acc, i2f(n))


#: Null-direction floor (FAST lane/apple-fast-bayesq, then every vendor,
#: every mode and the host column in lane/fix-bayes-null, 2026-10-03):
#: `bayes_eig_prep` drops the eigen-directions of the centered Gram below the
#: float32 noise floor d * eps * max ev (ev and V'X'y set to 0). The float32
#: Gram + Jacobi resolves eigenvalues only to about eps * max ev: on istella
#: (35 of 220 directions below 1e-12 max ev in float64: 19 constant columns
#: plus exact collinearities) the null directions came back as noise
#: eigenvalues of a few units with noise V'X'y, so z_k = vty_k / (ev_k +
#: lam/alpha) was huge there; the evidence iteration drove lambda to 1e-8 and
#: held-out r2 was -4.2e4 IDENTICAL, -3.2e4 FAST (scikit-learn float64:
#: 0.3287). See docs/apple-fast/notes/bayesq.md. One code path: the device
#: team kernel and the host column (nt = 1) both run it, same order.


def bayes_eig_prep(t: Team, fw: FP, fp: FP, d: Int, yvar: Float32) -> Tuple[Float32, Float32]:
    """The d x d half of `bayes_prep` (no row passes): the eigendecomposition
    of G on the team, the lead's eigenvalues and V'X'y, the starting alpha
    (the lead's, from yvar, when alpha_init is none) and lambda."""
    var gg = d
    var xty = gg + d * d
    var vv = xty + d
    var vty = vv + d * d
    var old = vty + d
    var tmp = old + d
    var alpha = ld(fp, 5)
    t_jacobi_eig(t, fw, gg, fw, vv, d, 60)
    if t.lead():
        for j in range(d):
            var ev = ld(fw, gg + j * d + j)
            st(fw, tmp + j, fmax(Float32(0), ev))
            var acc = Float32(0)
            for k in range(d):
                acc = fmad(ld(fw, vv + k * d + j), ld(fw, xty + k), acc)
            st(fw, vty + j, acc)
        # eigenvalues below the float32 noise floor (d eps max ev, the
        # pinv/matrix_rank cut) are null directions: ev 0 and V'X'y 0, so
        # z_k = 0 at every lam/alpha and gamma takes nothing from them
        var emax = Float32(0)
        for j in range(d):
            emax = fmax(emax, ld(fw, tmp + j))
        var tau = fm(fm(emax, i2f(d)), Float32(1.1920929e-07))
        for j in range(d):
            if ld(fw, tmp + j) <= tau:
                st(fw, tmp + j, Float32(0))
                st(fw, vty + j, Float32(0))
        if alpha < 0:
            alpha = fd(Float32(1), fa(yvar, Float32(1.1920929e-07)))
    alpha = t.bcast(alpha, 2)
    var lam = ld(fp, 6)
    if lam < 0:
        lam = Float32(1)
    return (alpha, lam)


def bayes_prep(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP) -> Tuple[Float32, Float32, Float32, Float32]:
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [tol, alpha_1, alpha_2, lambda_1,
    lambda_2, alpha_init (<0: none), lambda_init (<0: none)].
    With sample_weight, y = targets n | weights n: weighted centering, the
    sqrt(w)-rescaled Gram, X'y and sse, the weighted variance and sum(w) in
    the alpha update (theirs, sw_sum).
    res: coef d, intercept, alpha_, lambda_, n_iter.
    fw: xm d | G d*d | xty d | V d*d | vty d | old d | tmp d, then at
    3d^2 + 5d the host's sse scratch n (lane linear-cpu).
    Team form: the row passes (means, Gram, X'y, sse) across the team
    (x_linear/tops.mojo), the d x d algebra and the scalar updates on the lead."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var tol = ld(fp, 0)
    var a1 = ld(fp, 1)
    var a2 = ld(fp, 2)
    var l1 = ld(fp, 3)
    var l2 = ld(fp, 4)
    var xm = 0
    var gg = d
    var xty = gg + d * d
    var vv = xty + d
    var vty = vv + d * d
    var old = vty + d
    var tmp = old + d
    var sw = ldi(ip, 2) != 0
    var wsum = i2f(n)
    var ym = Float32(0)
    var wyvar = Float32(0)
    comptime if is_gpu():
        if sw:
            wsum = t_sum(t, y + n, n)
            # weighted means (their _preprocess_data with sample_weight)
            if fi:
                for j in range(t.tid, d, t.nt):
                    var acc = Float32(0)
                    for i in range(n):
                        acc = fmad(ld(y, n + i), ld(x, i * d + j), acc)
                    st(fw, xm + j, fd(acc, wsum))
                if t.lead():
                    var acc = Float32(0)
                    for i in range(n):
                        acc = fmad(ld(y, n + i), ld(y, i), acc)
                    ym = fd(acc, wsum)
                ym = t.bcast(ym, 1)
            else:
                if t.lead():
                    fill(fw, xm, d, Float32(0))
                t.sync()
            var cells = d * (d + 1) // 2
            for c in range(t.tid, cells, t.nt):
                var jk = upper_cell(c, d)
                var j = jk[0]
                var k = jk[1]
                var acc = Float32(0)
                for i in range(n):
                    acc = fmad(fm(ld(y, n + i), fs(ld(x, i * d + j), ld(fw, xm + j))), fs(ld(x, i * d + k), ld(fw, xm + k)), acc)
                st(fw, gg + j * d + k, acc)
                st(fw, gg + k * d + j, acc)
            t.sync()
        else:
            if fi:
                t_col_means(t, x, n, d, fw, xm)
                if t.lead():
                    ym = bayes_ymean(y, n, True)
                ym = t.bcast(ym, 1)
            else:
                if t.lead():
                    fill(fw, xm, d, Float32(0))
                t.sync()
            # ip[4] (device only, lane/neural-pass87): 1 when x_linear/device.mojo's
            # grid kernels already wrote this centered Gram into fw[gg, gg + d*d)
            if ldi(ip, 4) == 0:
                t_centered_gram(t, x, n, d, fw, xm, fw, gg)
        var yc = ym
        # X'y on centered data
        for j in range(t.tid, d, t.nt):
            var acc = Float32(0)
            var mj = ld(fw, xm + j)
            if sw:
                for i in range(n):
                    var xc = fs(ld(x, i * d + j), mj)
                    xc = fm(ld(y, n + i), xc)
                    acc = fmad(xc, fs(ld(y, i), yc), acc)
            else:
                acc = chain_cfmad(x, j, d, mj, y, 0, 1, yc, n)
            st(fw, xty + j, acc)
        t.sync()
    else:
        # the host's row passes: one pass per statistic, vector accumulators
        if sw:
            # cgr-linear: the weighted statistics in the blocked order (each
            # FOLD_BLOCK rows from zero, the blocks folded ascending), the
            # order the grid driver folds them (x_linear/device.mojo)
            wsum = fold_fa_blocked(y, n, 1, n)
            var scr = List[Float32](length=d * d + d, fill=Float32(0))
            var sp = FP(unsafe_from_address=Int(scr.unsafe_ptr()))
            var ywm = Float32(0)
            fill(fw, xm, d, Float32(0))
            var lo = 0
            while lo < n:
                var hi = min(lo + FOLD_BLOCK, n)
                ywm = fa(ywm, bayes_wy_part(y, n, lo, hi - lo))
                if fi:
                    fill(sp, 0, d, Float32(0))
                    for i in range(lo, hi):
                        axpy_acc(sp, 0, ld(y, n + i), x, i * d, d)
                    add_acc(fw, xm, sp, 0, d)
                lo = hi
            ywm = fd(ywm, wsum)
            if fi:
                ym = ywm
                for j in range(d):
                    st(fw, xm + j, fd(ld(fw, xm + j), wsum))
            fill(fw, gg, d * d, Float32(0))
            fill(fw, xty, d, Float32(0))
            lo = 0
            while lo < n:
                var hi = min(lo + FOLD_BLOCK, n)
                fill(sp, 0, d * d + d, Float32(0))
                for i in range(lo, hi):
                    var wi = ld(y, n + i)
                    for j in range(d):
                        var a = fm(wi, fs(ld(x, i * d + j), ld(fw, xm + j)))
                        axpy_centered(sp, j * d + j, a, x, i * d + j, fw, xm + j, d - j)
                    axpy_centered[True](sp, d * d, fs(ld(y, i), ym), x, i * d, fw, xm, d, wi)
                add_acc(fw, gg, sp, 0, d * d)
                add_acc(fw, xty, sp, d * d, d)
                lo = hi
            for j in range(d):
                for k in range(j + 1, d):
                    st(fw, gg + k * d + j, ld(fw, gg + j * d + k))
            if ld(fp, 5) < 0:
                # np.average((y - y_mean) ** 2, weights=sample_weight), blocked
                var acc = Float32(0)
                lo = 0
                while lo < n:
                    acc = fa(acc, bayes_wvar_part(y, n, ywm, lo, min(FOLD_BLOCK, n - lo)))
                    lo += FOLD_BLOCK
                wyvar = fd(acc, wsum)
        else:
            _ = _center(x, y, n, d, fi, fw, xm, iw)
            ym = bayes_ymean(y, n, fi)
            centered_gram(x, n, d, fw, xm, fw, gg)
            var yc = ym
            # X'y on centered data
            fill(fw, xty, d, Float32(0))
            for i in range(n):
                var b = fs(ld(y, i), yc)
                axpy_centered(fw, xty, b, x, i * d, fw, xm, d)
    var yvar = Float32(0)
    if t.lead() and ld(fp, 5) < 0:
        if sw:
            comptime if is_gpu():
                # np.average((y - y_mean) ** 2, weights=sample_weight)
                var m = Float32(0)
                for i in range(n):
                    m = fmad(ld(y, n + i), ld(y, i), m)
                m = fd(m, wsum)
                var acc = Float32(0)
                for i in range(n):
                    var r = fs(ld(y, i), m)
                    acc = fmad(fm(ld(y, n + i), r), r, acc)
                yvar = fd(acc, wsum)
            else:
                yvar = wyvar
        else:
            yvar = bayes_yvar(y, n)
    var al = bayes_eig_prep(t, fw, fp, d, yvar)
    var alpha = al[0]
    var lam = al[1]
    return (alpha, lam, ym, wsum)


@always_inline
def bayes_coef_one(fw: FP, d: Int, j: Int, ratio: Float32) -> Float32:
    """coef_j = sum_k V_jk vty_k / (ev_k+ + ratio), k ascending."""
    var vv = d + d * d + d
    var vty = vv + d * d
    var tmp = vty + 2 * d
    var acc = Float32(0)
    for k in range(d):
        acc = fmad(ld(fw, vv + j * d + k), fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio)), acc)
    return acc


def bayes_coef(fw: FP, res: FP, d: Int, lam: Float32, alpha: Float32):
    """coef = V diag(1/(ev + lam/alpha)) V' X'y (the lead's); the grid driver
    runs `bayes_coef_one` a thread a coefficient."""
    var ratio = fd(lam, alpha)
    for j in range(d):
        st(res, j, bayes_coef_one(fw, d, j, ratio))


def bayes_step(fw: FP, res: FP, d: Int, fp: FP, lam_in: Float32, alpha_in: Float32, sse: Float32, wsum: Float32,
               it: Int) -> Tuple[Float32, Float32, Int]:
    """One iteration's lead update from its sse: (lambda, alpha, stop); on a
    stop the final coefficients are written (their post-loop update)."""
    var tol = ld(fp, 0)
    var a1 = ld(fp, 1)
    var a2 = ld(fp, 2)
    var l1 = ld(fp, 3)
    var l2 = ld(fp, 4)
    var old = d + d * d + d + d * d + d
    var tmp = old + d
    var lam = lam_in
    var alpha = alpha_in
    var stop = 0
    var gamma = Float32(0)
    for k in range(d):
        var aev = fm(alpha, ld(fw, tmp + k))
        gamma = fa(gamma, fd(aev, fa(lam, aev)))
    var wn = Float32(0)
    for j in range(d):
        wn = fmad(ld(res, j), ld(res, j), wn)
    lam = fd(fa(gamma, fm(Float32(2), l1)), fa(wn, fm(Float32(2), l2)))
    alpha = fd(fa(fs(wsum, gamma), fm(Float32(2), a1)), fa(sse, fm(Float32(2), a2)))
    if it != 0:
        var delta = Float32(0)
        for j in range(d):
            delta = fa(delta, fabs(fs(ld(fw, old + j), ld(res, j))))
        if delta < tol:
            # their loop breaks here and the update below the loop runs
            bayes_coef(fw, res, d, lam, alpha)
            stop = 1
    if stop == 0:
        copy(fw, old, res, 0, d)
    return (lam, alpha, stop)


def bayes_finish(fw: FP, res: FP, d: Int, fi: Bool, ym: Float32, alpha: Float32, lam: Float32, iters: Int):
    st(res, d, _intercept(d, fw, 0, ym, res, 0) if fi else Float32(0))
    st(res, d + 1, alpha)
    st(res, d + 2, lam)
    st(res, d + 3, i2f(iters))


def bayes_ridge_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """See `bayes_prep` (the statistics, the eigendecomposition, the
    starting alpha and lambda); then the iterations: the lead's
    coefficients, the team's sse, the lead's update (`bayes_step`), and
    `bayes_finish` (lane/neural-pass131 factored them for the grid driver;
    the same statements in the same order)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var sw = ldi(ip, 2) != 0
    var xm = 0
    var st4 = bayes_prep(t, x, y, n, d, ip, fp, res, fw, iw)
    var alpha = st4[0]
    var lam = st4[1]
    var ym = st4[2]
    var wsum = st4[3]
    var gg = d
    var vty = gg + d * d + d + d * d
    var tmp = vty + 2 * d
    var gram_sse = False
    var have_ref = False
    var s0 = Float32(0)  # the reference row pass's sse, at z0
    var z0 = 3 * d * d + 5 * d  # the host's sse scratch (n >= d words), unused on the device
    var emax = Float32(0)
    comptime if is_gpu() and X_LINEAR_GRAM_SSE:
        if not sw and n >= d and ldi(ip, 5) != 0:
            gram_sse = True
            if t.lead():
                for k in range(d):
                    emax = fmax(emax, fabs(ld(fw, gg + k * d + k)))
    var iters = 0
    for it in range(max_iter + 1):
        if t.lead():
            bayes_coef(fw, res, d, lam, alpha)
        t.sync()
        if it == max_iter:
            break  # the last update after the loop
        iters = it + 1
        var sse: Float32
        comptime if is_gpu() and X_LINEAR_GRAM_SSE:
            if gram_sse:
                # w = V z, z_k = vty_k / (ev_k+ + lam/alpha); in the eigenbasis
                # (G = V diag(ev) V', q = V vty) the delta form is
                # s0 + sum_k dz_k (ev_k (z_k + z0_k) - 2 vty_k); its bound takes
                # every eigenvalue off by 2^-12 (|ev_k| + max |ev|)
                var s = Float32(0)
                var fresh = 1
                if t.lead() and have_ref:
                    var ratio = fd(lam, alpha)
                    var acc = Float32(0)
                    var mag = Float32(0)
                    for k in range(d):
                        var v = ld(fw, vty + k)
                        var e = ld(fw, gg + k * d + k)
                        var z = fd(v, fa(ld(fw, tmp + k), ratio))
                        var zo = ld(fw, z0 + k)
                        var dz = fs(z, zo)
                        var zs = fa(z, zo)
                        acc = fmad(dz, fs(fm(e, zs), fm(Float32(2), v)), acc)
                        mag = fmad(fabs(dz), fa(fm(fa(fabs(e), emax), fabs(zs)), fm(Float32(2), fabs(v))), mag)
                    s = fa(s0, acc)
                    if fm(mag, GRAM_SSE_TRUST) <= s:
                        fresh = 0
                if t.bcast_int(fresh, 0) == 0:
                    sse = t.bcast(s)
                else:
                    sse = _t_sse(t, x, y, n, d, fw, xm, ym, res, 0, sw, fw + 3 * d * d + 5 * d)
                    s0 = sse
                    have_ref = True
                    if t.lead():
                        var ratio = fd(lam, alpha)
                        for k in range(d):
                            st(fw, z0 + k, fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio)))
            else:
                sse = _t_sse(t, x, y, n, d, fw, xm, ym, res, 0, sw, fw + 3 * d * d + 5 * d)
        else:
            sse = _t_sse(t, x, y, n, d, fw, xm, ym, res, 0, sw, fw + 3 * d * d + 5 * d)
        var stop = 0
        if t.lead():
            var r = bayes_step(fw, res, d, fp, lam, alpha, sse, wsum, it)
            lam = r[0]
            alpha = r[1]
            stop = r[2]
        lam = t.bcast(lam, 1)
        alpha = t.bcast(alpha, 2)
        if t.bcast_int(stop, 3) == 1:
            break
    if t.lead():
        bayes_finish(fw, res, d, fi, ym, alpha, lam, iters)

#: QUALITY FIX (lane apple-fast-q-reg, 2026-10-04), the FAST default on every
#: vendor; `-D MOJOLEARN_ARD_SIGMA_QOLD` restores the old sigma. Audit
#: (board-quality-audit-2026-10-04, M3 0.8.34): ard istella r2 -0.1387
#: (FAST) / -0.1249 (IDENTICAL) vs scikit-learn 0.3274; taxi (11 features)
#: equal. Cause: `_ard_sigma` factors diag(lambda) + alpha G in float32 as
#: it stands and ignores `cholesky`'s False (`_ = cholesky(...)` below):
#: Istella's 220 columns span many orders of magnitude, so a pivot goes
#: non-positive, the factor is left half-written and sigma and the
#: coefficients are garbage (worse than the mean). scikit-learn takes pinvh
#: of the same matrix. Here, FAST: the matrix is equilibrated to unit
#: diagonal (A~ = S A S, S = diag(1 / sqrt(A_jj)), recomputed from G and
#: lambda), factored, and on a non-positive pivot refactored with a ridge
#: of dk * eps32 on A~'s diagonal (x10 a retry, ARD_EQ_TRIES); sigma =
#: S inv(A~) S. Same team split as before; no host step.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab5-ard): ard istella r2 0.2826 -> 0.3270 (scikit-learn 0.3274) BUT
#: 67.2 -> 1348.1 ms. KEPT for correctness; the speed fix is owed in lane
#: apple-fast-general-speed.
comptime ARD_FAST_EQ = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_ARD_SIGMA_QOLD"]()
comptime ARD_EQ_TRIES = 8


@always_inline
def _ard_eq_scale(fw: FP, gg: Int, lamo: Int, alpha: Float32, d: Int, j: Int) -> Float32:
    """1 / sqrt(alpha G_jj + lambda_j): feature j's equilibration factor."""
    return fd(Float32(1), fsqrt(fa(fm(alpha, ld(fw, gg + j * d + j)), ld(fw, lamo + j))))


@always_inline
def _ard_eq_row(fw: FP, gg: Int, aa: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int, d: Int, dk: Int,
                a: Int, ridge: Float32):
    """Row a of the equilibrated A~ (+ ridge on its diagonal)."""
    var ja = ldi(iw, keep + d + a)
    var sa = _ard_eq_scale(fw, gg, lamo, alpha, d, ja)
    for b in range(dk):
        var jb = ldi(iw, keep + d + b)
        var v = fm(alpha, ld(fw, gg + ja * d + jb))
        if a == b:
            v = fa(v, ld(fw, lamo + ja))
        v = fm(fm(v, sa), _ard_eq_scale(fw, gg, lamo, alpha, d, jb))
        if a == b:
            v = fa(v, ridge)
        st(fw, aa + a * dk + b, v)


@always_inline
def _ard_eq_col(fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int, d: Int,
                dk: Int, c: Int):
    """Column c of sigma = S inv(A~) S from the factor of A~."""
    for r in range(dk):
        st(fw, sg + c * dk + r, Float32(1) if r == c else Float32(0))
    chol_solve(fw, aa, dk, fw, sg + c * dk)
    var sc = _ard_eq_scale(fw, gg, lamo, alpha, d, ldi(iw, keep + d + c))
    for r in range(dk):
        var sr = _ard_eq_scale(fw, gg, lamo, alpha, d, ldi(iw, keep + d + r))
        st(fw, sg + c * dk + r, fm(fm(ld(fw, sg + c * dk + r), sr), sc))


@always_inline
def _ard_eq_next_ridge(ridge: Float32, dk: Int) -> Float32:
    if ridge == Float32(0):
        return fm(Float32(1.1920929e-07), i2f(max(dk, 1)))
    return fm(ridge, Float32(10))


def _ard_sigma_eq(d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int) -> Int:
    """ARD_FAST_EQ's sigma on one thread. Returns dk."""
    var dk = 0
    for j in range(d):
        if ldi(iw, keep + j) != 0:
            sti(iw, keep + d + dk, j)
            dk += 1
    var ridge = Float32(0)
    for _ in range(ARD_EQ_TRIES):
        for a in range(dk):
            _ard_eq_row(fw, gg, aa, lamo, alpha, iw, keep, d, dk, a, ridge)
        if cholesky(fw, aa, dk):
            break
        ridge = _ard_eq_next_ridge(ridge, dk)
    for c in range(dk):
        _ard_eq_col(fw, gg, aa, sg, lamo, alpha, iw, keep, d, dk, c)
    return dk


def _t_ard_sigma_eq(t: Team, d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP,
                    keep: Int) -> Int:
    """ARD_FAST_EQ's sigma on the team: rows and columns split as in
    `_t_ard_sigma`; `t_cholesky`'s result is uniform, so is the retry."""
    var dk = 0
    if t.lead():
        for j in range(d):
            if ldi(iw, keep + j) != 0:
                sti(iw, keep + d + dk, j)
                dk += 1
    dk = t.bcast_int(dk, 0)
    var ridge = Float32(0)
    for _ in range(ARD_EQ_TRIES):
        for a in range(t.tid, dk, t.nt):
            _ard_eq_row(fw, gg, aa, lamo, alpha, iw, keep, d, dk, a, ridge)
        t.sync()
        if t_cholesky(t, fw, aa, dk):
            break
        ridge = _ard_eq_next_ridge(ridge, dk)
    for c in range(t.tid, dk, t.nt):
        _ard_eq_col(fw, gg, aa, sg, lamo, alpha, iw, keep, d, dk, c)
    t.sync()
    return dk


def _ard_sigma(d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int) -> Int:
    """sigma (dk x dk, over the kept features in ascending order) = inv(diag(lambda) + alpha G). Returns dk."""
    comptime if ARD_FAST_EQ:
        return _ard_sigma_eq(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
    var dk = 0
    for j in range(d):
        if ldi(iw, keep + j) != 0:
            sti(iw, keep + d + dk, j)
            dk += 1
    for a in range(dk):
        var ja = ldi(iw, keep + d + a)
        for b in range(dk):
            var jb = ldi(iw, keep + d + b)
            var v = fm(alpha, ld(fw, gg + ja * d + jb))
            if a == b:
                v = fa(v, ld(fw, lamo + ja))
            st(fw, aa + a * dk + b, v)
    _ = cholesky(fw, aa, dk)
    for c in range(dk):
        for r in range(dk):
            st(fw, sg + c * dk + r, Float32(1) if r == c else Float32(0))
        chol_solve(fw, aa, dk, fw, sg + c * dk)
    return dk


comptime ARD_TEAM_MIN = 32


def _t_ard_sigma(t: Team, d: Int, fw: FP, gg: Int, aa: Int, sg: Int, lamo: Int, alpha: Float32, iw: IP, keep: Int) -> Int:
    """`_ard_sigma` on the team (lane/neural-pass86, 2026-10-01): the kept
    list on the lead, the rows of diag(lambda) + alpha G split across the
    team, the Cholesky by `t_cholesky`, then the dk columns of the inverse,
    each its own `chol_solve` reading only the factor, split across the team.
    Every word is `_ard_sigma`'s. On the device the whole of it ran on the
    lead thread every iteration (ARD at 220 features: about 57 ms an
    iteration on the M4's GPU). `-D MOJOLEARN_X_LINEAR_ARD_LEAD=1` restores
    the lead-only call."""
    comptime if is_defined["MOJOLEARN_X_LINEAR_ARD_LEAD"]():
        var dk0 = 0
        if t.lead():
            dk0 = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        t.sync()
        return dk0
    if t.nt <= 1:
        return _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
    comptime if ARD_FAST_EQ:
        if d >= ARD_TEAM_MIN:
            return _t_ard_sigma_eq(t, d, fw, gg, aa, sg, lamo, alpha, iw, keep)
    # below ARD_TEAM_MIN features the column barriers of `t_cholesky` cost
    # more than the lead's serial factor (ARD taxi, 11 features: 32.6 to
    # 36.7 ms on the L40S); the lead runs `_ard_sigma`, the same words
    if d < ARD_TEAM_MIN:
        var dk1 = 0
        if t.lead():
            dk1 = _ard_sigma(d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        return t.bcast_int(dk1, 0)
    var dk = 0
    if t.lead():
        for j in range(d):
            if ldi(iw, keep + j) != 0:
                sti(iw, keep + d + dk, j)
                dk += 1
    dk = t.bcast_int(dk, 0)
    for a in range(t.tid, dk, t.nt):
        var ja = ldi(iw, keep + d + a)
        for b in range(dk):
            var jb = ldi(iw, keep + d + b)
            var v = fm(alpha, ld(fw, gg + ja * d + jb))
            if a == b:
                v = fa(v, ld(fw, lamo + ja))
            st(fw, aa + a * dk + b, v)
    t.sync()
    _ = t_cholesky(t, fw, aa, dk)
    for c in range(t.tid, dk, t.nt):
        for r in range(dk):
            st(fw, sg + c * dk + r, Float32(1) if r == c else Float32(0))
        chol_solve(fw, aa, dk, fw, sg + c * dk)
    t.sync()
    return dk


def _t_ard_coef(t: Team, d: Int, dk: Int, fw: FP, sg: Int, xty: Int, alpha: Float32, iw: IP, keep: Int, res: FP):
    """`_ard_coef` with its dk outputs split across the team (each its own
    chain over b ascending); the same words."""
    if t.nt <= 1:
        _ard_coef(d, dk, fw, sg, xty, alpha, iw, keep, res)
        return
    for j in range(t.tid, d, t.nt):
        st(res, j, Float32(0))
    t.sync()
    for a in range(t.tid, dk, t.nt):
        var acc = Float32(0)
        for b in range(dk):
            acc = fmad(ld(fw, sg + b * dk + a), ld(fw, xty + ldi(iw, keep + d + b)), acc)
        st(res, ldi(iw, keep + d + a), fm(alpha, acc))
    t.sync()


def _ard_coef(d: Int, dk: Int, fw: FP, sg: Int, xty: Int, alpha: Float32, iw: IP, keep: Int, res: FP):
    fill(res, 0, d, Float32(0))
    for a in range(dk):
        var acc = Float32(0)
        for b in range(dk):
            acc = fmad(ld(fw, sg + b * dk + a), ld(fw, xty + ldi(iw, keep + d + b)), acc)
        st(res, ldi(iw, keep + d + a), fm(alpha, acc))


def ard_update(fw: FP, res: FP, iw: IP, d: Int, nf: Float32, dk: Int, sse: Float32, alpha_in: Float32,
               fp: FP, it: Int) -> Tuple[Float32, Int, Bool]:
    """One ARD iteration's lead update after its sse (nf = the row count):
    the lambdas, alpha, the pruning, the stop rule. Returns (alpha, stop,
    any_kept); the host loop and the grid driver (x_linear/ard_grid.mojo)
    run these statements."""
    var tol = ld(fp, 0)
    var a1 = ld(fp, 1)
    var a2 = ld(fp, 2)
    var l1 = ld(fp, 3)
    var l2 = ld(fp, 4)
    var thr = ld(fp, 5)
    var o = ard_layout(d)
    var sg = o[4]
    var lamo = o[5]
    var old = o[6]
    var keep = 0
    var stop = 0
    var gsum = Float32(0)
    for a in range(dk):
        var j = ldi(iw, keep + d + a)
        var gam = fs(Float32(1), fm(ld(fw, lamo + j), ld(fw, sg + a * dk + a)))
        gsum = fa(gsum, gam)
        var cj = ld(res, j)
        st(fw, lamo + j, fd(fa(gam, fm(Float32(2), l1)), fa(fm(cj, cj), fm(Float32(2), l2))))
    var alpha = fd(fa(fs(nf, gsum), fm(Float32(2), a1)), fa(sse, fm(Float32(2), a2)))
    var any_kept = False
    for j in range(d):
        var k = 1 if ld(fw, lamo + j) < thr else 0
        sti(iw, keep + j, k)
        if k == 0:
            st(res, j, Float32(0))
        else:
            any_kept = True
    if it > 0:
        var delta = Float32(0)
        for j in range(d):
            delta = fa(delta, fabs(fs(ld(fw, old + j), ld(res, j))))
        if delta < tol:
            stop = 1
    if stop == 0:
        copy(fw, old, res, 0, d)
        if not any_kept:
            stop = 1
    return (alpha, stop, any_kept)


@always_inline
def ard_layout(d: Int) -> InlineArray[Int, 8]:
    """fw: xm d | G d*d | xty d | A d*d | sigma d*d | lambda d | old d | sse scratch n."""
    var o = InlineArray[Int, 8](fill=0)
    o[0] = 0
    o[1] = d
    o[2] = d + d * d
    o[3] = o[2] + d
    o[4] = o[3] + d * d
    o[5] = o[4] + d * d
    o[6] = o[5] + d
    o[7] = o[6] + d
    return o^


def ard_finish(fw: FP, res: FP, d: Int, fi: Bool, ym: Float32, alpha: Float32, any_kept: Bool, iters: Int):
    """The result words after the last sigma and coefficients."""
    var o = ard_layout(d)
    if not any_kept:
        fill(res, 0, d, Float32(0))
    st(res, d, _intercept(d, fw, 0, ym, res, 0) if fi else Float32(0))
    st(res, d + 1, alpha)
    copy(res, d + 2, fw, o[5], d)
    st(res, d + 2 + d, i2f(iters))


def ard_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept]; fp: [tol, alpha_1, alpha_2, lambda_1,
    lambda_2, threshold_lambda].
    res: coef d, intercept, alpha_, lambda_ d, n_iter.
    fw: xm d | G d*d | xty d | A d*d | sigma d*d | lambda d | old d | sse scratch n (the host's).
    iw: keep d | kept index d.
    The host column (a team of one); the device binding runs
    x_linear/ard_grid.mojo (the row passes on the grid, sigma and the
    coefficients on one block team, `ard_update` on its lead)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var o = ard_layout(d)
    var xm = 0
    var gg = o[1]
    var xty = o[2]
    var aa = o[3]
    var sg = o[4]
    var lamo = o[5]
    var keep = 0
    var ym = _center(x, y, n, d, fi, fw, xm, iw)
    centered_gram(x, n, d, fw, xm, fw, gg)
    centered_xty(x, y, n, d, fw, xm, ym, fw, xty)
    # the starting alpha from the target variance in the blocked order
    # (cgr-linear: the grid driver folds the same partials)
    var alpha = fd(Float32(1), fa(bayes_yvar(y, n), Float32(1.1920929e-07)))
    fill(fw, lamo, d, Float32(1))
    fill(res, 0, d, Float32(0))
    for j in range(d):
        sti(iw, keep + j, 1)
    var iters = 0
    var any_kept = True
    for it in range(max_iter):
        iters = it + 1
        var dk = _t_ard_sigma(t, d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _t_ard_coef(t, d, dk, fw, sg, xty, alpha, iw, keep, res)
        var sse = _sse(x, y, n, d, fw, xm, ym, res, 0, fw + 3 * d * d + 4 * d)
        var r = ard_update(fw, res, iw, d, i2f(n), dk, sse, alpha, fp, it)
        alpha = r[0]
        any_kept = r[2]
        if r[1] == 1:
            break
    if any_kept:
        var dk = _t_ard_sigma(t, d, fw, gg, aa, sg, lamo, alpha, iw, keep)
        _t_ard_coef(t, d, dk, fw, sg, xty, alpha, iw, keep, res)
    ard_finish(fw, res, d, fi, ym, alpha, any_kept, iters)
