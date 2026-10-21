# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BayesianRidge's and ARDRegression's fits on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/bayes.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/rows_host.mojo) and no
GPU-reachable module runs host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, x_linear/ard_grid.mojo); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from x_linear.tops import FOLD_BLOCK, fold_blocks, fold_parts, fold_sq
from x_linear.rows_host import par_rows
from x_linear.ops import (
    FP,
    IP,
    centered_gram,
    centered_xty,
    fa,
    fabs,
    fd,
    fill,
    fm,
    fmad,
    fmax,
    fs,
    i2f,
    ld,
    ldi,
    st,
    sti,
)
from x_linear.team import Team
from std.sys.info import is_gpu
from x_linear.bayes import (
    GRAM_SSE_TRUST,
    X_LINEAR_GRAM_SSE,
    _center,
    _sse_blocked,
    _sse_part,
    _t_ard_coef,
    _t_ard_sigma,
    ard_finish,
    ard_layout,
    ard_update,
    bayes_coef,
    bayes_finish,
    bayes_prep,
    bayes_step,
    bayes_yvar,
)


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


def _sse(x: FP, y: FP, n: Int, d: Int, fw: FP, xm: Int, ym: Float32, coef: FP, coff: Int, sc: FP) -> Float32:
    _resid_rows(x, y, n, d, fw, xm, ym, coef, coff, sc)
    return _sse_blocked(sc, y, n, False)
    var acc = Float32(0)
    for i in range(n):
        var r = ld(sc, i)
        acc = fmad(r, r, acc)
    return acc


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
