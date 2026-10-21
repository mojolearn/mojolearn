# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HuberRegressor's fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/huber.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/ops.mojo) and no
GPU-reachable module calls host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `huber_fit_grid` in x_linear/huber_grid.mojo); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from x_linear.tops import FOLD_BLOCK
from x_linear.ops_host import par_rows
from x_linear.ops import (
    FP,
    IP,
    axpy_acc,
    fa,
    fabs,
    fd,
    fexp,
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
from x_linear.lbfgs import Objective, lbfgs, lbfgs_work
from x_linear.team import Team
from std.sys.info import is_gpu
from x_linear.huber import (
    _huber_objective_team,
    huber_finish,
)


def _huber_objective_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """Map (each row's residual into `sc`), then fold rows ascending."""
    var fi = ldi(ip, 1) != 0
    var eps = ld(fp, 0)
    var alpha = ld(fp, 1)
    var p = d + 2 if fi else d + 1
    var s = ld(th, toff + p - 1)
    var sigma = fexp(s)
    var b = ld(th, toff + d) if fi else Float32(0)
    fill(g, goff, p, Float32(0))
    var sq = Float32(0)
    var out_abs = Float32(0)
    var n_out = 0
    var sw = ldi(ip, 2) != 0
    var w_out = Float32(0)
    var w_all = Float32(0)
    var thr = fm(eps, sigma)
    var two_over_sigma = fd(Float32(2), sigma)
    var two_eps = fm(Float32(2), eps)

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm d, imm th, imm toff, imm b, imm sc}:
        row_dots(x, lo, hi, d, th, toff, sc)
        for i in range(lo, hi):
            st(sc, i, fs(fs(ld(y, i), ld(sc, i)), b))

    par_rows(rows_map, n)
    # the blocked order (lane/neural-pass97): every sum from zero over
    # FOLD_BLOCK rows, folded into its total at the block's end
    var cells = d + 1 if fi else d
    var pl = List[Float32](length=max(cells, 1), fill=Float32(0))
    var pg = FP(unsafe_from_address=Int(pl.unsafe_ptr()))
    var psq = Float32(0)
    var pout = Float32(0)
    var pwo = Float32(0)
    var pwa = Float32(0)
    for i in range(n):
        if i > 0 and i % FOLD_BLOCK == 0:
            for o in range(cells):
                st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
                st(pg, o, Float32(0))
            sq = fa(sq, psq)
            out_abs = fa(out_abs, pout)
            w_out = fa(w_out, pwo)
            w_all = fa(w_all, pwa)
            psq = Float32(0)
            pout = Float32(0)
            pwo = Float32(0)
            pwa = Float32(0)
        var r = ld(sc, i)
        var ar = fabs(r)
        var coefv: Float32
        if sw:
            var wi = ld(y, n + i)
            pwa = fa(pwa, wi)
            if ar > thr:
                pwo = fa(pwo, wi)
                pout = fmad(wi, ar, pout)
                coefv = fm(wi, -two_eps if r > 0 else two_eps)
            else:
                psq = fmad(fm(wi, r), r, psq)
                coefv = fm(-two_over_sigma, fm(wi, r))
        elif ar > thr:
            n_out += 1
            pout = fa(pout, ar)
            coefv = -two_eps if r > 0 else two_eps
        else:
            psq = fmad(r, r, psq)
            coefv = fm(-two_over_sigma, r)
        axpy_acc(pg, 0, coefv, x, i * d, d)
        if fi:
            st(pg, d, fa(ld(pg, d), coefv))
    if n > 0:
        for o in range(cells):
            st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
        sq = fa(sq, psq)
        out_abs = fa(out_abs, pout)
        w_out = fa(w_out, pwo)
        w_all = fa(w_all, pwa)
    _ = pl^
    return huber_finish(g, goff, th, toff, d, p, n, eps, alpha, sigma, two_eps, sw, sq, out_abs, n_out, w_out, w_all)


def huber_objective(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `sc` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _huber_objective_team(t, x, y, n, d, ip, fp, th, toff, g, goff)
    else:
        return _huber_objective_host(x, y, n, d, ip, fp, th, toff, g, goff, sc)


def huber_fit[obj: Objective = huber_objective](t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, sample_weight]; fp: [epsilon, alpha, tol].
    With sample_weight, y = targets n | weights n (their weighted objective).
    res: coef d, intercept 1, scale 1, n_iter 1 | theta scratch (P).
    fw: lbfgs_work(P) | objective scratch n (the host's map)."""
    var fi = ldi(ip, 1) != 0
    var p = d + 2 if fi else d + 1
    var th = d + 4
    if t.lead():
        fill(res, th, p, Float32(0))
    t.sync()
    var it = lbfgs[obj](t, x, y, n, d, ip, fp, res, th, p, ldi(ip, 0), ld(fp, 2), fw, 0, fw + lbfgs_work(p))
    if not t.lead():
        return
    for j in range(d):
        st(res, j, ld(res, th + j))
    st(res, d, ld(res, th + d) if fi else Float32(0))
    st(res, d + 1, fexp(ld(res, th + p - 1)))
    st(res, d + 2, i2f(it if it >= 0 else -it))
