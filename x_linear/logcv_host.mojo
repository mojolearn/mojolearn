# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegressionCV's fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/logcv.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/ops.mojo) and no
GPU-reachable module calls host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `logcv_fit_grid` in x_linear/logcv_grid.mojo); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from experiments.classical_identical_ideas.linear_controls import C13_LOGCV_WEIGHTS
from x_linear.tops import FOLD_BLOCK
from x_linear.ops_host import par_rows
from x_linear.ops import FP, IP, axpy_acc, fa, fexp, fill, flog, fm, fmax, fs, ld, ldi, row_dots, st
from x_linear.lbfgs import Objective
from x_linear.team import Team
from x_linear.cv_roles import cv_test, cv_train
from checks.numerics import ftz, identical_sigmoid, identical_softplus
from std.sys.info import is_gpu
from x_linear.logcv import (
    LCV_ROLES_AFTER,
    LcvRows,
    LcvScore,
    _logistic_objective_team,
    _predict_code,
    lcv_score_rows,
    logcv_finish,
    logcv_rows,
)


def _logistic_objective_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """Map, then fold (x_linear/ops.mojo, lane linear-cpu): each training
    row's loss term and gradient coefficients go to the scratch `sc`
    (L: n | R: n * K'), then one pass folds them in ascending row order,
    the order the one-pass loop used."""
    var kp = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var fold = ldi(ip, 2)
    var c = ld(fp, 0)
    var sw = ldi(ip, 3) != 0
    var stride = d + 1
    var p = kp * stride
    var sl = sc
    var sr = sl + n

    def rows_map(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm th, imm toff, imm kp, imm fi,
                                     imm fold, imm sw, imm stride, imm sl, imm sr}:
        # the linear predictors first (eight rows at a time on the host)
        if kp == 1:
            row_dots(x, lo, hi, d, th, toff, sl)
        else:
            for k in range(kp):
                row_dots(x, lo, hi, d, th, toff + k * stride, sl)
                for i in range(lo, hi):
                    st(sr, i * kp + k, fa(ld(sl, i), ld(th, toff + k * stride + d) if fi else Float32(0)))
        for i in range(lo, hi):
            if fold >= 0 and not cv_train(y + n, i, fold, LCV_ROLES_AFTER):
                continue
            var wi = ld(y, 2 * n + i) if sw else Float32(1)
            var label = Int(ld(y, i))
            if kp == 1:
                var z = fa(ld(sl, i), ld(th, toff + d) if fi else Float32(0))
                var yi = Float32(1) if label == 1 else Float32(0)
                var li = fs(ftz(identical_softplus(z)), fm(yi, z))
                var r = fs(ftz(identical_sigmoid(z)), yi)
                if sw:
                    li = fm(wi, li)
                    r = fm(wi, r)
                st(sl, i, li)
                st(sr, i, r)
            else:
                var zmax = Float32(-3.0e38)
                for k in range(kp):
                    zmax = fmax(zmax, ld(sr, i * kp + k))
                var se = Float32(0)
                var zy = Float32(0)
                for k in range(kp):
                    var z = ld(sr, i * kp + k)
                    se = fa(se, fexp(fs(z, zmax)))
                    if k == label:
                        zy = z
                var lse = fa(zmax, flog(se))
                st(sl, i, fm(wi, fs(lse, zy)) if sw else fs(lse, zy))
                for k in range(kp):
                    var r = fexp(fs(ld(sr, i * kp + k), lse))
                    if k == label:
                        r = fs(r, Float32(1))
                    if sw:
                        r = fm(wi, r)
                    st(sr, i * kp + k, r)

    par_rows(rows_map, n)
    fill(g, goff, p, Float32(0))
    var rows = 0
    var wrows = Float32(0)
    var acc = Float32(0)
    # the blocked order (lane/neural-pass97): every accumulator from zero
    # over FOLD_BLOCK training rows, folded into its total at each block's
    # end (the team's partials, folded blocks ascending)
    var pl = List[Float32](length=max(p, 1), fill=Float32(0))
    var pg = FP(unsafe_from_address=Int(pl.unsafe_ptr()))
    var pacc = Float32(0)
    var pw = Float32(0)
    for i in range(n):
        if fold >= 0 and not cv_train(y + n, i, fold, LCV_ROLES_AFTER):
            continue
        if rows > 0 and rows % FOLD_BLOCK == 0:
            for o in range(p):
                st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
                st(pg, o, Float32(0))
            acc = fa(acc, pacc)
            pacc = Float32(0)
            if sw and not C13_LOGCV_WEIGHTS:
                wrows = fa(wrows, pw)
                pw = Float32(0)
        rows += 1
        if sw and not C13_LOGCV_WEIGHTS:
            pw = fa(pw, ld(y, 2 * n + i))
        pacc = fa(pacc, ld(sl, i))
        for k in range(kp):
            var r = ld(sr, i * kp + k)
            axpy_acc(pg, k * stride, r, x, i * d, d)
            if fi:
                st(pg, k * stride + d, fa(ld(pg, k * stride + d), r))
    if rows > 0:
        for o in range(p):
            st(g, goff + o, fa(ld(g, goff + o), ld(pg, o)))
        acc = fa(acc, pacc)
        if sw and not C13_LOGCV_WEIGHTS:
            wrows = fa(wrows, pw)
    _ = pl^
    comptime if C13_LOGCV_WEIGHTS:
        wrows = ld(fp, 1) if sw else Float32(0)
    return logcv_finish(g, goff, th, toff, kp, d, sw, c, rows, acc, wrows)


def logistic_objective(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
    """The device runs the team schedule, the host the map-then-fold
    schedule over `sc` (lane linear-cpu). The same bits either way."""
    comptime if is_gpu():
        return _logistic_objective_team(t, x, y, n, d, ip, fp, th, toff, g, goff)
    else:
        return _logistic_objective_host(x, y, n, d, ip, fp, th, toff, g, goff, sc)


def lcv_hits_default(t: Team, x: FP, y: FP, n: Int, d: Int, kpp: Int, fi: Bool, fw: FP, th: Int, f: Int, hitr: FP):
    """Across the team on a device; on the host in row blocks (lane linear-cpu)."""
    comptime if is_gpu():
        for i in range(t.tid, n, t.nt):
            if cv_test(y + n, i, f, LCV_ROLES_AFTER):
                var hit = _predict_code(x, i, d, kpp, fi, fw, th) == Int(ld(y, i))
                st(hitr, i, Float32(1) if hit else Float32(0))
    else:
        var fwp = fw

        def rows_hit(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm kpp, imm fi, imm fwp, imm th,
                                        imm hitr, imm f}:
            for i in range(lo, hi):
                if cv_test(y + n, i, f, LCV_ROLES_AFTER):
                    var hit = _predict_code(x, i, d, kpp, fi, fwp, th) == Int(ld(y, i))
                    st(hitr, i, Float32(1) if hit else Float32(0))

        par_rows(rows_hit, n)


def lcv_score_default(t: Team, x: FP, y: FP, n: Int, d: Int, kpp: Int, fi: Bool, fw: FP, th: Int, f: Int,
                      hitr: FP, weighted: Bool) -> Float32:
    """Each held-out row's hit, then the lead's blocked score."""
    lcv_hits_default(t, x, y, n, d, kpp, fi, fw, th, f, hitr)
    t.sync()
    var s = Float32(0)
    if t.lead():
        s = lcv_score_rows(y, hitr, n, f, weighted)
    return s


def logcv_fit[obj: Objective = logistic_objective, rows: LcvRows = logcv_rows, score: LcvScore = lcv_score_default](
    t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP
