# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegressionCV (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_logistic.py`
(`LogisticRegressionCV.fit`, `_log_reg_scoring_path`,
`_logistic_regression_path`) and `sklearn/linear_model/_linear_loss.py`
(`LinearModelLoss`: objective (1/n) sum loss_i + 1/(2 C n) ||W||^2, the
intercepts unpenalized): binary log loss for two classes, the multinomial
(softmax) loss for more; the Cs path runs in the given order with warm
starts on each fold's training rows (their StratifiedKFold ids come from
Python); accuracy on the held-out rows; the first best mean over folds
wins. Their solver is L-BFGS and so is this one (x_linear/lbfgs.mojo).
Named difference: the refit on all rows starts from zero (theirs from the
mean of the folds' coefficients at the best C; the objective is convex, so
both reach the same minimizer). float32 throughout.

Per-call objective parameters: ip block [K', fit_intercept, fold (-1 all)],
fp block [C]; y = labels (0..K-1 as float32) | fold ids | weights (optional).
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fexp, flog, fmax, ld, st, ldi, sti, i2f, fill, copy, row_dot,
)
from x_linear.lbfgs import lbfgs, lbfgs_work
from x_linear.team import Team
from checks.numerics import identical_sigmoid, identical_softplus, ftz


def logistic_objective(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) -> Float32:
    """Team form: each row's residuals (row buffers 0..K'-1) and loss term
    (row buffer K') across the team; the lead folds the loss in ascending row
    order; one thread per gradient cell folds its rows ascending. The
    one-thread sequence, value for value. Skipped (held-out) rows are
    skipped, never added as zeros."""
    var kp = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var fold = ldi(ip, 2)
    var c = ld(fp, 0)
    var sw = ldi(ip, 3) != 0
    var stride = d + 1
    var p = kp * stride
    var lt = t.row(kp)
    for i in range(t.tid, n, t.nt):
        if fold >= 0 and Int(ld(y, n + i)) == fold:
            continue
        var wi = Float32(1)
        if sw:
            wi = ld(y, 2 * n + i)
        var label = Int(ld(y, i))
        if kp == 1:
            var z = fa(row_dot(x, i, d, th, toff), ld(th, toff + d) if fi else Float32(0))
            var yi = Float32(1) if label == 1 else Float32(0)
            var li = fs(ftz(identical_softplus(z)), fm(yi, z))
            var r = fs(ftz(identical_sigmoid(z)), yi)
            if sw:
                li = fm(wi, li)
                r = fm(wi, r)
            st(t.row(0), i, r)
            st(lt, i, li)
        else:
            # the logits are recomputed per pass (no scratch): max, sum, residuals
            var zmax = Float32(-3.0e38)
            for k in range(kp):
                var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
                zmax = fmax(zmax, z)
            var se = Float32(0)
            var zy = Float32(0)
            for k in range(kp):
                var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
                se = fa(se, fexp(fs(z, zmax)))
                if k == label:
                    zy = z
            var lse = fa(zmax, flog(se))
            st(lt, i, fm(wi, fs(lse, zy)) if sw else fs(lse, zy))
            for k in range(kp):
                var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
                var r = fexp(fs(z, lse))
                if k == label:
                    r = fs(r, Float32(1))
                if sw:
                    r = fm(wi, r)
                st(t.row(k), i, r)
    t.sync()
    for o in range(t.tid, p, t.nt):
        var k = o // stride
        var j = o - k * stride
        var rk = t.row(k)
        var acc = Float32(0)
        if j < d:
            for i in range(n):
                if fold >= 0 and Int(ld(y, n + i)) == fold:
                    continue
                acc = fmad(ld(rk, i), ld(x, i * d + j), acc)
        elif fi:
            for i in range(n):
                if fold >= 0 and Int(ld(y, n + i)) == fold:
                    continue
                acc = fa(acc, ld(rk, i))
        st(g, goff + o, acc)
    t.sync()
    var out = Float32(0)
    if t.lead():
        var rows = 0
        var wrows = Float32(0)
        var acc = Float32(0)
        for i in range(n):
            if fold >= 0 and Int(ld(y, n + i)) == fold:
                continue
            rows += 1
            if sw:
                wrows = fa(wrows, ld(y, 2 * n + i))
            acc = fa(acc, ld(lt, i))
        var cnt = wrows if sw else i2f(rows)
        var inv_n = fd(Float32(1), cnt)
        var lam = fd(Float32(1), fm(c, cnt))
        var reg = Float32(0)
        for k in range(kp):
            for j in range(stride):
                var o = k * stride + j
                var gv = fm(ld(g, goff + o), inv_n)
                if j < d:
                    var w = ld(th, toff + o)
                    reg = fmad(w, w, reg)
                    gv = fmad(lam, w, gv)
                st(g, goff + o, gv)
        out = fa(fm(acc, inv_n), fm(fm(Float32(0.5), lam), reg))
    return t.bcast(out)


def _predict_code(x: FP, i: Int, d: Int, kp: Int, fi: Bool, th: FP, toff: Int) -> Int:
    var stride = d + 1
    if kp == 1:
        var z = fa(row_dot(x, i, d, th, toff), ld(th, toff + d) if fi else Float32(0))
        return 1 if z > 0 else 0
    var best = 0
    var bz = Float32(0)
    for k in range(kp):
        var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
        if k == 0 or z > bz:  # DEVIATION 5005: the first max
            best = k
            bz = z
    return best


def logcv_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, K', n_Cs, n_folds, sample_weight]; fp: [tol, Cs...].
    With sample_weight, y = labels | folds | fit weights (sample x class) |
    score weights (sample): the loss sum and the penalty scale use sum(w)
    of the training rows (their LinearModelLoss), the held-out accuracy is
    weighted by the raw sample weights (their scorer's sample_weight).
    res: coef K'*d | intercept K' | C_ | n_iter | scores F*nC.
    fw: theta P | C 1 | lbfgs work.  iw: [K', fit_intercept, fold].
    Team rows: K' + 1 (logcv_team_rows)."""
    var max_iter = ldi(ip, 0)
    var fi = ldi(ip, 1)
    var kp = ldi(ip, 2)
    var nc = ldi(ip, 3)
    var nf = ldi(ip, 4)
    var tol = ld(fp, 0)
    var stride = d + 1
    var p = kp * stride
    var th = 0
    var cslot = p
    var work = p + 1
    var sc = kp * d + kp + 2
    if t.lead():
        sti(iw, 0, kp)
        sti(iw, 1, fi)
        sti(iw, 3, ldi(ip, 5))
    var cptr = fw + cslot
    var hitr = t.row(0)
    for f in range(nf):
        if t.lead():
            sti(iw, 2, f)
            fill(fw, th, p, Float32(0))
        t.sync()
        for ci in range(nc):
            if t.lead():
                st(fw, cslot, ld(fp, 1 + ci))
            t.sync()
            _ = lbfgs[logistic_objective](t, x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work)
            # each held-out row's hit (1) or miss (0) across the team, then
            # the lead counts them in ascending row order
            for i in range(t.tid, n, t.nt):
                if Int(ld(y, n + i)) == f:
                    var hit = _predict_code(x, i, d, kp if kp > 1 else 1, fi != 0, fw, th) == Int(ld(y, i))
                    st(hitr, i, Float32(1) if hit else Float32(0))
            t.sync()
            if t.lead():
                if ldi(ip, 5) != 0:
                    # their scorer gets sample_weight[test]: the raw weights, at y + 3n
                    var wh = Float32(0)
                    var wt = Float32(0)
                    for i in range(n):
                        if Int(ld(y, n + i)) == f:
                            var wi = ld(y, 3 * n + i)
                            wt = fa(wt, wi)
                            if ld(hitr, i) != 0:
                                wh = fa(wh, wi)
                    st(res, sc + f * nc + ci, fd(wh, wt) if wt > 0 else Float32(0))
                else:
                    var hit = 0
                    var cnt = 0
                    for i in range(n):
                        if Int(ld(y, n + i)) == f:
                            cnt += 1
                            if ld(hitr, i) != 0:
                                hit += 1
                    st(res, sc + f * nc + ci, fd(i2f(hit), i2f(cnt)) if cnt > 0 else Float32(0))
            t.sync()
    var best = 0
    if t.lead():
        var bs = Float32(0)
        for ci in range(nc):
            var acc = Float32(0)
            for f in range(nf):
                acc = fa(acc, ld(res, sc + f * nc + ci))
            var m = fd(acc, i2f(nf))
            if ci == 0 or m > bs:  # DEVIATION 5005: the first best C
                best = ci
                bs = m
        sti(iw, 2, -1)
        st(fw, cslot, ld(fp, 1 + best))
        fill(fw, th, p, Float32(0))
    best = t.bcast_int(best, 3)
    var it = lbfgs[logistic_objective](t, x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work)
    if not t.lead():
        return
    for k in range(kp):
        for j in range(d):
            st(res, k * d + j, ld(fw, th + k * stride + j))
        st(res, kp * d + k, ld(fw, th + k * stride + d) if fi != 0 else Float32(0))
    st(res, kp * d + kp, ld(fp, 1 + best))
    st(res, kp * d + kp + 1, i2f(it if it >= 0 else -it))


def logcv_team_rows(ip: IP) -> Int:
    """Row buffers a LogisticRegressionCV fit needs: K' residuals and the loss term."""
    return ldi(ip, 2) + 1
