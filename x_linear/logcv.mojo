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
    axpy_acc, par_rows,
)
from x_linear.lbfgs import lbfgs, lbfgs_work
from checks.numerics import identical_sigmoid, identical_softplus, ftz


def logistic_objective(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int, sc: FP) -> Float32:
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
        for i in range(lo, hi):
            if fold >= 0 and Int(ld(y, n + i)) == fold:
                continue
            var wi = ld(y, 2 * n + i) if sw else Float32(1)
            var label = Int(ld(y, i))
            if kp == 1:
                var z = fa(row_dot(x, i, d, th, toff), ld(th, toff + d) if fi else Float32(0))
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
                    var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
                    st(sr, i * kp + k, z)
                    zmax = fmax(zmax, z)
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
    for i in range(n):
        if fold >= 0 and Int(ld(y, n + i)) == fold:
            continue
        rows += 1
        if sw:
            wrows = fa(wrows, ld(y, 2 * n + i))
        acc = fa(acc, ld(sl, i))
        for k in range(kp):
            var r = ld(sr, i * kp + k)
            axpy_acc(g, goff + k * stride, r, x, i * d, d)
            if fi:
                st(g, goff + k * stride + d, fa(ld(g, goff + k * stride + d), r))
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
    return fa(fm(acc, inv_n), fm(fm(Float32(0.5), lam), reg))


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


def logcv_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [max_iter, fit_intercept, K', n_Cs, n_folds, sample_weight]; fp: [tol, Cs...].
    With sample_weight, y = labels | folds | fit weights (sample x class) |
    score weights (sample): the loss sum and the penalty scale use sum(w)
    of the training rows (their LinearModelLoss), the held-out accuracy is
    weighted by the raw sample weights (their scorer's sample_weight).
    res: coef K'*d | intercept K' | C_ | n_iter | scores F*nC.
    fw: theta P | C 1 | objective scratch n*(K'+1) | lbfgs work.  iw: [K', fit_intercept, fold]."""
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
    var work = p + 1 + n * (kp + 1)
    var sc = kp * d + kp + 2
    sti(iw, 0, kp)
    sti(iw, 1, fi)
    sti(iw, 3, ldi(ip, 5))
    var cptr = fw + cslot
    for f in range(nf):
        sti(iw, 2, f)
        fill(fw, th, p, Float32(0))
        for ci in range(nc):
            st(fw, cslot, ld(fp, 1 + ci))
            _ = lbfgs[logistic_objective](x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work, cptr + 1)
            var hit = 0
            var cnt = 0
            for i in range(n):
                if Int(ld(y, n + i)) == f:
                    cnt += 1
                    if _predict_code(x, i, d, kp if kp > 1 else 1, fi != 0, fw, th) == Int(ld(y, i)):
                        hit += 1
            if ldi(ip, 5) != 0:
                # their scorer gets sample_weight[test]: the raw weights, at y + 3n
                var wh = Float32(0)
                var wt = Float32(0)
                for i in range(n):
                    if Int(ld(y, n + i)) == f:
                        var wi = ld(y, 3 * n + i)
                        wt = fa(wt, wi)
                        if _predict_code(x, i, d, kp if kp > 1 else 1, fi != 0, fw, th) == Int(ld(y, i)):
                            wh = fa(wh, wi)
                st(res, sc + f * nc + ci, fd(wh, wt) if wt > 0 else Float32(0))
            else:
                st(res, sc + f * nc + ci, fd(i2f(hit), i2f(cnt)) if cnt > 0 else Float32(0))
    var best = 0
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
    var it = lbfgs[logistic_objective](x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work, cptr + 1)
    for k in range(kp):
        for j in range(d):
            st(res, k * d + j, ld(fw, th + k * stride + j))
        st(res, kp * d + k, ld(fw, th + k * stride + d) if fi != 0 else Float32(0))
    st(res, kp * d + kp, ld(fp, 1 + best))
    st(res, kp * d + kp + 1, i2f(it if it >= 0 else -it))
