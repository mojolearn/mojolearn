# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MINIBATCH SGD (lane/neural-pass103, 2026-10-01; Andrew: SGD on the device as
minibatch SGD with a fixed in-batch combine order, cuML MBSGD's form, batch
4096). SGDClassifier and SGDRegressor (losses and penalties of x_linear/sgd.mojo).

Per epoch the rows are visited in the per-sample fit's order (the splitmix
Fisher-Yates of x_linear/ops.mojo `shuffle`, seeded per problem as before),
MB_BATCH rows a batch. Per batch, at the weights the batch starts from:
  * every row's prediction p_i = fa(row_dot(x_i, w), b) and loss derivative
    dl_i = clip(sgd_dloss(y_i, p_i), +-1e12), times its class and sample
    weights (as the per-sample update);
  * every gradient sum G_j = sum_i dl_i x_ij (and G_b = sum_i dl_i) in the
    blocked order: the batch's rows MB_SUB at a time from zero (batch
    positions ascending), the partials folded ascending with fa;
  * the step, t the batch count from 1: w_j = w_j - eta_t (G_j / bs + the
    penalty's gradient: alpha w_j (l2), alpha sign(w_j) (l1), alpha (l1_ratio
    sign(w_j) + (1 - l1_ratio) w_j) (elasticnet)); b = b - eta_t G_b / bs;
    eta_t sgd_one's schedule with t counting batches.
The epoch objective (only with tol) is the per-batch loss sums in the same
blocked order, batches ascending, over n, plus alpha times the penalty at the
epoch's end; the stopping and the adaptive rate are sgd_one's. Every value
is a fixed chain of IDENTICAL operations: the host and every device give the
same words (the device runs a batch as three grid launches)."""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, fsign, ld, st, ldi, sti, i2f, fill, row_dot, shuffle
from x_linear.sgd import (
    sgd_loss, sgd_dloss, LR_OPTIMAL, LR_INVSCALING, LR_ADAPTIVE, P_NONE, P_L2, P_L1, P_EN,
)
from checks.numerics import identical_pow

comptime MB_BATCH_DEFAULT = 4096
comptime MB_SUB = 256


@always_inline
def mb_subs(bs: Int) -> Int:
    return (bs + MB_SUB - 1) // MB_SUB


@always_inline
def mb_optimal_init(loss: Int, alpha: Float32, eps: Float32) -> Float32:
    """sgd_one's optimal-rate offset."""
    var typw = fsqrt(fd(Float32(1), fsqrt(alpha)))
    var g0 = sgd_dloss(loss, Float32(1), -typw, eps)
    var initial_eta0 = fd(typw, fmax(Float32(1), g0))
    return fd(Float32(1), fm(initial_eta0, alpha))


@always_inline
def mb_eta(lr: Int, eta: Float32, eta0: Float32, alpha: Float32, power_t: Float32, opt_init: Float32, t: Int) -> Float32:
    """The rate of batch t (from 1): sgd_one's schedule with t counting batches;
    `eta` is the constant / adaptive rate."""
    if lr == LR_OPTIMAL:
        return fd(Float32(1), fm(alpha, fs(fa(opt_init, i2f(t)), Float32(1))))
    if lr == LR_INVSCALING:
        return fd(eta0, identical_pow(i2f(t), power_t))
    return eta


@always_inline
def mb_row(x: FP, ys: FP, i: Int, d: Int, w: FP, woff: Int, b: Float32, loss: Int, eps: Float32,
           swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool) -> Tuple[Float32, Float32]:
    """(dl_i weighted, loss_i) of row i at (w, b)."""
    var y = ld(ys, i)
    var p = fa(row_dot(x, i, d, w, woff), b)
    var dl = sgd_dloss(loss, y, p, eps)
    if dl < Float32(-1e12):
        dl = Float32(-1e12)
    elif dl > Float32(1e12):
        dl = Float32(1e12)
    if has_cw or has_sw:
        var cw = Float32(1)
        if has_cw:
            cw = wpos if y > 0 else wneg
        var swi = ld(swp, i) if has_sw else Float32(1)
        dl = fm(dl, fm(cw, swi))
    return (dl, sgd_loss(loss, y, p, eps))


@always_inline
def mb_part(x: FP, d: Int, idx: IP, start: Int, dlv: FP, lv: FP, j: Int, s: Int, bs: Int) -> Float32:
    """Sub-block s of the batch at position `start` (bs rows), from zero:
    j < d the column j's dl x chain, j == d the intercept's dl fold, j == d + 1
    the loss fold. dlv / lv are indexed by batch position."""
    var lo = s * MB_SUB
    var hi = min(bs, lo + MB_SUB)
    var acc = Float32(0)
    if j < d:
        for r in range(lo, hi):
            var i = Int(ldi(idx, start + r))
            acc = fmad(ld(dlv, r), ld(x, i * d + j), acc)
    elif j == d:
        for r in range(lo, hi):
            acc = fa(acc, ld(dlv, r))
    else:
        for r in range(lo, hi):
            acc = fa(acc, ld(lv, r))
    return acc


@always_inline
def mb_step(wj: Float32, g: Float32, bs: Int, eta: Float32, alpha: Float32, l1r: Float32, penalty: Int) -> Float32:
    """w_j after the batch: g the folded gradient sum."""
    var gm = fd(g, i2f(bs))
    if penalty == P_L2:
        gm = fmad(alpha, wj, gm)
    elif penalty == P_L1:
        gm = fa(gm, fm(alpha, fsign(wj)))
    elif penalty == P_EN:
        gm = fa(gm, fm(alpha, fa(fm(l1r, fsign(wj)), fm(fs(Float32(1), l1r), wj))))
    return fs(wj, fm(eta, gm))


@always_inline
def mb_penalty(w: FP, woff: Int, d: Int, alpha: Float32, l1r: Float32, penalty: Int) -> Float32:
    """alpha times the penalty (sgd_one's reg) at w."""
    if penalty == P_NONE:
        return Float32(0)
    var n2 = Float32(0)
    var n1 = Float32(0)
    for j in range(d):
        var wj = ld(w, woff + j)
        n2 = fmad(wj, wj, n2)
        n1 = fa(n1, fabs(wj))
    var l1 = Float32(0) if penalty == P_L2 else (Float32(1) if penalty == P_L1 else l1r)
    return fm(alpha, fa(fm(fm(fs(Float32(1), l1), Float32(0.5)), n2), fm(l1, n1)))


def sgd_mb_one(
    x: FP, ys: FP, n: Int, d: Int, loss: Int, penalty: Int, alpha: Float32, l1r: Float32,
    lr: Int, eta0: Float32, power_t: Float32, eps: Float32, fit_intercept: Bool, max_iter: Int, tol: Float32,
    nic: Int, do_shuffle: Bool, seed: UInt64, w: FP, woff: Int, b: FP, boff: Int, idx: IP,
    swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool, batch: Int, scratch: FP,
) -> Int:
    """One problem on the host (the CPU binding's form of the device's batches).
    scratch: dl batch | loss batch | partials (d + 2) * subs. Returns epochs,
    -1 on a non-finite weight."""
    var dlv = scratch
    var lv = scratch + batch
    var parts = lv + batch
    var nsub = mb_subs(batch)
    fill(w, woff, d, Float32(0))
    var bias = Float32(0)
    for i in range(n):
        sti(idx, i, i)
    var rng = seed
    var eta = eta0
    var opt_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
    var t = 1
    var best = Float32(3.0e38)
    var no_improve = 0
    var need_obj = tol > Float32(-3.0e38)
    var epochs = 0
    for epoch in range(max_iter):
        epochs = epoch + 1
        if do_shuffle:
            shuffle(idx, n, rng)
        var objective = Float32(0)
        var start = 0
        while start < n:
            var bs = min(batch, n - start)
            for r in range(bs):
                var dl_l = mb_row(x, ys, Int(ldi(idx, start + r)), d, w, woff, bias, loss, eps, swp, has_sw, wpos, wneg, has_cw)
                st(dlv, r, dl_l[0])
                st(lv, r, dl_l[1])
            var subs = mb_subs(bs)
            for j in range(d + 2):
                for s in range(subs):
                    st(parts, j * nsub + s, mb_part(x, d, idx, start, dlv, lv, j, s, bs))
            var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
            for j in range(d):
                var g = Float32(0)
                for s in range(subs):
                    g = fa(g, ld(parts, j * nsub + s))
                st(w, woff + j, mb_step(ld(w, woff + j), g, bs, et, alpha, l1r, penalty))
            if fit_intercept:
                var gb = Float32(0)
                for s in range(subs):
                    gb = fa(gb, ld(parts, d * nsub + s))
                bias = fs(bias, fm(et, fd(gb, i2f(bs))))
            if need_obj:
                var lb = Float32(0)
                for s in range(subs):
                    lb = fa(lb, ld(parts, (d + 1) * nsub + s))
                objective = fa(objective, lb)
            t += 1
            start += bs
        var finite = bias == bias and fabs(bias) < Float32(3.0e38)
        for j in range(d):
            var wj = ld(w, woff + j)
            if not (wj == wj and fabs(wj) < Float32(3.0e38)):
                finite = False
        if not finite:
            st(b, boff, Float32(0))
            fill(w, woff, d, Float32(0))
            return -1
        if need_obj:
            var mean_obj = fa(fd(objective, i2f(n)), mb_penalty(w, woff, d, alpha, l1r, penalty))
            if mean_obj > fs(best, tol):
                no_improve += 1
            else:
                no_improve = 0
            if mean_obj < best:
                best = mean_obj
            if no_improve >= nic:
                if lr == LR_ADAPTIVE and eta > Float32(1e-6):
                    eta = fd(eta, Float32(5))
                    no_improve = 0
                else:
                    break
    st(b, boff, bias)
    return epochs
