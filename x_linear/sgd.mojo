# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PLAIN SGD: SGDClassifier, SGDRegressor, Perceptron, PassiveAggressive*,
SGDOneClassSVM (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_sgd_fast.pyx.tp`, `_plain_sgd`
(the epoch loop, lines ~458-628: the optimal/invscaling rates, the PA-I/PA-II
step, weight decay, Tsuruoka's cumulative L1 penalty `l1penalty`, the
objective-based stopping with `n_iter_no_change`, the adaptive rate's divide
by five) and the loss classes at the top of that file. Differences, named:
  * float32 throughout (theirs accumulates in float64);
  * (lane/neural-pass139) the per-sample path keeps their lazy `wscale`
    (w = wscale * v; the decay multiplies wscale only, the step adds
    update / wscale * x to v, the fold into v below 1e-9) with wscale a
    float-float pair (hi, lo) built from fma's exact transforms
    (`ws_decay`, `ws_mul`, `ws_div`, `ws_clip`), since float32 cannot hold
    1 - eta * alpha at 1M rows x 20 epochs (eta * alpha ~ 5e-8); the
    minibatch path and the warp form still multiply w directly;
  * (lane/neural-pass139) the per-sample one-class intercept (near 1) is a
    float-float (`ff_add`), the hinge decided on the margin p - 1
    (`oc_hinge`), and every SGD path returns offset_ = 1 - intercept for
    k == 1 (`oc_offset`), since float32 near 1 drops the late steps;
  * the shuffle is Fisher-Yates over splitmix64 (x_linear/ops.mojo), not
    their `SequentialDataset.shuffle` over numpy's MT19937, so a fit agrees
    with theirs in quality, not in bits; the OvR problem c is seeded
    `seed + 1000003 * c`;
  * (lane/neural-pass139) each sample's predictor w . x_i is MB_DBLK-column
    blocks, each a chain from zero, folded ascending (`mb_dot` with
    MB_DBLK), and the objective's penalty norms the same blocks
    (`sgd_reg_blocked`): the host's `sgd_one` and the device's
    team-parallel per-sample kernel (x_linear/device.mojo `_sgd_ps_grid`)
    run the same chains.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fexp, flog, fabs, fmax, fmin,
    ld, st, ldi, sti, i2f, fill, row_dot, shuffle, axpy_acc, scale_acc, ftzv, par_rows, fz, xmad, fsign,
)
from std.sys.info import is_gpu
from std.sys.compile import is_defined
from std.sys.info import is_apple_gpu
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.team import Team
from checks.numerics import identical_pow

comptime L_HINGE = 0
comptime L_LOG = 1
comptime L_MODHUBER = 2
comptime L_SQHINGE = 3
comptime L_PERCEPTRON = 4
comptime L_SQUARED = 10
comptime L_HUBER = 11
comptime L_EPSINS = 12
comptime L_SQEPSINS = 13

comptime LR_CONSTANT = 0
comptime LR_OPTIMAL = 1
comptime LR_INVSCALING = 2
comptime LR_ADAPTIVE = 3
comptime LR_PA1 = 5
comptime LR_PA2 = 6

comptime P_NONE = 0
comptime P_L2 = 1
comptime P_L1 = 2
comptime P_EN = 3


def sgd_loss(kind: Int, y: Float32, p: Float32, eps: Float32) -> Float32:
    if kind == L_HINGE or kind == L_PERCEPTRON:
        var thr = Float32(1) if kind == L_HINGE else Float32(0)
        var z = fm(p, y)
        return fs(thr, z) if z <= thr else Float32(0)
    if kind == L_LOG:
        var z = fm(p, y)
        if z > 18:
            return fexp(-z)
        if z < -18:
            return -z
        return flog(fa(Float32(1), fexp(-z)))
    if kind == L_MODHUBER:
        var z = fm(p, y)
        if z >= 1:
            return Float32(0)
        if z >= -1:
            var t = fs(Float32(1), z)
            return fm(t, t)
        return fm(Float32(-4), z)
    if kind == L_SQHINGE:
        var z = fs(Float32(1), fm(p, y))
        return fm(z, z) if z > 0 else Float32(0)
    if kind == L_SQUARED:
        var r = fs(p, y)
        return fm(Float32(0.5), fm(r, r))
    if kind == L_HUBER:
        var r = fs(p, y)
        var ar = fabs(r)
        if ar <= eps:
            return fm(Float32(0.5), fm(r, r))
        return fs(fm(eps, ar), fm(Float32(0.5), fm(eps, eps)))
    if kind == L_EPSINS:
        var r = fs(fabs(fs(y, p)), eps)
        return r if r > 0 else Float32(0)
    # L_SQEPSINS
    var r = fs(fabs(fs(y, p)), eps)
    return fm(r, r) if r > 0 else Float32(0)


def sgd_dloss(kind: Int, y: Float32, p: Float32, eps: Float32) -> Float32:
    if kind == L_HINGE or kind == L_PERCEPTRON:
        var thr = Float32(1) if kind == L_HINGE else Float32(0)
        return -y if fm(p, y) <= thr else Float32(0)
    if kind == L_LOG:
        var z = fm(p, y)
        if z > 18:
            return fm(-y, fexp(-z))
        if z < -18:
            return -y
        return fd(-y, fa(fexp(z), Float32(1)))
    if kind == L_MODHUBER:
        var z = fm(p, y)
        if z >= 1:
            return Float32(0)
        if z >= -1:
            return fm(fm(Float32(2), fs(Float32(1), z)), -y)
        return fm(Float32(-4), y)
    if kind == L_SQHINGE:
        var z = fs(Float32(1), fm(p, y))
        return fm(fm(Float32(-2), y), z) if z > 0 else Float32(0)
    if kind == L_SQUARED:
        return fs(p, y)
    if kind == L_HUBER:
        var r = fs(p, y)
        if fabs(r) <= eps:
            return r
        return eps if r > 0 else -eps
    if kind == L_EPSINS:
        var z = fs(y, p)
        if z > eps:
            return Float32(-1)
        if z < -eps:
            return Float32(1)
        return Float32(0)
    var z = fs(y, p)
    if z > eps:
        return fm(Float32(-2), fs(z, eps))
    if z < -eps:
        return fm(Float32(2), fs(-z, eps))
    return Float32(0)


def sgd_one(
    x: FP, ys: FP, n: Int, d: Int,
    loss: Int, penalty: Int, alpha: Float32, l1_ratio_in: Float32,
    lr: Int, eta0: Float32, power_t: Float32, eps: Float32,
    fit_intercept: Bool, max_iter: Int, tol: Float32, n_iter_no_change: Int,
    do_shuffle: Bool, seed: UInt64, one_class: Bool,
    w: FP, woff: Int, b: FP, boff: Int, q: FP, idx: IP,
    swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool,
) -> Int:
    """One binary/regression problem on targets `ys`. Returns epochs run,
    or -1 on a non-finite weight (their ValueError)."""
    var l1_ratio = l1_ratio_in
    if penalty == P_L2:
        l1_ratio = Float32(0)
    elif penalty == P_L1:
        l1_ratio = Float32(1)
    fill(w, woff, d, Float32(0))
    fill(q, 0, d, Float32(0))
    var intercept = Float32(1) if one_class else Float32(0)
    var il = Float32(0)  # one-class: the intercept's low word
    for i in range(n):
        sti(idx, i, i)
    var rng = seed
    var eta = eta0
    var optimal_init = Float32(0)
    if lr == LR_OPTIMAL:
        var typw = fsqrt(fd(Float32(1), fsqrt(alpha)))
        var g0 = sgd_dloss(loss, Float32(1), -typw, eps)
        var initial_eta0 = fd(typw, fmax(Float32(1), g0))
        optimal_init = fd(Float32(1), fm(initial_eta0, alpha))
    var u = Float32(0)
    var t = 1
    var best = Float32(3.0e38)
    var no_improve = 0
    var decay_factor = fm(fs(Float32(1), l1_ratio), alpha)
    # their WeightVector's wscale, a float-float (hi, lo): w = wscale * v
    var whi = Float32(1)
    var wlo = Float32(0)
    var epochs = 0
    for epoch in range(max_iter):
        epochs = epoch + 1
        var objective = Float32(0)
        if do_shuffle:
            shuffle(idx, n, rng)
        for r in range(n):
            var i = ldi(idx, r)
            var y = ld(ys, i)
            # lane/neural-pass139: the predictor as MB_DBLK-column blocks,
            # each from zero, folded ascending (`mb_dot`; the device's
            # team-parallel per-sample kernel takes the same words)
            var dotw = ws_mul(mb_dot(x, i, d, w, woff, MB_DBLK), whi, wlo)
            # one-class: the intercept is the float-float (intercept, il)
            var p = fa(fa(dotw, il), intercept) if one_class else fa(dotw, intercept)
            if lr == LR_OPTIMAL:
                eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
            elif lr == LR_INVSCALING:
                eta = fd(eta0, identical_pow(i2f(t), power_t))
            var oc = one_class and loss == L_HINGE
            var och = oc_hinge(dotw, intercept, il)
            var cur = och[0] if oc else sgd_loss(loss, y, p, eps)
            objective = fa(objective, cur)
            if lr != LR_PA1 and lr != LR_PA2:
                if penalty != P_NONE and tol > Float32(-3.0e38):
                    # lane/neural-pass139: the norms as MB_DBLK-weight blocks
                    # (`sgd_reg_blocked`); only `tol` reads the objective
                    var nrm = sgd_reg_blocked(w, woff, d)
                    # wscale's norms: wscale^2 |v|^2 and wscale |v|_1
                    var n2 = fm(fm(whi, whi), nrm[0])
                    var n1 = fm(whi, nrm[1])
                    var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                    objective = fa(objective, fm(alpha, reg))
                if one_class:
                    objective = fa(objective, fm(intercept, alpha))
            var update: Float32
            if lr == LR_PA1 or lr == LR_PA2:
                var sq = Float32(0)
                for j in range(d):
                    var xj = ld(x, i * d + j)
                    sq = fmad(xj, xj, sq)
                if lr == LR_PA1:
                    if sq == 0:
                        continue
                    update = fmin(eta0, fd(cur, sq))
                else:
                    update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
                if loss == L_HINGE:
                    update = fm(update, y)
                elif fs(y, p) < 0:
                    update = -update
            else:
                var dl = och[1] if oc else sgd_dloss(loss, y, p, eps)
                if dl < Float32(-1e12):
                    dl = Float32(-1e12)
                elif dl > Float32(1e12):
                    dl = Float32(1e12)
                update = fm(-eta, dl)
            if has_cw or has_sw:
                # theirs: update *= class_weight * sample_weight
                var cw = Float32(1)
                if has_cw:
                    cw = wpos if y > 0 else wneg
                var swi = ld(swp, i) if has_sw else Float32(1)
                update = fm(update, fm(cw, swi))
            if penalty == P_L2 or penalty == P_EN:
                # their w.scale: wscale *= max(0, 1 - decay_factor * eta),
                # folded into v below 1e-9 (their reset_wscale)
                var ws = ws_decay(whi, wlo, fm(decay_factor, eta))
                whi = ws[0]
                wlo = ws[1]
                if whi < WS_RESET:
                    ws_fold(w, woff, d, whi, wlo)
                    whi = Float32(1)
                    wlo = Float32(0)
            if update != 0:
                axpy_acc(w, woff, ws_div(update, whi, wlo), x, i * d, d)
            if fit_intercept:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    if one_class:
                        var ia = ff_add(intercept, il, iu)
                        intercept = ia[0]
                        il = ia[1]
                    else:
                        intercept = fa(intercept, iu)
            if penalty == P_L1 or penalty == P_EN:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
                _l1_clip(w, woff, q, u, d, whi)
            t += 1
        # their floating-point under-/overflow check
        var finite = intercept == intercept and fabs(intercept) < Float32(3.0e38)
        for j in range(d):
            var wj = ws_mul(ld(w, woff + j), whi, wlo)
            if not (wj == wj and fabs(wj) < Float32(3.0e38)):
                finite = False
        if not finite:
            st(b, boff, Float32(0))
            fill(w, woff, d, Float32(0))
            return -1
        var mean_obj = fd(objective, i2f(n))
        if tol > Float32(-3.0e38) and mean_obj > fs(best, tol):
            no_improve += 1
        else:
            no_improve = 0
        if mean_obj < best:
            best = mean_obj
        if no_improve >= n_iter_no_change:
            if lr == LR_ADAPTIVE and eta > Float32(1e-6):
                eta = fd(eta, Float32(5))
                no_improve = 0
            else:
                break
    # coef = wscale * v
    for j in range(d):
        st(w, woff + j, ws_mul(ld(w, woff + j), whi, wlo))
    # one-class: the slot holds offset_ = 1 - intercept (`oc_offset`)
    st(b, boff, oc_offset(intercept, il) if one_class else intercept)
    return epochs


# `sgd_fit`'s target for problem c (the device per-sample kernel's too).
@always_inline
def _sgd_target(k: Int, c: Int, v: Float32) -> Float32:
    """`sgd_fit`'s `ys[i]` for label v (problem c of k classes)."""
    if k == 0:
        return v
    if k == 1:
        return Float32(1)
    if k == 2:
        return Float32(1) if v == Float32(1) else Float32(-1)
    return Float32(1) if v == i2f(c) else Float32(-1)


@always_inline
def sgd_mb_on(batch: Int, k: Int, lr: Int) -> Bool:
    """Whether a problem takes the minibatch form (lane/neural-pass103): a batch
    size was given. Since lane/neural-pass132 the one-class problem and the
    passive-aggressive rates take it too (see `mb_row`, `sgd_mb_one`)."""
    return batch > 0


def sgd_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_classes, loss, penalty, lr, fit_intercept, max_iter, n_iter_no_change,
    shuffle, seed_lo, seed_hi, sample_weight, class_weight]; with
    sample_weight y = labels n | weights n; with class_weight fp carries
    [.., pos weight per problem (P), neg weight per problem (P)]; n_classes 0 = regression, 1 = one-class,
    2 = binary (positive class = label 1), K > 2 = one-vs-rest.
    fp: [alpha, l1_ratio, eta0, power_t, epsilon, tol].
    y: labels as 0..K-1 (classification) or targets. res: coef (P*d),
    intercept (P; for k == 1 the one-class offset_ = 1 - intercept,
    lane/neural-pass139), n_iter (1), status (1: 0 ok, -1 non-finite).
    The host binding's fit (cpu-gpu-cleanup w2-linear: the device's warp
    and thread forms, which ran a problem on one warp / one thread, are
    deleted; the device runs `_sgd_mb_grid` / `_sgd_ps_grid`).
    Host form (lane linear-cpu): fw: per problem n (targets) + d (q), then P
    (epochs run); iw: per problem n (order); the problems run as independent
    units (par_rows)."""
    var k = ldi(ip, 0)
    var loss = ldi(ip, 1)
    var penalty = ldi(ip, 2)
    var lr = ldi(ip, 3)
    var fit_intercept = ldi(ip, 4) != 0
    var max_iter = ldi(ip, 5)
    var nic = ldi(ip, 6)
    var do_shuffle = ldi(ip, 7) != 0
    var seed = (UInt64(UInt32(ip.unsafe_load(9))) << 32) | UInt64(UInt32(ip.unsafe_load(8)))
    var alpha = ld(fp, 0)
    var l1r = ld(fp, 1)
    var eta0 = ld(fp, 2)
    var power_t = ld(fp, 3)
    var eps = ld(fp, 4)
    var tol = ld(fp, 5)
    var has_sw = ldi(ip, 10) != 0
    var has_cw = ldi(ip, 11) != 0
    # ip[12] (lane/neural-pass103): the minibatch size, 0 the per-sample fit
    var batch = ldi(ip, 12) if ldi(ip, 12) > 0 else 0
    # ip[13] (lane/neural-pass132): the per-sample-equivalent batch step (`mb_step`)
    var bsum = ldi(ip, 13) != 0
    var swp = y + n
    var problems = k if k > 2 else 1
    var epr = fw + problems * (n + d)  # the host's epochs run per problem
    # one-vs-rest problems are independent (their own targets, q, order and
    # result slots): the host may run them at once (lane linear-cpu)

    def run_problems(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm k, imm fw, imm iw, imm res, imm problems,
                                        imm loss, imm penalty, imm alpha, imm l1r, imm lr, imm eta0, imm power_t,
                                        imm eps, imm fit_intercept, imm max_iter, imm tol, imm nic, imm do_shuffle,
                                        imm seed, imm swp, imm has_sw, imm has_cw, imm fp, imm epr, imm batch, imm bsum}:
        for c in range(lo, hi):
            var ys = fw + c * (n + d)
            var q = ys + n
            var idx = iw + c * n
            for i in range(n):
                var v = ld(y, i)
                if k == 0:
                    st(ys, i, v)
                elif k == 1:
                    st(ys, i, Float32(1))
                elif k == 2:
                    st(ys, i, Float32(1) if v == Float32(1) else Float32(-1))
                else:
                    st(ys, i, Float32(1) if v == i2f(c) else Float32(-1))
            var ep: Int
            if sgd_mb_on(batch, k, lr):
                # lane/neural-pass103: the minibatch form (x_linear/sgd_mb.mojo)
                var sl = List[Float32](length=2 * batch + (d + 2) * mb_subs(batch, mb_sub_size(batch)), fill=Float32(0))
                ep = sgd_mb_one(
                    x, ys, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                    fit_intercept, max_iter, tol, nic, do_shuffle,
                    seed + UInt64(1000003) * UInt64(c), res, c * d, res, problems * d + c, idx,
                    swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                    ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw, batch,
                    FP(unsafe_from_address=Int(sl.unsafe_ptr())), k == 1, bsum,
                )
                _ = sl^
            else:
                ep = sgd_one(
                    x, ys, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                    fit_intercept, max_iter, tol, nic, do_shuffle,
                    seed + UInt64(1000003) * UInt64(c), k == 1,
                    res, c * d, res, problems * d + c, q, idx,
                    swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                    ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw,
                )
            st(epr, c, i2f(ep))

    par_rows(run_problems, problems, 1)
    if not t.lead():
        return
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        var ep = Int(ld(epr, c))
        if ep < 0:
            status = -1
        elif ep > max_epochs:
            max_epochs = ep
    st(res, problems * d + problems, i2f(max_epochs))
    st(res, problems * d + problems + 1, i2f(status))


@always_inline
def _clip_one(z: Float32, uq_pos: Float32, uq_neg: Float32) -> Float32:
    """Tsuruoka's cumulative clip of one weight (fa(u, q) and fs(u, q) given)."""
    if z > 0:
        return fmax(Float32(0), fs(z, uq_pos))
    if z < 0:
        return fmin(Float32(0), fa(z, uq_neg))
    return z


# ------------------------------------------------ wscale as float-float (lane/neural-pass139)
# Their WeightVector (`_weight_vector.pyx.tp`): w = wscale * v with wscale a
# float64 scalar. Ours is a float-float pair (hi, lo), hi + lo the value,
# |lo| <= ulp(hi) / 2, every statement one of the lane's identical ops
# (fa / fs / fm / fd / fmad: one rounding each, the flush around it), so
# every column rounds alike and Apple needs no FP64. With wscale = (1, 0)
# (the PA rates, penalty none and L1, which never scale) each statement
# reduces exactly: `ws_mul(a, 1, 0)` = fm(a, 1) = a, `ws_div(a, 1, 0)` =
# fd(a, 1) = a, `ws_clip(..., 1)` = the old clip. Host `sgd_one` and the
# device's `_sgd_ps_kernel_body` run the same statements.
comptime WS_RESET = Float32(1e-9)


@always_inline
def ws_mul(a: Float32, hi: Float32, lo: Float32) -> Float32:
    """a * (hi + lo) rounded once to float32: fmad(a, hi, fm(a, lo)); a zero
    lo is fm(a, hi) (so wscale = (1, 0) returns a, its sign and an infinity
    included)."""
    if lo == 0:
        return fm(a, hi)
    return fmad(a, hi, fm(a, lo))


@always_inline
def ws_div(a: Float32, hi: Float32, lo: Float32) -> Float32:
    """a / (hi + lo): c0 = fd(a, hi), then c0 (1 - lo / hi) as
    fmad(c0, fd(fs(0, lo), hi), c0); a zero lo is fd(a, hi)."""
    if lo == 0:
        return fd(a, hi)
    var c0 = fd(a, hi)
    return fmad(c0, fd(fs(Float32(0), lo), hi), c0)


@always_inline
def ws_decay(hi: Float32, lo: Float32, s: Float32) -> Tuple[Float32, Float32]:
    """Their `w.scale(max(0, 1 - s))` for s = decay_factor * eta (float32):
    (hi + lo) - s (hi + lo) in float-float. A factor fs(1, s) <= 0 is
    (0, 0) (their scale(0); the caller's reset then zeroes v).
      ph = fm(s, hi); pe = fmad(s, hi, -ph)       (two-prod: s hi = ph + pe)
      pl = fmad(s, lo, pe)                        (s lo + pe)
      sh = fs(hi, ph); bb = fs(sh, hi)            (two-sum of hi and -ph)
      er = fa(fs(hi, fs(sh, bb)), fs(-ph, bb))
      tl = fa(er, fs(lo, pl))
      nh = fa(sh, tl); nl = fs(tl, fs(nh, sh))    (fast two-sum)"""
    if fs(Float32(1), s) <= 0:
        return (Float32(0), Float32(0))
    var ph = fm(s, hi)
    var pe = fmad(s, hi, -ph)
    var pl = fmad(s, lo, pe)
    var sh = fs(hi, ph)
    var bb = fs(sh, hi)
    var er = fa(fs(hi, fs(sh, bb)), fs(-ph, bb))
    var tl = fa(er, fs(lo, pl))
    var nh = fa(sh, tl)
    var nl = fs(tl, fs(nh, sh))
    return (nh, nl)


@always_inline
def ff_add(h: Float32, l: Float32, a: Float32) -> Tuple[Float32, Float32]:
    """(h + l) + a in float-float (the one-class intercept's step):
      sh = fa(h, a); bb = fs(sh, h)               (two-sum of h and a)
      er = fa(fs(h, fs(sh, bb)), fs(a, bb))
      tl = fa(er, l)
      nh = fa(sh, tl); nl = fs(tl, fs(nh, sh))    (fast two-sum)"""
    var sh = fa(h, a)
    var bb = fs(sh, h)
    var er = fa(fs(h, fs(sh, bb)), fs(a, bb))
    var tl = fa(er, l)
    var nh = fa(sh, tl)
    var nl = fs(tl, fs(nh, sh))
    return (nh, nl)


@always_inline
def oc_hinge(dotw: Float32, ih: Float32, il: Float32) -> Tuple[Float32, Float32]:
    """The one-class hinge (y = 1, threshold 1) on p = dotw + (ih + il),
    decided on the margin m = p - 1 = fa(fa(fs(ih, 1), il), dotw) (ih - 1
    exact for ih in [0.5, 2]): (loss, dloss) = (fs(0, m), -1) when m <= 0,
    else (0, 0). Their dloss decides z <= 1 in float64; float32's p near 1
    cannot."""
    var m = fa(fa(fs(ih, Float32(1)), il), dotw)
    if m <= 0:
        return (fs(Float32(0), m), Float32(-1))
    return (Float32(0), Float32(0))


@always_inline
def oc_offset(ih: Float32, il: Float32) -> Float32:
    """offset_ = 1 - (ih + il) rounded once: fs(fs(1, ih), il) (1 - ih
    exact for ih in [0.5, 2])."""
    return fs(fs(Float32(1), ih), il)


def ws_fold(w: FP, woff: Int, d: Int, hi: Float32, lo: Float32):
    """Their `reset_wscale`: v_j = ws_mul(v_j, hi, lo) (the caller then sets
    wscale = (1, 0))."""
    for j in range(d):
        st(w, woff + j, ws_mul(ld(w, woff + j), hi, lo))


@always_inline
def ws_clip(z: Float32, u: Float32, qj: Float32, wf: Float32) -> Tuple[Float32, Float32]:
    """Their `l1penalty` for one weight v_j = z with wscale wf (> 0, hi of
    the pair): (the new v_j, the new q_j).
      z > 0: v = fmax(0, fs(z, fd(fa(u, q), wf)))
      z < 0: v = fmin(0, fa(z, fd(fs(u, q), wf)))
      q' = fa(q, fm(wf, fs(v, z)))
    wf = 1 is `_clip_one` and the old q statement, bit for bit."""
    var nz = z
    if z > 0:
        nz = fmax(Float32(0), fs(z, fd(fa(u, qj), wf)))
    elif z < 0:
        nz = fmin(Float32(0), fa(z, fd(fs(u, qj), wf)))
    return (nz, fa(qj, fm(wf, fs(nz, z))))


def _l1_clip(w: FP, woff: Int, q: FP, u: Float32, d: Int, wf: Float32 = Float32(1)):
    """The cumulative L1 penalty over every weight (their `l1penalty`); each
    weight and its q entry are updated on their own, so lanes are
    independent and the vector form is the scalar loop bit for bit. wf (the
    wscale's hi) != 1 runs `ws_clip` per weight, which at wf = 1 is the
    statements below exactly."""
    if wf != 1:
        for j in range(d):
            var r = ws_clip(ld(w, woff + j), u, ld(q, j), wf)
            st(w, woff + j, r[0])
            st(q, j, r[1])
        return
    comptime if is_gpu():
        for j in range(d):
            var z = ld(w, woff + j)
            var nz = _clip_one(z, fa(u, ld(q, j)), fs(u, ld(q, j)))
            st(w, woff + j, nz)
            st(q, j, fa(ld(q, j), fs(nz, z)))
    else:
        comptime V = 8
        var uv = ftzv[V](SIMD[DType.float32, V](u))
        var zero = SIMD[DType.float32, V](0)
        var j = 0
        while j + V <= d:
            var z = w.unsafe_load[width=V](woff + j)
            var qv = ftzv[V](q.unsafe_load[width=V](j))
            var zf = ftzv[V](z)
            var pos = ftzv[V](zf - ftzv[V](uv + qv))   # fs(z, fa(u, q))
            var neg = ftzv[V](zf + ftzv[V](uv - qv))   # fa(z, fs(u, q))
            var pmax = zero.ge(pos).select(zero, pos)  # fmax(0, pos)
            var nmin = zero.le(neg).select(zero, neg)  # fmin(0, neg)
            var nz = z.gt(zero).select(pmax, z.lt(zero).select(nmin, z))
            w.unsafe_store[width=V](woff + j, nz)
            q.unsafe_store[width=V](j, ftzv[V](qv + ftzv[V](ftzv[V](nz) - zf)))
            j += V
        while j < d:
            var z = ld(w, woff + j)
            var nz = _clip_one(z, fa(u, ld(q, j)), fs(u, ld(q, j)))
            st(w, woff + j, nz)
            st(q, j, fa(ld(q, j), fs(nz, z)))
            j += 1


def sgd_team_rows(ip: IP) -> Int:
    """Row buffers of an SGD fit's team scratch (the host binding's; the
    device runs SGD on the grid, x_linear/device.mojo `_sgd_mb_grid` /
    `_sgd_ps_grid`, and never a team)."""
    var k = ldi(ip, 0)
    var problems = k if k > 2 else 1
    return 2 * problems + 1

# ------------------------------------------------ minibatch SGD (lane/neural-pass103)
# MINIBATCH SGD (lane/neural-pass103, 2026-10-01; Andrew: SGD on the device as
# minibatch SGD with a fixed in-batch combine order, cuML MBSGD's form, batch
# 4096). SGDClassifier and SGDRegressor (losses and penalties of x_linear/sgd.mojo).
#
# Per epoch the rows are visited in the per-sample fit's order (the splitmix
# Fisher-Yates of x_linear/ops.mojo `shuffle`, seeded per problem as before),
# MB_BATCH rows a batch. Per batch, at the weights the batch starts from:
#   * every row's prediction p_i = fa(row_dot(x_i, w), b) and loss derivative
#     dl_i = clip(sgd_dloss(y_i, p_i), +-1e12), times its class and sample
#     weights (as the per-sample update);
#   * every gradient sum G_j = sum_i dl_i x_ij (and G_b = sum_i dl_i) in the
#     blocked order: the batch's rows MB_SUB at a time from zero (batch
#     positions ascending), the partials folded ascending with fa;
#   * the step, t the batch count from 1: w_j = w_j - eta_t (G_j / bs + the
#     penalty's gradient: alpha w_j (l2), alpha sign(w_j) (l1), alpha (l1_ratio
#     sign(w_j) + (1 - l1_ratio) w_j) (elasticnet)); b = b - eta_t G_b / bs;
#     eta_t sgd_one's schedule with t counting batches.
# lane/neural-pass132 (2026-10-02; GPU-only rule: Perceptron, PA and the
# one-class SVM ran per sample on the host inside the device binding):
#   * passive-aggressive (pa1 / pa2): row i's own PA step u_i (theirs, at the
#     batch's starting weights) is its dl_i = -u_i and the rate is 1, so the
#     batch moves by the MEAN of its rows' PA steps (no penalty, as theirs);
#   * one-class: the intercept starts at 1 and every batch also takes their
#     offset step, b = b - eta_t G_b / bs - eta_t alpha; the epoch objective
#     adds alpha b (theirs adds alpha times the intercept per row);
#   * Perceptron is the perceptron loss at a constant rate (already covered).
# The epoch objective (only with tol) is the per-batch loss sums in the same
# blocked order, batches ascending, over n, plus alpha times the penalty at the
# epoch's end; the stopping and the adaptive rate are sgd_one's. Every value
# is a fixed chain of IDENTICAL operations: the host and every device give the
# same words (the device runs a batch as three grid launches).

comptime MB_BATCH_DEFAULT = 4096
comptime MB_SUB = 256


@always_inline
def mb_subs(bs: Int, sub: Int = MB_SUB) -> Int:
    return (bs + sub - 1) // sub


# lane/neural-pass135 (2026-10-02; new bits for batch <= MB_SMALL_BATCH, the
# Perceptron / PA / opt-in one-class defaults): the column partials over
# MB_SUB_SMALL-row sub-blocks and each row's linear predictor as MB_DBLK-
# column blocks, each from zero, folded ascending. Every dependent chain is
# about 8x shorter (the peer's MI325X ran these one-block batches
# latency-bound: perceptron istella 10.7 s against the host's 9.6).
# SGDClassifier / SGDRegressor at their batch 4096 keep #103's chains.
comptime MB_SMALL_BATCH = 256
comptime MB_SUB_SMALL = 32
comptime MB_DBLK = 32


@always_inline
def mb_sub_size(batch: Int) -> Int:
    return MB_SUB_SMALL if batch <= MB_SMALL_BATCH else MB_SUB


@always_inline
def mb_dblk(batch: Int) -> Int:
    return MB_DBLK if batch <= MB_SMALL_BATCH else 0


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
    `eta` is the constant / adaptive rate; 1 for the PA rates (the step is in
    each row's dl, `mb_row`)."""
    if lr == LR_PA1 or lr == LR_PA2:
        return Float32(1)
    if lr == LR_OPTIMAL:
        return fd(Float32(1), fm(alpha, fs(fa(opt_init, i2f(t)), Float32(1))))
    if lr == LR_INVSCALING:
        return fd(eta0, identical_pow(i2f(t), power_t))
    return eta


@always_inline
def mb_row(x: FP, ys: FP, i: Int, d: Int, w: FP, woff: Int, b: Float32, loss: Int, eps: Float32,
           swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool,
           lr: Int, eta0: Float32, dblk: Int = 0) -> Tuple[Float32, Float32]:
    """(dl_i weighted, loss_i) of row i at (w, b); with a PA rate dl_i is minus
    the row's PA step (sgd_one's statements, eta0 = C)."""
    return mb_row_dot(x, ys, i, d, mb_dot(x, i, d, w, woff, dblk), b, loss, eps, swp, has_sw, wpos, wneg, has_cw,
                      lr, eta0, swp, False)


@always_inline
def mb_rowsq(x: FP, i: Int, d: Int) -> Float32:
    """Row i's squared norm, j ascending, one fmad a term (the PA rates' |x|^2)."""
    var sq = Float32(0)
    for j in range(d):
        var xj = ld(x, i * d + j)
        sq = fmad(xj, xj, sq)
    return sq


@always_inline
def mb_block_dot(x: FP, i: Int, d: Int, w: FP, woff: Int, blk: Int) -> Float32:
    """Block blk (MB_DBLK columns) of row i's predictor from zero: `mb_dot`'s
    chain for that block, with all MB_DBLK loads issued before it (the
    device's one-task-per-block form, lane/neural-pass132 AMD fix)."""
    var j0 = blk * MB_DBLK
    var m = min(MB_DBLK, d - j0)
    var xv = SIMD[DType.float32, MB_DBLK]()
    var wv = SIMD[DType.float32, MB_DBLK]()
    comptime for u in range(MB_DBLK):
        if u < m:
            xv[u] = ld(x, i * d + j0 + u)
            wv[u] = ld(w, woff + j0 + u)
    var acc = Float32(0)
    comptime for u in range(MB_DBLK):
        if u < m:
            acc = fmad(xv[u], wv[u], acc)
    return acc


@always_inline
def sgd_reg_block(w: FP, woff: Int, d: Int, blk: Int) -> Tuple[Float32, Float32]:
    """Block blk (MB_DBLK weights) of sgd_one's penalty norms from zero
    (lane/neural-pass139): (the fmad chain of w_j^2, the fa chain of |w_j|),
    j ascending, with all the block's loads issued first."""
    var j0 = blk * MB_DBLK
    var m = min(MB_DBLK, d - j0)
    var wv = SIMD[DType.float32, MB_DBLK]()
    comptime for u in range(MB_DBLK):
        if u < m:
            wv[u] = ld(w, woff + j0 + u)
    var n2 = Float32(0)
    var n1 = Float32(0)
    comptime for u in range(MB_DBLK):
        if u < m:
            n2 = fmad(wv[u], wv[u], n2)
            n1 = fa(n1, fabs(wv[u]))
    return (n2, n1)


@always_inline
def sgd_reg_blocked(w: FP, woff: Int, d: Int) -> Tuple[Float32, Float32]:
    """sgd_one's (sum w_j^2, sum |w_j|): the `sgd_reg_block` partials folded
    ascending from zero with fa, as the device folds them."""
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var n2 = Float32(0)
    var n1 = Float32(0)
    for b in range(nb):
        var pr = sgd_reg_block(w, woff, d, b)
        n2 = fa(n2, pr[0])
        n1 = fa(n1, pr[1])
    return (n2, n1)


@always_inline
def mb_row_dot(x: FP, ys: FP, i: Int, d: Int, dot: Float32, b: Float32, loss: Int, eps: Float32,
               swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool,
               lr: Int, eta0: Float32, sqp: FP, has_sq: Bool) -> Tuple[Float32, Float32]:
    """`mb_row` from the row's predictor dot (w . x_i, before the intercept);
    has_sq: the PA rates' |x_i|^2 read from sqp[i] (`mb_rowsq`, the same
    chain computed once a fit)."""
    var y = ld(ys, i)
    var p = fa(dot, b)
    var dl: Float32
    if lr == LR_PA1 or lr == LR_PA2:
        var cur = sgd_loss(loss, y, p, eps)
        var sq = ld(sqp, i) if has_sq else mb_rowsq(x, i, d)
        var update = Float32(0)
        if lr == LR_PA1:
            if sq != 0:
                update = fmin(eta0, fd(cur, sq))
        else:
            update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
        if loss == L_HINGE:
            update = fm(update, y)
        elif fs(y, p) < 0:
            update = -update
        dl = -update
    else:
        dl = sgd_dloss(loss, y, p, eps)
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


comptime MB_PF = 16


@always_inline
def mb_dot(x: FP, i: Int, d: Int, w: FP, woff: Int, blk: Int = 0) -> Float32:
    """`row_dot` (j ascending, one fmad a term) with MB_PF terms' loads in
    flight on the device (lane/neural-pass134); the host's is row_dot.
    blk > 0 (lane/neural-pass135): blk-column blocks, each from zero, folded
    ascending; up to 8 blocks advance together (independent chains), each
    block's terms in column order: host and device the same words."""
    if blk > 0:
        var base = i * d
        var nb = (d + blk - 1) // blk
        var total = Float32(0)
        var g0 = 0
        while g0 < nb:
            var accs = SIMD[DType.float32, 8](0)
            for k in range(blk):
                comptime for b in range(8):
                    var j = (g0 + b) * blk + k
                    if g0 + b < nb and j < d:
                        accs[b] = fmad(ld(x, base + j), ld(w, woff + j), accs[b])
            comptime for b in range(8):
                if g0 + b < nb:
                    total = fa(total, accs[b])
            g0 += 8
        return total
    comptime if is_gpu():
        var acc = Float32(0)
        var j = 0
        var base = i * d
        while j + MB_PF <= d:
            var xv = SIMD[DType.float32, MB_PF]()
            var wv = SIMD[DType.float32, MB_PF]()
            comptime for u in range(MB_PF):
                xv[u] = ld(x, base + j + u)
                wv[u] = ld(w, woff + j + u)
            comptime for u in range(MB_PF):
                acc = fmad(xv[u], wv[u], acc)
            j += MB_PF
        while j < d:
            acc = fmad(ld(x, base + j), ld(w, woff + j), acc)
            j += 1
        return acc
    return row_dot(x, i, d, w, woff)


@always_inline
def mb_part(x: FP, d: Int, idx: IP, start: Int, dlv: FP, lv: FP, j: Int, s: Int, bs: Int,
            sub: Int = MB_SUB) -> Float32:
    """Sub-block s (`sub` rows) of the batch at position `start` (bs rows),
    from zero: j < d the column j's dl x chain, j == d the intercept's dl
    fold, j == d + 1 the loss fold. dlv / lv are indexed by batch position."""
    var lo = s * sub
    var hi = min(bs, lo + sub)
    var acc = Float32(0)
    if j < d:
        var r = lo
        comptime if is_gpu():
            # lane/neural-pass134: MB_PF rows' loads in flight before their
            # fmads (the same fmads in the same order; the chain no longer
            # waits on an idx load and then an x load each step)
            while r + MB_PF <= hi:
                var xv = SIMD[DType.float32, MB_PF]()
                var dv = SIMD[DType.float32, MB_PF]()
                comptime for u in range(MB_PF):
                    xv[u] = ld(x, Int(ldi(idx, start + r + u)) * d + j)
                    dv[u] = ld(dlv, r + u)
                comptime for u in range(MB_PF):
                    acc = fmad(dv[u], xv[u], acc)
                r += MB_PF
        while r < hi:
            var i = Int(ldi(idx, start + r))
            acc = fmad(ld(dlv, r), ld(x, i * d + j), acc)
            r += 1
    elif j == d:
        for r in range(lo, hi):
            acc = fa(acc, ld(dlv, r))
    else:
        for r in range(lo, hi):
            acc = fa(acc, ld(lv, r))
    return acc


@always_inline
def mb_step(wj: Float32, g: Float32, bs: Int, eta: Float32, alpha: Float32, l1r: Float32, penalty: Int,
            bsum: Bool = False) -> Float32:
    """w_j after the batch: g the folded gradient sum. `bsum` (lane/neural-
    pass132, Perceptron and the one-class SVM) is the per-sample fit's step
    over the batch with its rows' gradients taken at the batch's start: the
    SUM of the gradients, the L2 part as sgd_one's per-row decay applied bs
    times (max(0, 1 - eta (1 - l1_ratio) alpha)^bs), the L1 part bs times
    alpha l1_ratio sign(w_j)."""
    if bsum:
        var w = wj
        if penalty == P_L2 or penalty == P_EN:
            var decay = fm(fs(Float32(1), l1r), alpha)
            w = fm(w, identical_pow(fmax(Float32(0), fs(Float32(1), fm(decay, eta))), i2f(bs)))
        var gs = g
        if penalty == P_L1 or penalty == P_EN:
            gs = fa(gs, fm(fm(i2f(bs), fm(alpha, l1r)), fsign(wj)))
        return fs(w, fm(eta, gs))
    var gm = fd(g, i2f(bs))
    if penalty == P_L2:
        gm = fmad(alpha, wj, gm)
    elif penalty == P_L1:
        gm = fa(gm, fm(alpha, fsign(wj)))
    elif penalty == P_EN:
        gm = fa(gm, fm(alpha, fa(fm(l1r, fsign(wj)), fm(fs(Float32(1), l1r), wj))))
    return fs(wj, fm(eta, gm))


@always_inline
def mb_bias_step(b: Float32, gb: Float32, bs: Int, eta: Float32, alpha: Float32, one_class: Bool,
                 bsum: Bool = False) -> Float32:
    """The intercept after the batch: gb the folded dl sum; the one-class
    problem also takes their offset step (bs of them with `bsum`)."""
    if bsum:
        var nb = fs(b, fm(eta, gb))
        if one_class:
            nb = fs(nb, fm(fm(i2f(bs), eta), alpha))
        return nb
    var nb = fs(b, fm(eta, fd(gb, i2f(bs))))
    if one_class:
        nb = fs(nb, fm(eta, alpha))
    return nb


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
    one_class: Bool = False, bsum: Bool = False,
) -> Int:
    """One problem on the host (the CPU binding's form of the device's batches).
    scratch: dl batch | loss batch | partials (d + 2) * subs. Returns epochs,
    -1 on a non-finite weight."""
    var dlv = scratch
    var lv = scratch + batch
    var parts = lv + batch
    var sub = mb_sub_size(batch)
    var dblk = mb_dblk(batch)
    var nsub = mb_subs(batch, sub)
    fill(w, woff, d, Float32(0))
    var bias = Float32(1) if one_class else Float32(0)
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
                var dl_l = mb_row(x, ys, Int(ldi(idx, start + r)), d, w, woff, bias, loss, eps, swp, has_sw, wpos, wneg, has_cw,
                                  lr, eta0, dblk)
                st(dlv, r, dl_l[0])
                st(lv, r, dl_l[1])
            var subs = mb_subs(bs, sub)
            for j in range(d + 2):
                for s in range(subs):
                    st(parts, j * nsub + s, mb_part(x, d, idx, start, dlv, lv, j, s, bs, sub))
            var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
            for j in range(d):
                var g = Float32(0)
                for s in range(subs):
                    g = fa(g, ld(parts, j * nsub + s))
                st(w, woff + j, mb_step(ld(w, woff + j), g, bs, et, alpha, l1r, penalty, bsum))
            if fit_intercept:
                var gb = Float32(0)
                for s in range(subs):
                    gb = fa(gb, ld(parts, d * nsub + s))
                bias = mb_bias_step(bias, gb, bs, et, alpha, one_class, bsum)
            if need_obj:
                var lb = Float32(0)
                for s in range(subs):
                    lb = fa(lb, ld(parts, (d + 1) * nsub + s))
                objective = fa(objective, lb)
            t += bs if bsum else 1  # bsum: the per-sample schedule counts rows
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
            if one_class:
                mean_obj = fa(mean_obj, fm(alpha, bias))
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
    st(b, boff, fs(Float32(1), bias) if one_class else bias)
    return epochs
