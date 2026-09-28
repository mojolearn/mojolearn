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
  * the weight decay multiplies w directly instead of their lazy `wscale`;
  * the shuffle is Fisher-Yates over splitmix64 (x_linear/ops.mojo), not
    their `SequentialDataset.shuffle` over numpy's MT19937, so a fit agrees
    with theirs in quality, not in bits; the OvR problem c is seeded
    `seed + 1000003 * c`.
"""
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fsqrt, fexp, flog, fabs, fmax, fmin,
    ld, st, ldi, sti, i2f, fill, row_dot, shuffle, axpy_acc, scale_acc, ftzv, par_rows,
)
from std.sys.info import is_gpu
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
    var epochs = 0
    for epoch in range(max_iter):
        epochs = epoch + 1
        var objective = Float32(0)
        if do_shuffle:
            shuffle(idx, n, rng)
        for r in range(n):
            var i = ldi(idx, r)
            var y = ld(ys, i)
            var p = fa(row_dot(x, i, d, w, woff), intercept)
            if lr == LR_OPTIMAL:
                eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
            elif lr == LR_INVSCALING:
                eta = fd(eta0, identical_pow(i2f(t), power_t))
            var cur = sgd_loss(loss, y, p, eps)
            objective = fa(objective, cur)
            if lr != LR_PA1 and lr != LR_PA2:
                if penalty != P_NONE:
                    var n2 = Float32(0)
                    var n1 = Float32(0)
                    for j in range(d):
                        var wj = ld(w, woff + j)
                        n2 = fmad(wj, wj, n2)
                        n1 = fa(n1, fabs(wj))
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
                var dl = sgd_dloss(loss, y, p, eps)
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
                var scale = fmax(Float32(0), fs(Float32(1), fm(decay_factor, eta)))
                scale_acc(w, woff, scale, d)
            if update != 0:
                axpy_acc(w, woff, update, x, i * d, d)
            if fit_intercept:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    intercept = fa(intercept, iu)
            if penalty == P_L1 or penalty == P_EN:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
                _l1_clip(w, woff, q, u, d)
            t += 1
        # their floating-point under-/overflow check
        var finite = intercept == intercept and fabs(intercept) < Float32(3.0e38)
        for j in range(d):
            var wj = ld(w, woff + j)
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
    st(b, boff, intercept)
    return epochs


def sgd_fit(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_classes, loss, penalty, lr, fit_intercept, max_iter, n_iter_no_change,
    shuffle, seed_lo, seed_hi, sample_weight, class_weight]; with
    sample_weight y = labels n | weights n; with class_weight fp carries
    [.., pos weight per problem (P), neg weight per problem (P)]; n_classes 0 = regression, 1 = one-class,
    2 = binary (positive class = label 1), K > 2 = one-vs-rest.
    fp: [alpha, l1_ratio, eta0, power_t, epsilon, tol].
    y: labels as 0..K-1 (classification) or targets. res: coef (P*d),
    intercept (P), n_iter (1), status (1: 0 ok, -1 non-finite).
    fw: per problem n (targets) + d (q), then P (epochs run); iw: per problem n (order)."""
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
    var swp = y + n
    var problems = k if k > 2 else 1
    # one-vs-rest problems are independent (their own targets, q, order and
    # result slots): the host may run them at once (lane linear-cpu)
    var eps_run = fw + problems * (n + d)  # epochs run per problem

    def run_problems(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm k, imm fw, imm iw, imm res, imm problems,
                                        imm loss, imm penalty, imm alpha, imm l1r, imm lr, imm eta0, imm power_t,
                                        imm eps, imm fit_intercept, imm max_iter, imm tol, imm nic, imm do_shuffle,
                                        imm seed, imm swp, imm has_sw, imm has_cw, imm fp, imm eps_run}:
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
            var ep = sgd_one(
                x, ys, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                fit_intercept, max_iter, tol, nic, do_shuffle,
                seed + UInt64(1000003) * UInt64(c), k == 1,
                res, c * d, res, problems * d + c, q, idx,
                swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw,
            )
            st(eps_run, c, i2f(ep))

    par_rows(run_problems, problems, 1)
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        var ep = Int(ld(eps_run, c))
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


def _l1_clip(w: FP, woff: Int, q: FP, u: Float32, d: Int):
    """The cumulative L1 penalty over every weight (their `l1penalty`); each
    weight and its q entry are updated on their own, so lanes are
    independent and the vector form is the scalar loop bit for bit."""
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
