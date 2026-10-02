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
    ld, st, ldi, sti, i2f, fill, row_dot, shuffle, axpy_acc, scale_acc, ftzv, par_rows, fz, xmad, fsign,
)
from std.sys.info import is_gpu
from std.gpu import WARP_SIZE
from std.gpu.primitives.warp import shuffle_idx, shuffle_xor
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


# ------------------------------------------------ the warp form (lane/linear-apple)
# One SGD problem on ONE warp instead of one thread, the same words
# (lane/linear-apple, 2026-09-28). The row pass is serial in the rows and a
# thread spent it waiting on d dependent loads per row (sgd-clf 100k x 28,
# 5 epochs: 8.2 s on the M3 Ultra, 30x the host). Here:
#   * weight j lives in a REGISTER of lane j mod W (chunk j / W), with its
#     L1 history q; every elementwise step (decay, the axpy, the L1 clip) is
#     that lane's, with the same expression the thread applied to w[j];
#   * row i's x is read by the warp at once, x_j in lane j mod W;
#   * every fold (the row dot, the penalty norms, PA's |x|^2, the finite
#     check) runs in EVERY lane over j ascending, each x_j / w_j fetched from
#     its lane with `shuffle_idx` (a register copy): the same fmad sequence,
#     so every lane holds the thread's scalar and the scalar control flow is
#     uniform;
#   * the row order is lane 0's (its Fisher-Yates in `idx`, which only lane 0
#     reads and writes), broadcast per row with `shuffle_idx`; the target is
#     computed from the label in every lane (the thread read it from `ys`,
#     which the loop that filled it computed the same way).
# No lane reads a device word another lane wrote. `sgd_one` stays the host's
# and the fallback for d > SGD_WARP_MAX_CHUNKS * W.

comptime SGD_WARP_MAX_CHUNKS = 8


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
def _fmad_flushed(a: Float32, b: Float32, c: Float32) -> Float32:
    """`fmad` for operands that are already flushed words (fz is idempotent:
    fz(fz(v)) == fz(v)), so the chain does not flush its accumulator twice."""
    return fz(xmad(a, b, c))


@always_inline
def _warp_row_folds[K: Int, NORMS: Bool, SQ: Bool](
    xr: InlineArray[Float32, K], wr: InlineArray[Float32, K], d: Int,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """One row's folds over j ascending in every lane (x_j, w_j fetched from
    lane j mod W): the row dot, and when asked the penalty norms (sum w_j^2,
    sum |w_j|) and PA's |x|^2. Each is its own chain with the thread's
    expressions; running them in one loop only interleaves independent
    chains. xr and wr hold flushed words."""
    comptime W = WARP_SIZE
    var acc = Float32(0)
    var n2 = Float32(0)
    var n1 = Float32(0)
    var sq = Float32(0)
    # Fully unrolled over the K * W slots (lane/linear-apple2): every
    # shuffle's source lane is a constant and the fetches do not wait on the
    # chain. A slot j >= d computes and is not taken (an integer select, so
    # the chain is the same fmad sequence over j < d, ascending).
    comptime for kk in range(K):
        comptime for l in range(W):
            comptime j = kk * W + l
            var live = j < d
            var xj = shuffle_idx(xr[kk], UInt32(l))
            var wj = shuffle_idx(wr[kk], UInt32(l))
            var a2 = _fmad_flushed(xj, wj, acc)
            acc = a2 if live else acc
            comptime if NORMS:
                var b2 = _fmad_flushed(wj, wj, n2)
                var c2 = fa(n1, fabs(wj))
                n2 = b2 if live else n2
                n1 = c2 if live else n1
            comptime if SQ:
                var s2 = _fmad_flushed(xj, xj, sq)
                sq = s2 if live else sq
    return (acc, n2, n1, sq)


@always_inline
def _warp_row_folds_tree[K: Int, NORMS: Bool, SQ: Bool](
    xr: InlineArray[Float32, K], wr: InlineArray[Float32, K],
) -> Tuple[Float32, Float32, Float32, Float32]:
    """FAST on Apple, opt-in `-D MOJOLEARN_SGD_FAST_TREE=1` (lane/linear-apple3,
    WIP): one row's folds as a warp reduction. Each lane multiplies its own
    K slots (a slot j >= d holds x = 0 and w = 0), then a `shuffle_xor`
    butterfly sums the lanes: log2(W) steps instead of a chain of K * W,
    and every lane ends with the same word (the two lanes of a pair add the
    same two words). The grouping of each sum differs from the chain's, so
    FAST words change; IDENTICAL never compiles this."""
    comptime W = WARP_SIZE
    var acc = Float32(0)
    var n2 = Float32(0)
    var n1 = Float32(0)
    var sq = Float32(0)
    comptime for kk in range(K):
        acc = xmad(xr[kk], wr[kk], acc)
        comptime if NORMS:
            n2 = xmad(wr[kk], wr[kk], n2)
            n1 = n1 + fabs(wr[kk])
        comptime if SQ:
            sq = xmad(xr[kk], xr[kk], sq)
    comptime for s in range(7):
        comptime off = 1 << s
        comptime if off < W:
            acc = acc + shuffle_xor(acc, UInt32(off))
            comptime if NORMS:
                n2 = n2 + shuffle_xor(n2, UInt32(off))
                n1 = n1 + shuffle_xor(n1, UInt32(off))
            comptime if SQ:
                sq = sq + shuffle_xor(sq, UInt32(off))
    return (acc, n2, n1, sq)


@always_inline
def _row_folds[K: Int, NORMS: Bool, SQ: Bool](
    xr: InlineArray[Float32, K], wr: InlineArray[Float32, K], d: Int,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """`_warp_row_folds`, or FAST on Apple with the define its tree form."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and is_apple_gpu() and is_defined["MOJOLEARN_SGD_FAST_TREE"]():
        return _warp_row_folds_tree[K, NORMS, SQ](xr, wr)
    return _warp_row_folds[K, NORMS, SQ](xr, wr, d)


comptime SPLITMIX_GAMMA = UInt64(0x9E3779B97F4A7C15)


@always_inline
def _splitmix_at(seed: UInt64, k: Int) -> UInt64:
    """The word the k-th rng_next call from state `seed` returns (k >= 1)."""
    var z = seed + UInt64(k) * SPLITMIX_GAMMA
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _warp_draws(lane: Int, seed: UInt64, n: Int, epoch: Int, dst: IP):
    """Epoch `epoch`'s Fisher-Yates draws j_t = draw mod (n - t), t < n - 1,
    lanes striding over t; draw k = epoch * (n - 1) + t + 1."""
    var base = epoch * (n - 1)
    var tt = lane
    while tt < n - 1:
        var z = _splitmix_at(seed, base + tt + 1)
        dst.unsafe_store(tt, Int32(Int(z % UInt64(n - tt))))
        tt += WARP_SIZE


def _shuffle_drawn(idx: IP, n: Int, dr: IP):
    """ops.shuffle with its draws read from `dr` (the same swaps)."""
    for tt in range(n - 1):
        var i = n - 1 - tt
        var j = Int(dr.unsafe_load(tt))
        var a = ldi(idx, i)
        sti(idx, i, ldi(idx, j))
        sti(idx, j, a)


def sgd_one_warp[K: Int](
    lane: Int, x: FP, y: FP, k: Int, c: Int, n: Int, d: Int,
    loss: Int, penalty: Int, alpha: Float32, l1_ratio_in: Float32,
    lr: Int, eta0: Float32, power_t: Float32, eps: Float32,
    fit_intercept: Bool, max_iter: Int, tol: Float32, n_iter_no_change: Int,
    do_shuffle: Bool, seed: UInt64, one_class: Bool,
    w: FP, woff: Int, b: FP, boff: Int, idx: IP,
    swp: FP, has_sw: Bool, wpos: Float32, wneg: Float32, has_cw: Bool,
    pipe: Bool, t_team: Team, warp: Int, idx_b: IP, draws0: IP, draws1: IP,
) -> Int:
    """`sgd_one` on the warp of `lane` (see above). d <= K * WARP_SIZE.

    `pipe` (lane/linear-apple2; one problem, shuffle on, every thread of the
    block calls this): warp 0 computes, warp 1's lane 0 shuffles the NEXT
    epoch's order (a copy of this epoch's, then Fisher-Yates on the same
    draws in the same sequence) into the other of `idx` / `idx_b` while
    warp 0 runs this epoch; every warp meets at one broadcast per epoch
    (warp 0's stop verdict). The orders and every word computed from them
    are the ones the serial form produced."""
    comptime W = WARP_SIZE
    var l1_ratio = l1_ratio_in
    if penalty == P_L2:
        l1_ratio = Float32(0)
    elif penalty == P_L1:
        l1_ratio = Float32(1)
    var wr = InlineArray[Float32, K](fill=Float32(0))
    var qr = InlineArray[Float32, K](fill=Float32(0))
    var xr = InlineArray[Float32, K](fill=Float32(0))
    var intercept = Float32(1) if one_class else Float32(0)
    var rng = seed
    var is_comp = (not pipe) or warp == 0
    var is_shuf = pipe and warp == 1
    var is_draw = pipe and warp == 2
    if not pipe:
        if lane == 0:
            for i in range(n):
                sti(idx, i, i)
    else:
        # lane/linear-apple2: warp 2's lanes compute each epoch's draws in
        # parallel (draw k of the stream is splitmix64 at seed + k * gamma,
        # the word rng_next's k-th call returns), two epochs ahead; warp 1's
        # lane 0 runs Fisher-Yates on them.
        if is_draw:
            _warp_draws(lane, seed, n, 0, draws0)
            if max_iter > 1:
                _warp_draws(lane, seed, n, 1, draws1)
        t_team.sync()
        if is_shuf and lane == 0:
            for i in range(n):
                sti(idx, i, i)
            _shuffle_drawn(idx, n, draws0)
        t_team.sync()
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
    # lane/linear-apple2: the penalty norms feed only the objective, and the
    # objective only the stopping test; with tol None (-3e38) nothing reads
    # them, so they are not folded.
    var need_obj = tol > Float32(-3.0e38)
    var pa = lr == LR_PA1 or lr == LR_PA2
    var fold_norms = need_obj and not pa and penalty != P_NONE
    var stop = 0
    for epoch in range(max_iter):
        var order = idx
        if pipe:
            if epoch % 2 == 1:
                order = idx_b
            if is_shuf and lane == 0 and epoch + 1 < max_iter:
                var nxt = idx_b if epoch % 2 == 0 else idx
                # the copy loads 16 words before storing them (one lane)
                var i = 0
                while i + 16 <= n:
                    var cp = InlineArray[Int32, 16](fill=Int32(0))
                    comptime for u in range(16):
                        cp[u] = order.unsafe_load(i + u)
                    comptime for u in range(16):
                        nxt.unsafe_store(i + u, cp[u])
                    i += 16
                while i < n:
                    nxt.unsafe_store(i, order.unsafe_load(i))
                    i += 1
                _shuffle_drawn(nxt, n, draws1 if epoch % 2 == 0 else draws0)
            if is_draw and epoch + 2 < max_iter:
                _warp_draws(lane, seed, n, epoch + 2, draws0 if epoch % 2 == 0 else draws1)
        if is_comp:
            epochs = epoch + 1
            var objective = Float32(0)
            if do_shuffle and lane == 0 and not pipe:
                shuffle(idx, n, rng)
            # lane/linear-apple2: row r + 1's index, target and x are loaded while
            # row r is computed (row r + 2's index one step earlier still), so a
            # row no longer waits on its own loads. x is flushed once, as the row
            # starts (not at the load, which would wait for it).
            var raw_next = Int32(0)
            if lane == 0:
                raw_next = order.unsafe_load(0)
            var ci = Int(shuffle_idx(raw_next, UInt32(0)))
            var cx = InlineArray[Float32, K](fill=Float32(0))
            comptime for kk in range(K):
                var j = lane + kk * W
                cx[kk] = ld(x, ci * d + j) if j < d else Float32(0)
            var cy = ld(y, ci)
            if lane == 0 and n > 1:
                raw_next = order.unsafe_load(1)
            for r in range(n):
                var i = ci
                comptime for kk in range(K):
                    xr[kk] = fz(cx[kk])
                var yv = _sgd_target(k, c, cy)
                if r + 1 < n:
                    ci = Int(shuffle_idx(raw_next, UInt32(0)))
                    comptime for kk in range(K):
                        var j = lane + kk * W
                        cx[kk] = ld(x, ci * d + j) if j < d else Float32(0)
                    cy = ld(y, ci)
                    if lane == 0 and r + 2 < n:
                        raw_next = order.unsafe_load(r + 2)
                var folds: Tuple[Float32, Float32, Float32, Float32]
                if fold_norms:
                    folds = _row_folds[K, True, False](xr, wr, d)
                elif pa:
                    folds = _row_folds[K, False, True](xr, wr, d)
                else:
                    folds = _row_folds[K, False, False](xr, wr, d)
                var p = fa(folds[0], intercept)
                if lr == LR_OPTIMAL:
                    eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
                elif lr == LR_INVSCALING:
                    eta = fd(eta0, identical_pow(i2f(t), power_t))
                var cur = sgd_loss(loss, yv, p, eps)
                objective = fa(objective, cur)
                if not pa and need_obj:
                    if penalty != P_NONE:
                        var n2 = folds[1]
                        var n1 = folds[2]
                        var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                        objective = fa(objective, fm(alpha, reg))
                    if one_class:
                        objective = fa(objective, fm(intercept, alpha))
                var update: Float32
                if pa:
                    var sq = folds[3]
                    if lr == LR_PA1:
                        if sq == 0:
                            continue
                        update = fmin(eta0, fd(cur, sq))
                    else:
                        update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
                    if loss == L_HINGE:
                        update = fm(update, yv)
                    elif fs(yv, p) < 0:
                        update = -update
                else:
                    var dl = sgd_dloss(loss, yv, p, eps)
                    if dl < Float32(-1e12):
                        dl = Float32(-1e12)
                    elif dl > Float32(1e12):
                        dl = Float32(1e12)
                    update = fm(-eta, dl)
                if has_cw or has_sw:
                    var cw = Float32(1)
                    if has_cw:
                        cw = wpos if yv > 0 else wneg
                    var swi = ld(swp, i) if has_sw else Float32(1)
                    update = fm(update, fm(cw, swi))
                if penalty == P_L2 or penalty == P_EN:
                    var scale = fmax(Float32(0), fs(Float32(1), fm(decay_factor, eta)))
                    comptime for kk in range(K):
                        if lane + kk * W < d:
                            wr[kk] = fm(wr[kk], scale)
                if update != 0:
                    comptime for kk in range(K):
                        if lane + kk * W < d:
                            wr[kk] = fmad(update, xr[kk], wr[kk])
                if fit_intercept:
                    var iu = update
                    if one_class:
                        iu = fs(iu, fm(eta, alpha))
                    if iu != 0:
                        intercept = fa(intercept, iu)
                if penalty == P_L1 or penalty == P_EN:
                    u = fa(u, fm(fm(l1_ratio, eta), alpha))
                    comptime for kk in range(K):
                        if lane + kk * W < d:
                            var z = wr[kk]
                            var nz = _clip_one(z, fa(u, qr[kk]), fs(u, qr[kk]))
                            wr[kk] = nz
                            qr[kk] = fa(qr[kk], fs(nz, z))
                t += 1
            var finite = intercept == intercept and fabs(intercept) < Float32(3.0e38)
            for j in range(d):
                var src = UInt32(j % W)
                comptime for kk in range(K):
                    if j // W == kk:
                        var wj = shuffle_idx(wr[kk], src)
                        if not (wj == wj and fabs(wj) < Float32(3.0e38)):
                            finite = False
            if not finite:
                if lane == 0:
                    st(b, boff, Float32(0))
                comptime for kk in range(K):
                    if lane + kk * W < d:
                        st(w, woff + lane + kk * W, Float32(0))
                stop = 2
            if stop == 0:
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
                        stop = 1
        if pipe:
            stop = t_team.bcast_int(stop)
        if stop != 0:
            break
    if not is_comp:
        return 0
    if stop == 2:
        return -1
    comptime for kk in range(K):
        if lane + kk * W < d:
            st(w, woff + lane + kk * W, wr[kk])
    if lane == 0:
        st(b, boff, intercept)
    return epochs



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
    intercept (P), n_iter (1), status (1: 0 ok, -1 non-finite).
    Team form: the P one-vs-rest problems are independent; thread c runs
    problem c (targets in team row 2c, order in team row 2c + 1, q in its
    own d words, epochs in team row 2P), then the lead folds the epochs.
    Each problem is the one-thread sequence.
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
    comptime if is_gpu():
        var q = t.own()
        epr = t.row(2 * problems)
        # lane/linear-apple: a problem per WARP (sgd_one_warp) when d fits
        # the warp's registers; the thread-per-problem form otherwise.
        var chunks = (d + WARP_SIZE - 1) // WARP_SIZE
        if chunks <= SGD_WARP_MAX_CHUNKS and t.nt >= WARP_SIZE and t.nt % WARP_SIZE == 0:
            var lane = t.tid % WARP_SIZE
            var nw = t.nt // WARP_SIZE
            # lane/linear-apple2: one problem with a shuffle: every warp takes
            # part (warp 0 computes, warp 1 shuffles the next epoch's order).
            var pipe = problems == 1 and do_shuffle and nw >= 3
            var warp = t.tid // WARP_SIZE
            var idx_b = t.row(2 * problems + 1).bitcast[Int32]()
            var draws0 = t.row(2 * problems + 2).bitcast[Int32]()
            var draws1 = t.row(2 * problems + 3).bitcast[Int32]()
            var first = 0 if pipe else warp
            var step = 1 if pipe else nw
            for c in range(first, problems, step):
                var order = t.row(2 * c + 1).bitcast[Int32]()
                var cw_pos = ld(fp, 6 + c) if has_cw else Float32(1)
                var cw_neg = ld(fp, 6 + problems + c) if has_cw else Float32(1)
                var ep: Int
                if chunks <= 1:
                    ep = sgd_one_warp[1](
                        lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                        fit_intercept, max_iter, tol, nic, do_shuffle,
                        seed + UInt64(1000003) * UInt64(c), k == 1,
                        res, c * d, res, problems * d + c, order, swp, has_sw, cw_pos, cw_neg, has_cw,
                        pipe, t, warp, idx_b, draws0, draws1,
                    )
                elif chunks <= 2:
                    ep = sgd_one_warp[2](
                        lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                        fit_intercept, max_iter, tol, nic, do_shuffle,
                        seed + UInt64(1000003) * UInt64(c), k == 1,
                        res, c * d, res, problems * d + c, order, swp, has_sw, cw_pos, cw_neg, has_cw,
                        pipe, t, warp, idx_b, draws0, draws1,
                    )
                elif chunks <= 4:
                    ep = sgd_one_warp[4](
                        lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                        fit_intercept, max_iter, tol, nic, do_shuffle,
                        seed + UInt64(1000003) * UInt64(c), k == 1,
                        res, c * d, res, problems * d + c, order, swp, has_sw, cw_pos, cw_neg, has_cw,
                        pipe, t, warp, idx_b, draws0, draws1,
                    )
                else:
                    ep = sgd_one_warp[SGD_WARP_MAX_CHUNKS](
                        lane, x, y, k, c, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                        fit_intercept, max_iter, tol, nic, do_shuffle,
                        seed + UInt64(1000003) * UInt64(c), k == 1,
                        res, c * d, res, problems * d + c, order, swp, has_sw, cw_pos, cw_neg, has_cw,
                        pipe, t, warp, idx_b, draws0, draws1,
                    )
                if lane == 0 and (not pipe or warp == 0):
                    st(epr, c, i2f(ep))
            t.sync()
        else:
            for c in range(t.tid, problems, t.nt):
                var ys = t.row(2 * c)
                var order = t.row(2 * c + 1).bitcast[Int32]()
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
                    res, c * d, res, problems * d + c, q, order,
                    swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                    ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw,
                )
                st(epr, c, i2f(ep))
            t.sync()
    else:
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
                    var sl = List[Float32](length=2 * batch + (d + 2) * mb_subs(batch), fill=Float32(0))
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


def sgd_team_rows(ip: IP) -> Int:
    """Row buffers an SGD fit needs: targets and order per problem, the
    epochs, and the warp form's second order buffer."""
    var k = ldi(ip, 0)
    var problems = k if k > 2 else 1
    return 2 * problems + 4  # + the second order buffer and two draw buffers (lane/linear-apple2)

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
           lr: Int, eta0: Float32) -> Tuple[Float32, Float32]:
    """(dl_i weighted, loss_i) of row i at (w, b); with a PA rate dl_i is minus
    the row's PA step (sgd_one's statements, eta0 = C)."""
    var y = ld(ys, i)
    var p = fa(row_dot(x, i, d, w, woff), b)
    var dl: Float32
    if lr == LR_PA1 or lr == LR_PA2:
        var cur = sgd_loss(loss, y, p, eps)
        var sq = Float32(0)
        for j in range(d):
            var xj = ld(x, i * d + j)
            sq = fmad(xj, xj, sq)
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
    var nsub = mb_subs(batch)
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
                                  lr, eta0)
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
    st(b, boff, bias)
    return epochs
