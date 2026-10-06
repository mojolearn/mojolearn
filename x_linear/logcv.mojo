# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegressionCV (lane/algos-linear, 2026-09-27).

Reference: scikit-learn `sklearn/linear_model/_logistic.py`
(`LogisticRegressionCV.fit`, `_log_reg_scoring_path`,
`_logistic_regression_path`) and `sklearn/linear_model/_linear_loss.py`
(`LinearModelLoss`: objective (1/n) sum loss_i + 1/(2 C n) ||W||^2, the
intercepts unpenalized): binary log loss for two classes, the multinomial
(softmax) loss for more; the Cs path runs in the given order with warm
starts on each fold's training rows (their StratifiedKFold ids, built
from the labels by `lcv_fold_table` on whichever column fits, cgr-linear);
accuracy on the held-out rows; the first best mean over folds
wins. Their solver is L-BFGS and so is this one (x_linear/lbfgs.mojo).
Named difference: the refit on all rows starts from zero (theirs from the
mean of the folds' coefficients at the best C; the objective is convex, so
both reach the same minimizer). float32 throughout.

Per-call objective parameters: ip block [K', fit_intercept, fold (-1 all)],
fp block [C]; y = labels (0..K-1 as float32) | fold ids | weights (optional).
The caller passes zeros for the fold ids; the host binding fills them
(`logcv_fold_ids`) and the device binding builds them on the grid
(x_linear/logcv_grid.mojo) from the same table.
"""
from experiments.classical_identical_ideas.linear_controls import C13_LOGCV_WEIGHTS
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fexp, flog, fmax, ld, st, ldi, sti, i2f, fill, copy, row_dot,
    axpy_acc, par_rows, row_dots,
)
from std.sys.info import is_gpu
from x_linear.lbfgs import lbfgs, lbfgs_work, Objective
from x_linear.team import Team, team_at
from x_linear.vfold import vwsq
from std.gpu import WARP_SIZE
from x_linear.tops import fold_fa, fold_fa_ix, chain_fmad, chain_fmad_ix, fold_parts, fold_blocks, FOLD_BLOCK
from checks.numerics import identical_sigmoid, identical_softplus, ftz


# ------------------------------------------------ StratifiedKFold ids (cgr-linear)
# scikit-learn's StratifiedKFold(n_splits=k, shuffle=False)._make_test_folds
# in integers: classes renumbered by first appearance; their sorted labels
# dealt round-robin to the k folds give alloc[f][class]; within a class the
# rows take folds 0, 0, .., 1, 1, .. in row order, alloc[f][class] of each.
# A class's table row holds the fold bounds (k + 1 cumulative counts), so a
# row's fold is the f with bound[f] <= its rank in its class < bound[f + 1].
@always_inline
def _lcv_mod_count(x: Int, f: Int, k: Int) -> Int:
    """#{p in [0, x): p % k == f}."""
    return (x - f + k - 1) // k if x > f else 0


def lcv_fold_table(counts: IP, first: IP, K: Int, k: Int, order: IP, table: IP):
    """table[c * (k + 1) + f]: class c's fold bounds, from the class counts and
    first rows (order: K words of scratch). K x k integer work, one thread."""
    for c in range(K):
        sti(order, c, c)
    # classes by first appearance (insertion sort; firsts are distinct)
    for a in range(1, K):
        var c = ldi(order, a)
        var b = a - 1
        while b >= 0 and ldi(first, ldi(order, b)) > ldi(first, c):
            sti(order, b + 1, ldi(order, b))
            b -= 1
        sti(order, b + 1, c)
    var s0 = 0
    for e in range(K):
        var c = ldi(order, e)
        var s1 = s0 + ldi(counts, c)
        var acc = 0
        for f in range(k):
            sti(table, c * (k + 1) + f, acc)
            acc += _lcv_mod_count(s1, f, k) - _lcv_mod_count(s0, f, k)
        sti(table, c * (k + 1) + k, acc)
        s0 = s1


@always_inline
def lcv_fold_of(table: IP, c: Int, k: Int, rank: Int) -> Int:
    var f = 0
    while f + 1 < k and ldi(table, c * (k + 1) + f + 1) <= rank:
        f += 1
    return f


def logcv_fold_ids(y: FP, n: Int, ip: IP):
    """The host column's fold ids into y[n, 2n) (ip[2] = K', ip[4] = k)."""
    var K = max(ldi(ip, 2), 2)
    var k = ldi(ip, 4)
    var counts = List[Int32](length=K, fill=Int32(0))
    var first = List[Int32](length=K, fill=Int32(n))
    var order = List[Int32](length=K, fill=Int32(0))
    var table = List[Int32](length=K * (k + 1), fill=Int32(0))
    var cp = IP(unsafe_from_address=Int(counts.unsafe_ptr()))
    var fp_ = IP(unsafe_from_address=Int(first.unsafe_ptr()))
    for i in range(n):
        var c = Int(ld(y, i))
        if ldi(cp, c) == 0:
            sti(fp_, c, i)
        sti(cp, c, ldi(cp, c) + 1)
    var tp = IP(unsafe_from_address=Int(table.unsafe_ptr()))
    lcv_fold_table(cp, fp_, K, k, IP(unsafe_from_address=Int(order.unsafe_ptr())), tp)
    for c in range(K):
        sti(cp, c, 0)
    for i in range(n):
        var c = Int(ld(y, i))
        var r = ldi(cp, c)
        st(y, n + i, i2f(lcv_fold_of(tp, c, k, r)))
        sti(cp, c, r + 1)
    _ = counts^
    _ = first^
    _ = order^
    _ = table^


# ------------------------------------------------ held-out score (cgr-linear)
@always_inline
def lcv_score_part(y: FP, hitr: FP, n: Int, f: Int, weighted: Bool, lo: Int, cnt: Int) -> Tuple[Float32, Float32]:
    """Rows [lo, lo + cnt) of fold f, ascending, from zero: (sum w hit, sum w)
    with the raw weights at y + 3n, else (hits, rows) as exact counts."""
    var a = Float32(0)
    var b = Float32(0)
    for i in range(lo, lo + cnt):
        if Int(ld(y, n + i)) == f:
            var wi = ld(y, 3 * n + i) if weighted else Float32(1)
            b = fa(b, wi)
            if ld(hitr, i) != 0:
                a = fa(a, wi)
    return (a, b)


@always_inline
def lcv_score_final(a: Float32, b: Float32) -> Float32:
    return fd(a, b) if b > 0 else Float32(0)


def lcv_score_rows(y: FP, hitr: FP, n: Int, f: Int, weighted: Bool) -> Float32:
    """The held-out score in the blocked order: FOLD_BLOCK rows from zero,
    the partials folded blocks ascending (exact counts when unweighted)."""
    var a = Float32(0)
    var b = Float32(0)
    var lo = 0
    while lo < n:
        var p = lcv_score_part(y, hitr, n, f, weighted, lo, min(FOLD_BLOCK, n - lo))
        a = fa(a, p[0])
        b = fa(b, p[1])
        lo += FOLD_BLOCK
    return lcv_score_final(a, b)


def logcv_rows(t: Team, y: FP, n: Int, fold: Int, kp: Int) -> Int:
    """The lead writes the training rows of `fold` (every row with another
    fold id), ascending, into team row buffer K' + 1 as int32; returns their
    count to every thread. Rows of fold < 0: every row, no list."""
    if fold < 0:
        return n
    var ix = t.row(kp + 1).bitcast[Int32]()
    var cnt = 0
    if t.lead():
        for i in range(n):
            if Int(ld(y, n + i)) != fold:
                ix.unsafe_store(cnt, Int32(i))
                cnt += 1
    return t.bcast_int(cnt, 5)


@always_inline

def lcv_disjoint_weight(y: FP, n: Int, fold: Int, weighted: Bool) -> Float32:
    """C13 weighted LogCV preparation: immutable fold sum, rows ascending.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    var total = Float32(0)
    for i in range(n):
        if Int(ld(y, n + i)) == fold:
            total = fa(total, ld(y, 2 * n + i) if weighted else Float32(1))
    return total


def lcv_retained_weight(cache: FP, folds: Int, held: Int) -> Float32:
    var total = Float32(0)
    for f in range(folds):
        if f != held:
            total = fa(total, ld(cache, f))
    return total


def _logcv_part(o: Int, x: FP, y: FP, n: Int, d: Int, kp: Int, fi: Bool, sw: Bool, fold: Int, ix: IP, cnt: Int,
                bk: Int, t: Team) -> Float32:
    return logcv_part_rows(o, x, y, n, d, kp, fi, sw, fold, ix, cnt, bk, t.row(0))


@always_inline
def logcv_part_rows(o: Int, x: FP, y: FP, n: Int, d: Int, kp: Int, fi: Bool, sw: Bool, fold: Int, ix: IP, cnt: Int,
                    bk: Int, rows: FP) -> Float32:
    """Task o of block bk (training positions [bk*B, ...)) from zero: o < p the
    gradient cell (class k, column j; j == d the intercept), o == p the loss,
    o == p + 1 the weight sum (lane/neural-pass97 blocked order)."""
    var stride = d + 1
    var p = kp * stride
    var lo = bk * FOLD_BLOCK
    var cb = min(FOLD_BLOCK, cnt - lo)
    if o < p:
        var k = o // stride
        var j = o - k * stride
        var rk = rows + k * n
        if j < d:
            if fold >= 0:
                return chain_fmad_ix(rk, x, j, d, ix + lo, cb)
            return chain_fmad(rk, lo, 1, x, lo * d + j, d, cb)
        if not fi:
            return Float32(0)
        if fold >= 0:
            return fold_fa_ix(rk, ix + lo, cb)
        return fold_fa(rk, lo, 1, cb)
    if o == p:
        var lt = rows + kp * n
        if fold >= 0:
            return fold_fa_ix(lt, ix + lo, cb)
        return fold_fa(lt, lo, 1, cb)
    if not sw or C13_LOGCV_WEIGHTS:
        return Float32(0)
    if fold >= 0:
        return fold_fa_ix(y, ix + lo, cb, 2 * n)
    return fold_fa(y, 2 * n + lo, 1, cb)



@always_inline
def logcv_map_row(i: Int, x: FP, y: FP, n: Int, d: Int, kp: Int, fi: Bool, sw: Bool, th: FP, toff: Int, rows: FP):
    """Training row i's residuals (rows[k * n + i]) and loss term
    (rows[kp * n + i]): the team map's statements (lane/neural-pass127: the
    grid objective runs the same function)."""
    var stride = d + 1
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
        st(rows, i, r)
        st(rows + kp * n, i, li)
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
        st(rows + kp * n, i, fm(wi, fs(lse, zy)) if sw else fs(lse, zy))
        for k in range(kp):
            var z = fa(row_dot(x, i, d, th, toff + k * stride), ld(th, toff + k * stride + d) if fi else Float32(0))
            var r = fexp(fs(z, lse))
            if k == label:
                r = fs(r, Float32(1))
            if sw:
                r = fm(wi, r)
            st(rows + k * n, i, r)



def logcv_finish_t(v: Team, g: FP, goff: Int, th: FP, toff: Int, kp: Int, d: Int, sw: Bool, c: Float32,
                   cnt: Int, acc: Float32, wrows: Float32, parts: FP) -> Float32:
    """The objective's last step on the folded sums, on a team (the device
    L-BFGS's finish block) or a team of one: the gradient scaled and
    penalized in place a thread a cell, ||W||^2 in the vfold order (lane
    cgr4-device-optim; it was one ascending chain). Every thread returns
    the loss."""
    var stride = d + 1
    var cntf = wrows if sw else i2f(cnt)
    var inv_n = fd(Float32(1), cntf)
    var lam = fd(Float32(1), fm(c, cntf))
    var reg = vwsq(v, th, toff, d, stride, kp, parts)
    for o in range(v.tid, kp * stride, v.nt):
        var j = o % stride
        var gv = fm(ld(g, goff + o), inv_n)
        if j < d:
            gv = fmad(lam, ld(th, toff + o), gv)
        st(g, goff + o, gv)
    v.sync()
    return fa(fm(acc, inv_n), fm(fm(Float32(0.5), lam), reg))


def logcv_finish(g: FP, goff: Int, th: FP, toff: Int, kp: Int, d: Int, sw: Bool, c: Float32, cnt: Int,
                 acc: Float32, wrows: Float32) -> Float32:
    """`logcv_finish_t` on one thread."""
    return logcv_finish_t(team_at(0, 1, g, 0, 0, 0), g, goff, th, toff, kp, d, sw, c, cnt, acc, wrows, g)


def _logistic_objective_team(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) -> Float32:
    """Team form: each training row's residuals (row buffers 0..K'-1) and
    loss term (row buffer K') across the team; the lead folds the loss in
    ascending row order; one thread per gradient cell folds its rows
    ascending (x_linear/tops.mojo chains). The one-thread sequence, value
    for value. Held-out rows are skipped, never added as zeros: a fold's
    training rows are the ascending list `logcv_rows` left in row buffer
    K' + 1, its count in ip[4]."""
    var kp = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var fold = ldi(ip, 2)
    var c = ld(fp, 0)
    var sw = ldi(ip, 3) != 0
    var stride = d + 1
    var p = kp * stride
    var lt = t.row(kp)
    var ix = t.row(kp + 1).bitcast[Int32]()
    var cnt = ldi(ip, 4) if fold >= 0 else n
    for q in range(t.tid, cnt, t.nt):
        var i = Int(ix.unsafe_load(q)) if fold >= 0 else q
        logcv_map_row(i, x, y, n, d, kp, fi, sw, th, toff, t.row(0))
    t.sync()
    var sl = t.slot_at.unsafe_origin_cast[MutAnyOrigin]()
    var base = ((p + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    var roles = t.nt >= base + 2 * WARP_SIZE
    var nbk = fold_blocks(cnt)
    var tasks = (p + 2) * nbk
    # the blocked order (lane/neural-pass97): (cell, block) tasks across the
    # team, the gradient cells, the loss (p) and the weight sum (p + 1),
    # each FOLD_BLOCK training rows from zero into team row K' + 2, then
    # each cell's partials folded blocks ascending
    var scr = t.row(kp + 2)
    if tasks <= n:
        for q in range(t.tid, tasks, t.nt):
            var bk = q // (p + 2)
            var o = q - bk * (p + 2)
            st(scr, o * nbk + bk, _logcv_part(o, x, y, n, d, kp, fi, sw, fold, ix, cnt, bk, t))
        t.sync()
        for o in range(t.tid, p, t.nt):
            st(g, goff + o, fold_parts(scr, o * nbk, nbk))
    else:
        for o in range(t.tid, p, t.nt):
            var accb = Float32(0)
            for bk in range(nbk):
                accb = fa(accb, _logcv_part(o, x, y, n, d, kp, fi, sw, fold, ix, cnt, bk, t))
            st(g, goff + o, accb)
    t.sync()
    var out = Float32(0)
    if t.lead():
        var wrows = Float32(0)
        var acc = Float32(0)
        var scr = t.row(kp + 2)
        if tasks <= n:
            acc = fold_parts(scr, p * nbk, nbk)
            if sw:
                wrows = fold_parts(scr, (p + 1) * nbk, nbk)
        else:
            for bk in range(nbk):
                acc = fa(acc, _logcv_part(p, x, y, n, d, kp, fi, sw, fold, ix, cnt, bk, t))
                if sw:
                    wrows = fa(wrows, _logcv_part(p + 1, x, y, n, d, kp, fi, sw, fold, ix, cnt, bk, t))
        comptime if C13_LOGCV_WEIGHTS:
            wrows = ld(fp, 1) if sw else Float32(0)
        out = logcv_finish(g, goff, th, toff, kp, d, sw, c, cnt, acc, wrows)
    return t.bcast(out)


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
            if fold >= 0 and Int(ld(y, n + i)) == fold:
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
        if fold >= 0 and Int(ld(y, n + i)) == fold:
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


#: The fold's training rows step of `logcv_fit` (count to every thread; the
#: list for the objective) and its held-out score step (the lead's value);
#: the grid fit passes device forms (x_linear/logcv_grid.mojo, cgr-linear).
comptime LcvRows = def(Team, FP, Int, Int, Int) thin -> Int
comptime LcvScore = def(Team, FP, FP, Int, Int, Int, Bool, FP, Int, Int, FP, Bool) thin -> Float32


def lcv_hits_default(t: Team, x: FP, y: FP, n: Int, d: Int, kpp: Int, fi: Bool, fw: FP, th: Int, f: Int, hitr: FP):
    """Across the team on a device; on the host in row blocks (lane linear-cpu)."""
    comptime if is_gpu():
        for i in range(t.tid, n, t.nt):
            if Int(ld(y, n + i)) == f:
                var hit = _predict_code(x, i, d, kpp, fi, fw, th) == Int(ld(y, i))
                st(hitr, i, Float32(1) if hit else Float32(0))
    else:
        var fwp = fw

        def rows_hit(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm kpp, imm fi, imm fwp, imm th,
                                        imm hitr, imm f}:
            for i in range(lo, hi):
                if Int(ld(y, n + i)) == f:
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
):
    """ip: [max_iter, fit_intercept, K', n_Cs, n_folds, sample_weight]; fp: [tol, Cs...].
    With sample_weight, y = labels | folds | fit weights (sample x class) |
    score weights (sample): the loss sum and the penalty scale use sum(w)
    of the training rows (their LinearModelLoss), the held-out accuracy is
    weighted by the raw sample weights (their scorer's sample_weight).
    res: coef K'*d | intercept K' | C_ | n_iter | scores F*nC.
    fw: theta P | C 1 | objective scratch n*(K'+1) (the host's map) | lbfgs work.  iw: [K', fit_intercept, fold, sample_weight, training rows].
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
    var extra = nf + 2 if C13_LOGCV_WEIGHTS else 1
    var work = p + extra + n * (kp + 1)
    var sc = kp * d + kp + 2
    if t.lead():
        sti(iw, 0, kp)
        sti(iw, 1, fi)
        sti(iw, 3, ldi(ip, 5))
    var cptr = fw + cslot
    var hitr = t.row(0)
    comptime if C13_LOGCV_WEIGHTS:
        for f in range(t.tid, nf, t.nt):
            st(cptr, 2 + f, lcv_disjoint_weight(y, n, f, ldi(ip, 5) != 0))
        t.sync()
    for f in range(nf):
        var cnt = rows(t, y, n, f, kp)
        if t.lead():
            comptime if C13_LOGCV_WEIGHTS:
                st(cptr, 1, lcv_retained_weight(cptr + 2, nf, f))
            sti(iw, 2, f)
            sti(iw, 4, cnt)
            fill(fw, th, p, Float32(0))
        t.sync()
        for ci in range(nc):
            if t.lead():
                st(fw, cslot, ld(fp, 1 + ci))
            t.sync()
            _ = lbfgs[obj](t, x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work, cptr + extra)
            # the held-out score: each row's hit (1) or miss (0), the lead's
            # blocked fold (weighted by the raw sample weights at y + 3n)
            var sc_v = score(t, x, y, n, d, kp if kp > 1 else 1, fi != 0, fw, th, f, hitr, ldi(ip, 5) != 0)
            if t.lead():
                st(res, sc + f * nc + ci, sc_v)
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
        comptime if C13_LOGCV_WEIGHTS:
            st(cptr, 1, lcv_retained_weight(cptr + 2, nf, -1))
        sti(iw, 2, -1)
        st(fw, cslot, ld(fp, 1 + best))
        fill(fw, th, p, Float32(0))
    best = t.bcast_int(best, 3)
    var it = lbfgs[obj](t, x, y, n, d, iw, cptr, fw, th, p, max_iter, tol, fw, work, cptr + extra)
    if not t.lead():
        return
    for k in range(kp):
        for j in range(d):
            st(res, k * d + j, ld(fw, th + k * stride + j))
        st(res, kp * d + k, ld(fw, th + k * stride + d) if fi != 0 else Float32(0))
    st(res, kp * d + kp, ld(fp, 1 + best))
    st(res, kp * d + kp + 1, i2f(it if it >= 0 else -it))


def logcv_team_rows(ip: IP) -> Int:
    """Row buffers a LogisticRegressionCV fit needs: K' residuals, the loss
    term, the fold's training row list."""
    return ldi(ip, 2) + 3  # + the blocked partials (lane/neural-pass97 order)
