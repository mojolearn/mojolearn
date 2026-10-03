# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CalibratedClassifierCV(GaussianNB, sigmoid, ensemble=True) in ONE
program (lane apple-fast-meta, -D MOJOLEARN_CALIB_GNB_FOLDS). FAST + Apple
ONLY: x_prep/units.mojo references these units only under CALIB_FOLDS, so
the IDENTICAL binding never compiles them.

The reference route (python/mojolearn/_expansion_trees.py) gathers each
fold's rows on the host, fits one GaussianNB per fold (X uploaded per fold),
scores the held-out rows (uploaded again), fits Platt's sigmoid on the host
in float64, and at predict time scores X once per member. Here X goes up
once per program:

  fit     cal_fold_*       StratifiedKFold(shuffle=False)'s test fold of every
                           row (sklearn `_make_test_folds`), from per-class
                           ranks in row order
          class_stats      per (fold, class) and per fold column statistics
                           (the lane's existing unit, pseudo-classes
                           fold * K + class)
          cal_lofo_merge   each fold's TRAIN statistics = Chan's merge of
                           the other folds' (gnb_merge's rule, so no
                           E[x^2] - E[x]^2 cancellation)
          cal_eps_folds / cal_params_folds   GaussianNB's epsilon, priors
                           and constants per fold
          cal_jll_folds    every row scored by the model that left it out
          row_softmax      the member's predict_proba
          cal_platt_*      Platt's sigmoid per (fold, column): Newton with
                           backtracking (xtrees/ops.mojo platt_fit's
                           objective and rule) in float32, every iteration
                           a blocked reduction over the fold's rows and a
                           per-problem step; T step lengths 1, 1/2, ..
                           evaluated in one pass, the first that satisfies
                           Armijo taken (the sequential rule's answer when
                           it stops within T halvings)
  predict cal_jll_folds    every row under every fold's model
          row_softmax
          cal_sigmoid_avg  each member's calibrated row (binary: (1-p, p);
                           else per-class sigmoids normalised), averaged
                           over the members

Integer words (codes, folds, ranks, counts) are read and written with
ldi / sti; the statistics are float32 through the lane's add/sub/mul/div.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP, p, ld, st, ldi, sti
from x_prep.prims import add, sub, mul, div, logf, expf
from x_prep.transform import log1pf

#: the switch: FAST + Apple, on by default since the M3 A/B (lane/apple-fast-meta
#: 18b4150df, taxi CalibratedClassifierCV(GaussianNB) 839.9 -> 49.8 ms, acc
#: .7553 / logloss .5507 identical); -D MOJOLEARN_CALIB_GNB_FOLDS_OFF turns it
#: off, the old -D MOJOLEARN_CALIB_GNB_FOLDS is harmless
comptime CALIB_FOLDS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CALIB_GNB_FOLDS_OFF"]()
)
#: words per Platt problem in STATE: A, B, fval, done, hi, lo, da, db, gd, nrows
comptime CAL_ST = 10
#: the Platt line search's step lengths 1, 1/2, .., 2^-(CAL_LS - 1)
comptime CAL_LS = 16
#: the Newton Hessian's diagonal start (platt_fit's 1e-12 is below float32)
comptime CAL_HDIAG = Float32(1e-12)


def _ge_mod_count(hi: Int, f: Int, F: Int) -> Int:
    """The number of p in [0, hi] with p mod F == f."""
    if hi < f:
        return 0
    return (hi - f) // F + 1


def cal_fold_part_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, XB, PCNT, PMIN]; t = b * K + k: the rows of block b
    (rows b*XB .. b*XB+XB-1) with code k: their count and the smallest row
    index (n when none). Integer words."""
    var n = p(q, 1)
    var K = p(q, 2)
    var XB = p(q, 3)
    var b = t // K
    var k = t % K
    var lo = b * XB
    var hi = lo + XB
    if hi > n:
        hi = n
    var cnt = 0
    var first = n
    for i in range(lo, hi):
        if ldi(f, p(q, 0) + i) == k:
            if cnt == 0:
                first = i
            cnt += 1
    sti(f, p(q, 4) + t, cnt)
    sti(f, p(q, 5) + t, first)


def cal_fold_scan_unit(t: Int, f: FP, q: IP):
    """q = [PCNT, PMIN, NB, K, CCNT, CFIRST, PREFIX]; t = k: PREFIX[b, k] =
    rows of class k before block b; CCNT[k] the class count, CFIRST[k] its
    first row (n when absent). Integer words."""
    var NB = p(q, 2)
    var K = p(q, 3)
    var run = 0
    var first = -1
    for b in range(NB):
        sti(f, p(q, 6) + b * K + t, run)
        run += ldi(f, p(q, 0) + b * K + t)
        var m = ldi(f, p(q, 1) + b * K + t)
        if first < 0 or m < first:
            first = m
    sti(f, p(q, 4) + t, run)
    sti(f, p(q, 5) + t, first)


def cal_fold_rank_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, XB, PREFIX, RANK]; t = b * K + k: RANK[i] = the
    number of earlier rows with the same code, for block b's rows of code
    k. Integer words."""
    var n = p(q, 1)
    var K = p(q, 2)
    var XB = p(q, 3)
    var b = t // K
    var k = t % K
    var lo = b * XB
    var hi = lo + XB
    if hi > n:
        hi = n
    var r = ldi(f, p(q, 4) + t)
    for i in range(lo, hi):
        if ldi(f, p(q, 0) + i) == k:
            sti(f, p(q, 5) + i, r)
            r += 1


def cal_fold_assign_unit(t: Int, f: FP, q: IP):
    """q = [CODES, RANK, n, K, CCNT, CFIRST, F, FOLD, PCODE, FOLDF]; t = row
    i. sklearn StratifiedKFold(shuffle=False)._make_test_folds: classes
    renumbered by first appearance, the allocation of fold f to class k =
    the positions p = f, f + F, .. of the sorted encoded labels inside the
    class's run [s, s + c), the class's rows taking fold 0 alloc[0] times,
    then fold 1, .. in row order. FOLD[i] (integer), PCODE[i] = fold * K +
    code and FOLDF[i] = fold (float values for class_stats)."""
    var K = p(q, 3)
    var F = p(q, 6)
    var k = ldi(f, p(q, 0) + t)
    var r = ldi(f, p(q, 1) + t)
    var first_k = ldi(f, p(q, 5) + k)
    var s = 0
    for kk in range(K):
        if ldi(f, p(q, 5) + kk) < first_k:
            s += ldi(f, p(q, 4) + kk)
    var c = ldi(f, p(q, 4) + k)
    var hi = s + c - 1
    var cum = 0
    var fold = F - 1
    for ff in range(F):
        cum += _ge_mod_count(hi, ff, F) - _ge_mod_count(s - 1, ff, F)
        if r < cum:
            fold = ff
            break
    sti(f, p(q, 7) + t, fold)
    st(f, p(q, 8) + t, Float32(fold * K + k))
    st(f, p(q, 9) + t, Float32(fold))


def cal_lofo_merge_unit(t: Int, f: FP, q: IP):
    """q = [CNT, MEAN, VAR, F, G, d, OCNT, OMEAN, OVAR]; t = (f * G + g) * d
    + c: the count, mean and population variance of group g over every fold
    but f (Chan's merge in ascending fold order, gnb_merge's rule); OCNT
    written for c == 0."""
    var F = p(q, 3)
    var G = p(q, 4)
    var d = p(q, 5)
    var fg = t // d
    var c = t % d
    var fo = fg // G
    var g = fg % G
    var cnt = Float32(0)
    var m = Float32(0)
    var v = Float32(0)
    for h in range(F):
        if h == fo:
            continue
        var hg = h * G + g
        var nn = ld(f, p(q, 0) + hg)
        if nn == Float32(0):
            continue
        var nmu = ld(f, p(q, 1) + hg * d + c)
        var nva = ld(f, p(q, 2) + hg * d + c)
        if cnt == Float32(0):
            m = nmu
            v = nva
        else:
            var tot = add(cnt, nn)
            var e = sub(m, nmu)
            var ssd = add(add(mul(cnt, v), mul(nn, nva)), mul(div(mul(nn, cnt), tot), mul(e, e)))
            m = div(add(mul(nn, nmu), mul(cnt, m)), tot)
            v = div(ssd, tot)
        cnt = add(cnt, nn)
    st(f, p(q, 7) + t, m)
    st(f, p(q, 8) + t, v)
    if c == 0:
        st(f, p(q, 6) + fg, cnt)


def cal_eps_folds_unit(t: Int, f: FP, q: IP):
    """q = [VAR, F, d, EPS, VS]; t = fold f: EPS[f] = VS * max_c VAR[f, c]
    (gnb_eps over the fold's train column variances)."""
    var d = p(q, 2)
    var m = Float32(0)
    for c in range(d):
        var v = ld(f, p(q, 0) + t * d + c)
        if v > m:
            m = v
    st(f, p(q, 3) + t, mul(ld(f, p(q, 4)), m))


def cal_params_folds_unit(t: Int, f: FP, q: IP):
    """q = [CNT, VAR, K, d, NTR, EPS, PRIOR, CONST]; t = f * K + k:
    gnb_params for fold f: VAR[t, :] += EPS[f] (<= 0 becomes 1), PRIOR[t] =
    CNT[t] / NTR[f], CONST[t] = log PRIOR - 0.5 sum_c log(2 pi VAR)."""
    var K = p(q, 2)
    var d = p(q, 3)
    var fo = t // K
    var eps = ld(f, p(q, 5) + fo)
    var sl = Float32(0)
    for c in range(d):
        var v = add(ld(f, p(q, 1) + t * d + c), eps)
        if v <= Float32(0):
            v = Float32(1)
        st(f, p(q, 1) + t * d + c, v)
        sl = add(sl, logf(mul(Float32(6.2831855), v)))
    var prior = div(ld(f, p(q, 0) + t), ld(f, p(q, 4) + fo))
    st(f, p(q, 6) + t, prior)
    st(f, p(q, 7) + t, sub(logf(prior), mul(Float32(0.5), sl)))


def cal_jll_folds_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, THETA, VAR, CONST, K, FOLD, OUT, F]. FOLD >= 0: t = i *
    K + k, row i under the model of its own fold FOLD[i] (the held-out
    score). FOLD < 0: t = (i * F + f) * K + k, row i under every fold's
    model. gnb_jll's value."""
    var d = p(q, 2)
    var K = p(q, 6)
    var i: Int
    var model: Int
    if p(q, 7) >= 0:
        i = t // K
        model = ldi(f, p(q, 7) + i)
    else:
        var F = p(q, 9)
        var row_f = t // K
        i = row_f // F
        model = row_f % F
    var k = t % K
    var mk = model * K + k
    var s = Float32(0)
    for c in range(d):
        var e = sub(ld(f, p(q, 0) + i * d + c), ld(f, p(q, 3) + mk * d + c))
        s = add(s, div(mul(e, e), ld(f, p(q, 4) + mk * d + c)))
    st(f, p(q, 8) + t, sub(ld(f, p(q, 5) + mk), mul(Float32(0.5), s)))


@always_inline
def _cal_target_class(c: Int, j: Int) -> Int:
    """The code a calibrator's column j scores: class 1 when binary (c == 1)."""
    return 1 if c == 1 else j


def cal_platt_init_unit(t: Int, f: FP, q: IP):
    """q = [CODES, FOLD, n, F, c, PB, PART]; t = b * (F * c) + f * c + j:
    block b's rows of fold f: the positives (code == target of column j)
    and the row count, PART[t * 2 + 0 / 1] (float values)."""
    var n = p(q, 2)
    var F = p(q, 3)
    var c = p(q, 4)
    var PB = p(q, 5)
    var FC = F * c
    var b = t // FC
    var fj = t % FC
    var fo = fj // c
    var cls = _cal_target_class(c, fj % c)
    var lo = b * PB
    var hi = lo + PB
    if hi > n:
        hi = n
    var pos = 0
    var cnt = 0
    for i in range(lo, hi):
        if ldi(f, p(q, 1) + i) == fo:
            cnt += 1
            if ldi(f, p(q, 0) + i) == cls:
                pos += 1
    st(f, p(q, 6) + t * 2, Float32(pos))
    st(f, p(q, 6) + t * 2 + 1, Float32(cnt))


def cal_platt_setup_unit(t: Int, f: FP, q: IP):
    """q = [PART, NBP, F, c, STATE]; t = problem fj: platt_fit's start:
    T+ = (N+ + 1) / (N+ + 2), T- = 1 / (N- + 2), A = 0, B = log((N- + 1) /
    (N+ + 1)); STATE[fj] = [A, B, fval, done, hi, lo, da, db, gd, nrows]."""
    var NBP = p(q, 1)
    var FC = p(q, 2) * p(q, 3)
    var pos = Float32(0)
    var cnt = Float32(0)
    for b in range(NBP):
        pos = add(pos, ld(f, p(q, 0) + (b * FC + t) * 2))
        cnt = add(cnt, ld(f, p(q, 0) + (b * FC + t) * 2 + 1))
    var neg = sub(cnt, pos)
    var S = p(q, 4) + t * CAL_ST
    st(f, S + 0, Float32(0))
    st(f, S + 1, logf(div(add(neg, Float32(1)), add(pos, Float32(1)))))
    st(f, S + 2, Float32(0))
    st(f, S + 3, Float32(0))
    st(f, S + 4, div(add(pos, Float32(1)), add(pos, Float32(2))))
    st(f, S + 5, div(Float32(1), add(neg, Float32(2))))
    st(f, S + 6, Float32(0))
    st(f, S + 7, Float32(0))
    st(f, S + 8, Float32(0))
    st(f, S + 9, cnt)


@always_inline
def _log1pexp32(z: Float32) -> Float32:
    """log(1 + exp(z)) without overflow."""
    if z >= Float32(0):
        return add(z, log1pf(expf(sub(Float32(0), z))))
    return log1pf(expf(z))


def cal_platt_part_unit(t: Int, f: FP, q: IP):
    """q = [S, CS, CODES, FOLD, n, F, c, PB, STATE, PART]; t = b * (F * c) +
    fj: over block b's rows of fold f, at the problem's current (A, B):
    the Newton sums h11, h22, h21, g1, g2 and the objective value, PART[t *
    6 ..]. The score of row i is S[i * CS + col], col = 1 when binary else
    j. Nothing when the problem is done."""
    var n = p(q, 4)
    var F = p(q, 5)
    var c = p(q, 6)
    var PB = p(q, 7)
    var FC = F * c
    var b = t // FC
    var fj = t % FC
    var fo = fj // c
    var j = fj % c
    var col = _cal_target_class(c, j)
    var cls = col
    var S = p(q, 8) + fj * CAL_ST
    if ld(f, S + 3) != Float32(0):
        return
    var a = ld(f, S + 0)
    var bb = ld(f, S + 1)
    var hi_t = ld(f, S + 4)
    var lo_t = ld(f, S + 5)
    var lo = b * PB
    var hi = lo + PB
    if hi > n:
        hi = n
    var h11 = Float32(0)
    var h22 = Float32(0)
    var h21 = Float32(0)
    var g1 = Float32(0)
    var g2 = Float32(0)
    var val = Float32(0)
    for i in range(lo, hi):
        if ldi(f, p(q, 3) + i) != fo:
            continue
        var fi = ld(f, p(q, 0) + i * p(q, 1) + col)
        var ti = hi_t if ldi(f, p(q, 2) + i) == cls else lo_t
        var z = add(mul(fi, a), bb)
        var pp: Float32
        var qq: Float32
        if z >= Float32(0):
            var e = expf(sub(Float32(0), z))
            var den = add(Float32(1), e)
            pp = div(e, den)
            qq = div(Float32(1), den)
        else:
            var e = expf(z)
            var den = add(Float32(1), e)
            pp = div(Float32(1), den)
            qq = div(e, den)
        var d2 = mul(pp, qq)
        h11 = add(h11, mul(mul(fi, fi), d2))
        h22 = add(h22, d2)
        h21 = add(h21, mul(fi, d2))
        var d1 = sub(ti, pp)
        g1 = add(g1, mul(fi, d1))
        g2 = add(g2, d1)
        val = add(val, add(mul(ti, _log1pexp32(z)), mul(sub(Float32(1), ti), _log1pexp32(sub(Float32(0), z)))))
    var O = p(q, 9) + t * 6
    st(f, O + 0, h11)
    st(f, O + 1, h22)
    st(f, O + 2, h21)
    st(f, O + 3, g1)
    st(f, O + 4, g2)
    st(f, O + 5, val)


def cal_platt_step_unit(t: Int, f: FP, q: IP):
    """q = [PART, NBP, F, c, STATE, TOL]; t = problem fj: folds the blocks'
    sums in block order, stops the problem when |g| < TOL * rows, else the
    Newton direction (da, db) and gd = g . d into STATE, and fval = the
    objective at (A, B)."""
    var NBP = p(q, 1)
    var FC = p(q, 2) * p(q, 3)
    var S = p(q, 4) + t * CAL_ST
    if ld(f, S + 3) != Float32(0):
        return
    var h11 = CAL_HDIAG
    var h22 = CAL_HDIAG
    var h21 = Float32(0)
    var g1 = Float32(0)
    var g2 = Float32(0)
    var val = Float32(0)
    for b in range(NBP):
        var O = p(q, 0) + (b * FC + t) * 6
        h11 = add(h11, ld(f, O + 0))
        h22 = add(h22, ld(f, O + 1))
        h21 = add(h21, ld(f, O + 2))
        g1 = add(g1, ld(f, O + 3))
        g2 = add(g2, ld(f, O + 4))
        val = add(val, ld(f, O + 5))
    st(f, S + 2, val)
    var tol = mul(ld(f, p(q, 5)), ld(f, S + 9))
    if abs(g1) < tol and abs(g2) < tol:
        st(f, S + 3, Float32(1))
        return
    var det = sub(mul(h11, h22), mul(h21, h21))
    var da = sub(Float32(0), div(sub(mul(h22, g1), mul(h21, g2)), det))
    var db = sub(Float32(0), div(sub(mul(h11, g2), mul(h21, g1)), det))
    st(f, S + 6, da)
    st(f, S + 7, db)
    st(f, S + 8, add(mul(g1, da), mul(g2, db)))


def cal_platt_ls_part_unit(t: Int, f: FP, q: IP):
    """q = [S, CS, CODES, FOLD, n, F, c, PB, STATE, PART, T]; t = b * (F * c)
    + fj: the objective over block b's rows of fold f at (A + s da, B + s
    db) for the T steps s = 1, 1/2, .., PART[t * T + u]. Nothing when done."""
    var n = p(q, 4)
    var F = p(q, 5)
    var c = p(q, 6)
    var PB = p(q, 7)
    var T = p(q, 10)
    var FC = F * c
    var b = t // FC
    var fj = t % FC
    var fo = fj // c
    var col = _cal_target_class(c, fj % c)
    var cls = col
    var S = p(q, 8) + fj * CAL_ST
    if ld(f, S + 3) != Float32(0):
        return
    var a = ld(f, S + 0)
    var bb = ld(f, S + 1)
    var hi_t = ld(f, S + 4)
    var lo_t = ld(f, S + 5)
    var da = ld(f, S + 6)
    var db = ld(f, S + 7)
    var lo = b * PB
    var hi = lo + PB
    if hi > n:
        hi = n
    var O = p(q, 9) + t * T
    for u in range(T):
        st(f, O + u, Float32(0))
    for i in range(lo, hi):
        if ldi(f, p(q, 3) + i) != fo:
            continue
        var fi = ld(f, p(q, 0) + i * p(q, 1) + col)
        var ti = hi_t if ldi(f, p(q, 2) + i) == cls else lo_t
        var one_m = sub(Float32(1), ti)
        var step = Float32(1)
        for u in range(T):
            var z = add(mul(fi, add(a, mul(step, da))), add(bb, mul(step, db)))
            var v = add(mul(ti, _log1pexp32(z)), mul(one_m, _log1pexp32(sub(Float32(0), z))))
            st(f, O + u, add(ld(f, O + u), v))
            step = mul(step, Float32(0.5))


def cal_platt_ls_pick_unit(t: Int, f: FP, q: IP):
    """q = [PART, NBP, F, c, STATE, T]; t = problem fj: the first step s
    (1, 1/2, ..) whose objective is below fval + 1e-4 s gd (Armijo) moves
    (A, B); none within T halvings stops the problem (platt_fit's `not
    moved`)."""
    var NBP = p(q, 1)
    var FC = p(q, 2) * p(q, 3)
    var T = p(q, 5)
    var S = p(q, 4) + t * CAL_ST
    if ld(f, S + 3) != Float32(0):
        return
    var fval = ld(f, S + 2)
    var gd = ld(f, S + 8)
    var step = Float32(1)
    for u in range(T):
        var nf = Float32(0)
        for b in range(NBP):
            nf = add(nf, ld(f, p(q, 0) + (b * FC + t) * T + u))
        if nf < add(fval, mul(mul(Float32(0.0001), step), gd)):
            st(f, S + 0, add(ld(f, S + 0), mul(step, ld(f, S + 6))))
            st(f, S + 1, add(ld(f, S + 1), mul(step, ld(f, S + 7))))
            st(f, S + 2, nf)
            return
        step = mul(step, Float32(0.5))
    st(f, S + 3, Float32(1))


@always_inline
def _platt_p(fi: Float32, a: Float32, b: Float32) -> Float32:
    """1 / (1 + exp(A f + B)) without overflow."""
    var z = add(mul(fi, a), b)
    if z >= Float32(0):
        var e = expf(sub(Float32(0), z))
        return div(e, add(Float32(1), e))
    return div(Float32(1), add(Float32(1), expf(z)))


def cal_sigmoid_avg_unit(t: Int, f: FP, q: IP):
    """q = [PROBA, n, F, K, STATE, c, OUT]; t = row i. PROBA is (n * F, K),
    row i * F + f the member f's predict_proba of row i. Binary (c == 1):
    p = mean_f sigmoid_f(PROBA[.., 1]), OUT[i] = (1 - p, p). Else each
    member's per-class sigmoids are normalised over the row (a zero row is
    uniform, xtrees normalize_rows' rule) and averaged: OUT[i, k]."""
    var F = p(q, 2)
    var K = p(q, 3)
    var c = p(q, 5)
    var O = p(q, 6) + t * K
    var inv_f = div(Float32(1), Float32(F))
    if c == 1:
        var acc = Float32(0)
        for fo in range(F):
            var S = p(q, 4) + fo * CAL_ST
            acc = add(acc, _platt_p(ld(f, p(q, 0) + (t * F + fo) * K + 1), ld(f, S + 0), ld(f, S + 1)))
        var pr = mul(acc, inv_f)
        st(f, O + 1, pr)
        st(f, O + 0, sub(Float32(1), pr))
        return
    for k in range(K):
        st(f, O + k, Float32(0))
    for fo in range(F):
        var s = Float32(0)
        for k in range(K):
            var S = p(q, 4) + (fo * c + k) * CAL_ST
            s = add(s, _platt_p(ld(f, p(q, 0) + (t * F + fo) * K + k), ld(f, S + 0), ld(f, S + 1)))
        for k in range(K):
            var S = p(q, 4) + (fo * c + k) * CAL_ST
            var v = _platt_p(ld(f, p(q, 0) + (t * F + fo) * K + k), ld(f, S + 0), ld(f, S + 1))
            var nv = div(v, s) if s > Float32(0) else div(Float32(1), Float32(K))
            st(f, O + k, add(ld(f, O + k), mul(nv, inv_f)))
