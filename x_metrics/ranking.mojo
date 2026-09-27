# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ranking and per-sample units (scikit-learn 1.9 sklearn/metrics/_ranking.py
and _classification.py).

`bin_curve_unit` is `confusion_matrix_at_thresholds`: a stable sort of the
scores DESCENDING (ties keep ascending row order, DEVIATION 6101), then the
cumulative (weighted) true and false positive counts at every distinct score.
Unweighted counts are exact integers; weighted ones are a SEQUENTIAL ascending
Float32 prefix (DEVIATION 6107). Rows of zero weight are skipped, as
scikit-learn drops them before sorting.

`row_metric_unit` computes one per-sample score per row (top-k hit, Brier
term, log-loss term, hinge term, DCG, the label-ranking scores); their
(weighted) means are group.mojo's PairSum.
"""
from std.memory import bitcast
from x_metrics.common import FP, IP, p, ld, st, ldi, sti, key, fadd, PairSum
from checks.numerics import ftz, identical_mul, identical_div, identical_log

comptime ROW_TOPK = 0
comptime ROW_BRIER = 1
comptime ROW_LOGLOSS = 2
comptime ROW_HINGE_BIN = 3
comptime ROW_HINGE_MC = 4
comptime ROW_DCG = 5
comptime ROW_COVERAGE = 6
comptime ROW_LRAP = 7
comptime ROW_RANKLOSS = 8
comptime ROW_DCG_IGNORE = 9

comptime F32_EPS = Float32(1.1920929e-07)


@always_inline
def vkey(x: Float32) -> UInt32:
    """`key` with -0.0 read as +0.0: the VALUE order numpy's sorts and
    `unique` use, where the two zeros tie."""
    return key(x if x != Float32(0) else Float32(0))


@always_inline
def _desc_before(f: FP, S: Int, SS: Int, i: Int, j: Int) -> Bool:
    """Row i before row j: larger score first, ties by ascending row."""
    var ki = key(ld(f, S + i * SS))
    var kj = key(ld(f, S + j * SS))
    return ki > kj or (ki == kj and i < j)


def _sift_desc(f: FP, S: Int, SS: Int, O: Int, start: Int, n: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= n:
            return
        if child + 1 < n and _desc_before(f, S, SS, ldi(f, O + child), ldi(f, O + child + 1)):
            child += 1
        if _desc_before(f, S, SS, ldi(f, O + root), ldi(f, O + child)):
            var tmp = ldi(f, O + root)
            sti(f, O + root, ldi(f, O + child))
            sti(f, O + child, tmp)
            root = child
        else:
            return


def sort_desc(f: FP, S: Int, SS: Int, O: Int, n: Int):
    """Heapsort of the row indices in ORD[O .. O+n) (already filled) into the
    order `_desc_before` defines."""
    if n < 2:
        return
    var start = n // 2 - 1
    while start >= 0:
        _sift_desc(f, S, SS, O, start, n)
        start -= 1
    var end = n - 1
    while end > 0:
        var tmp = ldi(f, O)
        sti(f, O, ldi(f, O + end))
        sti(f, O + end, tmp)
        _sift_desc(f, S, SS, O, 0, end)
        end -= 1


def bin_curve_unit(t: Int, f: FP, q: IP):
    """q = [S, SS, POS, W, n, ORD, FPS, TPS, THR, CNT]; t = problem.
    Problem t reads scores S + t + r*SS (column t of a row-major matrix, or
    one vector with t = 0) and positive flags POS + t*n + r (int 0/1); its
    outputs are at t*n offsets and CNT + t (the number of thresholds)."""
    var S = p(q, 0) + t
    var SS = p(q, 1)
    var POS = p(q, 2) + t * p(q, 4)
    var W = p(q, 3)
    var n = p(q, 4)
    var O = p(q, 5) + t * n
    var FPS = p(q, 6) + t * n
    var TPS = p(q, 7) + t * n
    var THR = p(q, 8) + t * n
    var m = 0
    for r in range(n):
        if W >= 0 and ld(f, W + r) == Float32(0):
            continue
        sti(f, O + m, r)
        m += 1
    sort_desc(f, S, SS, O, m)
    var tp_i = 0
    var fp_i = 0
    var tp_w = Float32(0)
    var fp_w = Float32(0)
    var cnt = 0
    for i in range(m):
        var r = ldi(f, O + i)
        var pos = ldi(f, POS + r)
        if W >= 0:
            var w = ld(f, W + r)
            if pos == 1:
                tp_w = fadd(tp_w, w)
            else:
                fp_w = fadd(fp_w, w)
        else:
            if pos == 1:
                tp_i += 1
            else:
                fp_i += 1
        var s = ld(f, S + r * SS)
        var last = i == m - 1
        if not last:
            var s2 = ld(f, S + ldi(f, O + i + 1) * SS)
            last = s2 != s
        if last:
            if W >= 0:
                st(f, TPS + cnt, tp_w)
                st(f, FPS + cnt, fp_w)
            else:
                st(f, TPS + cnt, Float32(tp_i))
                st(f, FPS + cnt, Float32(fp_i))
            st(f, THR + cnt, s)
            cnt += 1
    sti(f, p(q, 9) + t, cnt)


def _clip01(x: Float32) -> Float32:
    var lo = F32_EPS
    var hi = ftz(Float32(1) - F32_EPS)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def row_metric_unit(t: Int, f: FP, q: IP):
    """q = [S, k, Y, OUT, kind, K, D]; t = row. S is the (n, k) row-major
    score / probability matrix (k = 1: a vector), Y the true class code
    (int) or, for the label-ranking kinds and DCG, the (n, k) relevance /
    indicator matrix (float). K is top-k's k or DCG's cutoff (<= 0: none);
    D is the DCG discount table (k floats, 1 / log_b(i + 2))."""
    var S = p(q, 0) + t * p(q, 1)
    var k = p(q, 1)
    var Y = p(q, 2)
    var OUT = p(q, 3)
    var kind = p(q, 4)
    var K = p(q, 5)
    var D = p(q, 6)
    if kind == ROW_TOPK:
        var j = ldi(f, Y + t)
        var sj = ld(f, S + j)
        var ahead = 0
        for c in range(k):
            var sc = ld(f, S + c)
            if vkey(sc) > vkey(sj) or (vkey(sc) == vkey(sj) and c > j):
                ahead += 1
        st(f, OUT + t, Float32(1) if ahead < K else Float32(0))
    elif kind == ROW_BRIER:
        var j = ldi(f, Y + t)
        var acc = PairSum()
        for c in range(k):
            var d = ftz((Float32(1) if c == j else Float32(0)) - ld(f, S + c))
            acc.add(identical_mul(d, d))
        st(f, OUT + t, acc.result())
    elif kind == ROW_LOGLOSS:
        var j = ldi(f, Y + t)
        st(f, OUT + t, ftz(Float32(0) - identical_log(_clip01(ld(f, S + j)))))
    elif kind == ROW_HINGE_BIN:
        var y = Float32(1) if ldi(f, Y + t) == 1 else Float32(-1)
        var m = ftz(Float32(1) - identical_mul(y, ld(f, S)))
        st(f, OUT + t, m if m > Float32(0) else Float32(0))
    elif kind == ROW_HINGE_MC:
        var j = ldi(f, Y + t)
        var best = Float32(0)
        var have = False
        for c in range(k):
            if c == j:
                continue
            var v = ld(f, S + c)
            if not have or v > best:
                best = v
                have = True
        var m = ftz(Float32(1) - ftz(ld(f, S + j) - best))
        st(f, OUT + t, m if m > Float32(0) else Float32(0))
    elif kind == ROW_DCG:
        st(f, OUT + t, _dcg_row(f, S, Y + t * k, k, K, D))
    elif kind == ROW_DCG_IGNORE:
        st(f, OUT + t, _dcg_row_ranked(f, S, Y + t * k, k, K, D))
    elif kind == ROW_COVERAGE:
        # the largest rank of a true label: #{c : s_c >= min over true s}
        var have = False
        var mn = Float32(0)
        for c in range(k):
            if ld(f, Y + t * k + c) != Float32(0):
                var v = ld(f, S + c)
                if not have or v < mn:
                    mn = v
                    have = True
        var cov = 0
        if have:
            for c in range(k):
                if ld(f, S + c) >= mn:
                    cov += 1
        st(f, OUT + t, Float32(cov))
    elif kind == ROW_LRAP:
        var yb = Y + t * k
        var nrel = 0
        for c in range(k):
            if ld(f, yb + c) != Float32(0):
                nrel += 1
        if nrel == 0 or nrel == k:
            st(f, OUT + t, Float32(1))
            return
        var acc = PairSum()
        for c in range(k):
            if ld(f, yb + c) == Float32(0):
                continue
            var sc = ld(f, S + c)
            var rank = 0
            var lrank = 0
            for d in range(k):
                if ld(f, S + d) >= sc:
                    rank += 1
                    if ld(f, yb + d) != Float32(0):
                        lrank += 1
            acc.add(identical_div(Float32(lrank), Float32(rank)))
        st(f, OUT + t, identical_div(acc.result(), Float32(nrel)))
    elif kind == ROW_RANKLOSS:
        # ROW_RANKLOSS: #{(a, b): a true, b false, s_a <= s_b} / (|T| |F|)
        var yb = Y + t * k
        var nrel = 0
        for c in range(k):
            if ld(f, yb + c) != Float32(0):
                nrel += 1
        if nrel == 0 or nrel == k:
            st(f, OUT + t, Float32(0))
            return
        var bad = 0
        for a in range(k):
            if ld(f, yb + a) == Float32(0):
                continue
            var sa = ld(f, S + a)
            for b in range(k):
                if ld(f, yb + b) != Float32(0):
                    continue
                if sa <= ld(f, S + b):
                    bad += 1
        st(f, OUT + t, identical_div(Float32(bad), Float32(nrel * (k - nrel))))


def _dcg_row(f: FP, S: Int, Yr: Int, k: Int, K: Int, D: Int) -> Float32:
    """scikit-learn `_tie_averaged_dcg` (ties among scores share the mean
    gain over the discount slots they occupy), with the discount zeroed past
    K. Groups of equal score are visited in descending score order; each
    group's gain sum and discount sum are fixed-order folds."""
    var acc = PairSum()
    var pos = 0
    # visit distinct scores descending: repeatedly take the largest key below the last
    var have_last = False
    var last_key = UInt32(0)
    while pos < k:
        var best = UInt32(0)
        var found = False
        for c in range(k):
            var kc = vkey(ld(f, S + c))
            if have_last and kc >= last_key:
                continue
            if not found or kc > best:
                best = kc
                found = True
        if not found:
            break
        var gains = PairSum()
        var cnt = 0
        for c in range(k):
            if vkey(ld(f, S + c)) == best:
                gains.add(ld(f, Yr + c))
                cnt += 1
        var disc = PairSum()
        for i in range(pos, pos + cnt):
            if K <= 0 or i < K:
                disc.add(ld(f, D + i))
        var g = identical_div(gains.result(), Float32(cnt))
        acc.add(identical_mul(g, disc.result()))
        pos += cnt
        have_last = True
        last_key = best
    return acc.result()


def _dcg_row_ranked(f: FP, S: Int, Yr: Int, k: Int, K: Int, D: Int) -> Float32:
    """DCG with ties broken by DESCENDING column index: position i holds the
    i-th item in (value desc, index desc) order."""
    var acc = PairSum()
    var lim = k if K <= 0 or K > k else K
    var have_last = False
    var last_key = UInt32(0)
    var last_idx = 0
    for i in range(lim):
        var best = -1
        var bkey = UInt32(0)
        for c in range(k):
            var kc = vkey(ld(f, S + c))
            if have_last and (kc > last_key or (kc == last_key and c >= last_idx)):
                continue
            if best < 0 or kc > bkey or (kc == bkey and c > best):
                best = c
                bkey = kc
        acc.add(identical_mul(ld(f, Yr + best), ld(f, D + i)))
        have_last = True
        last_key = bkey
        last_idx = best
    return acc.result()
