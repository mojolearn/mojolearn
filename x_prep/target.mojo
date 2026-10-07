# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TargetEncoder units, in the prep lane's program model (x_prep/common.mojo).

Reference: scikit-learn 1.9 `sklearn/preprocessing/_target_encoder.py`
(`fit_transform` cross fitting, `_fit_encodings_all`, `_transform_X_ordinal`)
and `_target_encoder_fast.pyx` (`_fit_encoding_fast`,
`_fit_encoding_fast_auto_smooth`). A fold index `fi` in [0, F) means "every
row whose fold is not fi" (the cross-fit training rows of fold fi); fi == F
means every row (plain `fit`). T target columns: 1 for a binary or
continuous target, one per class (one-vs-rest) for a multiclass one.
Where the reference's lambda is NaN (an empty category, or zero variance and
zero spread) it returns the target mean; so does this, by test, not by NaN.
"""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, RUN, run_block
from checks.numerics import ftz
from x_prep.prims import add, acc_add, sub, mul, div
from x_prep.idn_fold import (
    IDN_TE_GLOBAL_TREE, IDN_TE_ENC_TREE, IDN_TE_BLOCKED, TREE_W, TEB, LTLanes, lt_zero, lt_tree, blt_close,
)


def te_global_lt(t: Int, f: FP, q: IP):
    """`te_global_unit`'s q and t in the lane-tree order: lane r mod TREE_W
    takes row r's target (rows of fold fi skipped) ascending, then the tree;
    the count is exact; the squared deviations from the mean likewise.
    x_prep/idn_tree.mojo `te_global_lt_kernel` is the device's spelling."""
    var n = p(q, 1)
    var T = p(q, 2)
    var Y = p(q, 0)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var a = LTLanes(fill=Float32(0))
    var cnt = 0
    for i in range(n):
        if Int(ld(f, FO + i)) == fi:
            continue
        var l = i % TREE_W
        a[l] = add(a[l], ld(f, Y + i * T + tt))
        cnt += 1
    var s = lt_tree(a)
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        lt_zero(a)
        for i in range(n):
            if Int(ld(f, FO + i)) == fi:
                continue
            var l = i % TREE_W
            var e = sub(ld(f, Y + i * T + tt), mean)
            a[l] = add(a[l], mul(e, e))
        ss = div(lt_tree(a), Float32(cnt))
    st(f, p(q, 4) + 2 * t, mean)
    st(f, p(q, 4) + 2 * t + 1, ss)


def te_enc_lt_fold(f: FP, rp: FP, BF: Int, BY: Int, R: Int, Y: Int, T: Int, tt: Int, FO: Int, lo: Int, hi: Int,
                   fi: Int, smooth: Float32, mut s: Float32, mut cnt: Int, mut mean: Float32, mut ssd: Float32):
    """One category's te_enc folds in the lane-tree order over its bucket
    positions k in [lo, hi) (rank r = k - lo): BF >= 0 reads the gathered
    folds BF[k] and targets BY[k] (`te_gather`), else row i = the int word
    rp[R + k] (the unit: the arena's `te_bucket` ROWS; the host column: its
    own bucket)'s FOLD[FO + i] and Y[Y + i*T + tt]. Rows of fold fi add
    nothing. The sum
    and count; with smooth < 0 and a count, the mean and the squared
    deviations the same way (else they stay 0)."""
    var a = LTLanes(fill=Float32(0))
    var c = 0
    for k in range(lo, hi):
        var fo: Int
        var y: Float32
        if BF >= 0:
            fo = Int(ld(f, BF + k))
            y = ld(f, BY + k)
        else:
            var i = ldi(rp, R + k)
            fo = Int(ld(f, FO + i))
            y = ld(f, Y + i * T + tt)
        if fo == fi:
            continue
        var l = (k - lo) % TREE_W
        a[l] = add(a[l], y)
        c += 1
    s = lt_tree(a)
    cnt = c
    mean = Float32(0)
    ssd = Float32(0)
    if smooth < Float32(0) and c > 0:
        mean = div(s, Float32(c))
        lt_zero(a)
        for k in range(lo, hi):
            var fo: Int
            var y: Float32
            if BF >= 0:
                fo = Int(ld(f, BF + k))
                y = ld(f, BY + k)
            else:
                var i = ldi(rp, R + k)
                fo = Int(ld(f, FO + i))
                y = ld(f, Y + i * T + tt)
            if fo == fi:
                continue
            var l = (k - lo) % TREE_W
            var e = sub(y, mean)
            a[l] = add(a[l], mul(e, e))
        ssd = lt_tree(a)


def te_enc_lt(t: Int, f: FP, q: IP):
    """`te_enc_unit`'s q and t in the lane-tree order: rank r of a row among
    its category's rows (all folds, ascending), lane r mod TREE_W. With BK > 0
    the bucket gives the ranks (k - lo); else every row is scanned and the
    rank counted. x_prep/idn_tree.mojo `te_enc_lt_kernel` is the device's
    spelling, x_prep/host/target.mojo the host column's."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var smooth = ld(f, p(q, 9))
    var s = Float32(0)
    var cnt = 0
    var mean = Float32(0)
    var ssd = Float32(0)
    var bk = p(q, 11)
    if bk > 0:
        var S = bk - 1 + j * (cmax + 1)
        var lo = ldi(f, S + cat)
        var hi = ldi(f, S + cat + 1)
        var gb = p(q, 13)
        var BF = -1
        var BY = 0
        if gb > 0:
            BF = gb - 1 + j * n
            BY = gb - 1 + d * n + (j * T + tt) * n
        te_enc_lt_fold(f, f, BF, BY, p(q, 12) + j * n, p(q, 3), T, tt, p(q, 5), lo, hi, fi, smooth, s, cnt,
                       mean, ssd)
        st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))
        return
    # no buckets: every row, the rank counted over the category's rows
    var a = LTLanes(fill=Float32(0))
    var rank = 0
    for i in range(n):
        if Int(ld(f, p(q, 0) + i * d + j)) != cat:
            continue
        var rk = rank
        rank += 1
        if Int(ld(f, p(q, 5) + i)) == fi:
            continue
        var l = rk % TREE_W
        a[l] = add(a[l], ld(f, p(q, 3) + i * T + tt))
        cnt += 1
    s = lt_tree(a)
    if smooth < Float32(0) and cnt > 0:
        mean = div(s, Float32(cnt))
        lt_zero(a)
        rank = 0
        for i in range(n):
            if Int(ld(f, p(q, 0) + i * d + j)) != cat:
                continue
            var rk = rank
            rank += 1
            if Int(ld(f, p(q, 5) + i)) == fi:
                continue
            var l = rk % TREE_W
            var e = sub(ld(f, p(q, 3) + i * T + tt), mean)
            a[l] = add(a[l], mul(e, e))
        ssd = lt_tree(a)
    st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))


# ------------------------------------------------- the blocked order (BLT)
def te_global_blt(t: Int, f: FP, q: IP):
    """`te_global_unit`'s q and t in the blocked order (IDN_TE_BLOCKED,
    x_prep/idn_fold.mojo `BLT`): rows in TEB-row blocks, block m's tree
    (lane i mod TREE_W) closing into lane m mod TREE_W of the outer tree;
    the count exact; then the squared deviations from the mean the same
    way. x_prep/idn_blocked.mojo spells it on the device (part kernel per
    block, fold kernel per unit)."""
    var n = p(q, 1)
    var T = p(q, 2)
    var Y = p(q, 0)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var a = LTLanes(fill=Float32(0))
    var b = LTLanes(fill=Float32(0))
    var cnt = 0
    var nb = (n + TEB - 1) // TEB
    for m in range(nb):
        for i in range(m * TEB, min(n, (m + 1) * TEB)):
            if Int(ld(f, FO + i)) == fi:
                continue
            var l = i % TREE_W
            a[l] = add(a[l], ld(f, Y + i * T + tt))
            cnt += 1
        blt_close(a, b, m)
    var s = lt_tree(b)
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        lt_zero(b)
        for m in range(nb):
            for i in range(m * TEB, min(n, (m + 1) * TEB)):
                if Int(ld(f, FO + i)) == fi:
                    continue
                var l = i % TREE_W
                var e = sub(ld(f, Y + i * T + tt), mean)
                a[l] = add(a[l], mul(e, e))
            blt_close(a, b, m)
        ss = div(lt_tree(b), Float32(cnt))
    st(f, p(q, 4) + 2 * t, mean)
    st(f, p(q, 4) + 2 * t + 1, ss)


@always_inline
def _te_pos(f: FP, rp: FP, BF: Int, BY: Int, R: Int, Y: Int, T: Int, tt: Int, FO: Int, k: Int) -> Tuple[Int, Float32]:
    """Bucket position k's (fold, target) as `te_enc_lt_fold` reads them:
    the gathered words (BF >= 0) or row rp[R + k]'s."""
    if BF >= 0:
        return (Int(ld(f, BF + k)), ld(f, BY + k))
    var i = ldi(rp, R + k)
    return (Int(ld(f, FO + i)), ld(f, Y + i * T + tt))


def te_enc_blt_fold(f: FP, rp: FP, BF: Int, BY: Int, R: Int, Y: Int, T: Int, tt: Int, FO: Int, lo: Int, hi: Int,
                    fi: Int, smooth: Float32, mut s: Float32, mut cnt: Int, mut mean: Float32, mut ssd: Float32):
    """`te_enc_lt_fold`'s arguments and outputs in the blocked order (`BLT`):
    the category's bucket positions k in [lo, hi) cut at the TEB-aligned
    block boundaries of the column's bucket array, block m's tree (lane
    k mod TREE_W) closing into lane m mod TREE_W of the outer tree; rows of
    fold fi add nothing. The sum and count; with smooth < 0 and a count,
    the mean and the squared deviations the same way (else they stay 0)."""
    var a = LTLanes(fill=Float32(0))
    var b = LTLanes(fill=Float32(0))
    var c = 0
    if hi > lo:
        for m in range(lo // TEB, (hi - 1) // TEB + 1):
            for k in range(max(lo, m * TEB), min(hi, (m + 1) * TEB)):
                var v = _te_pos(f, rp, BF, BY, R, Y, T, tt, FO, k)
                if v[0] == fi:
                    continue
                var l = k % TREE_W
                a[l] = add(a[l], v[1])
                c += 1
            blt_close(a, b, m)
    s = lt_tree(b)
    cnt = c
    mean = Float32(0)
    ssd = Float32(0)
    if smooth < Float32(0) and c > 0:
        mean = div(s, Float32(c))
        lt_zero(b)
        for m in range(lo // TEB, (hi - 1) // TEB + 1):
            for k in range(max(lo, m * TEB), min(hi, (m + 1) * TEB)):
                var v = _te_pos(f, rp, BF, BY, R, Y, T, tt, FO, k)
                if v[0] == fi:
                    continue
                var l = k % TREE_W
                var e = sub(v[1], mean)
                a[l] = add(a[l], mul(e, e))
            blt_close(a, b, m)
        ssd = lt_tree(b)


def te_enc_blt(t: Int, f: FP, q: IP):
    """`te_enc_unit`'s q and t in the blocked order (`BLT`). With BK > 0 the
    bucket gives the positions; else every row is scanned: the category's
    first position is the number of rows of lower categories, and its rows'
    positions follow in row order, so the blocks are the bucket's. The
    device spelling is x_prep/idn_blocked.mojo, the host column's
    x_prep/host/target.mojo (its own bucket, the same positions)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var smooth = ld(f, p(q, 9))
    var s = Float32(0)
    var cnt = 0
    var mean = Float32(0)
    var ssd = Float32(0)
    var bk = p(q, 11)
    if bk > 0:
        var S = bk - 1 + j * (cmax + 1)
        var lo = ldi(f, S + cat)
        var hi = ldi(f, S + cat + 1)
        var gb = p(q, 13)
        var BF = -1
        var BY = 0
        if gb > 0:
            BF = gb - 1 + j * n
            BY = gb - 1 + d * n + (j * T + tt) * n
        te_enc_blt_fold(f, f, BF, BY, p(q, 12) + j * n, p(q, 3), T, tt, p(q, 5), lo, hi, fi, smooth, s, cnt,
                        mean, ssd)
        st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))
        return
    # no buckets: position lo + rank, lo = the rows of lower categories
    var lo = 0
    for i in range(n):
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code >= 0 and code < cat:
            lo += 1
    var a = LTLanes(fill=Float32(0))
    var b = LTLanes(fill=Float32(0))
    var k = lo
    var m = lo // TEB
    for i in range(n):
        if Int(ld(f, p(q, 0) + i * d + j)) != cat:
            continue
        if k // TEB != m:
            blt_close(a, b, m)
            m = k // TEB
        if Int(ld(f, p(q, 5) + i)) != fi:
            var l = k % TREE_W
            a[l] = add(a[l], ld(f, p(q, 3) + i * T + tt))
            cnt += 1
        k += 1
    if k > lo:
        blt_close(a, b, m)
    s = lt_tree(b)
    if smooth < Float32(0) and cnt > 0:
        mean = div(s, Float32(cnt))
        lt_zero(b)
        k = lo
        m = lo // TEB
        for i in range(n):
            if Int(ld(f, p(q, 0) + i * d + j)) != cat:
                continue
            if k // TEB != m:
                blt_close(a, b, m)
                m = k // TEB
            if Int(ld(f, p(q, 5) + i)) != fi:
                var l = k % TREE_W
                var e = sub(ld(f, p(q, 3) + i * T + tt), mean)
                a[l] = add(a[l], mul(e, e))
            k += 1
        if k > lo:
            blt_close(a, b, m)
        ssd = lt_tree(b)
    st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))


def te_global_unit(t: Int, f: FP, q: IP):
    """q = [Y, n, T, FOLD, META]; t = fi*T + tt. META[2t] = mean,
    META[2t+1] = population variance of target column tt over fold fi's rows
    (rows loaded RUN at a time, folded ascending). IDN_TE_GLOBAL_TREE: both
    folds in the lane-tree order (x_prep/idn_fold.mojo, `te_global_lt`);
    IDN_TE_BLOCKED: the blocked order (`te_global_blt`)."""
    comptime if IDN_TE_BLOCKED:
        te_global_blt(t, f, q)
        return
    comptime if IDN_TE_GLOBAL_TREE:
        te_global_lt(t, f, q)
        return
    var n = p(q, 1)
    var T = p(q, 2)
    var Y = p(q, 0)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var s = Float32(0)
    var cnt = 0
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var bf = run_block[RUN](f, FO + i0, 1)
        var by = run_block[RUN](f, Y + i0 * T + tt, T)
        comptime for u in range(RUN):
            if Int(ftz(bf[u])) != fi:
                s = acc_add(s, ftz(by[u]))
                cnt += 1
    for i in range(full, n):
        if Int(ld(f, FO + i)) == fi:
            continue
        s = acc_add(s, ld(f, Y + i * T + tt))
        cnt += 1
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
        for i0 in range(0, full, RUN):
            var bf = run_block[RUN](f, FO + i0, 1)
            var by = run_block[RUN](f, Y + i0 * T + tt, T)
            comptime for u in range(RUN):
                if Int(ftz(bf[u])) != fi:
                    var e = sub(ftz(by[u]), mean)
                    ss = acc_add(ss, mul(e, e))
        for i in range(full, n):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            ss = acc_add(ss, mul(e, e))
        ss = div(ss, Float32(cnt))
    st(f, p(q, 4) + 2 * t, mean)
    st(f, p(q, 4) + 2 * t + 1, ss)


#: categories whose bucket counters `te_bucket` keeps in a thread-private array
comptime TE_REG = 256


def _te_bucket_reg(t: Int, f: FP, n: Int, d: Int, cmax: Int, C: Int, S: Int, R: Int):
    """`te_bucket_unit` with its counters in a thread-private array instead of
    arena words (lane prep-apple2: consecutive rows of one category made each
    count a load after a store to the same arena word). The same START and
    ROWS words."""
    var cnt = InlineArray[Int32, TE_REG + 1](fill=Int32(0))
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                cnt[code + 1] += 1
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            cnt[code + 1] += 1
    for c in range(cmax):
        cnt[c + 1] += cnt[c]
    for c in range(cmax + 1):
        sti(f, S + c, Int(cnt[c]))
    # cnt[c] is now category c's fill position
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                sti(f, R + Int(cnt[code]), i0 + u)
                cnt[code] += 1
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            sti(f, R + Int(cnt[code]), i)
            cnt[code] += 1


def te_bucket_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, CMAX, START, ROWS]; t = column j (lane prep-apple).
    The rows of each category of column j, ascending: START[j*(CMAX+1) + c]
    .. START[j*(CMAX+1) + c + 1] index ROWS[j*n + ...], which holds the row
    numbers whose code is c, in row order (int words). A code outside
    [0, CMAX) is in no bucket. `te_enc` then walks one category's rows
    instead of every row: the same rows in the same order."""
    var n = p(q, 1)
    var d = p(q, 2)
    var cmax = p(q, 3)
    var C = p(q, 0)
    var S = p(q, 4) + t * (cmax + 1)
    var R = p(q, 5) + t * n
    if cmax <= TE_REG:
        _te_bucket_reg(t, f, n, d, cmax, C, S, R)
        return
    for c in range(cmax + 1):
        sti(f, S + c, 0)
    var full = n - n % RUN
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                sti(f, S + code + 1, ldi(f, S + code + 1) + 1)
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            sti(f, S + code + 1, ldi(f, S + code + 1) + 1)
    for c in range(cmax):
        sti(f, S + c + 1, ldi(f, S + c + 1) + ldi(f, S + c))
    # scatter: START[c] is advanced as its bucket fills, then restored
    for i0 in range(0, full, RUN):
        var bc = run_block[RUN](f, C + i0 * d + t, d)
        comptime for u in range(RUN):
            var code = Int(ftz(bc[u]))
            if code >= 0 and code < cmax:
                var at = ldi(f, S + code)
                sti(f, R + at, i0 + u)
                sti(f, S + code, at + 1)
    for i in range(full, n):
        var code = Int(ld(f, C + i * d + t))
        if code >= 0 and code < cmax:
            var at = ldi(f, S + code)
            sti(f, R + at, i)
            sti(f, S + code, at + 1)
    for c in range(cmax, 0, -1):
        sti(f, S + c, ldi(f, S + c - 1))
    sti(f, S, 0)


# ------------------------------------------------ te_bucket in parallel (lane prep-apple2)
# te_bucket is one thread per column walking every row twice (0.26 s of a
# 1M row TargetEncoder on the M4 Pro). The same START and ROWS as a stable
# counting sort by chunks: each (column, chunk) counts its categories
# (te_hist), each (column, category) turns its chunk counts into offsets
# (te_hsum), each column lays out the category starts (te_hstart), and each
# (column, chunk) places its rows at their offsets in row order
# (te_hscatter). Chunks are consecutive row ranges, so every bucket lists its
# rows ascending, as te_bucket's single walk does.


@always_inline
def _te_chunk(n: Int, ch_n: Int, ch: Int, mut lo: Int, mut hi: Int):
    var cs = (n + ch_n - 1) // ch_n
    lo = min(ch * cs, n)
    hi = min(lo + cs, n)


def te_hist_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, CMAX, CH, H]; t = j*CH + ch: H[t*CMAX + c] = how many
    rows of chunk ch have code c in column j (int words)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var cmax = p(q, 3)
    var ch_n = p(q, 4)
    var j = t // ch_n
    var ch = t % ch_n
    var Hb = p(q, 5) + t * cmax
    for c in range(cmax):
        sti(f, Hb + c, 0)
    var lo = 0
    var hi = 0
    _te_chunk(n, ch_n, ch, lo, hi)
    for i in range(lo, hi):
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code >= 0 and code < cmax:
            sti(f, Hb + code, ldi(f, Hb + code) + 1)


def te_hsum_unit(t: Int, f: FP, q: IP):
    """q = [CMAX, CH, H, TOT]; t = j*CMAX + c: category c's chunk counts
    become the rows of c before each chunk; TOT[t] = c's total."""
    var cmax = p(q, 0)
    var ch_n = p(q, 1)
    var j = t // cmax
    var c = t % cmax
    var run = 0
    for ch in range(ch_n):
        var at = p(q, 2) + (j * ch_n + ch) * cmax + c
        var v = ldi(f, at)
        sti(f, at, run)
        run += v
    sti(f, p(q, 3) + t, run)


def te_hstart_unit(t: Int, f: FP, q: IP):
    """q = [CMAX, START, TOT]; t = column j: START[j*(CMAX+1) + c] = the rows
    of categories below c (te_bucket's START)."""
    var cmax = p(q, 0)
    var S = p(q, 1) + t * (cmax + 1)
    var acc = 0
    for c in range(cmax):
        sti(f, S + c, acc)
        acc += ldi(f, p(q, 2) + t * cmax + c)
    sti(f, S + cmax, acc)


def te_hscatter_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, CMAX, CH, H, START, ROWS]; t = j*CH + ch: the rows of
    chunk ch in row order, each at its category's start plus the rows of that
    category before it (te_bucket's ROWS)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var cmax = p(q, 3)
    var ch_n = p(q, 4)
    var j = t // ch_n
    var ch = t % ch_n
    var Hb = p(q, 5) + t * cmax
    var S = p(q, 6) + j * (cmax + 1)
    var R = p(q, 7) + j * n
    var lo = 0
    var hi = 0
    _te_chunk(n, ch_n, ch, lo, hi)
    for i in range(lo, hi):
        var code = Int(ld(f, p(q, 0) + i * d + j))
        if code >= 0 and code < cmax:
            var k = ldi(f, Hb + code)
            sti(f, R + ldi(f, S + code) + k, i)
            sti(f, Hb + code, k + 1)


def te_enc_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, Y, T, FOLD, CMAX, NCAT, META, SMOOTH, ENC, BK, ROWS, GB];
    t = ((fi*d + j)*CMAX + cat)*T + tt. SMOOTH < 0: the empirical Bayes
    ("auto") encoding; else (sum + s*mean) / (count + s). With BK > 0 the
    rows walked are category cat's bucket (`te_bucket`, ascending), else
    every row; either way the rows of cat outside fold fi, in row order.
    IDN_TE_ENC_TREE: the lane-tree order (`te_enc_lt`); IDN_TE_BLOCKED: the
    blocked order (`te_enc_blt`)."""
    comptime if IDN_TE_BLOCKED:
        te_enc_blt(t, f, q)
        return
    comptime if IDN_TE_ENC_TREE:
        te_enc_lt(t, f, q)
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var s = Float32(0)
    var cnt = 0
    # BK = START + 1 of `te_bucket` (0: no buckets, every row is scanned)
    var bk = p(q, 11)
    var lo = 0
    var hi = n
    var R = 0
    if bk > 0:
        var S = bk - 1 + j * (cmax + 1)
        lo = ldi(f, S + cat)
        hi = ldi(f, S + cat + 1)
        R = p(q, 12) + j * n
    var smooth = ld(f, p(q, 9))
    var mean = Float32(0)
    var ssd = Float32(0)
    # GB = `te_gather`'s offset + 1 (lane prep-apple2): the bucket's folds and
    # targets in bucket order, streamed RUN at a time; the same rows in the
    # same order, so the same words
    var gb = p(q, 13) if bk > 0 else 0
    if gb > 0:
        var BF = gb - 1 + j * n
        var BY = gb - 1 + d * n + (j * T + tt) * n
        var cut = lo + (hi - lo) - (hi - lo) % RUN
        for k0 in range(lo, cut, RUN):
            var bf = run_block[RUN](f, BF + k0, 1)
            var by = run_block[RUN](f, BY + k0, 1)
            comptime for u in range(RUN):
                if Int(ftz(bf[u])) != fi:
                    s = acc_add(s, ftz(by[u]))
                    cnt += 1
        for k in range(cut, hi):
            if Int(ld(f, BF + k)) == fi:
                continue
            s = acc_add(s, ld(f, BY + k))
            cnt += 1
        if smooth < Float32(0) and cnt > 0:
            mean = div(s, Float32(cnt))
            for k0 in range(lo, cut, RUN):
                var bf = run_block[RUN](f, BF + k0, 1)
                var by = run_block[RUN](f, BY + k0, 1)
                comptime for u in range(RUN):
                    if Int(ftz(bf[u])) != fi:
                        var e = sub(ftz(by[u]), mean)
                        ssd = acc_add(ssd, mul(e, e))
            for k in range(cut, hi):
                if Int(ld(f, BF + k)) == fi:
                    continue
                var e = sub(ld(f, BY + k), mean)
                ssd = acc_add(ssd, mul(e, e))
        st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))
        return
    for k in range(lo, hi):
        var i = k
        if bk > 0:
            i = ldi(f, R + k)
        if Int(ld(f, p(q, 5) + i)) == fi:
            continue
        if bk == 0 and Int(ld(f, p(q, 0) + i * d + j)) != cat:
            continue
        s = add(s, ld(f, p(q, 3) + i * T + tt))
        cnt += 1
    if smooth < Float32(0) and cnt > 0:
        mean = div(s, Float32(cnt))
        for k in range(lo, hi):
            var i = k
            if bk > 0:
                i = ldi(f, R + k)
            if Int(ld(f, p(q, 5) + i)) == fi:
                continue
            if bk == 0 and Int(ld(f, p(q, 0) + i * d + j)) != cat:
                continue
            var e = sub(ld(f, p(q, 3) + i * T + tt), mean)
            ssd = add(ssd, mul(e, e))
    st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, s, cnt, mean, ssd))


def te_gather_unit(t: Int, f: FP, q: IP):
    """q = [ROWS, n, d, Y, T, FOLD, G]; t = j*n + k (lane prep-apple2). The
    k-th row of column j's buckets (`te_bucket`'s ROWS), i: G[j*n + k] =
    FOLD[i], G[d*n + (j*T + tt)*n + k] = Y[i*T + tt], raw words (`te_enc`
    flushes them as it reads them). A slot past the bucketed rows (codes
    outside [0, CMAX)) is never read."""
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var j = t // n
    var k = t % n
    var G = p(q, 6)
    var i = ldi(f, p(q, 0) + t)
    if i < 0 or i >= n:
        return
    f.unsafe_store(G + t, raw(f, p(q, 5) + i))
    for tt in range(T):
        f.unsafe_store(G + d * n + (j * T + tt) * n + k, raw(f, p(q, 3) + i * T + tt))


@always_inline
def te_value(ymean: Float32, yvar: Float32, smooth: Float32, s: Float32, cnt: Int, mean: Float32,
             ssd: Float32) -> Float32:
    """One category's encoding from its target sum s over cnt rows and, for
    the "auto" encoding (smooth < 0) with cnt > 0, its mean s / cnt and
    squared deviations ssd (both folded in ascending row order). The host's
    one-pass-per-feature spelling (x_prep/host/target.mojo) calls this too."""
    var enc = ymean
    if smooth >= Float32(0):
        var den = add(Float32(cnt), smooth)
        if den > Float32(0):
            enc = div(add(s, mul(smooth, ymean)), den)
    elif cnt > 0:
        var vc = mul(yvar, Float32(cnt))
        var den = add(vc, div(ssd, Float32(cnt)))
        if den > Float32(0):
            var lam = div(vc, den)
            enc = add(mul(lam, mean), mul(sub(Float32(1), lam), ymean))
    return enc


def te_apply_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, d, T, FOLD, ENC, CMAX, META, F, OUT]; t = (i*d + j)*T + tt.
    Row i reads the encodings of its own fold (FOLD < 0: fi = F, the full
    fit); an unknown code (-1) reads that fold's target mean."""
    var d = p(q, 2)
    var T = p(q, 3)
    var tt = t % T
    var ij = t // T
    var j = ij % d
    var i = ij // d
    var fi = p(q, 8)
    if p(q, 4) >= 0:
        fi = Int(ld(f, p(q, 4) + i))
    var code = Int(ld(f, p(q, 0) + i * d + j))
    var v: Float32
    if code < 0:
        v = ld(f, p(q, 7) + 2 * (fi * T + tt))
    else:
        v = ld(f, p(q, 5) + ((fi * d + j) * p(q, 6) + code) * T + tt)
    st(f, p(q, 9) + t, v)
