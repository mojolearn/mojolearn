# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BLOCKED ROW FOLDS of the prep lane (lane gap-nb-maxabs-grp, 2026-10-02).

`col_stats` and `class_stats` (x_prep/prims.mojo) fold a whole column on ONE
thread per (class, column): at the board's 1M rows that is a million
dependent adds on a few hundred threads, a handful of SMs busy and the rest
idle. Here the rows are cut into XB-row blocks: one unit per (block, class,
column) folds its block ascending from zero (the serial unit's chain over
the block's rows), and one unit per (class, column) folds the block partials
ascending from zero (`add`). The same units run on the host (the program
model, x_prep/common.mojo), so the host gets the same order.

BITS. Counts are integers (any order is exact); minimum, maximum and max |x|
are order-free and exact (the first-seen value of an equal pair is kept, as
in the serial fold, because blocks are visited ascending and every compare
is strict). Sums, means and variances take the blocked order: a NEW order
for n > XB (at n <= XB there is one block and the words are the serial
unit's). The blocked count of a block is exact in float32 (XB < 2^24).

A partial table lives in device scratch (`_Prog.work`): a stage writes it
whole before the next stage reads it.
"""
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, st, RUN, run_block, is_nan
from x_prep.prims import add, sub, mul, div, zero_to_one, acc_add, _cs_take, _ss_take

#: rows a block unit folds (python/mojolearn/_expansion_prep.py `_XB`)
comptime XB = 2048


@always_inline
def _span(b: Int, n: Int) -> Tuple[Int, Int]:
    var lo = b * XB
    return (lo, min(n, lo + XB))


# ---------------------------------------------------------------- columns
def colb_part_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, P, nb]; t = b*d + c. Over rows [b*XB, min(n, (b+1)*XB))
    ascending, col_stats' first pass from zero: P rows of nb*d are the
    count, sum, min, max and max |x| of the non-NaN entries."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var P = p(q, 3)
    var nbd = p(q, 4) * d
    var b = t // d
    var c = t % d
    var r = _span(b, n)
    var lo_i = r[0]
    var hi_i = r[1]
    var cnt = 0
    var s = Float32(0)
    var lo = Float32(0)
    var hi = Float32(0)
    var ma = Float32(0)
    var full = lo_i + (hi_i - lo_i) - (hi_i - lo_i) % RUN
    for i0 in range(lo_i, full, RUN):
        var blk = run_block[RUN](f, X + i0 * d + c, d)
        comptime for u in range(RUN):
            _cs_take(ftz(blk[u]), cnt, s, lo, hi, ma)
    for i in range(full, hi_i):
        _cs_take(ld(f, X + i * d + c), cnt, s, lo, hi, ma)
    st(f, P + t, Float32(cnt))
    st(f, P + nbd + t, s)
    st(f, P + 2 * nbd + t, lo)
    st(f, P + 3 * nbd + t, hi)
    st(f, P + 4 * nbd + t, ma)


def colb_fold_unit(t: Int, f: FP, q: IP):
    """q = [P, nb, d, OUT]; t = column. colb_part's partials, blocks
    ascending: OUT's count, mean (the partial sums folded from zero, over
    the count), min, max and max |x| rows (col_stats' layout; the variance
    row is colb_var's). An empty column writes zeros."""
    var P = p(q, 0)
    var nb = p(q, 1)
    var d = p(q, 2)
    var O = p(q, 3)
    var nbd = nb * d
    var c = t
    var cnt = 0
    var s = Float32(0)
    var lo = Float32(0)
    var hi = Float32(0)
    var ma = Float32(0)
    var seen = False
    for b in range(nb):
        var at = b * d + c
        var pc = Int(ld(f, P + at))
        s = add(s, ld(f, P + nbd + at))
        if pc == 0:
            continue
        cnt += pc
        var bl = ld(f, P + 2 * nbd + at)
        var bh = ld(f, P + 3 * nbd + at)
        if not seen:
            lo = bl
            hi = bh
            seen = True
        else:
            if bl < lo:
                lo = bl
            if bh > hi:
                hi = bh
        var bm = ld(f, P + 4 * nbd + at)
        if bm > ma:
            ma = bm
    var mean = Float32(0)
    if cnt > 0:
        mean = div(s, Float32(cnt))
    st(f, O + c, Float32(cnt))
    st(f, O + d + c, mean)
    st(f, O + 3 * d + c, lo)
    st(f, O + 4 * d + c, hi)
    st(f, O + 5 * d + c, ma)


def colb_ss_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, OUT, P, nb]; t = b*d + c: the block's sum of squared
    deviations from OUT's mean row (col_stats' second pass from zero) into
    P[t]; an empty column writes zero."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var O = p(q, 3)
    var c = t % d
    var ss = Float32(0)
    if ld(f, O + c) > Float32(0):
        var mean = ld(f, O + d + c)
        var r = _span(t // d, n)
        var full = r[0] + (r[1] - r[0]) - (r[1] - r[0]) % RUN
        for i0 in range(r[0], full, RUN):
            var blk = run_block[RUN](f, X + i0 * d + c, d)
            comptime for u in range(RUN):
                _ss_take(ftz(blk[u]), mean, ss)
        for i in range(full, r[1]):
            _ss_take(ld(f, X + i * d + c), mean, ss)
    st(f, p(q, 4) + t, ss)


def colb_var_unit(t: Int, f: FP, q: IP):
    """q = [P, nb, d, OUT]; t = column: OUT's variance row, colb_ss's
    partials folded ascending from zero over the count (zero when empty)."""
    var P = p(q, 0)
    var nb = p(q, 1)
    var d = p(q, 2)
    var O = p(q, 3)
    var cnt = ld(f, O + t)
    var v = Float32(0)
    if cnt > Float32(0):
        var ss = Float32(0)
        for b in range(nb):
            ss = add(ss, ld(f, P + b * d + t))
        v = div(ss, cnt)
    st(f, O + 2 * d + t, v)


def maxabs_fold_unit(t: Int, f: FP, q: IP):
    """q = [P, nb, d, MA, SCALE]; t = column (MaxAbsScaler): the largest of
    colb_part's max |x| partials (exact: a maximum has one answer) into MA,
    and `_handle_zeros_in_scale` of it into SCALE, scale_params kind 1's
    words."""
    var P = p(q, 0)
    var nb = p(q, 1)
    var d = p(q, 2)
    var nbd = nb * d
    var ma = Float32(0)
    for b in range(nb):
        var bm = ld(f, P + 4 * nbd + b * d + t)
        if bm > ma:
            ma = bm
    st(f, p(q, 3) + t, ma)
    st(f, p(q, 4) + t, zero_to_one(ld(f, p(q, 3) + t)))


# ---------------------------------------------------------------- classes
def csb_part_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, PS, PC, nb, W]; t = (b*K + k)*d + c. Over the
    block's rows of class k, ascending from zero: PS[t] = the sum of column c
    (W >= 0: of w x) and PC[t] = the row count (the weight sum), class_stats
    and class_stats_w's first-pass chains."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var W = p(q, 8)
    var c = t % d
    var bk = t // d
    var k = bk % K
    var r = _span(bk // K, n)
    var s = Float32(0)
    if W < 0:
        var cnt = 0
        var full = r[0] + (r[1] - r[0]) - (r[1] - r[0]) % RUN
        for i0 in range(r[0], full, RUN):
            var by = run_block[RUN](f, Y + i0, 1)
            var bx = run_block[RUN](f, X + i0 * d + c, d)
            comptime for u in range(RUN):
                if Int(ftz(by[u])) == k:
                    s = acc_add(s, ftz(bx[u]))
                    cnt += 1
        for i in range(full, r[1]):
            if Int(ld(f, Y + i)) == k:
                s = add(s, ld(f, X + i * d + c))
                cnt += 1
        st(f, p(q, 6) + t, Float32(cnt))
    else:
        var sw = Float32(0)
        for i in range(r[0], r[1]):
            if Int(ld(f, Y + i)) != k:
                continue
            var w = ld(f, W + i)
            s = add(s, mul(w, ld(f, X + i * d + c)))
            sw = add(sw, w)
        st(f, p(q, 6) + t, sw)
    st(f, p(q, 5) + t, s)


def csb_fold_unit(t: Int, f: FP, q: IP):
    """q = [PS, PC, nb, K, d, CN, CNT, MEAN, SUM, W]; t = k*d + c: csb_part's
    partials, blocks ascending. The count (an exact integer sum; W >= 0: the
    weights folded from zero) into CN[t] and, for c == 0, CNT[k]; the sum
    folded from zero into SUM and over the count into MEAN (zero for an
    empty class). Offsets < 0 are not written."""
    var PS = p(q, 0)
    var PC = p(q, 1)
    var nb = p(q, 2)
    var K = p(q, 3)
    var d = p(q, 4)
    var W = p(q, 9)
    var kd = K * d
    var s = Float32(0)
    var cw = Float32(0)
    if W < 0:
        var cnt = 0
        for b in range(nb):
            cnt += Int(ld(f, PC + b * kd + t))
            s = add(s, ld(f, PS + b * kd + t))
        cw = Float32(cnt)
    else:
        for b in range(nb):
            cw = add(cw, ld(f, PC + b * kd + t))
            s = add(s, ld(f, PS + b * kd + t))
    var mean = Float32(0)
    if cw != Float32(0):
        mean = div(s, cw)
    st(f, p(q, 5) + t, cw)
    if t % d == 0 and p(q, 6) >= 0:
        st(f, p(q, 6) + t // d, cw)
    if p(q, 7) >= 0:
        st(f, p(q, 7) + t, mean)
    if p(q, 8) >= 0:
        st(f, p(q, 8) + t, s)


def csb_ss_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, MEAN, CN, PS, nb, W]; t = (b*K + k)*d + c: the
    block's sum of squared deviations of class k's rows from MEAN (W >= 0:
    weighted, w (x - mean)^2), ascending from zero, into PS[t]; zero when
    the class is empty."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var W = p(q, 9)
    var c = t % d
    var bk = t // d
    var k = bk % K
    var kc = k * d + c
    var ss = Float32(0)
    if ld(f, p(q, 6) + kc) != Float32(0):
        var mean = ld(f, p(q, 5) + kc)
        var r = _span(bk // K, n)
        if W < 0:
            var full = r[0] + (r[1] - r[0]) - (r[1] - r[0]) % RUN
            for i0 in range(r[0], full, RUN):
                var by = run_block[RUN](f, Y + i0, 1)
                var bx = run_block[RUN](f, X + i0 * d + c, d)
                comptime for u in range(RUN):
                    if Int(ftz(by[u])) == k:
                        var e = sub(ftz(bx[u]), mean)
                        ss = acc_add(ss, mul(e, e))
            for i in range(full, r[1]):
                if Int(ld(f, Y + i)) == k:
                    var e = sub(ld(f, X + i * d + c), mean)
                    ss = add(ss, mul(e, e))
        else:
            for i in range(r[0], r[1]):
                if Int(ld(f, Y + i)) != k:
                    continue
                var e = sub(ld(f, X + i * d + c), mean)
                ss = add(ss, mul(ld(f, W + i), mul(e, e)))
    st(f, p(q, 7) + t, ss)


def csb_var_unit(t: Int, f: FP, q: IP):
    """q = [PS, nb, K, d, CN, VAR]; t = k*d + c: csb_ss's partials folded
    ascending from zero, over the count CN[t] (zero for an empty class)."""
    var PS = p(q, 0)
    var nb = p(q, 1)
    var kd = p(q, 2) * p(q, 3)
    var cw = ld(f, p(q, 4) + t)
    var v = Float32(0)
    if cw != Float32(0):
        var ss = Float32(0)
        for b in range(nb):
            ss = add(ss, ld(f, PS + b * kd + t))
        v = div(ss, cw)
    st(f, p(q, 5) + t, v)


# ---------------------------------------------------------------- categories
def cat_hpart_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, NCAT, CMAX, W, H, R]; t = b*d + j (CategoricalNB
    `_count`): H's K*CMAX words of unit t are zeroed, then over rows
    [b*R, min(n, (b+1)*R)) ascending each row of class k whose feature j
    holds v (0 <= v < NCAT[j]) adds one (W >= 0: its weight) to slot
    k*CMAX + v. R is the caller's block (a function of the shape only)."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var cmax = p(q, 6)
    var W = p(q, 7)
    var R = p(q, 9)
    var j = t % d
    var base = p(q, 8) + t * K * cmax
    for s in range(K * cmax):
        st(f, base + s, Float32(0))
    var ncat = Int(ld(f, p(q, 5) + j))
    var lo = (t // d) * R
    var hi = min(n, lo + R)
    for i in range(lo, hi):
        var v = Int(ld(f, X + i * d + j))
        if v < 0 or v >= ncat:
            continue
        var at = base + Int(ld(f, Y + i)) * cmax + v
        if W >= 0:
            st(f, at, add(ld(f, at), ld(f, W + i)))
        else:
            st(f, at, ld(f, at) + Float32(1))


def cat_hfold_unit(t: Int, f: FP, q: IP):
    """q = [H, nb, d, K, NCAT, CMAX, W, OUT]; t = (j*K + k)*CMAX + v: the
    block histograms of (feature j, class k, category v), blocks ascending:
    an exact integer count (W >= 0: the weights folded from zero) into
    OUT[t], cat_counts' layout. Slots v >= NCAT[j] are left as they are."""
    var H = p(q, 0)
    var nb = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 3)
    var cmax = p(q, 5)
    var v = t % cmax
    var jk = t // cmax
    var k = jk % K
    var j = jk // K
    if v >= Int(ld(f, p(q, 4) + j)):
        return
    var slot = k * cmax + v
    var stride = d * K * cmax
    var at = H + j * K * cmax + slot
    if p(q, 6) >= 0:
        var cw = Float32(0)
        for b in range(nb):
            cw = add(cw, ld(f, at + b * stride))
        st(f, p(q, 7) + t, cw)
    else:
        var m = 0
        for b in range(nb):
            m += Int(ld(f, at + b * stride))
        st(f, p(q, 7) + t, Float32(m))
