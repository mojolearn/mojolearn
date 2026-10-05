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
from std.sys.compile import is_defined
from checks.numerics import ftz, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, st, raw, RUN, run_block, is_nan, ldi
from x_prep.prims import add, sub, mul, div, zero_to_one, acc_add, _cs_take, _ss_take

#: rows a block unit folds (python/mojolearn/_expansion_prep.py `_XB`)
comptime XB = 2048

#: lane idn-int-prep (2026-10-04), IDENTICAL on every vendor and the host
#: column, ON by default: the discrete naive Bayes count pass as ONE thread
#: per (block, column) (`csb1_part`, op 162) instead of one per (block, class,
#: column), so X is read once, not once per class; BernoulliNB's binarize is
#: folded into it (no n x d binarized copy) and its unused column-stats pass
#: is not staged; Multinomial / Complement take their negative-input check
#: from the same pass (`csb1_neg`, op 163) instead of a column-stats pass.
#: Every class chain is csb_part's chain (the class's rows of the block,
#: ascending from zero), so the words are csb_part's.
#: -D MOJOLEARN_IDN_NB_ONEPASS_OFF restores csb_part + colb_part + binarize.
comptime IDN_NB_ONEPASS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_NB_ONEPASS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
#: classes csb1_part keeps in registers (more: the partial table itself)
comptime CSB1_REG = 8

#: lane fam-prep-metrics (2026-10-04), IDENTICAL on every vendor and the host
#: column, ON by default: every `col_stats` / `class_stats` stage the Python
#: layer still staged as one thread per (class, column) over all n rows
#: (SimpleImputer, KBinsDiscretizer, QuantileTransformer, PowerTransformer,
#: SplineTransformer, IterativeImputer, VarianceThreshold, LDA, QDA, chi2,
#: f_classif, mutual information, the weight check) is staged as this file's
#: blocked folds (colb_* / csb_*). No kernel changes: the switch is a bit of
#: `x_prep_idn_fam` (bindings/_mojolearn_x_prep*.mojo) that
#: python/mojolearn/_expansion_prep.py `_idn_fam` reads, so the device and the
#: host column stage the same program. BITS: sums, means and variances take
#: the blocked order for n > XB (counts, minima, maxima unchanged).
#: -D MOJOLEARN_IDN_STATS_BLOCKED_OFF restores the serial stages.
comptime IDN_STATS_BLOCKED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_STATS_BLOCKED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)

#: lane fam-prep-metrics (2026-10-04), IDENTICAL on every vendor and the host
#: column, ON by default: the blocked class statistics walk X ONCE per (block,
#: column) instead of once per (block, class, column): the sums by
#: `csb1_part` (op 162, so it needs IDN_NB_ONEPASS) and the squared deviations
#: by `csb1_ss` (op 165). Each class chain is csb_part's / csb_ss's (the
#: class's rows of the block, ascending from zero), so the words are theirs:
#: no bit moves against the blocked order.
#: -D MOJOLEARN_IDN_CLASS_ONEPASS_OFF restores csb_part / csb_ss.
comptime IDN_CLASS_ONEPASS = (
    IDN_NB_ONEPASS
    and not (is_defined["MOJOLEARN_IDN_CLASS_ONEPASS_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


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


@always_inline
def _csb1_x(f: FP, at: Int, THR: Int, thr: Float32) -> Float32:
    """The flushed word csb_part would load: X's, or `binarize_unit`'s of it
    when THR >= 0 (1 when X > THR, else 0; a NaN is kept)."""
    return _csb1_v(raw(f, at), THR, thr)


@always_inline
def _csb1_v(x: Float32, THR: Int, thr: Float32) -> Float32:
    """`_csb1_x` of an already loaded raw word."""
    if THR >= 0 and not is_nan(x):
        return Float32(1) if ftz(x) > thr else Float32(0)
    return ftz(x)


@always_inline
def _csb1_reg_take(
    x: Float32, k: Int, w: Float32, W: Int,
    mut s: SIMD[DType.float32, CSB1_REG], mut cw: SIMD[DType.float32, CSB1_REG],
):
    """One row of csb1_part's register path (K <= CSB1_REG): the row's word
    onto its class's chain (csb_part's adds: acc_add on the unweighted sum,
    whose accumulator is already flushed, as csb_part)."""
    if W < 0:
        comptime for u in range(CSB1_REG):
            if k == u:
                s[u] = acc_add(s[u], x)
                cw[u] = cw[u] + Float32(1)
    else:
        comptime for u in range(CSB1_REG):
            if k == u:
                s[u] = add(s[u], mul(w, x))
                cw[u] = add(cw[u], w)


@always_inline
def _csb1_mem_take(
    f: FP, x: Float32, k: Int, w: Float32, W: Int, PS: Int, PC: Int, base: Int, d: Int, K: Int
):
    """One row of csb1_part's table path (K > CSB1_REG): the same adds on
    the partial table's words."""
    if k < 0 or k >= K:
        return
    var at = base + k * d
    if W < 0:
        st(f, PS + at, add(ld(f, PS + at), x))
        st(f, PC + at, ld(f, PC + at) + Float32(1))
    else:
        st(f, PS + at, add(ld(f, PS + at), mul(w, x)))
        st(f, PC + at, add(ld(f, PC + at), w))


def csb1_part_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, PS, PC, nb, W, THR, NG]; t = b*d + c (lane
    idn-int-prep). csb_part's partials of EVERY class of (block b, column c)
    from one walk of the block's rows, ascending: PS[(b*K + k)*d + c] = the
    sum of column c over the block's rows of class k from zero (W >= 0: of
    w x) and PC[...] = their count (the weight sum). THR >= 0: X is binarized
    at the scalar THR first (`binarize_unit`'s word). NG >= 0: NG[t] = 1 when
    a word of the block's column is below zero, else 0. Each class's chain is
    csb_part's (same rows, same order, same adds), so the words are its."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var PS = p(q, 5)
    var PC = p(q, 6)
    var W = p(q, 8)
    var THR = p(q, 9)
    var NG = p(q, 10)
    var c = t % d
    var b = t // d
    var r = _span(b, n)
    var base = b * K * d + c
    var thr = Float32(0)
    if THR >= 0:
        thr = ld(f, THR)
    var neg = False
    # lane idn-regress (2026-10-05): RUN rows' X, label and weight words are
    # loaded together before any is used (`run_block`, csb_part's loads):
    # one GPU thread per (block, column) otherwise waits one memory latency
    # per row (M3 IDENTICAL MultinomialNB / ComplementNB fit 2-3x slower
    # than csb_part's RUN loads). Each word still goes onto its class's chain
    # one row at a time, ascending: the same adds, the same bits.
    var full = r[0] + (r[1] - r[0]) - (r[1] - r[0]) % RUN
    if K <= CSB1_REG:
        var s = SIMD[DType.float32, CSB1_REG](0)
        var cw = SIMD[DType.float32, CSB1_REG](0)
        for i0 in range(r[0], full, RUN):
            var bx = run_block[RUN](f, X + i0 * d + c, d)
            var by = run_block[RUN](f, Y + i0, 1)
            var bw = SIMD[DType.float32, RUN](0)
            if W >= 0:
                bw = run_block[RUN](f, W + i0, 1)
            comptime for u in range(RUN):
                var x = _csb1_v(bx[u], THR, thr)
                if x < Float32(0):
                    neg = True
                _csb1_reg_take(x, Int(ftz(by[u])), ftz(bw[u]), W, s, cw)
        for i in range(full, r[1]):
            var x = _csb1_x(f, X + i * d + c, THR, thr)
            if x < Float32(0):
                neg = True
            var w = Float32(0)
            if W >= 0:
                w = ld(f, W + i)
            _csb1_reg_take(x, Int(ld(f, Y + i)), w, W, s, cw)
        comptime for u in range(CSB1_REG):
            if u < K:
                st(f, PS + base + u * d, s[u])
                st(f, PC + base + u * d, cw[u])
    else:
        for k in range(K):
            st(f, PS + base + k * d, Float32(0))
            st(f, PC + base + k * d, Float32(0))
        for i0 in range(r[0], full, RUN):
            var bx = run_block[RUN](f, X + i0 * d + c, d)
            var by = run_block[RUN](f, Y + i0, 1)
            var bw = SIMD[DType.float32, RUN](0)
            if W >= 0:
                bw = run_block[RUN](f, W + i0, 1)
            comptime for u in range(RUN):
                var x = _csb1_v(bx[u], THR, thr)
                if x < Float32(0):
                    neg = True
                _csb1_mem_take(f, x, Int(ftz(by[u])), ftz(bw[u]), W, PS, PC, base, d, K)
        for i in range(full, r[1]):
            var x = _csb1_x(f, X + i * d + c, THR, thr)
            if x < Float32(0):
                neg = True
            var w = Float32(0)
            if W >= 0:
                w = ld(f, W + i)
            _csb1_mem_take(f, x, Int(ld(f, Y + i)), w, W, PS, PC, base, d, K)
    if NG >= 0:
        st(f, NG + t, Float32(1) if neg else Float32(0))


def csb1_neg_unit(t: Int, f: FP, q: IP):
    """q = [NG, nb, d, OUT]; t = column: OUT[t] = -1 when a block of
    csb1_part raised the column's negative flag, else 0 (the word the
    naive Bayes fit reads where it read the column minimum: only its sign
    is used)."""
    var NG = p(q, 0)
    var d = p(q, 2)
    var neg = False
    for b in range(p(q, 1)):
        if ld(f, NG + b * d + t) != Float32(0):
            neg = True
    st(f, p(q, 3) + t, Float32(-1) if neg else Float32(0))


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


def csb1_ss_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, MEAN, CN, PS, nb, W]; t = b*d + c (lane
    fam-prep-metrics, IDN_CLASS_ONEPASS). csb_ss's partials of EVERY class of
    (block b, column c) from one walk of the block's rows, ascending:
    PS[(b*K + k)*d + c] = the block's sum of squared deviations of class k's
    rows from MEAN[k*d + c] (W >= 0: w (x - mean)^2) from zero; zero when
    CN[k*d + c] is zero. Each class's chain is csb_ss's (same rows, same
    order, same operations), so the words are its."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var Y = p(q, 3)
    var K = p(q, 4)
    var MEAN = p(q, 5)
    var CN = p(q, 6)
    var PS = p(q, 7)
    var W = p(q, 9)
    var c = t % d
    var b = t // d
    var r = _span(b, n)
    var base = b * K * d + c
    if K <= CSB1_REG:
        var mean = SIMD[DType.float32, CSB1_REG](0)
        var live = SIMD[DType.float32, CSB1_REG](0)
        var ss = SIMD[DType.float32, CSB1_REG](0)
        comptime for u in range(CSB1_REG):
            if u < K:
                mean[u] = ld(f, MEAN + u * d + c)
                live[u] = ld(f, CN + u * d + c)
        for i in range(r[0], r[1]):
            var x = ld(f, X + i * d + c)
            var k = Int(ld(f, Y + i))
            if W < 0:
                comptime for u in range(CSB1_REG):
                    if k == u:
                        var e = sub(x, mean[u])
                        ss[u] = add(ss[u], mul(e, e))
            else:
                var w = ld(f, W + i)
                comptime for u in range(CSB1_REG):
                    if k == u:
                        var e = sub(x, mean[u])
                        ss[u] = add(ss[u], mul(w, mul(e, e)))
        comptime for u in range(CSB1_REG):
            if u < K:
                if live[u] != Float32(0):
                    st(f, PS + base + u * d, ss[u])
                else:
                    st(f, PS + base + u * d, Float32(0))
    else:
        for k in range(K):
            st(f, PS + base + k * d, Float32(0))
        for i in range(r[0], r[1]):
            var k = Int(ld(f, Y + i))
            if k < 0 or k >= K:
                continue
            if ld(f, CN + k * d + c) == Float32(0):
                continue
            var e = sub(ld(f, X + i * d + c), ld(f, MEAN + k * d + c))
            var at = base + k * d
            if W < 0:
                st(f, PS + at, add(ld(f, PS + at), mul(e, e)))
            else:
                st(f, PS + at, add(ld(f, PS + at), mul(ld(f, W + i), mul(e, e))))


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


#: lane idn-all (2026-10-04, the review of idn-int-prep): the IDENTICAL CSR
#: naive Bayes route's fallback (non-integer values, counts >= 2^24, rows not
#: canonical) densified its input on the host (`X.toarray()`) in the middle
#: of a fit. Here the dense block is built ON THE DEVICE from the CSR arrays
#: (op 164, one thread per row) and the dense program runs on it: nothing
#: comes back to the host between the flag and the dense stages. The words
#: are `toarray`'s for float32 data: zeros, each entry added in its row's
#: stored order (a duplicate column sums, -0.0 becomes +0.0, NaN stays).
#: -D MOJOLEARN_IDN_NB_CSR_DENSE_OFF restores the host densify.
comptime IDN_NB_CSR_DENSE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_NB_CSR_DENSE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


def csr_dense_unit(t: Int, f: FP, q: IP):
    """q = [INDPTR, INDICES, DATA, d, OUT]; t = row. INDPTR (n + 1) and
    INDICES (nnz) are int32 words, DATA (nnz) float32. OUT's row t (d words,
    written whole by this thread alone): zeros, then OUT[t, INDICES[e]] +=
    DATA[e] for the row's entries e ascending. An index outside [0, d) is
    skipped (the CSR entry checks the indices before this stage runs)."""
    var d = p(q, 3)
    var o = p(q, 4) + t * d
    for j in range(d):
        st(f, o + j, Float32(0))
    var lo = ldi(f, p(q, 0) + t)
    var hi = ldi(f, p(q, 0) + t + 1)
    for e in range(lo, hi):
        var c = ldi(f, p(q, 1) + e)
        if c >= 0 and c < d:
            st(f, o + c, ftz(ld(f, o + c) + ld(f, p(q, 2) + e)))
