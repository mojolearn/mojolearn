# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l7-metrics (2026-10-04): op 59 `cm_epi`, THE CONFUSION-MATRIX
TAILS ON THE DEVICE.

multilabel_confusion_matrix, matthews_corrcoef, confusion_matrix's
normalize, cohen_kappa_score, hamming_loss / zero_one_loss / accuracy,
class_likelihood_ratios and classification_report's support total read the
per-label sums (or the k x k cells) back and finished them over Python lists
(python/mojolearn/_expansion_metrics.py). Here the same binary64 operations
run in the program that made the sums, as integer arithmetic on the binary64
encoding (checks/soft_f64.mojo: correctly rounded add, sub, mul, div, sqrt),
so the Apple GPU (no float64), NVIDIA, AMD and the host column compute the
same word. Inputs are a `group_sort` OFF table (unweighted: exact integer
group sizes, below 2^31) or Float32 PairSums at a group_sum OUT (weighted),
read with `sum_at` (x_metrics/tail.mojo), exactly as Python widened them.

q[0] = KIND. One unit (t = 0) for an O(k) fold; one unit per label / row /
column where the tail is a map (MLCM, NORM, the kappa row passes). Every fold
runs in ascending index order. Unweighted integer outputs stay exact (Int64
words, two Float32 words each, low first, as Python's array('q')).

ON in both numeric modes, the only route (the Python tails left the GPU route).
"""
from checks.soft_f64 import (
    SF64_NAN, SF64_ZERO, SF64_ONE, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_sqrt, sf64_lt, sf64_is_nan,
    sf64_from_int, sf64_from_f32,
)
from x_metrics.common import FP, IP, p, ld, ldi, sti
from x_metrics.tail import st64, ld64, is0, cnt_at, sum_at

comptime OP_CM_EPI = 59

comptime KIND_MLCM = 0
comptime KIND_MCC = 1
comptime KIND_COUNT = 2
comptime KIND_NORM = 3
comptime KIND_KAPPA = 4
comptime KIND_CLR = 5
comptime KIND_TOTAL = 6

#: NORM modes: the 'all' total (one unit), then 'all' / 'true' / 'pred' /
#: none (one unit per row, row, column, row)
comptime NORM_ALL_SUM = 0
comptime NORM_ALL = 1
comptime NORM_TRUE = 2
comptime NORM_PRED = 3
comptime NORM_NONE = 4

#: kappa weights: none, linear, quadratic
comptime KW_NONE = 0
comptime KW_LINEAR = 1
comptime KW_QUADRATIC = 2


@always_inline
def _st_i64(f: FP, at: Int, v: Int):
    """A non-negative integer as an Int64 (two words, low first)."""
    st64(f, at, UInt64(v))


@always_inline
def _total(f: FP, tot: Int, weighted: Bool) -> UInt64:
    """The weight total (the total grouping's one PairSum at `tot`), or the
    row count `tot` itself (unweighted)."""
    if weighted:
        return sf64_from_f32(ld(f, tot))
    return sf64_from_int(tot)


@always_inline
def _mlcm(t: Int, f: FP, q: IP):
    """q = [KIND, TP, TR, PR, k, WEIGHTED, TOT, OUT]; unit t < k writes label
    t's [[tn, fp], [fn, tp]] at OUT + 8t: Int64 counts (unweighted, TOT = n)
    or binary64 (weighted, TOT = the total's PairSum): fp = pred - tp,
    fn = true - tp, tn = total - tp - fp - fn (left to right)."""
    var k = p(q, 4)
    if t >= k:
        return
    var TP = p(q, 1)
    var TR = p(q, 2)
    var PR = p(q, 3)
    var weighted = p(q, 5) != 0
    var tot = p(q, 6)
    var at = p(q, 7) + 8 * t
    if not weighted:
        var tp = cnt_at(f, TP, t)
        var fp = cnt_at(f, PR, t) - tp
        var fn_ = cnt_at(f, TR, t) - tp
        var tn = tot - tp - fp - fn_
        _st_i64(f, at, tn)
        _st_i64(f, at + 2, fp)
        _st_i64(f, at + 4, fn_)
        _st_i64(f, at + 6, tp)
        return
    var tp64 = sum_at(f, TP, t, True)
    var fp64 = sf64_sub(sum_at(f, PR, t, True), tp64)
    var fn64 = sf64_sub(sum_at(f, TR, t, True), tp64)
    var tn64 = sf64_sub(sf64_sub(sf64_sub(sf64_from_f32(ld(f, tot)), tp64), fp64), fn64)
    st64(f, at, tn64)
    st64(f, at + 2, fp64)
    st64(f, at + 4, fn64)
    st64(f, at + 6, tp64)


@always_inline
def _mcc(f: FP, q: IP):
    """q = [KIND, TP, TR, PR, k, WEIGHTED, OUT]; one unit. OUT[0] = 1 when
    the covariance product is negative (Python's sqrt raised); OUT[2..4)
    the coefficient (0.0 when the product is zero)."""
    var TP = p(q, 1)
    var TR = p(q, 2)
    var PR = p(q, 3)
    var k = p(q, 4)
    var weighted = p(q, 5) != 0
    var OUT = p(q, 6)
    var n_correct = SF64_ZERO
    for i in range(k):
        n_correct = sf64_add(n_correct, sum_at(f, TP, i, weighted))
    var n = SF64_ZERO
    for i in range(k):
        n = sf64_add(n, sum_at(f, PR, i, weighted))
    var tp_dot = SF64_ZERO
    var pp_dot = SF64_ZERO
    var tt_dot = SF64_ZERO
    for i in range(k):
        var a = sum_at(f, TR, i, weighted)
        var b = sum_at(f, PR, i, weighted)
        tp_dot = sf64_add(tp_dot, sf64_mul(a, b))
        pp_dot = sf64_add(pp_dot, sf64_mul(b, b))
        tt_dot = sf64_add(tt_dot, sf64_mul(a, a))
    var cov_ytyp = sf64_sub(sf64_mul(n_correct, n), tp_dot)
    var cov_ypyp = sf64_sub(sf64_mul(n, n), pp_dot)
    var cov_ytyt = sf64_sub(sf64_mul(n, n), tt_dot)
    var prod = sf64_mul(cov_ypyp, cov_ytyt)
    sti(f, OUT, 0)
    if is0(prod):
        st64(f, OUT + 2, SF64_ZERO)
        return
    if not sf64_is_nan(prod) and sf64_lt(prod, SF64_ZERO):
        sti(f, OUT, 1)
        st64(f, OUT + 2, SF64_NAN)
        return
    st64(f, OUT + 2, sf64_div(cov_ytyp, sf64_sqrt(prod)))


@always_inline
def _count(f: FP, q: IP):
    """q = [KIND, TP, k, WEIGHTED, TOT, OUT]; one unit. hit = the matches'
    sum over the k labels (ascending, from 0.0). OUT: [0] 1 when the total
    is zero, [1] hit and [10] n - hit (Int32, unweighted), [2..4) hit,
    [4..6) total - hit, [6..8) (total - hit) / total, [8..10) hit / total
    (unweighted: hit / max(n, 1)), binary64."""
    var TP = p(q, 1)
    var k = p(q, 2)
    var weighted = p(q, 3) != 0
    var tot = p(q, 4)
    var OUT = p(q, 5)
    var hit = SF64_ZERO
    var ihit = 0
    for i in range(k):
        hit = sf64_add(hit, sum_at(f, TP, i, weighted))
        if not weighted:
            ihit += cnt_at(f, TP, i)
    var total = _total(f, tot, weighted)
    var miss = sf64_sub(total, hit)
    var zero = is0(total)
    sti(f, OUT, 1 if zero else 0)
    sti(f, OUT + 1, ihit)
    sti(f, OUT + 10, tot - ihit if not weighted else 0)
    st64(f, OUT + 2, hit)
    st64(f, OUT + 4, miss)
    if zero:
        st64(f, OUT + 6, SF64_NAN)
    else:
        st64(f, OUT + 6, sf64_div(miss, total))
    if not weighted and tot == 0:
        st64(f, OUT + 8, sf64_div(hit, SF64_ONE))
    elif zero:
        st64(f, OUT + 8, SF64_NAN)
    else:
        st64(f, OUT + 8, sf64_div(hit, total))


@always_inline
def _norm(t: Int, f: FP, q: IP):
    """q = [KIND, CELLS, k, WEIGHTED, MODE, OUT, TS]; the k x k cells
    (row-major, rows true) normalized into OUT (binary64, row-major). Each
    sum runs from 0.0 in ascending index; a zero sum gives 0.0 cells.
    NORM_ALL_SUM (one unit) writes the total of every cell at TS; NORM_ALL
    then divides row t by it; NORM_TRUE divides row t by its sum, NORM_PRED
    column t by its sum; NORM_NONE widens row t."""
    var k = p(q, 2)
    var C = p(q, 1)
    var weighted = p(q, 3) != 0
    var mode = p(q, 4)
    var OUT = p(q, 5)
    var TS = p(q, 6)
    if mode == NORM_ALL_SUM:
        if t != 0:
            return
        var tsum = SF64_ZERO
        for c in range(k * k):
            tsum = sf64_add(tsum, sum_at(f, C, c, weighted))
        st64(f, TS, tsum)
        return
    if t >= k:
        return
    if mode == NORM_NONE:
        for j in range(k):
            st64(f, OUT + 2 * (t * k + j), sum_at(f, C, t * k + j, weighted))
        return
    var s = SF64_ZERO
    if mode == NORM_ALL:
        s = ld64(f, TS)
    else:
        for j in range(k):
            var idx = t * k + j if mode == NORM_TRUE else j * k + t
            s = sf64_add(s, sum_at(f, C, idx, weighted))
    var z = is0(s)
    for j in range(k):
        var idx = t * k + j if mode != NORM_PRED else j * k + t
        var v = SF64_ZERO
        if not z:
            v = sf64_div(sum_at(f, C, idx, weighted), s)
        st64(f, OUT + 2 * idx, v)


@always_inline
def _kw(i: Int, j: Int, wmode: Int) -> UInt64:
    if wmode == KW_NONE:
        return SF64_ZERO if i == j else SF64_ONE
    var d = i - j
    if wmode == KW_LINEAR:
        return sf64_from_int(d if d >= 0 else -d)
    return sf64_from_int(d * d)


@always_inline
def _kappa(t: Int, f: FP, q: IP):
    """q = [KIND, CELLS, k, WEIGHTED, PHASE, WMODE, OUT, S]. S (scratch,
    binary64): sum0 (column sums) at S, sum1 (row sums) at S + 2k, the
    per-row numerator / denominator partials at S + 4k / S + 6k, den at
    S + 8k. OUT: [0] flags (bit 0 den == 0, bit 1 den_k == 0), [2..4) kappa.
    PHASE 0 (unit t < k): sum0[t] over rows, sum1[t] over columns.
    PHASE 1 (one unit): den = the sum of sum0. PHASE 2 (unit t < k): row t's
    sum of w * cm and of w * (sum0[t] * sum1[j] / den). PHASE 3 (one unit):
    num_k, den_k = the sums of the row partials (blocked by row, ascending),
    kappa = 1 - num_k / den_k."""
    var C = p(q, 1)
    var k = p(q, 2)
    var weighted = p(q, 3) != 0
    var phase = p(q, 4)
    var wmode = p(q, 5)
    var OUT = p(q, 6)
    var S = p(q, 7)
    if phase == 0:
        if t >= k:
            return
        var s0 = SF64_ZERO
        var s1 = SF64_ZERO
        for i in range(k):
            s0 = sf64_add(s0, sum_at(f, C, i * k + t, weighted))
            s1 = sf64_add(s1, sum_at(f, C, t * k + i, weighted))
        st64(f, S + 2 * t, s0)
        st64(f, S + 2 * k + 2 * t, s1)
        return
    if phase == 1:
        if t != 0:
            return
        var d = SF64_ZERO
        for i in range(k):
            d = sf64_add(d, ld64(f, S + 2 * i))
        st64(f, S + 8 * k, d)
        sti(f, OUT, 1 if is0(d) else 0)
        return
    var den = ld64(f, S + 8 * k)
    if is0(den):
        return
    if phase == 2:
        if t >= k:
            return
        var a = ld64(f, S + 2 * t)
        var num = SF64_ZERO
        var dk = SF64_ZERO
        for j in range(k):
            var wm = _kw(t, j, wmode)
            num = sf64_add(num, sf64_mul(wm, sum_at(f, C, t * k + j, weighted)))
            dk = sf64_add(dk, sf64_mul(wm, sf64_div(sf64_mul(a, ld64(f, S + 2 * k + 2 * j)), den)))
        st64(f, S + 4 * k + 2 * t, num)
        st64(f, S + 6 * k + 2 * t, dk)
        return
    if t != 0:
        return
    var num_k = SF64_ZERO
    var den_k = SF64_ZERO
    for i in range(k):
        num_k = sf64_add(num_k, ld64(f, S + 4 * k + 2 * i))
        den_k = sf64_add(den_k, ld64(f, S + 6 * k + 2 * i))
    if is0(den_k):
        sti(f, OUT, 2)
        return
    st64(f, OUT + 2, sf64_sub(SF64_ONE, sf64_div(num_k, den_k)))


@always_inline
def _clr(f: FP, q: IP):
    """q = [KIND, CELLS, WEIGHTED, OUT]; one unit over the 2 x 2 cells
    tn, fp, fn, tp. OUT: [0] flags (bit 0 no positive support, bit 1
    fp == 0, bit 2 tn == 0), [2..4) LR+, [4..6) LR-."""
    var C = p(q, 1)
    var weighted = p(q, 2) != 0
    var OUT = p(q, 3)
    var tn = sum_at(f, C, 0, weighted)
    var fp = sum_at(f, C, 1, weighted)
    var fn_ = sum_at(f, C, 2, weighted)
    var tp = sum_at(f, C, 3, weighted)
    var support_pos = sf64_add(tp, fn_)
    var support_neg = sf64_add(tn, fp)
    var flags = 0
    if is0(support_pos):
        flags |= 1
    if is0(fp):
        flags |= 2
    else:
        st64(f, OUT + 2, sf64_div(sf64_mul(tp, support_neg), sf64_mul(fp, support_pos)))
    if is0(tn):
        flags |= 4
    else:
        st64(f, OUT + 4, sf64_div(sf64_mul(fn_, support_neg), sf64_mul(tn, support_pos)))
    sti(f, OUT, flags)


@always_inline
def _total_of(f: FP, q: IP):
    """q = [KIND, SRC, k, WEIGHTED, OUT]; one unit: the sum of k values
    from 0 in ascending order: Int32 words into an Int64 (unweighted), or
    Float32 words widened into binary64 (weighted), at OUT (two words)."""
    var src = p(q, 1)
    var k = p(q, 2)
    var OUT = p(q, 4)
    if p(q, 3) != 0:
        var s = SF64_ZERO
        for i in range(k):
            s = sf64_add(s, sf64_from_f32(ld(f, src + i)))
        st64(f, OUT, s)
        return
    var tot = 0
    for i in range(k):
        tot += ldi(f, src + i)
    st64(f, OUT, UInt64(tot))


def cm_epi_unit(t: Int, f: FP, q: IP):
    """q[0] = KIND (see the functions above for each layout)."""
    var kind = p(q, 0)
    if kind == KIND_MLCM:
        _mlcm(t, f, q)
    elif kind == KIND_NORM:
        _norm(t, f, q)
    elif kind == KIND_KAPPA:
        _kappa(t, f, q)
    elif t != 0:
        return
    elif kind == KIND_MCC:
        _mcc(f, q)
    elif kind == KIND_COUNT:
        _count(f, q)
    elif kind == KIND_CLR:
        _clr(f, q)
    elif kind == KIND_TOTAL:
        _total_of(f, q)
