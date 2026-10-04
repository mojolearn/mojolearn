# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE fam2-prep-metrics (2026-10-04): THE SET-WISE CLASSIFICATION EPILOGUE
ON THE DEVICE (`cls_epi`, op 55).

jaccard_score, balanced_accuracy_score and precision_recall_fscore_support
read their per-label tp / true / pred sums back and ran the ratios, the
micro sums and the averages over Python lists
(python/mojolearn/_expansion_metrics.py, DEVIATION 6106: the binary64
epilogue). Here the same binary64 operations, in the same order, run in the
program that made the sums, as integer arithmetic on the binary64 encoding
(checks/soft_f64.mojo: correctly rounded add, sub, mul, div, the IEEE result
a hardware double returns), so the Apple GPU, which has no float64, NVIDIA,
AMD and the host column all compute the word Python computed. No value
moves: an unweighted sum is an exact integer (below 2^31, so its binary64
image is exact, and Python's int / int is the correctly rounded quotient of
those images), a weighted one is the Float32 PairSum widened exactly, as
Python's float() widened it.

One unit (t = 0) over k <= L labels: an O(k) fold next to the O(n) sums.

ON in both numeric modes, the only route (lane cpu2-l7-metrics, 2026-10-04:
the Python epilogue left the GPU route; MOJOLEARN_IDN_CLS_EPI_OFF and the
IDENTICAL-only gate are gone). `IDN_CLS_EPI` stays True for the bindings'
`x_metrics_idn_fam2` report.
"""
from checks.soft_f64 import (
    SF64_NAN, SF64_ZERO, SF64_ONE, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_is_nan, sf64_from_int, sf64_from_f32,
)
from x_metrics.common import FP, IP, p, ld, ldi, sti, ldu, stu

comptime IDN_CLS_EPI = True
comptime OP_CLS_EPI = 55

comptime KIND_JACCARD = 0
comptime KIND_BALANCED = 1
comptime KIND_PRF = 2
#: AVG: 0 = per label (average=None), 1 = micro, 2 = the plain mean of the
#: non-NaN labels (macro, binary), 3 = weighted by the true sums
comptime AVG_NONE = 0
comptime AVG_MICRO = 1
comptime AVG_MEAN = 2
comptime AVG_WEIGHTED = 3
#: words before the per-label values: flags, kept count, three binary64 scalars
comptime CLS_HEAD = 8


@always_inline
def _st64(f: FP, at: Int, v: UInt64):
    """A binary64 word as two UInt32 words, low first (array('d') layout)."""
    stu(f, at, UInt32(v & UInt64(0xFFFFFFFF)))
    stu(f, at + 1, UInt32(v >> UInt64(32)))


@always_inline
def _ld64(f: FP, at: Int) -> UInt64:
    return (UInt64(ldu(f, at + 1)) << UInt64(32)) | UInt64(ldu(f, at))


@always_inline
def _is0(v: UInt64) -> Bool:
    return (v << UInt64(1)) == UInt64(0)


@always_inline
def _sum_at(f: FP, base: Int, i: Int, weighted: Bool) -> UInt64:
    """Label i's sum as binary64: the Float32 PairSum widened (weighted), or
    the group size OFF[i + 1] - OFF[i] (an exact integer)."""
    if weighted:
        return sf64_from_f32(ld(f, base + i))
    return sf64_from_int(ldi(f, base + i + 1) - ldi(f, base + i))


@always_inline
def _zd(code: Int) -> UInt64:
    if code == 1:
        return SF64_ONE
    if code == 2:
        return SF64_NAN
    return SF64_ZERO


@always_inline
def _nanaverage(f: FP, V: Int, k: Int, WB: Int, weighted: Bool, use_w: Bool) -> UInt64:
    """`_nanaverage(values, weights)` of the k binary64 values at V (two
    words each): the non-NaN values' mean, weighted by the sums at WB when
    use_w and their total is not zero; NaN when no value is kept."""
    var cnt = 0
    for i in range(k):
        if not sf64_is_nan(_ld64(f, V + 2 * i)):
            cnt += 1
    if cnt == 0:
        return SF64_NAN
    if use_w:
        var sw = SF64_ZERO
        for i in range(k):
            if not sf64_is_nan(_ld64(f, V + 2 * i)):
                sw = sf64_add(sw, _sum_at(f, WB, i, weighted))
        if not _is0(sw):
            var s = SF64_ZERO
            for i in range(k):
                var v = _ld64(f, V + 2 * i)
                if not sf64_is_nan(v):
                    s = sf64_add(s, sf64_mul(v, _sum_at(f, WB, i, weighted)))
            return sf64_div(s, sw)
    var s2 = SF64_ZERO
    for i in range(k):
        var v = _ld64(f, V + 2 * i)
        if not sf64_is_nan(v):
            s2 = sf64_add(s2, v)
    return sf64_div(s2, sf64_from_int(cnt))


@always_inline
def _jaccard(f: FP, q: IP):
    var TP = p(q, 0)
    var TR = p(q, 1)
    var PR = p(q, 2)
    var k = p(q, 3)
    var weighted = p(q, 4) != 0
    var avg = p(q, 6)
    var zv = _zd(p(q, 7))
    var OUT = p(q, 8)
    var V = OUT + CLS_HEAD
    var bad = 0
    if avg == AVG_MICRO:
        var a = SF64_ZERO
        var b = SF64_ZERO
        for i in range(k):
            var tp = _sum_at(f, TP, i, weighted)
            a = sf64_add(a, tp)
            b = sf64_add(b, sf64_sub(sf64_add(_sum_at(f, PR, i, weighted), _sum_at(f, TR, i, weighted)), tp))
        var r = zv
        if _is0(b):
            bad = 1
        else:
            r = sf64_div(a, b)
        # `_nanaverage([r])`: (0.0 + r) / 1, NaN when r is NaN
        if not sf64_is_nan(r):
            r = sf64_div(sf64_add(SF64_ZERO, r), SF64_ONE)
        _st64(f, OUT + 2, r)
        sti(f, OUT, bad)
        return
    for i in range(k):
        var tp = _sum_at(f, TP, i, weighted)
        var den = sf64_sub(sf64_add(_sum_at(f, PR, i, weighted), _sum_at(f, TR, i, weighted)), tp)
        var r = zv
        if _is0(den):
            bad = 1
        else:
            r = sf64_div(tp, den)
        _st64(f, V + 2 * i, r)
    sti(f, OUT, bad)
    if avg == AVG_NONE:
        return
    if avg == AVG_WEIGHTED:
        var some = False
        for i in range(k):
            if not _is0(_sum_at(f, TR, i, weighted)):
                some = True
        if some:
            # `_average_plain`: no NaN filter
            var s = SF64_ZERO
            var sw = SF64_ZERO
            for i in range(k):
                var w = _sum_at(f, TR, i, weighted)
                s = sf64_add(s, sf64_mul(_ld64(f, V + 2 * i), w))
                sw = sf64_add(sw, w)
            _st64(f, OUT + 2, sf64_div(s, sw))
            return
    _st64(f, OUT + 2, _nanaverage(f, V, k, TR, weighted, False))


@always_inline
def _balanced(f: FP, q: IP):
    """The mean recall over the labels of nonzero true sum (m of them, at
    OUT[1]); with ADJ (q[7]) the chance adjustment. m == 0, or ADJ with
    m == 1, is the caller's NaN: nothing is divided then."""
    var TP = p(q, 0)
    var TR = p(q, 1)
    var k = p(q, 3)
    var weighted = p(q, 4) != 0
    var OUT = p(q, 8)
    var m = 0
    var score = SF64_ZERO
    for i in range(k):
        var tr = _sum_at(f, TR, i, weighted)
        if not _is0(tr):
            score = sf64_add(score, sf64_div(_sum_at(f, TP, i, weighted), tr))
            m += 1
    sti(f, OUT, 0)
    sti(f, OUT + 1, m)
    if m == 0:
        _st64(f, OUT + 2, SF64_NAN)
        return
    score = sf64_div(score, sf64_from_int(m))
    if p(q, 7) != 0:
        if m == 1:
            _st64(f, OUT + 2, SF64_NAN)
            return
        var chance = sf64_div(SF64_ONE, sf64_from_int(m))
        score = sf64_sub(score, chance)
        score = sf64_div(score, sf64_sub(SF64_ONE, chance))
    _st64(f, OUT + 2, score)


@always_inline
def _ratio(num: UInt64, den: UInt64, zv: UInt64, mut bad: Int, bit: Int) -> UInt64:
    if _is0(den):
        bad |= bit
        return zv
    return sf64_div(num, den)


@always_inline
def _prf(f: FP, q: IP):
    """q[9] = FMODE (0: f-score = recall, beta infinite; 1: = precision,
    beta 0; 2: the formula), q[10] = B: two binary64 words, 1 + beta^2 then
    beta^2. Values: precision at V, recall at V + 2k, f-score at V + 4k
    (one value each under micro); scalars at OUT + 2, + 4, + 6."""
    var TP = p(q, 0)
    var TR = p(q, 1)
    var PR = p(q, 2)
    var k = p(q, 3)
    var weighted = p(q, 4) != 0
    var avg = p(q, 6)
    var zv = _zd(p(q, 7))
    var OUT = p(q, 8)
    var fmode = p(q, 9)
    var one_b2 = SF64_ONE
    var b2 = SF64_ZERO
    if fmode == 2:
        one_b2 = _ld64(f, p(q, 10))
        b2 = _ld64(f, p(q, 10) + 2)
    var V = OUT + CLS_HEAD
    var bad = 0
    var kk = k
    if avg == AVG_MICRO:
        kk = 1
    for i in range(kk):
        var tp = SF64_ZERO
        var ps = SF64_ZERO
        var ts = SF64_ZERO
        if avg == AVG_MICRO:
            for u in range(k):
                tp = sf64_add(tp, _sum_at(f, TP, u, weighted))
                ps = sf64_add(ps, _sum_at(f, PR, u, weighted))
                ts = sf64_add(ts, _sum_at(f, TR, u, weighted))
        else:
            tp = _sum_at(f, TP, i, weighted)
            ps = _sum_at(f, PR, i, weighted)
            ts = _sum_at(f, TR, i, weighted)
        var pr = _ratio(tp, ps, zv, bad, 1)
        var rc = _ratio(tp, ts, zv, bad, 2)
        var fs = rc
        if fmode == 1:
            fs = pr
        elif fmode == 2:
            fs = _ratio(sf64_mul(one_b2, tp), sf64_add(sf64_mul(b2, ts), ps), zv, bad, 4)
        _st64(f, V + 2 * i, pr)
        _st64(f, V + 2 * kk + 2 * i, rc)
        _st64(f, V + 4 * kk + 2 * i, fs)
    sti(f, OUT, bad)
    if avg == AVG_NONE:
        return
    var use_w = avg == AVG_WEIGHTED
    _st64(f, OUT + 2, _nanaverage(f, V, kk, TR, weighted, use_w))
    _st64(f, OUT + 4, _nanaverage(f, V + 2 * kk, kk, TR, weighted, use_w))
    _st64(f, OUT + 6, _nanaverage(f, V + 4 * kk, kk, TR, weighted, use_w))


def cls_epi_unit(t: Int, f: FP, q: IP):
    """q = [TP, TR, PR, k, WEIGHTED, KIND, AVG, ZD, OUT, FMODE, B]; one unit.
    TP / TR / PR: the match, true and pred groupings' sums over the labels,
    WEIGHTED 0: each a `group_sort` OFF table (L + 1 int32 offsets), 1: each
    L Float32 PairSums. The first k labels are the chosen ones. ZD: the
    zero_division value (0: 0.0, 1: 1.0, 2: NaN); balanced accuracy reads it
    as `adjusted`. OUT: [0] flags (bit 0: a zero denominator; PRF: bit 0
    precision, bit 1 recall, bit 2 f-score), [1] balanced accuracy's kept
    labels, [2..8) up to three binary64 scalars, [8..) the per-label
    binary64 values (two words each, low word first)."""
    var kind = p(q, 5)
    if kind == KIND_JACCARD:
        _jaccard(f, q)
    elif kind == KIND_BALANCED:
        _balanced(f, q)
    else:
        _prf(f, q)
