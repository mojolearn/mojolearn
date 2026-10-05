# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu4-python (2026-10-04): THE CURVE ARRAYS ON THE DEVICE.

`roc_curve`, `precision_recall_curve` (sample_weight or drop_intermediate)
and `det_curve` formatted their Float64 outputs on the host after the device
sort and cumulative counts (x_metrics/epilogue.mojo roc_arrays, pr_arrays,
det_arrays: one host walk over the curve's points). Here they are units of
the metric program, so the host column (x_metrics/host/program.mojo) runs the
same code and every vendor writes the same words:

- `curve_out` (op 63, the caller's): q = [kind, n, FPS, TPS, THR, CNT, DROP,
  OUT, LEN], one problem (total 1). kind 0 ROC, 1 precision-recall, 2 DET.
  The planner (x_metrics/plan.mojo) expands it; its own unit runs nothing.
  - DROP (kinds 1, 2): `co_keep` (op 64) flags the points the
    precision-recall / DET drop rule keeps (the first, the last, every i
    with tps[i] != tps[i-1] or tps[i+1] != tps[i], when c > 2), and the
    curve compaction units (x_metrics/par.mojo ck_cnt, ck_off, ck_fill)
    copy the kept points in order. The ROC drop is bin_curve's own
    compaction (the caller passes the compacted words and DROP 0).
  - kind 2: `co_det` (op 66, one unit) finds the DET slice [first, last)
    by the same two binary searches over the kept points behind a leading 0.
  - `co_emit` (op 65, n + 1 units): one output index each, binary64 words
    (two Float32 words, low first) at OUT + 2 * j, OUT + 2 * (n + 1) + 2 * j,
    OUT + 4 * (n + 1) + 2 * j for the three arrays.
  LEN (6 words) = [the longest array's length, status (0 ok, 1 an empty
  curve, 2 a DET class of zero weight), F (binary64), T (binary64)], F and T
  the last fps and tps.

Arithmetic: every Float32 word widens exactly from its bits (no flush, as
the host epilogue's Float64() widening); the divisions, the DET subtraction
and the precision denominator are correctly rounded binary64 operations in
checks/soft_f64.mojo, the same IEEE results the host epilogue's hardware
operations gave for finite operands, on every vendor including the Apple
GPU (no Float64). A NaN result is the canonical quiet NaN everywhere.
Comparisons of Float32 curve words are on their bits (+0 == -0, a NaN
equals nothing), so a device flushing subnormals decides as the host does.
ON in both numeric modes: this is the only route (no `_OFF` arm; the Python
call of the host epilogue left the GPU route).
"""
from std.memory import bitcast
from checks.soft_f64 import (
    SF64_NAN, SF64_INF, SF64_ZERO, SF64_ONE, SF64_SIGN, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_lt,
    sf64_is_nan, sf64_neg, sf64_from_f32,
)
from x_metrics.common import FP, IP, p, ldi, sti, ldu
from x_metrics.tail import st64, ld64, is0

comptime OP_CURVE_OUT = 63
comptime OP_CO_KEEP = 64
comptime OP_CO_EMIT = 65
comptime OP_CO_DET = 66
comptime OP_AUC_XY = 67
comptime OP_AX_CHUNK = 68
comptime OP_AX_FINAL = 69
#: words per chunk record of auc_xy: hi (2), lo (2), flags (1)
comptime AX_REC = 5
#: auc_xy flags: a negative step, a positive (or NaN) step, a NaN term, a
#: +inf term or overflow, a -inf term or overflow
comptime AX_NEG = 1
comptime AX_POS = 2
comptime AX_NAN = 4
comptime AX_PINF = 8
comptime AX_NINF = 16

comptime CO_ROC = 0
comptime CO_PR = 1
comptime CO_DET = 2

comptime CO_OK = 0
comptime CO_EMPTY = 1
comptime CO_ZERO_CLASS = 2


@always_inline
def w64(f: FP, i: Int) -> UInt64:
    """Arena word i (a Float32), widened exactly from its bits."""
    return sf64_from_f32(bitcast[DType.float32](ldu(f, i)))


@always_inline
def f32_eq(f: FP, i: Int, j: Int) -> Bool:
    """Float32 words i and j compare equal (on the bits: +0 == -0, a NaN
    equals nothing), as their exact binary64 widenings do on the host."""
    var a = ldu(f, i)
    var b = ldu(f, j)
    var m = UInt32(0x7FFFFFFF)
    var inf = UInt32(0x7F800000)
    if (a & m) > inf or (b & m) > inf:
        return False
    return a == b or ((a | b) & m) == UInt32(0)


@always_inline
def gt0(v: UInt64) -> Bool:
    """v > 0.0 (False for a NaN, +0 and -0)."""
    return not sf64_is_nan(v) and (v >> UInt64(63)) == UInt64(0) and not is0(v)


def curve_out_unit(t: Int, f: FP, q: IP):
    """The caller's stage: the planner expands it (x_metrics/plan.mojo);
    run unplanned it writes nothing."""
    pass


def co_keep_unit(t: Int, f: FP, q: IP):
    """q = [n, TPS, CNT, KEEP]; unit t < CNT[0]: KEEP[t] = 1 when the
    precision-recall / DET drop rule keeps point t, else 0."""
    var n = p(q, 0)
    if t >= n:
        return
    var c = ldi(f, p(q, 2))
    if t >= c:
        return
    var keep = 1
    if c > 2 and t > 0 and t < c - 1:
        var T = p(q, 1) + t
        if f32_eq(f, T, T - 1) and f32_eq(f, T + 1, T):
            keep = 0
    sti(f, p(q, 3) + t, keep)


@always_inline
def _det_f(f: FP, fps: Int, i: Int) -> UInt64:
    """`det_arrays`' list f (or t): 0.0 at i == 0, else the kept word i - 1."""
    if i == 0:
        return SF64_ZERO
    return w64(f, fps + i - 1)


def co_det_unit(t: Int, f: FP, q: IP):
    """q = [FPS, TPS, CNT, LEN, B]; unit 0: over the m = c + 1 values f, t
    (a leading 0 then the c kept points), F = f[m-1], T = t[m-1]; status 1
    when c <= 0, 2 when F or T is zero; else first = bisect_right(f, f[0])
    - 1 (at least 0), last = min(bisect_left(t, T) + 1, m), LEN[0] = the
    slice length, B[0] = first. The binary searches are `det_arrays`'
    (x_metrics/epilogue.mojo), step for step."""
    if t != 0:
        return
    var fps = p(q, 0)
    var tps = p(q, 1)
    var c = ldi(f, p(q, 2))
    var LEN = p(q, 3)
    if c <= 0:
        sti(f, LEN, 0)
        sti(f, LEN + 1, CO_EMPTY)
        return
    var m = c + 1
    var F = w64(f, fps + c - 1)
    var T = w64(f, tps + c - 1)
    st64(f, LEN + 2, F)
    st64(f, LEN + 4, T)
    if is0(T) or is0(F):
        sti(f, LEN, 0)
        sti(f, LEN + 1, CO_ZERO_CLASS)
        return
    # bisect_right(f, f[0]): x < v[mid] moves hi
    var x = SF64_ZERO
    var lo = 0
    var hi = m
    while lo < hi:
        var mid = (lo + hi) // 2
        if sf64_lt(x, _det_f(f, fps, mid)):
            hi = mid
        else:
            lo = mid + 1
    var first = lo - 1 if lo > 0 else 0
    # bisect_left(t, T): v[mid] < x moves lo
    lo = 0
    hi = m
    while lo < hi:
        var mid = (lo + hi) // 2
        if sf64_lt(_det_f(f, tps, mid), T):
            lo = mid + 1
        else:
            hi = mid
    var last = min(lo + 1, m)
    var L = last - first if last > first else 0
    sti(f, LEN, L)
    sti(f, LEN + 1, CO_OK)
    sti(f, p(q, 4), first)


def co_emit_unit(t: Int, f: FP, q: IP):
    """q = [kind, n, FPS, TPS, THR, CNT, OUT, LEN, B]; unit t <= n writes
    output index t of the three arrays (A0 at OUT, A1 at OUT + 2(n + 1),
    A2 at OUT + 4(n + 1), binary64 each):

    - ROC (c points): t == 0 is (0, 0, +inf) (NaN rates for an empty
      class: F <= 0 or T <= 0); t in 1..c is (fps / F, tps / T, thr) of
      point t - 1. Length c + 1.
    - precision-recall (m = c kept points, reversed): index m - 1 - j of
      point j is (tps / (tps + fps), 0.0 when the sum is zero; tps / T, 1.0
      when T is zero; thr); precision[m] = 1, recall[m] = 0. Lengths m + 1,
      m + 1, m (LEN[0] = m + 1).
    - DET (after co_det): q = first + j over the slice of length L, index
      L - 1 - j: (f / F, (T - t) / T, +inf at i == 0 else thr[i - 1]).

    Unit 0 writes LEN for ROC and precision-recall (co_det wrote DET's)."""
    var n = p(q, 1)
    if t > n:
        return
    var kind = p(q, 0)
    var fps = p(q, 2)
    var tps = p(q, 3)
    var thr = p(q, 4)
    var c = ldi(f, p(q, 5))
    var OUT = p(q, 6)
    var LEN = p(q, 7)
    var A0 = OUT
    var A1 = OUT + 2 * (n + 1)
    var A2 = OUT + 4 * (n + 1)
    if kind == CO_DET:
        if ldi(f, LEN + 1) != CO_OK:
            return
        var L = ldi(f, LEN)
        if t >= L:
            return
        var i = ldi(f, p(q, 8)) + t
        var r = L - 1 - t
        var F = w64(f, fps + c - 1)
        var T = w64(f, tps + c - 1)
        st64(f, A0 + 2 * r, sf64_div(_det_f(f, fps, i), F))
        st64(f, A1 + 2 * r, sf64_div(sf64_sub(T, _det_f(f, tps, i)), T))
        st64(f, A2 + 2 * r, SF64_INF if i == 0 else w64(f, thr + i - 1))
        return
    if t == 0:
        sti(f, LEN, 0 if c <= 0 else c + 1)
        sti(f, LEN + 1, CO_EMPTY if c <= 0 else CO_OK)
        if c > 0:
            st64(f, LEN + 2, w64(f, fps + c - 1))
            st64(f, LEN + 4, w64(f, tps + c - 1))
    if c <= 0:
        return
    var F = w64(f, fps + c - 1)
    var T = w64(f, tps + c - 1)
    if kind == CO_ROC:
        if t > c:
            return
        var f_ok = gt0(F)
        var t_ok = gt0(T)
        if t == 0:
            st64(f, A0, SF64_ZERO if f_ok else SF64_NAN)
            st64(f, A1, SF64_ZERO if t_ok else SF64_NAN)
            st64(f, A2, SF64_INF)
            return
        var j = t - 1
        st64(f, A0 + 2 * t, sf64_div(w64(f, fps + j), F) if f_ok else SF64_NAN)
        st64(f, A1 + 2 * t, sf64_div(w64(f, tps + j), T) if t_ok else SF64_NAN)
        st64(f, A2 + 2 * t, w64(f, thr + j))
        return
    # precision-recall: m = c kept points
    var m = c
    if t == m:
        st64(f, A0 + 2 * m, SF64_ONE)
        st64(f, A1 + 2 * m, SF64_ZERO)
        return
    if t > m:
        return
    var tt = w64(f, tps + t)
    var d = sf64_add(tt, w64(f, fps + t))
    var r = m - 1 - t
    st64(f, A0 + 2 * r, SF64_ZERO if is0(d) else sf64_div(tt, d))
    st64(f, A1 + 2 * r, sf64_div(tt, T) if not is0(T) else SF64_ONE)
    st64(f, A2 + 2 * r, w64(f, thr + t))


# ---------------------------------------------------------------------------
# The public `auc` (x, y): the trapezoid sum on the device (lane cpu4-python)
# ---------------------------------------------------------------------------
# x_metrics/epilogue.mojo auc_xy summed the terms (x[i] - x[i-1]) *
# (y[i] + y[i-1]) / 2 by a correctly rounded host fsum. Here: each term is
# the same correctly rounded binary64 operations (soft f64), summed as a
# float-float (two-sum, hi + lo) in ascending order within fixed chunks of
# CH terms, the chunk sums met in ascending chunk order (a function of n and
# CH only, so every vendor and the host column fold the same way); the
# answer is hi + lo. Non-finite terms and an overflowing partial sum are
# flagged instead of summed: a NaN term, or +inf with -inf, gives NaN, else
# the infinity's sign (the IEEE sum's answer).


@always_inline
def _is_inf(v: UInt64) -> Bool:
    return (v & ~SF64_SIGN) == SF64_INF


@always_inline
def _finite(v: UInt64) -> Bool:
    return (v & SF64_INF) != SF64_INF


@always_inline
def _ff_step(mut hi: UInt64, mut lo: UInt64, x: UInt64) -> Int:
    """hi + lo += x (finite x) by a two-sum; returns the overflow flag
    (AX_PINF / AX_NINF) when hi + x is not finite, leaving hi, lo as they were."""
    var s = sf64_add(hi, x)
    if not _finite(s):
        return AX_NINF if (s >> UInt64(63)) != UInt64(0) else AX_PINF
    var bb = sf64_sub(s, hi)
    var e = sf64_add(sf64_sub(hi, sf64_sub(s, bb)), sf64_sub(x, bb))
    lo = sf64_add(lo, e)
    hi = s
    return 0


def auc_xy_unit(t: Int, f: FP, q: IP):
    """The caller's stage q = [n, X, Y, OUT, CH] (X, Y: n binary64 values
    each, two words per value, low first; OUT: the binary64 answer then the
    flags word): the planner expands it (ax_chunk, ax_final); run unplanned
    it writes nothing."""
    pass


def ax_chunk_unit(t: Int, f: FP, q: IP):
    """q = [n, X, Y, S, C, CH]; unit t < C: the terms i in [max(1, t*CH),
    min(n, (t+1)*CH)) summed as above into S + AX_REC * t (hi, lo, flags)."""
    var C = p(q, 4)
    if t >= C:
        return
    var n = p(q, 0)
    var X = p(q, 1)
    var Y = p(q, 2)
    var CH = p(q, 5)
    var hi = SF64_ZERO
    var lo = SF64_ZERO
    var flags = 0
    var two = UInt64(0x4000000000000000)
    for i in range(max(1, t * CH), min(n, t * CH + CH)):
        var d = sf64_sub(ld64(f, X + 2 * i), ld64(f, X + 2 * i - 2))
        if sf64_is_nan(d):
            flags |= AX_POS
        elif not is0(d):
            flags |= AX_NEG if (d >> UInt64(63)) != UInt64(0) else AX_POS
        var term = sf64_div(sf64_mul(d, sf64_add(ld64(f, Y + 2 * i), ld64(f, Y + 2 * i - 2))), two)
        if sf64_is_nan(term):
            flags |= AX_NAN
        elif _is_inf(term):
            flags |= AX_NINF if (term >> UInt64(63)) != UInt64(0) else AX_PINF
        else:
            flags |= _ff_step(hi, lo, term)
    var S = p(q, 3) + AX_REC * t
    st64(f, S, hi)
    st64(f, S + 2, lo)
    sti(f, S + 4, flags)


def ax_final_unit(t: Int, f: FP, q: IP):
    """q = [S, C, OUT]; unit 0: the chunk sums met in ascending order
    (hi by a two-sum, the lows and the errors into lo), flags or-ed;
    OUT = the direction (-1 when some step is negative and none positive)
    times the sum, OUT + 2 = the flags (AX_NEG | AX_POS both: the caller
    raises its non-monotonic error)."""
    if t != 0:
        return
    var S = p(q, 0)
    var C = p(q, 1)
    var hi = SF64_ZERO
    var lo = SF64_ZERO
    var flags = 0
    for c in range(C):
        var r = S + AX_REC * c
        flags |= ldi(f, r + 4)
        flags |= _ff_step(hi, lo, ld64(f, r))
        lo = sf64_add(lo, ld64(f, r + 2))
    var v: UInt64
    if (flags & AX_NAN) != 0 or ((flags & AX_PINF) != 0 and (flags & AX_NINF) != 0):
        v = SF64_NAN
    elif (flags & AX_PINF) != 0:
        v = SF64_INF
    elif (flags & AX_NINF) != 0:
        v = SF64_INF | SF64_SIGN
    else:
        v = sf64_add(hi, lo)
    if (flags & AX_NEG) != 0 and (flags & AX_POS) == 0 and not sf64_is_nan(v):
        v = sf64_neg(v)
    var OUT = p(q, 2)
    st64(f, OUT, v)
    sti(f, OUT + 2, flags)
