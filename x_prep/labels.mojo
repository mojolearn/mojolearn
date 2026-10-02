# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LabelEncoder / LabelBinarizer on a numeric label BUFFER (lane
neural-pass137). The labels cross as their own bytes (int32, uint32, int64,
float32 or float64 words), `lab_load` turns each into the float32 word the
Python route would have built (NaN where float32 cannot hold the label
exactly, so the caller takes the Python route), and the distinct sorted
words come out of a chunked run scan in three parallel stages instead of one
thread walking every row (`unique_cols`). Integer arithmetic only: the same
words on every vendor and the host, and the same words as `unique_cols`."""
from std.memory import bitcast
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, key, canonical_nan


@always_inline
def _int_word(neg: Bool, m: UInt64) -> Float32:
    """The float32 holding the integer (-1)^neg * m exactly, else NaN."""
    if m == UInt64(0):
        return Float32(0)
    var msb = 63
    while (m >> UInt64(msb)) == UInt64(0):
        msb -= 1
    var mant: UInt64
    if msb > 23:
        var drop = UInt64(msb - 23)
        if (m & ((UInt64(1) << drop) - UInt64(1))) != UInt64(0):
            return canonical_nan()
        mant = m >> drop
    else:
        mant = m << UInt64(23 - msb)
    var bits = (UInt32(msb + 127) << 23) | (UInt32(mant) & UInt32(0x7FFFFF))
    if neg:
        bits |= UInt32(0x80000000)
    return bitcast[DType.float32](bits)


def lab_load_unit(t: Int, f: FP, q: IP):
    """q = [RAW, KIND, OUT]; t = label. KIND 0 float32, 1 int32, 2 uint32
    (one word a label), 3 int64, 4 float64 (two little-endian words a label).
    OUT[t] = the label as a float32 word when float32 holds it exactly as a
    zero, normal or infinite value (a float keeps its sign bit), else the
    canonical NaN (a NaN label, a float32 subnormal, an inexact value)."""
    var R = p(q, 0)
    var kind = p(q, 1)
    var v = canonical_nan()
    if kind == 0:
        var b = bitcast[DType.uint32](raw(f, R + t))
        var e = (b >> 23) & UInt32(0xFF)
        if e == UInt32(0xFF):
            if (b & UInt32(0x7FFFFF)) == UInt32(0):
                v = bitcast[DType.float32](b)
        elif e != UInt32(0) or (b & UInt32(0x7FFFFF)) == UInt32(0):
            v = bitcast[DType.float32](b)
    elif kind == 1:
        var w = Int64(Int32(bitcast[DType.int32](raw(f, R + t))))
        v = _int_word(w < 0, UInt64(-w) if w < 0 else UInt64(w))
    elif kind == 2:
        v = _int_word(False, UInt64(bitcast[DType.uint32](raw(f, R + t))))
    elif kind == 3:
        var lo = UInt64(bitcast[DType.uint32](raw(f, R + 2 * t)))
        var hi = UInt64(bitcast[DType.uint32](raw(f, R + 2 * t + 1)))
        var u = (hi << 32) | lo
        var neg = (hi >> 31) != UInt64(0)
        v = _int_word(neg, (~u + UInt64(1)) if neg else u)
    elif kind == 4:
        var lo = bitcast[DType.uint32](raw(f, R + 2 * t))
        var hi = bitcast[DType.uint32](raw(f, R + 2 * t + 1))
        var sign = hi & UInt32(0x80000000)
        var e = Int((hi >> 20) & UInt32(0x7FF))
        var mh = hi & UInt32(0xFFFFF)
        if e == 0x7FF:
            if mh == UInt32(0) and lo == UInt32(0):
                v = bitcast[DType.float32](sign | UInt32(0x7F800000))
        elif e == 0:
            if mh == UInt32(0) and lo == UInt32(0):
                v = bitcast[DType.float32](sign)
        elif e >= 897 and e <= 1150 and (lo & UInt32(0x1FFFFFFF)) == UInt32(0):
            v = bitcast[DType.float32](sign | (UInt32(e - 896) << 23) | (mh << 3) | (lo >> 29))
    f.unsafe_store(p(q, 2) + t, v)


@always_inline
def _starts_run(f: FP, S: Int, i: Int) -> Bool:
    """Row i of the sorted column S opens a run of equal `key`s
    (`unique_cols`' test)."""
    return i == 0 or key(raw(f, S + i)) != key(raw(f, S + i - 1))


def uniq_count_unit(t: Int, f: FP, q: IP):
    """q = [S, n, CH, CNT]; t = chunk of CH sorted rows: how many runs open in
    it, int32 bits."""
    var S = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = 0
    for i in range(lo, hi):
        if _starts_run(f, S, i):
            k += 1
    sti(f, p(q, 3) + t, k)


def uniq_scan_unit(t: Int, f: FP, q: IP):
    """q = [CNT, nch, OFF, TOT]; one unit: OFF = the exclusive prefix sums of
    CNT (int32 bits), TOT = their total as a float (`unique_cols`' count)."""
    var s = 0
    for c in range(p(q, 1)):
        sti(f, p(q, 2) + c, s)
        s += ldi(f, p(q, 0) + c)
    st(f, p(q, 3), Float32(s))


def uniq_write_unit(t: Int, f: FP, q: IP):
    """q = [S, n, CH, OFF, U]; t = chunk: the first word of every run that
    opens in the chunk, at U[OFF[t] ..], bit for bit."""
    var S = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var U = p(q, 4)
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = ldi(f, p(q, 3) + t)
    for i in range(lo, hi):
        if _starts_run(f, S, i):
            f.unsafe_store(U + k, raw(f, S + i))
            k += 1


def chunk_neg_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, CH, OUT]; t = chunk: how many float codes in it are
    negative, int32 bits."""
    var C = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = 0
    for i in range(lo, hi):
        if ld(f, C + i) < Float32(0):
            k += 1
    sti(f, p(q, 3) + t, k)
