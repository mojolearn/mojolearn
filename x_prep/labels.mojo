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
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, key, canonical_nan

# Lane lab-binarizer (2026-10-11), ideas-dense-linalg G1-G3, IDENTICAL only, every
# vendor and the host column, DEFAULT OFF (A/B owed on NV + AMD, label-binarizer /
# label-encoder taxi + istella):
#   G1 MOJOLEARN_IDN_LAB_UNIQUE_HIST: integral labels whose range from their device
#     minimum fits R flags (R chosen in python/mojolearn/_expansion_prep.py
#     `_lab_hist_range` from the row count and a 2^24 cap, never a label range)
#     take min + presence flags + a flag scan for the sorted distinct words: no
#     radix sort and no run scan (about 6 launches instead of the sort's ~20). The
#     flags, scan and words are integers, so classes are the sort route's words
#     (as IDN_LABEL's [0, 4096) presence route, x_prep/label_fast.mojo).
#     Expected: the fit drops from sort-bound to two n-row passes (several x on fit).
#   G2 MOJOLEARN_IDN_LAB_CODES_DIRECT (needs G1): each label's code is RK[v - min],
#     the flag scan's rank, instead of a binary search among the K classes. Same
#     codes by construction (rank among the ascending distinct values).
#   G3 MOJOLEARN_IDN_LAB_ONE_ENTRY (needs G2): LabelBinarizer.fit_transform takes
#     the fit program's codes into the dense output instead of a second
#     upload + lab_load + lookup over the raw labels. The int32 output dtype is
#     kept (the public API's); a narrower dtype would change what callers get.
comptime IDN_LAB_UNIQUE_HIST = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_IDN_LAB_UNIQUE_HIST"]()
comptime IDN_LAB_CODES_DIRECT = IDN_LAB_UNIQUE_HIST and is_defined["MOJOLEARN_IDN_LAB_CODES_DIRECT"]()
comptime IDN_LAB_ONE_ENTRY = IDN_LAB_CODES_DIRECT and is_defined["MOJOLEARN_IDN_LAB_ONE_ENTRY"]()
#: ops F2_BASE + LAB_FIRST .. (x_prep/fam2.mojo `run_f2_unit`): lab_min_part,
#: lab_min_fold, lab_zero, lab_present, lab_flag_count, lab_hist_write, lab_code_direct
comptime LAB_N = 7
#: |label| bound of the integral test: int32 arithmetic on v - min never overflows
comptime _LAB_ABS = 1 << 30


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


def row_ones_unit(t: Int, f: FP, q: IP):
    """q = [CODES, OFF, W, OUT]; t = row (MultiLabelBinarizer's indicator,
    lane gap-prep2): int32 1 at OUT[t*W + code] for each known code of the
    row's entries [OFF[t], OFF[t+1]) (OFF int32 words, `ldi`). The words of
    `scatter_ones` with ROWS[j] = t, with no per-entry row vector built on
    the host."""
    var C = p(q, 0)
    var W = p(q, 2)
    var O = p(q, 3) + t * W
    var lo = ldi(f, p(q, 1) + t)
    var hi = ldi(f, p(q, 1) + t + 1)
    for j in range(lo, hi):
        var code = Int(ld(f, C + j))
        if code >= 0:
            sti(f, O + code, 1)


@always_inline
def _lab_int(x: Float32) -> Tuple[Bool, Int]:
    """(ok, v): the float32 word x is the integer v exactly with |v| < 2^30
    (-0.0 is 0), by bits (Metal flushes compare operands); NaN, infinities,
    fractions and larger words are not ok."""
    var b = bitcast[DType.uint32](x)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    var a = b & UInt32(0x7FFFFFFF)
    if a >= bitcast[DType.uint32](Float32(_LAB_ABS)):
        return (False, 0)
    var v = Int(bitcast[DType.float32](b))
    return (bitcast[DType.uint32](Float32(v)) == b, v)


def lab_min_part_unit(t: Int, f: FP, q: IP):
    """G1. q = [X, n, CH, PMIN, BAD]; t = chunk of CH rows: the least integral
    label of the chunk at PMIN[t] (int32 bits; 2^30 when none), BAD[0] = 1 when
    a row is not integral (every writer stores the same word)."""
    var X = p(q, 0)
    var lo = t * p(q, 2)
    var hi = min(lo + p(q, 2), p(q, 1))
    var m = _LAB_ABS
    var bad = False
    for i in range(lo, hi):
        var r = _lab_int(raw(f, X + i))
        if r[0]:
            m = min(m, r[1])
        else:
            bad = True
    sti(f, p(q, 3) + t, m)
    if bad:
        sti(f, p(q, 4), 1)


def lab_min_fold_unit(t: Int, f: FP, q: IP):
    """G1. q = [PMIN, nch, MN]; one unit: MN = the least of the nch chunk
    minima (int32 bits; nch is about sqrt(n), `_label_chunk`, as `uniq_scan`)."""
    var m = _LAB_ABS
    for c in range(p(q, 1)):
        m = min(m, ldi(f, p(q, 0) + c))
    sti(f, p(q, 2), m)


def lab_zero_unit(t: Int, f: FP, q: IP):
    """G1. q = [OUT]; OUT[t] = 0 (int32 bits)."""
    sti(f, p(q, 0) + t, 0)


def lab_present_unit(t: Int, f: FP, q: IP):
    """G1. q = [X, MN, R, FL, BAD]; t = row: FL[v - MN] = 1 for an integral
    label v with v - MN < R, else BAD[0] = 1 (every writer stores 1)."""
    var r = _lab_int(raw(f, p(q, 0) + t))
    var d = r[1] - ldi(f, p(q, 1))
    if r[0] and d >= 0 and d < p(q, 2):
        sti(f, p(q, 3) + d, 1)
    else:
        sti(f, p(q, 4), 1)


def lab_flag_count_unit(t: Int, f: FP, q: IP):
    """G1. q = [FL, R, CH, CNT]; t = chunk of CH flags: how many are set."""
    var lo = t * p(q, 2)
    var hi = min(lo + p(q, 2), p(q, 1))
    var k = 0
    for r in range(lo, hi):
        if ldi(f, p(q, 0) + r) != 0:
            k += 1
    sti(f, p(q, 3) + t, k)


def lab_hist_write_unit(t: Int, f: FP, q: IP):
    """G1/G2. q = [FL, R, CH, OFF, U, MN, RK]; t = chunk: every set flag r of
    the chunk writes the float word of r + MN at U[OFF[t] ..], ascending (an
    exact float32 integer: it is a label's own word); RK >= 0 (G2): RK[r] = its
    rank among the distinct labels (int32 bits; unset flags are never read)."""
    var lo = t * p(q, 2)
    var hi = min(lo + p(q, 2), p(q, 1))
    var k = ldi(f, p(q, 3) + t)
    var mn = ldi(f, p(q, 5))
    var RK = p(q, 6)
    for r in range(lo, hi):
        if ldi(f, p(q, 0) + r) != 0:
            f.unsafe_store(p(q, 4) + k, Float32(r + mn))
            if RK >= 0:
                sti(f, RK + r, k)
            k += 1


def lab_code_direct_unit(t: Int, f: FP, q: IP):
    """G2. q = [X, MN, R, RK, C]; t = row: C[t] = the float code RK[v - MN]
    (`lookup`'s word: the label's index among the ascending classes); -1 for
    a row outside the flags (BAD is then set and the caller discards C)."""
    var r = _lab_int(raw(f, p(q, 0) + t))
    var d = r[1] - ldi(f, p(q, 1))
    var c = -1
    if r[0] and d >= 0 and d < p(q, 2):
        c = ldi(f, p(q, 3) + d)
    st(f, p(q, 4) + t, Float32(c))


def run_lab_unit[K: Int](t: Int, f: FP, q: IP):
    """K = op - F2_BASE - LAB_FIRST (x_prep/fam2.mojo)."""
    comptime if K == 0:
        lab_min_part_unit(t, f, q)
    comptime if K == 1:
        lab_min_fold_unit(t, f, q)
    comptime if K == 2:
        lab_zero_unit(t, f, q)
    comptime if K == 3:
        lab_present_unit(t, f, q)
    comptime if K == 4:
        lab_flag_count_unit(t, f, q)
    comptime if K == 5:
        lab_hist_write_unit(t, f, q)
    comptime if K == 6:
        lab_code_direct_unit(t, f, q)
