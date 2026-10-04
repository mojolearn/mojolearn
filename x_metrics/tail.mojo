# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l7-metrics (2026-10-04): THE METRIC TAILS AND SCANS ON THE DEVICE.

The re-audit (cpu-reaudit-2026-10-04 section 2, Metrics) found host work on
the default GPU route after the device folds: per-label count differences,
the O(n) validation scans (the log-domain refusal, the 0/1 relevance check,
the probability rows' check and binary pack) and the O(classes) / O(outputs)
binary64 finalizations. This file holds the shared pieces:

- binary64 words in the float32 arena (two words, low first, as Python's
  array('d') lays them out), read and written as integers so the Apple GPU,
  which has no float64, computes the same word as NVIDIA, AMD and the host
  column (checks/soft_f64.mojo: correctly rounded + - * / sqrt);
- `off_diff` (op 56): a `group_sort` OFF table's group sizes, one unit per group;
- `flag_scan` (op 57): one unit per value; a value that fails the test stores
  1 in the flag word. Every writer stores the same word, so the result is
  the same on every vendor and thread count; the caller reads one word;
- `proba_rows` (op 58): one unit per row; the check `probability_rows_f32`
  (bindings/_mojolearn.mojo) made on the host, here in the metric program,
  three flag words (non-finite, outside [0, 1], row sum off one), and the
  binary column packed into `[1 - p, p]` rows in the arena.

Every op here is ON in both numeric modes (FAST and IDENTICAL): these are the
only route (the Python and host-binding forms left the GPU route).
"""
from std.memory import bitcast
from std.math import isfinite
from checks.soft_f64 import SF64_NAN, SF64_ZERO, SF64_ONE, SF64_SIGN, sf64_add, sf64_sub, sf64_gt, sf64_from_int, sf64_from_f32
from x_metrics.common import FP, IP, p, ld, ldi, sti, ldu, stu

comptime OP_OFF_DIFF = 56
comptime OP_FLAG_SCAN = 57
comptime OP_PROBA_ROWS = 58

#: flag_scan tests: 0 the log domain (v <= -1), 1 a relevance indicator
#: (v not 0 and not 1)
comptime SCAN_LOG_DOMAIN = 0
comptime SCAN_INDICATOR = 1

#: sqrt(Float32 epsilon) rounded to Float32, widened: 0.0003452669770922512
comptime PROBA_SUM_TOL = UInt64(0x3F36A09E60000000)


@always_inline
def st64(f: FP, at: Int, v: UInt64):
    """A binary64 word as two UInt32 words, low first (array('d') layout)."""
    stu(f, at, UInt32(v & UInt64(0xFFFFFFFF)))
    stu(f, at + 1, UInt32(v >> UInt64(32)))


@always_inline
def ld64(f: FP, at: Int) -> UInt64:
    return (UInt64(ldu(f, at + 1)) << UInt64(32)) | UInt64(ldu(f, at))


@always_inline
def is0(v: UInt64) -> Bool:
    """+0.0 or -0.0."""
    return (v << UInt64(1)) == UInt64(0)


@always_inline
def abs64(v: UInt64) -> UInt64:
    return v & ~SF64_SIGN


@always_inline
def cnt_at(f: FP, off: Int, i: Int) -> Int:
    """Group i's size in a `group_sort` OFF table (L + 1 Int32 offsets)."""
    return ldi(f, off + i + 1) - ldi(f, off + i)


@always_inline
def sum_at(f: FP, base: Int, i: Int, weighted: Bool) -> UInt64:
    """Group i's sum as binary64: the Float32 PairSum widened exactly
    (weighted: `base` is the group_sum OUT), or the group size, an exact
    integer (`base` is the OFF table)."""
    if weighted:
        return sf64_from_f32(ld(f, base + i))
    return sf64_from_int(cnt_at(f, base, i))


@always_inline
def f32_64(f: FP, i: Int) -> UInt64:
    """Arena word i (a Float32 result, flushed as every unit loads it), widened exactly."""
    return sf64_from_f32(ld(f, i))


def off_diff_unit(t: Int, f: FP, q: IP):
    """q = [OFF, m, OUT]; unit t < m: OUT[t] = OFF[t + 1] - OFF[t] (Int32)."""
    var m = p(q, 1)
    if t >= m:
        return
    var off = p(q, 0)
    sti(f, p(q, 2) + t, cnt_at(f, off, t))


def flag_scan_unit(t: Int, f: FP, q: IP):
    """q = [SRC, n, MODE, FLAG]; unit t < n tests SRC[t] (the raw Float32
    word: no flush, a subnormal is neither -1 nor 0 nor 1, as on the host)
    and stores 1 at FLAG when it fails. NaN fails neither test (numpy's
    `min() <= -1` and `v != 0 and v != 1` see a NaN as the host did:
    the indicator test fails it, the log-domain test does not)."""
    var n = p(q, 1)
    if t >= n:
        return
    var v = f.unsafe_load(p(q, 0) + t)
    var mode = p(q, 2)
    var bad = False
    if mode == SCAN_LOG_DOMAIN:
        bad = v <= Float32(-1)
    else:
        bad = v != Float32(0) and v != Float32(1)
    if bad:
        sti(f, p(q, 3), 1)


def proba_rows_unit(t: Int, f: FP, q: IP):
    """q = [SRC, n, k, BIN, DST, FLAGS]; unit t < n checks row t of the
    (n, k) Float32 probabilities (k = 1 when BIN): FLAGS[0] = 1 for a
    non-finite value, FLAGS[1] = 1 for a value outside [0, 1], FLAGS[2] = 1
    for a multiclass row whose binary64 sum (ascending columns, correctly
    rounded adds of the exact widenings) is more than sqrt(Float32 eps)
    from one. BIN: DST[2t] = 1 - p (Float32), DST[2t + 1] = p. The caller
    reports the first set flag in that order, as probability_rows_f32 did."""
    var n = p(q, 1)
    if t >= n:
        return
    var src = p(q, 0)
    var k = p(q, 2)
    var binary = p(q, 3) != 0
    var flags = p(q, 5)
    var total = SF64_ZERO
    var nonfinite = False
    var outside = False
    for c in range(k):
        var v = f.unsafe_load(src + t * k + c)
        if not isfinite(v):
            nonfinite = True
        if v < Float32(0) or v > Float32(1):
            outside = True
        total = sf64_add(total, sf64_from_f32(v))
        if binary:
            var dst = p(q, 4)
            f.unsafe_store(dst + 2 * t, Float32(1) - v)
            f.unsafe_store(dst + 2 * t + 1, v)
    if nonfinite:
        sti(f, flags, 1)
    if outside:
        sti(f, flags + 1, 1)
    if not binary and sf64_gt(abs64(sf64_sub(total, SF64_ONE)), PROBA_SUM_TOL):
        sti(f, flags + 2, 1)
