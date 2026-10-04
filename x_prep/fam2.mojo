# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE fam2-prep-metrics (2026-10-04): the host numeric work that still sat
inside the prep lane's GPU routes, as x_prep units (x_prep/common.mojo's
program model: one GPU thread per unit, the host binding the same units in a
loop, so NVIDIA, AMD, Apple and the host column share one arithmetic).
IDENTICAL only; each group has its own switch, ON by default, an `_OFF`
define that restores the old route, and turns off under MOJOLEARN_IDN_ALL_OFF.

  IDN_WDRAW (-D MOJOLEARN_IDN_WDRAW_OFF): KBinsDiscretizer's weighted
      with-replacement subsample. Was the base binding's host helper
      `weighted_draw_rows_i32` (binary64 running sums and a bisect per draw).
      f2_wblk / f2_wscan / f2_wdraw: the weights' sums by XB-row block, the
      inclusive prefix of the block sums, then one thread per draw: one
      uniform picks the block (first block whose prefix exceeds u1 * total),
      a second picks the row inside it (first row whose in-block running sum
      exceeds u2 * block sum). Two-level, so the float32 sums keep their
      relative precision inside each level whatever n is. Uniforms are the
      top 24 bits of splitmix64's closed form (draws 2k and 2k+1).
  IDN_PERM_DRAW (-D MOJOLEARN_IDN_PERM_DRAW_OFF): QuantileTransformer's
      without-replacement subsample. Was the host helper
      `draw_rows_without_replacement_i32` (a serial partial Fisher-Yates).
      f2_perm_rows: draw k is the image of k under a keyed permutation of
      [0, n) (a 6-round Feistel network on the next even power of two, cycle
      walked into range), so every draw is its own thread, the k rows are
      distinct by construction, and the arithmetic is integers only.
  IDN_WPICK (-D MOJOLEARN_IDN_WPICK_OFF): IterativeImputer's
      n_nearest_features draws. Was the host helper `weighted_pick_i32` over
      a downloaded |corr| matrix, one call per (round, feature). f2_wpick:
      one thread per (round, feature) call draws its k features without
      replacement from the |corr| column that stays on the device; float32
      running sums in ascending feature order.
  IDN_PARTIAL_CODES (-D MOJOLEARN_IDN_PARTIAL_CODES_OFF): naive Bayes
      partial_fit label codes. Was `_partial_codes`' Python walk over the
      labels. The fit program now stages lab_load + lookup + count_neg and
      f2_clamp0 (an unknown label's -1 becomes class 0 so no unit indexes
      out of its table; the count refuses the batch after the run).

Ops F2_BASE .. F2_BASE + F2_N - 1 are a range of their own (as P2M_BASE's).
An op whose switch is off compiles to nothing; the Python layer never stages
it then (`x_prep_idn_fam2`, bindings/_mojolearn_x_prep*.mojo).
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, raw, ldi, sti, ld, st
from x_prep.prims import add, mul
from x_prep.py2mojo import splitmix_at
from x_prep.blocked import XB

comptime _F2_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime IDN_WDRAW = _F2_IDN and not is_defined["MOJOLEARN_IDN_WDRAW_OFF"]()
comptime IDN_PERM_DRAW = _F2_IDN and not is_defined["MOJOLEARN_IDN_PERM_DRAW_OFF"]()
comptime IDN_WPICK = _F2_IDN and not is_defined["MOJOLEARN_IDN_WPICK_OFF"]()
comptime IDN_PARTIAL_CODES = _F2_IDN and not is_defined["MOJOLEARN_IDN_PARTIAL_CODES_OFF"]()

comptime F2_BASE = 230
comptime F2_N = 6


@always_inline
def is_f2_op(op: Int) -> Bool:
    return op >= F2_BASE and op < F2_BASE + F2_N


@always_inline
def _seed64(f: FP, at: Int) -> UInt64:
    """Two int32 words (low, high) as the uint64 seed (p2m_smrows' layout)."""
    var lo = UInt64(f.bitcast[UInt32]().unsafe_load(at))
    var hi = UInt64(f.bitcast[UInt32]().unsafe_load(at + 1))
    return (hi << 32) | lo


@always_inline
def _u24(z: UInt64) -> Float32:
    """The top 24 bits of a splitmix64 word as a uniform in [0, 1): an exact
    float32 (an integer below 2^24 times 2^-24)."""
    return Float32(Int(z >> 40)) * Float32(5.9604644775390625e-08)


# ---------------------------------------------------------------- weighted draws with replacement
def f2_wblk_unit(t: Int, f: FP, q: IP):
    """q = [W, n, S]; t = block of XB rows: S[t] = the block's positive
    weights added ascending from zero."""
    var n = p(q, 1)
    var lo = t * XB
    var hi = min(lo + XB, n)
    var acc = Float32(0)
    for i in range(lo, hi):
        var w = ld(f, p(q, 0) + i)
        if w > Float32(0):
            acc = add(acc, w)
    st(f, p(q, 2) + t, acc)


def f2_wscan_unit(t: Int, f: FP, q: IP):
    """q = [S, nb, BP]; one unit: BP[b] = S[0] + .. + S[b], ascending from
    zero (nb = ceil(n / XB) adds)."""
    var nb = p(q, 1)
    var acc = Float32(0)
    for b in range(nb):
        acc = add(acc, ld(f, p(q, 0) + b))
        st(f, p(q, 2) + b, acc)


def f2_wdraw_unit(t: Int, f: FP, q: IP):
    """q = [W, n, S, BP, nb, SEED, ROWS]; t = draw: ROWS[t] (int32 bits) = a
    row drawn with probability proportional to its weight. The block is the
    first whose prefix BP exceeds u1 * total (the last block of positive sum
    when rounding leaves none), the row the first of the block whose running
    sum exceeds u2 * S[block] (the block's last positive row when none). A
    row of zero weight is never drawn: a running sum only exceeds the target
    at a row that raised it."""
    var W = p(q, 0)
    var n = p(q, 1)
    var S = p(q, 2)
    var BP = p(q, 3)
    var nb = p(q, 4)
    var s = _seed64(f, p(q, 5))
    var u1 = _u24(splitmix_at(s, 2 * t))
    var u2 = _u24(splitmix_at(s, 2 * t + 1))
    var tgt = mul(u1, ld(f, BP + nb - 1))
    var lo = 0
    var hi = nb
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(f, BP + mid) > tgt:
            hi = mid
        else:
            lo = mid + 1
    var b = lo
    if b >= nb:
        b = nb - 1
        while b > 0 and not (ld(f, S + b) > Float32(0)):
            b -= 1
    var r0 = b * XB
    var r1 = min(r0 + XB, n)
    var tgt2 = mul(u2, ld(f, S + b))
    var acc = Float32(0)
    var pick = -1
    var last = r0
    for i in range(r0, r1):
        var w = ld(f, W + i)
        if w > Float32(0):
            acc = add(acc, w)
            last = i
            if pick < 0 and acc > tgt2:
                pick = i
    if pick < 0:
        pick = last
    sti(f, p(q, 6) + t, pick)


# ---------------------------------------------------------------- draws without replacement
comptime _PERM_ROUNDS = 6
comptime _PERM_KEY = UInt64(0xD1B54A32D192ED03)


def f2_perm_rows_unit(t: Int, f: FP, q: IP):
    """q = [SEED, n, ROWS]; t = draw (t < n): ROWS[t] (int32 bits) = pi(t)
    for a seed-keyed permutation pi of [0, n): a balanced Feistel network of
    _PERM_ROUNDS rounds on 2h bits (the smallest even width covering n - 1;
    round function = splitmix64's closed form under the round's key, cut to
    h bits), applied again while the image is >= n (cycle walking: the
    domain is below 4n, so few steps, and the walk of a point below n ends
    below n). Distinct t give distinct rows; integers only."""
    var n = p(q, 1)
    var s = _seed64(f, p(q, 0))
    var h = 1
    while (1 << (2 * h)) < n:
        h += 1
    var mask = (UInt64(1) << UInt64(h)) - UInt64(1)
    var x = UInt64(t)
    var again = True
    while again:
        var L = x >> UInt64(h)
        var R = x & mask
        for r in range(_PERM_ROUNDS):
            var F = splitmix_at(s + UInt64(r + 1) * _PERM_KEY, Int(R)) & mask
            var nl = R
            R = L ^ F
            L = nl
        x = (L << UInt64(h)) | R
        again = x >= UInt64(n)
    sti(f, p(q, 2) + t, Int(x))


# ---------------------------------------------------------------- IterativeImputer feature draws
def f2_wpick_unit(t: Int, f: FP, q: IP):
    """q = [M, dk, JS, k, SEED, FL, OUT]; t = call c (one (round, feature)
    step): k of the dk features drawn without replacement with probability
    M[:, j] (j = JS[c], int32 bits; M row-major dk x dk float32, a zero
    diagonal). Draw i takes the uniform of splitmix64 word c*k + i times the
    total of the positive weights not yet drawn (ascending features, from
    zero) and picks the first such feature whose running sum exceeds it (the
    last one when none does). FL[c*dk ..] = this call's drawn flags (the
    unit zeroes them first); OUT[c*(k+1) ..] = the picks ascending (int32
    bits), OUT[c*(k+1) + k] = how many were drawn (below k when the column
    ran out of positive weights)."""
    var M = p(q, 0)
    var dk = p(q, 1)
    var j = ldi(f, p(q, 2) + t)
    var k = p(q, 3)
    var s = _seed64(f, p(q, 4))
    var fl = p(q, 5) + t * dk
    for a in range(dk):
        f.unsafe_store(fl + a, Float32(0))
    var got = 0
    for i in range(k):
        var tot = Float32(0)
        var last = -1
        for a in range(dk):
            var w = ld(f, M + a * dk + j)
            if w > Float32(0) and raw(f, fl + a) == Float32(0):
                tot = add(tot, w)
                last = a
        if last >= 0:
            var tgt = mul(_u24(splitmix_at(s, t * k + i)), tot)
            var cum = Float32(0)
            var pick = -1
            for a in range(dk):
                var w = ld(f, M + a * dk + j)
                if w > Float32(0) and raw(f, fl + a) == Float32(0):
                    cum = add(cum, w)
                    if pick < 0 and cum > tgt:
                        pick = a
            if pick < 0:
                pick = last
            f.unsafe_store(fl + pick, Float32(1))
            got += 1
    var o = p(q, 6) + t * (k + 1)
    var at = 0
    for a in range(dk):
        if raw(f, fl + a) != Float32(0):
            sti(f, o + at, a)
            at += 1
    sti(f, o + k, got)


# ---------------------------------------------------------------- partial_fit codes
def f2_clamp0_unit(t: Int, f: FP, q: IP):
    """q = [SRC, DST]; t = row: DST[t] = the float code SRC[t], a negative
    code (lookup's -1 for a label outside classes_) as class 0. The batch is
    refused after the run when count_neg saw one; until then no class table
    is indexed out of range."""
    var v = ld(f, p(q, 0) + t)
    if v < Float32(0):
        v = Float32(0)
    st(f, p(q, 1) + t, v)


@always_inline
def run_f2_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if IDN_WDRAW:
        comptime if OP == F2_BASE + 0:
            f2_wblk_unit(t, f, q)
        comptime if OP == F2_BASE + 1:
            f2_wscan_unit(t, f, q)
        comptime if OP == F2_BASE + 2:
            f2_wdraw_unit(t, f, q)
    comptime if IDN_PERM_DRAW:
        comptime if OP == F2_BASE + 3:
            f2_perm_rows_unit(t, f, q)
    comptime if IDN_WPICK:
        comptime if OP == F2_BASE + 4:
            f2_wpick_unit(t, f, q)
    comptime if IDN_PARTIAL_CODES:
        comptime if OP == F2_BASE + 5:
            f2_clamp0_unit(t, f, q)
