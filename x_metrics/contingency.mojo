# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu4-python (2026-10-04): THE CONTINGENCY STATISTICS ON THE DEVICE.

The clustering metrics (contingency_matrix, pair_confusion_matrix,
normalized and adjusted mutual information) read the device grouping's
offsets back and formed everything else on the host (x_metrics/epilogue.mojo
contingency_stats, mi_contingency, expected_mi: host walks over the ka x kb
cells and the hypergeometric supports). Here they are units of the metric
program, so the host column (x_metrics/host/program.mojo) runs the same code:

- `cont_stats` (op 70, the caller's): q = [OFF, ka, kb, kk, EPS, C, F, R, K,
  P, E, MI, EMI, n], one problem. The planner (x_metrics/plan.mojo) expands
  it; its own unit runs nothing.
  - `ct_cell` (71): C[i, j] = the group size of key i * kk + j (Int64, two
    words, low first); F = C + eps (binary64) when F >= 0 (EPS: eps's two
    words).
  - `ct_row` (72), `ct_col` (73): the row and column sums R, K (Int64) and
    each row's sum of squares (scratch).
  - `ct_isum` (74), `ct_pairs` (75): the exact Int64 sums of rows^2, cols^2
    and cells^2 in chunks, then scikit-learn's pair confusion P (2 x 2
    Int64): every count is an exact integer, the same on every vendor.
  - `ct_ent` (76): one entropy term per row and per column, (c / n) * (log c
    - log n) (0 for c == 0), then `ff_chunk` / `ff_fin` sum each and negate:
    E = [H(rows), H(cols)] (binary64; 1.0 when n == 0).
  - MI >= 0: `mi_cell` (79), one term per cell, `_mi_from_contingency`'s
    nm * (log v - log n) + nm * ((-log(pi pj) + log n) + log n), |t| < 2^-52
    to 0, then summed, max(sum, 0) (0 when ka == 1 or kb == 1).
  - EMI >= 0: `emi_cell` (80), one (a, b) cell of the expected MI each: the
    hypergeometric ratio walks from the mode (each stops at the first exact
    0), their float-float sum z, then the terms (nij / n) * (log(n nij) -
    log a - log b) * (u / z), summed; then over the cells.
- `ff_chunk` (77), `ff_fin` (78): the shared fold of float-float records
  (hi, lo binary64, four words): records in ascending order within fixed
  chunks of CT_CH, the chunk sums in ascending chunk order (a function of
  the sizes only, so every vendor and the host column fold the same way).

Bits: the host epilogue's correctly rounded fsums become these float-float
folds (two-sum, then hi + lo), on every vendor and in the host column
together; every term is the same correctly rounded binary64 operations
(checks/soft_f64.mojo; `sf64_log` is `_portable_math.log`), Python's int ->
float conversions correctly rounded (`i2f`). ON in both numeric modes: the
only route (the Python call of the host epilogue left the GPU route).
"""
from checks.soft_f64 import (
    SF64_ZERO, SF64_ONE, SF64_SIGN, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_log, sf64_lt, sf64_neg,
    sf64_from_int,
)
from x_metrics.common import FP, IP, p, ldi
from x_metrics.tail import st64, ld64, is0

comptime OP_CONT_STATS = 70
comptime OP_CT_CELL = 71
comptime OP_CT_ROW = 72
comptime OP_CT_COL = 73
comptime OP_CT_ISUM = 74
comptime OP_CT_PAIRS = 75
comptime OP_CT_ENT = 76
comptime OP_FF_CHUNK = 77
comptime OP_FF_FIN = 78
comptime OP_MI_CELL = 79
comptime OP_EMI_CELL = 80

#: records per chunk of the folds
comptime CT_CH = 1024
#: words per ct_isum chunk record: rows^2, cols^2, cells^2 (Int64 each)
comptime CT_IREC = 6
#: words per float-float record: hi, lo (binary64 each)
comptime FF_REC = 4
#: ff_fin modes: 0 an entropy (-sum; 1.0 when X == 0, X = n), 1 the MI
#: (max(sum, 0); 0 when X != 0), 2 the sum
comptime FF_ENT = 0
comptime FF_MI = 1
comptime FF_SUM = 2

comptime _TWO32 = UInt64(0x41F0000000000000)
#: 2.220446049250313e-16 (2^-52), `_mi_from_contingency`'s epsilon
comptime _EPS52 = UInt64(0x3CB0000000000000)


@always_inline
def st_i64(f: FP, at: Int, v: Int):
    st64(f, at, UInt64(v))


@always_inline
def ld_i64(f: FP, at: Int) -> Int:
    return Int(ld64(f, at))


def i2f(i: Int) -> UInt64:
    """Float64(i) for 0 <= i < 2^62, correctly rounded (Python's int ->
    float): exact below 2^53, else the exact high part (i >> 32) * 2^32 plus
    the exact low 32 bits, one correctly rounded add."""
    if i < (1 << 53):
        return sf64_from_int(i)
    return sf64_add(sf64_mul(sf64_from_int(i >> 32), _TWO32), sf64_from_int(i & 0xFFFFFFFF))


@always_inline
def ff_add(mut hi: UInt64, mut lo: UInt64, x: UInt64):
    """hi + lo += x (finite values) by a two-sum, the error into lo."""
    var s = sf64_add(hi, x)
    var bb = sf64_sub(s, hi)
    var e = sf64_add(sf64_sub(hi, sf64_sub(s, bb)), sf64_sub(x, bb))
    lo = sf64_add(lo, e)
    hi = s


def cont_stats_unit(t: Int, f: FP, q: IP):
    """The caller's stage: the planner expands it (x_metrics/plan.mojo);
    run unplanned it writes nothing."""
    pass


def ct_cell_unit(t: Int, f: FP, q: IP):
    """q = [OFF, kb, kk, EPS, C, F, ka]; unit t < ka * kb, cell (i, j)."""
    var kb = p(q, 1)
    var ka = p(q, 6)
    if t >= ka * kb:
        return
    var i = t // kb
    var j = t - i * kb
    var off = p(q, 0) + i * p(q, 2) + j
    var v = ldi(f, off + 1) - ldi(f, off)
    st_i64(f, p(q, 4) + 2 * t, v)
    var F = p(q, 5)
    if F >= 0:
        st64(f, F + 2 * t, sf64_add(i2f(v), ld64(f, p(q, 3))))


def ct_row_unit(t: Int, f: FP, q: IP):
    """q = [C, ka, kb, R, SQ]; unit t < ka: R[t] = the row's sum, SQ[t] =
    its sum of squares (Int64)."""
    if t >= p(q, 1):
        return
    var C = p(q, 0)
    var kb = p(q, 2)
    var s = 0
    var sq = 0
    for j in range(kb):
        var v = ld_i64(f, C + 2 * (t * kb + j))
        s += v
        sq += v * v
    st_i64(f, p(q, 3) + 2 * t, s)
    st_i64(f, p(q, 4) + 2 * t, sq)


def ct_col_unit(t: Int, f: FP, q: IP):
    """q = [C, ka, kb, K]; unit t < kb: K[t] = the column's sum (Int64)."""
    var kb = p(q, 2)
    if t >= kb:
        return
    var C = p(q, 0)
    var s = 0
    for i in range(p(q, 1)):
        s += ld_i64(f, C + 2 * (i * kb + t))
    st_i64(f, p(q, 3) + 2 * t, s)


def ct_isum_unit(t: Int, f: FP, q: IP):
    """q = [R, K, SQ, ka, kb, S, CH, NC]; unit t < NC: over indices [t*CH,
    t*CH + CH), the sums of R[i]^2 (i < ka), K[j]^2 (j < kb) and SQ[i]
    (i < ka), Int64, at S + CT_IREC * t."""
    if t >= p(q, 7):
        return
    var R = p(q, 0)
    var K = p(q, 1)
    var SQ = p(q, 2)
    var ka = p(q, 3)
    var kb = p(q, 4)
    var CH = p(q, 6)
    var r2 = 0
    var c2 = 0
    var sq = 0
    for i in range(t * CH, t * CH + CH):
        if i < ka:
            var r = ld_i64(f, R + 2 * i)
            r2 += r * r
            sq += ld_i64(f, SQ + 2 * i)
        if i < kb:
            var c = ld_i64(f, K + 2 * i)
            c2 += c * c
    var S = p(q, 5) + CT_IREC * t
    st_i64(f, S, r2)
    st_i64(f, S + 2, c2)
    st_i64(f, S + 4, sq)


def ct_pairs_unit(t: Int, f: FP, q: IP):
    """q = [S, NC, n, P]; unit 0: P = [n^2 - c01 - c10 - sq, c01, c10, sq -
    n] (Int64), c01 = sum cols^2 - sq, c10 = sum rows^2 - sq, as
    contingency_stats formed them."""
    if t != 0:
        return
    var S = p(q, 0)
    var r2 = 0
    var c2 = 0
    var sq = 0
    for c in range(p(q, 1)):
        r2 += ld_i64(f, S + CT_IREC * c)
        c2 += ld_i64(f, S + CT_IREC * c + 2)
        sq += ld_i64(f, S + CT_IREC * c + 4)
    var n = p(q, 2)
    var c01 = c2 - sq
    var c10 = r2 - sq
    var P = p(q, 3)
    st_i64(f, P, n * n - c01 - c10 - sq)
    st_i64(f, P + 2, c01)
    st_i64(f, P + 4, c10)
    st_i64(f, P + 6, sq - n)


def ct_ent_unit(t: Int, f: FP, q: IP):
    """q = [R, K, ka, kb, n, T]; unit t < ka + kb: the entropy term of row
    t (t < ka) or column t - ka, (c / n) * (log c - log n), 0 for c == 0,
    as a float-float record at T + FF_REC * t."""
    var ka = p(q, 2)
    if t >= ka + p(q, 3):
        return
    var c = ld_i64(f, p(q, 0) + 2 * t) if t < ka else ld_i64(f, p(q, 1) + 2 * (t - ka))
    var n = p(q, 4)
    var v = SF64_ZERO
    if c != 0 and n > 0:
        var fn_ = i2f(n)
        var fc = i2f(c)
        v = sf64_mul(sf64_div(fc, fn_), sf64_sub(sf64_log(fc), sf64_log(fn_)))
    var T = p(q, 5) + FF_REC * t
    st64(f, T, v)
    st64(f, T + 2, SF64_ZERO)


def mi_cell_unit(t: Int, f: FP, q: IP):
    """q = [C, R, K, ka, kb, n, T]; unit t < ka * kb, cell (i, j) with
    count v > 0: nm = v / n, t = nm * (log v - log n) + nm * ((-log(pi pj)
    + log n) + log n), 0 when |t| < 2^-52; a float-float record at T +
    FF_REC * t (0 for v == 0)."""
    var ka = p(q, 3)
    var kb = p(q, 4)
    if t >= ka * kb:
        return
    var i = t // kb
    var j = t - i * kb
    var v = ld_i64(f, p(q, 0) + 2 * t)
    var n = p(q, 5)
    var term = SF64_ZERO
    if v > 0 and n > 0:
        var pi = ld_i64(f, p(q, 1) + 2 * i)
        var pj = ld_i64(f, p(q, 2) + 2 * j)
        var lt = sf64_log(i2f(n))
        var nm = sf64_div(i2f(v), i2f(n))
        var outer = sf64_add(sf64_add(sf64_neg(sf64_log(i2f(pi * pj))), lt), lt)
        term = sf64_add(sf64_mul(nm, sf64_sub(sf64_log(i2f(v)), lt)), sf64_mul(nm, outer))
        if sf64_lt(term & ~SF64_SIGN, _EPS52):
            term = SF64_ZERO
    var T = p(q, 6) + FF_REC * t
    st64(f, T, term)
    st64(f, T + 2, SF64_ZERO)


@always_inline
def _up_step(v: UInt64, x: Int, a: Int, b: Int, n: Int) -> UInt64:
    return sf64_div(sf64_mul(v, i2f((a - x) * (b - x))), i2f((x + 1) * (n - a - b + x + 1)))


@always_inline
def _down_step(v: UInt64, x: Int, a: Int, b: Int, n: Int) -> UInt64:
    return sf64_div(sf64_mul(v, i2f(x * (n - a - b + x))), i2f((a - x + 1) * (b - x + 1)))


@always_inline
def _emi_term(mut hi: UInt64, mut lo: UInt64, nij: Int, u: UInt64, z: UInt64, n: Int, la: UInt64,
              lb: UInt64):
    if nij < 1:
        return
    var pr = sf64_div(u, z)
    if is0(pr):
        return
    var qv = sf64_div(i2f(nij), i2f(n))
    var d = sf64_sub(sf64_sub(sf64_log(i2f(n * nij)), la), lb)
    ff_add(hi, lo, sf64_mul(sf64_mul(qv, d), pr))


def emi_cell_unit(t: Int, f: FP, q: IP):
    """q = [R, K, ka, kb, n, T]; unit t < ka * kb, cell (i, j), a = R[i], b
    = K[j]: lo = max(0, a + b - n), hi = min(a, b), mode = min(max((a + 1)
    (b + 1) // (n + 2), lo), hi); the ratio walks up and down from the mode
    (u = 1 at the mode; each walk stops at its first exact 0), z = their
    float-float sum (the up walk ascending, then the down walk descending);
    the terms over the same points in the same order, summed into the
    record at T + FF_REC * t. The walks are recomputed, not stored: the same
    operations give the same words."""
    var ka = p(q, 2)
    var kb = p(q, 3)
    if t >= ka * kb:
        return
    var i = t // kb
    var j = t - i * kb
    var a = ld_i64(f, p(q, 0) + 2 * i)
    var b = ld_i64(f, p(q, 1) + 2 * j)
    var n = p(q, 4)
    var hi = SF64_ZERO
    var lo = SF64_ZERO
    if a > 0 and b > 0 and n > 0:
        var lo_n = max(0, a + b - n)
        var hi_n = min(a, b)
        var mode = min(max(((a + 1) * (b + 1)) // (n + 2), lo_n), hi_n)
        # z: the pmf's normalizer
        var zh = SF64_ONE
        var zl = SF64_ZERO
        var x = mode
        var v = SF64_ONE
        while x < hi_n:
            v = _up_step(v, x, a, b, n)
            if is0(v):
                break
            ff_add(zh, zl, v)
            x += 1
        x = mode
        v = SF64_ONE
        while x > lo_n:
            v = _down_step(v, x, a, b, n)
            if is0(v):
                break
            ff_add(zh, zl, v)
            x -= 1
        var z = sf64_add(zh, zl)
        var la = sf64_log(i2f(a))
        var lb = sf64_log(i2f(b))
        # the terms, the same points in the same order
        _emi_term(hi, lo, mode, SF64_ONE, z, n, la, lb)
        x = mode
        v = SF64_ONE
        while x < hi_n:
            v = _up_step(v, x, a, b, n)
            if is0(v):
                break
            _emi_term(hi, lo, x + 1, v, z, n, la, lb)
            x += 1
        x = mode
        v = SF64_ONE
        while x > lo_n:
            v = _down_step(v, x, a, b, n)
            if is0(v):
                break
            _emi_term(hi, lo, x - 1, v, z, n, la, lb)
            x -= 1
    var T = p(q, 5) + FF_REC * t
    st64(f, T, hi)
    st64(f, T + 2, lo)


def ff_chunk_unit(t: Int, f: FP, q: IP):
    """q = [SRC, m, S, CH, NC]; unit t < NC: the records [t*CH, min(m,
    t*CH + CH)) of SRC met in ascending order (hi by a two-sum, the lows
    and errors into lo) into the record S + FF_REC * t."""
    if t >= p(q, 4):
        return
    var SRC = p(q, 0)
    var m = p(q, 1)
    var CH = p(q, 3)
    var hi = SF64_ZERO
    var lo = SF64_ZERO
    for k in range(t * CH, min(m, t * CH + CH)):
        ff_add(hi, lo, ld64(f, SRC + FF_REC * k))
        lo = sf64_add(lo, ld64(f, SRC + FF_REC * k + 2))
    var S = p(q, 2) + FF_REC * t
    st64(f, S, hi)
    st64(f, S + 2, lo)


def ff_fin_unit(t: Int, f: FP, q: IP):
    """q = [S, NC, OUT, MODE, X]; unit 0: the NC chunk records met in
    ascending order, s = hi + lo; OUT (binary64) = -s (FF_ENT; 1.0 when X
    == 0), max(s, 0) (FF_MI; +0 when X != 0) or s (FF_SUM)."""
    if t != 0:
        return
    var S = p(q, 0)
    var hi = SF64_ZERO
    var lo = SF64_ZERO
    for c in range(p(q, 1)):
        ff_add(hi, lo, ld64(f, S + FF_REC * c))
        lo = sf64_add(lo, ld64(f, S + FF_REC * c + 2))
    var s = sf64_add(hi, lo)
    var mode = p(q, 3)
    var X = p(q, 4)
    var v = s
    if mode == FF_ENT:
        v = SF64_ONE if X == 0 else sf64_neg(s)
    elif mode == FF_MI:
        v = s if (X == 0 and sf64_lt(SF64_ZERO, s)) else SF64_ZERO
    st64(f, p(q, 2), v)
