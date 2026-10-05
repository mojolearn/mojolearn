# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE fam2-prep-metrics (2026-10-04): THE IDENTICAL TILED GRAM of the
discriminant analyses, in the prep lane's program model (x_prep/common.mojo:
the device runs one thread per unit, the host column the same units, so the
four columns share one arithmetic and one order).

LDA's Z'Z (`matmul`, x_prep/prims.mojo) and QDA's class covariances
(`qda_cov`, naive_bayes/da.mojo) were one thread per OUTPUT CELL folding all
n rows: d^2 (or K d^2) threads each with an n-long dependent chain, every
cell computed twice (a,b and b,a) and, for QDA, X read once per class. Here
the rows are cut into BR-row blocks:

  gb_part / qcb_part   one thread per (block, a, b >= a): the block's rows
                       ascending from zero (matmul's `mm_step` chain, qda_cov's
                       centred product chain); qcb_part walks the block ONCE
                       for every class (the class sums in registers for
                       K <= GRAM_REG classes, else in the partial table)
  gb_fold / qcb_fold   one thread per output cell: the block partials of
                       (min(a,b), max(a,b)) ascending from zero, so the two
                       mirrored cells are the same word (qcb_fold then
                       divides by the class count, as qda_cov)
  gb_part_row          CANDIDATE schedule of gb_part: one thread per
                       (block, a) walks its rows once and advances every
                       b >= a (row-contiguous reads; the accumulators live in
                       the partial table). Each cell's chain is gb_part's, so
                       the words are gb_part's: a pure scheduling arm.

BITS: a new order for n > BR (blocked sums of products); at n <= BR one block,
the old chain. Products are rounded before their add (`mul`, DEVIATION 5401)
as before. All four columns run these units.

SWITCHES (IDENTICAL only):
  IDN_GRAM_BLOCKED, ON by default; -D MOJOLEARN_IDN_GRAM_BLOCKED_OFF (or
      MOJOLEARN_IDN_ALL_OFF) restores matmul / qda_cov.
  candidates, default OFF: -D MOJOLEARN_IDN_GRAM_ROWTILE (gb_part_row for the
      LDA Gram), -D MOJOLEARN_IDN_GRAM_ROWS_512 / _8192 (rows per block; the
      default is XB = 2048; a different block length is a different order).
The Python layer reads them from `x_prep_idn_fam2` and stages accordingly.
"""
from std.sys.compile import is_defined
from checks.numerics import ftz, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, st, raw, RUN, run_block
from x_prep.prims import add, sub, mul, div, mm_step
from x_prep.blocked import XB

comptime IDN_GRAM_BLOCKED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_GRAM_BLOCKED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime IDN_GRAM_ROWTILE = IDN_GRAM_BLOCKED and is_defined["MOJOLEARN_IDN_GRAM_ROWTILE"]()
comptime IDN_GRAM_ROWS = (
    512 if is_defined["MOJOLEARN_IDN_GRAM_ROWS_512"]() else (8192 if is_defined["MOJOLEARN_IDN_GRAM_ROWS_8192"]() else XB)
)
#: classes qcb_part keeps in registers (more: the partial table itself)
comptime GRAM_REG = 8


def gb_part_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, P, nb, BR]; t = (blk*d + a)*d + b. For b >= a:
    P[t] = sum over the block's rows i (ascending, from zero) of
    Z[i*d + a] * Z[i*d + b], each product rounded before its add. b < a:
    nothing (gb_fold reads the mirrored cell)."""
    var d = p(q, 2)
    var b = t % d
    var a = (t // d) % d
    if b < a:
        return
    var blk = t // (d * d)
    var n = p(q, 1)
    var br = p(q, 5)
    var lo = blk * br
    var hi = min(n, lo + br)
    var Z = p(q, 0)
    var acc = Float32(0)
    var full = lo + ((hi - lo) - (hi - lo) % RUN)
    for i0 in range(lo, full, RUN):
        var ba = run_block[RUN](f, Z + i0 * d + a, d)
        var bb = run_block[RUN](f, Z + i0 * d + b, d)
        comptime for u in range(RUN):
            acc = mm_step(acc, ftz(ba[u]), ftz(bb[u]))
    for i in range(full, hi):
        acc = mm_step(acc, ld(f, Z + i * d + a), ld(f, Z + i * d + b))
    st(f, p(q, 3) + t, acc)


def gb_part_row_unit(t: Int, f: FP, q: IP):
    """q = [Z, n, d, P, nb, BR]; t = blk*d + a (the candidate schedule):
    every P[(blk*d + a)*d + b], b >= a, from one walk of the block's rows;
    each cell's chain is gb_part's."""
    var d = p(q, 2)
    var a = t % d
    var blk = t // d
    var n = p(q, 1)
    var br = p(q, 5)
    var lo = blk * br
    var hi = min(n, lo + br)
    var Z = p(q, 0)
    var base = p(q, 3) + t * d
    for b in range(a, d):
        f.unsafe_store(base + b, Float32(0))
    for i in range(lo, hi):
        var za = ld(f, Z + i * d + a)
        for b in range(a, d):
            st(f, base + b, mm_step(raw(f, base + b), za, ld(f, Z + i * d + b)))


def gb_fold_unit(t: Int, f: FP, q: IP):
    """q = [P, nb, d, G]; t = a*d + b: G[t] = the block partials of
    (min(a,b), max(a,b)) added ascending from zero."""
    var nb = p(q, 1)
    var d = p(q, 2)
    var a = t // d
    var b = t % d
    var lo = min(a, b)
    var hi = max(a, b)
    var P = p(q, 0)
    var acc = Float32(0)
    for blk in range(nb):
        acc = add(acc, ld(f, P + (blk * d + lo) * d + hi))
    st(f, p(q, 3) + t, acc)


def qcb_part_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, Y, K, MEAN, P, nb, BR]; t = (blk*d + a)*d + b. For
    b >= a, every class k from one walk of the block's rows, ascending:
    P[((blk*K + k)*d + a)*d + b] = the sum from zero of
    (X[i,a] - MEAN[k,a]) * (X[i,b] - MEAN[k,b]) over the block's rows of class
    k (qda_cov's term). A code outside [0, K) is skipped."""
    var d = p(q, 2)
    var b = t % d
    var a = (t // d) % d
    if b < a:
        return
    var blk = t // (d * d)
    var X = p(q, 0)
    var n = p(q, 1)
    var Y = p(q, 3)
    var K = p(q, 4)
    var MEAN = p(q, 5)
    var br = p(q, 8)
    var lo = blk * br
    var hi = min(n, lo + br)
    var base = p(q, 6) + (blk * K * d + a) * d + b
    var kstep = d * d
    if K <= GRAM_REG:
        var ma = SIMD[DType.float32, GRAM_REG](0)
        var mb = SIMD[DType.float32, GRAM_REG](0)
        var s = SIMD[DType.float32, GRAM_REG](0)
        comptime for u in range(GRAM_REG):
            if u < K:
                ma[u] = ld(f, MEAN + u * d + a)
                mb[u] = ld(f, MEAN + u * d + b)
        for i in range(lo, hi):
            var k = Int(ld(f, Y + i))
            var xa = ld(f, X + i * d + a)
            var xb = ld(f, X + i * d + b)
            comptime for u in range(GRAM_REG):
                if k == u:
                    s[u] = add(s[u], mul(sub(xa, ma[u]), sub(xb, mb[u])))
        comptime for u in range(GRAM_REG):
            if u < K:
                st(f, base + u * kstep, s[u])
    else:
        for k in range(K):
            f.unsafe_store(base + k * kstep, Float32(0))
        for i in range(lo, hi):
            var k = Int(ld(f, Y + i))
            if k < 0 or k >= K:
                continue
            var ea = sub(ld(f, X + i * d + a), ld(f, MEAN + k * d + a))
            var eb = sub(ld(f, X + i * d + b), ld(f, MEAN + k * d + b))
            st(f, base + k * kstep, add(raw(f, base + k * kstep), mul(ea, eb)))


def qcb_fold_unit(t: Int, f: FP, q: IP):
    """q = [P, nb, K, d, CNT, COV]; t = (k*d + a)*d + b: COV[t] = the block
    partials of class k's (min(a,b), max(a,b)) added ascending from zero,
    over CNT[k] (qda_cov's divisor)."""
    var nb = p(q, 1)
    var K = p(q, 2)
    var d = p(q, 3)
    var b = t % d
    var a = (t // d) % d
    var k = t // (d * d)
    var lo = min(a, b)
    var hi = max(a, b)
    var P = p(q, 0)
    var acc = Float32(0)
    for blk in range(nb):
        acc = add(acc, ld(f, P + ((blk * K + k) * d + lo) * d + hi))
    st(f, p(q, 5) + t, div(acc, ld(f, p(q, 4) + k)))
