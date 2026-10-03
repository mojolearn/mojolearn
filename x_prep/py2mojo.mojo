# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE apple-fast-py2mojo-prep (2026-10-03): the data-path loops that ran in
python/mojolearn/_expansion_prep.py, as x_prep units (x_prep/common.mojo's
program model: one GPU thread per unit, the host binding the same units in a
loop). Every unit here moves words or counts integers, so the device, the
host column and the Python loops it replaces give the same bits by
construction.

  p2m_ccount / p2m_cscan / p2m_cstart / p2m_cwrite
      a stable counting sort of rows by class code, chunk parallel: the
      class counts and the rows grouped by class, ascending within a class
      (LDA / QDA `covariance_estimator`'s per-class rows, `_class_counts`).
  p2m_rgather
      rows of a row-major block by an int32 row list (`_gather_rows`).
  p2m_smrows
      draw k of `splitmix64(seed)` mod n, every draw on its own thread
      (state k = seed + (k + 1) * golden, the stream's closed form):
      KBinsDiscretizer's with-replacement subsample rows.
  p2m_sel_count / p2m_sel_scan / p2m_sel_write
      per-column stable compaction, chunk parallel: the non-NaN words of
      each column (SimpleImputer strategy=<callable>), or the rows whose
      word is positive (the nonzero-weight rows of KBins / Spline).
  p2m_transpose
      a row-major (n, w) block written column-major (SplineTransformer
      order='F').

Ops P2M_BASE .. P2M_BASE + P2M_N - 1 are a separate range, outside the
0 .. N_OPS - 1 table other lanes grow; built with
-D MOJOLEARN_PY2MOJO_prep_OFF there are none (P2M_N = 0), the binding does not
export `x_prep_py2mojo`, and Python takes its old loops (the A/B arm A).
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from x_prep.common import FP, IP, p, raw, ldi, sti, ld

comptime PY2MOJO_PREP = not is_defined["MOJOLEARN_PY2MOJO_prep_OFF"]()
comptime P2M_BASE = 200
comptime P2M_N = 10 if PY2MOJO_PREP else 0


@always_inline
def is_p2m_op(op: Int) -> Bool:
    return op >= P2M_BASE and op < P2M_BASE + P2M_N


@always_inline
def _bits(f: FP, i: Int) -> UInt32:
    return f.bitcast[UInt32]().unsafe_load(i)


# ---------------------------------------------------------------- counting sort by class
def p2m_ccount_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, CH, CNT]; t = chunk of CH rows. CODES are float
    class values; a code in [0, K) adds one to the int32 count CNT[t*K + code]
    (CNT arrives zeroed); any other code is skipped."""
    var n = p(q, 1)
    var K = p(q, 2)
    var ch = p(q, 3)
    var base = p(q, 4) + t * K
    var lo = t * ch
    var hi = min(lo + ch, n)
    for i in range(lo, hi):
        var k = Int(ld(f, p(q, 0) + i))
        if k >= 0 and k < K:
            sti(f, base + k, ldi(f, base + k) + 1)


def p2m_cscan_unit(t: Int, f: FP, q: IP):
    """q = [CNT, nch, K, TOT]; t = class k: CNT[c*K + k] becomes the
    exclusive prefix over chunks c (in place) and TOT[k] the class count,
    int32 bits."""
    var nch = p(q, 1)
    var K = p(q, 2)
    var s = 0
    for c in range(nch):
        var at = p(q, 0) + c * K + t
        var v = ldi(f, at)
        sti(f, at, s)
        s += v
    sti(f, p(q, 3) + t, s)


def p2m_cstart_unit(t: Int, f: FP, q: IP):
    """q = [TOT, K, START, FTOT]; one unit over the K class counts (O(K)):
    START[k] = the exclusive prefix (int32 bits), FTOT[k] = the count as a
    float (FTOT < 0: not written)."""
    var K = p(q, 1)
    var s = 0
    for k in range(K):
        var v = ldi(f, p(q, 0) + k)
        sti(f, p(q, 2) + k, s)
        if p(q, 3) >= 0:
            f.unsafe_store(p(q, 3) + k, Float32(v))
        s += v


def p2m_cwrite_unit(t: Int, f: FP, q: IP):
    """q = [CODES, n, K, CH, CNT, START, ROWS]; t = chunk: each row i of the
    chunk, ascending, with code k in [0, K) goes to ROWS[START[k] + CNT[t*K + k]]
    (int32 bits) and bumps that chunk-owned offset. Over every chunk: the
    rows grouped by class, ascending within each class."""
    var n = p(q, 1)
    var K = p(q, 2)
    var ch = p(q, 3)
    var cb = p(q, 4) + t * K
    var lo = t * ch
    var hi = min(lo + ch, n)
    for i in range(lo, hi):
        var k = Int(ld(f, p(q, 0) + i))
        if k >= 0 and k < K:
            var o = ldi(f, cb + k)
            sti(f, p(q, 6) + ldi(f, p(q, 5) + k) + o, i)
            sti(f, cb + k, o + 1)


# ---------------------------------------------------------------- row gather
def p2m_rgather_unit(t: Int, f: FP, q: IP):
    """q = [X, d, ROWS, DST]; t = r*d + c: DST[t] = X[ROWS[r], c], the word as
    it is (ROWS int32 bits)."""
    var d = p(q, 1)
    var r = t // d
    var c = t - r * d
    var i = ldi(f, p(q, 2) + r)
    f.unsafe_store(p(q, 3) + t, raw(f, p(q, 0) + i * d + c))


# ---------------------------------------------------------------- splitmix64 draws
comptime _GOLDEN = UInt64(0x9E3779B97F4A7C15)


@always_inline
def splitmix_at(seed: UInt64, k: Int) -> UInt64:
    """The (k+1)-th output of python/mojolearn/_expansion_prep.py
    `_splitmix64` from state `seed`: its state after k+1 steps is
    seed + (k+1) * golden (mod 2**64), then the mix."""
    var z = seed + UInt64(k + 1) * _GOLDEN
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def p2m_smrows_unit(t: Int, f: FP, q: IP):
    """q = [SEED, n, ROWS]; t = draw k: ROWS[k] = splitmix64 output k mod n
    (int32 bits). SEED = two int32 words, the low and high halves."""
    var s = (UInt64(_bits(f, p(q, 0) + 1)) << 32) | UInt64(_bits(f, p(q, 0)))
    var z = splitmix_at(s, t)
    sti(f, p(q, 2) + t, Int(z % UInt64(p(q, 1))))


# ---------------------------------------------------------------- per-column compaction
@always_inline
def _sel(b: UInt32, mode: Int) -> Bool:
    """MODE 0: the word is not NaN; MODE 1 (rows): positive (int32 bits above
    zero and not NaN, so +0.0, -0.0 and negatives fail and a subnormal
    passes, as Python's `v > 0` on the float32 value). Integer tests: no
    flush can move them."""
    var mag = b & UInt32(0x7FFFFFFF)
    if mode == 0:
        return mag <= UInt32(0x7F800000)
    return (b & UInt32(0x80000000)) == UInt32(0) and b != UInt32(0) and mag <= UInt32(0x7F800000)


def p2m_sel_count_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, CH, nch, MODE, CNT]; t = j*nch + c: how many rows of
    chunk c of column j are selected (`_sel`), int32 bits at CNT[t]."""
    var n = p(q, 1)
    var d = p(q, 2)
    var ch = p(q, 3)
    var nch = p(q, 4)
    var j = t // nch
    var c = t - j * nch
    var lo = c * ch
    var hi = min(lo + ch, n)
    var k = 0
    for i in range(lo, hi):
        if _sel(_bits(f, p(q, 0) + i * d + j), p(q, 5)):
            k += 1
    sti(f, p(q, 6) + t, k)


def p2m_sel_scan_unit(t: Int, f: FP, q: IP):
    """q = [CNT, nch, TOT]; t = column j: CNT[j*nch + c] becomes the exclusive
    prefix over its chunks (in place), TOT[j] the column's count (int32 bits)."""
    var nch = p(q, 1)
    var s = 0
    for c in range(nch):
        var at = p(q, 0) + t * nch + c
        var v = ldi(f, at)
        sti(f, at, s)
        s += v
    sti(f, p(q, 2) + t, s)


def p2m_sel_write_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, CH, nch, MODE, CNT, DST]; t = j*nch + c: the selected rows
    of chunk c of column j, ascending, at DST[j*n + CNT[t] ..]: MODE 0 the
    words as they are, MODE 1 the row indices (int32 bits)."""
    var n = p(q, 1)
    var d = p(q, 2)
    var ch = p(q, 3)
    var nch = p(q, 4)
    var mode = p(q, 5)
    var j = t // nch
    var c = t - j * nch
    var lo = c * ch
    var hi = min(lo + ch, n)
    var k = p(q, 7) + j * n + ldi(f, p(q, 6) + t)
    for i in range(lo, hi):
        var b = _bits(f, p(q, 0) + i * d + j)
        if _sel(b, mode):
            if mode == 0:
                f.bitcast[UInt32]().unsafe_store(k, b)
            else:
                sti(f, k, i)
            k += 1


# ---------------------------------------------------------------- column-major copy
def p2m_transpose_unit(t: Int, f: FP, q: IP):
    """q = [SRC, n, w, DST]; t = j*n + i: DST[t] = SRC[i*w + j], the word as it is."""
    var n = p(q, 1)
    var j = t // n
    var i = t - j * n
    f.unsafe_store(p(q, 3) + t, raw(f, p(q, 0) + i * p(q, 2) + j))


@always_inline
def run_p2m_unit[OP: Int](t: Int, f: FP, q: IP):
    comptime if PY2MOJO_PREP:
        comptime if OP == P2M_BASE + 0:
            p2m_ccount_unit(t, f, q)
        comptime if OP == P2M_BASE + 1:
            p2m_cscan_unit(t, f, q)
        comptime if OP == P2M_BASE + 2:
            p2m_cstart_unit(t, f, q)
        comptime if OP == P2M_BASE + 3:
            p2m_cwrite_unit(t, f, q)
        comptime if OP == P2M_BASE + 4:
            p2m_rgather_unit(t, f, q)
        comptime if OP == P2M_BASE + 5:
            p2m_smrows_unit(t, f, q)
        comptime if OP == P2M_BASE + 6:
            p2m_sel_count_unit(t, f, q)
        comptime if OP == P2M_BASE + 7:
            p2m_sel_scan_unit(t, f, q)
        comptime if OP == P2M_BASE + 8:
            p2m_sel_write_unit(t, f, q)
        comptime if OP == P2M_BASE + 9:
            p2m_transpose_unit(t, f, q)
