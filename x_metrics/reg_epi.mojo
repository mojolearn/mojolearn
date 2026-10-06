# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE cpu2-l7-metrics (2026-10-04): op 60 `reg_epi`, THE REGRESSION
PER-OUTPUT TAILS ON THE DEVICE.

The regression metrics (python/mojolearn/_expansion_metrics.py, `_Reg`) read
their per-output Float32 PairSums back and finished them in Python binary64
(DEVIATION 6106): the means (sum / weight total), the roots, the percentile
NaN flags, the Float32 broadcast means the next program uploaded, the
r2 / explained variance / d2 assembly and the multioutput averages; the
Tweedie domain took an O(n) host min. Here the same binary64 operations, in
the same order, run in the program that made the sums, as integer arithmetic
on the binary64 encoding (checks/soft_f64.mojo: correctly rounded + - * /
sqrt, what a hardware double returns), so the Apple GPU (no float64), NVIDIA,
AMD and the host column write the word Python wrote. Binary64 values are two
arena words, low first (x_metrics/tail.mojo st64 / ld64 = array('d')).

KIND (q[0]) selects the tail; the per-output kinds are maps (one unit per
output t < D), AVG is one unit (t = 0) folding D values in ascending order:

- MEAN q = [0, D, SUM, SW, n, DST, D32, MODE]: v = SUM[t] widened; MODE & 1:
  v / (SW's Float32 total widened, or n when SW < 0); MODE & 2: sqrt(v).
  DST[2t] = v; D32 >= 0: D32[t] = the Float32 nearest v (round to nearest even).
- PCT q = [1, D, OUT, CDF, n, DST, D32]: the `wpercentile` value OUT[t]
  widened, NaN when its flag CDF[t*n] is -1 (an all-zero weight column);
  D32 as MEAN.
- ASM q = [2, D, NUM, DEN, FF, DST]: the score 1 - NUM/DEN of `_assemble`
  (force_finite FF: 1.0 for 0/0, 0.0 for x/0; else NaN for 0/0, -inf for x/0).
- AVG q = [3, D, SRC, MODE, W, DST, NANRULE]: MODE 1 the plain mean
  (s += v, then s / D), 2 weighted by the D binary64 words at W (s += v * w,
  sw += w, s / sw), 3 weighted by W unless every W is zero (then the plain
  mean). NANRULE: a NaN score, or an infinite one with weight zero, answers
  NaN (the `_assemble` rule).
- SIGN q = [4, count, SRC, FLAGS]: one unit per value; the raw Float32 word
  (no flush, as numpy's min saw it): v < 0 stores 1 at FLAGS[0], v <= 0
  stores 1 at FLAGS[1] (the Tweedie domain).

ON in both numeric modes, the only route (the Python tails left the GPU route).
"""
from checks.soft_f64 import (
    SF64_NAN, SF64_ZERO, SF64_ONE, SF64_INF, SF64_SIGN, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_sqrt,
    sf64_is_nan, sf64_from_int, sf64_from_f32, sf64_to_f32,
)
from x_metrics.common import FP, IP, p, ld, ldi, sti
from x_metrics.tail import st64, ld64, is0, abs64
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

# AFCL-P08: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified.
# Two software-binary64 chains can overlap long integer-emulation dependencies
# in multioutput score averaging. This is an epilogue experiment, not a new
# per-row metric fold. Preserve every weight and NANRULE before final division.
comptime AFCL_P08 = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_AFCL_P08"]()

comptime OP_REG_EPI = 60

comptime KIND_MEAN = 0
comptime KIND_PCT = 1
comptime KIND_ASM = 2
comptime KIND_AVG = 3
comptime KIND_SIGN = 4

comptime MEAN_DIV = 1
comptime MEAN_ROOT = 2

comptime AVG_UNIFORM = 1
comptime AVG_CUSTOM = 2
comptime AVG_VARIANCE = 3


@always_inline
def _is_inf(v: UInt64) -> Bool:
    return abs64(v) == SF64_INF


@always_inline
def _put(f: FP, t: Int, dst: Int, d32: Int, v: UInt64):
    st64(f, dst + 2 * t, v)
    if d32 >= 0:
        f.unsafe_store(d32 + t, sf64_to_f32(v))


def _mean(t: Int, f: FP, q: IP):
    var sums = p(q, 2)
    var sw = p(q, 3)
    var mode = p(q, 7)
    var v = sf64_from_f32(ld(f, sums + t))
    if (mode & MEAN_DIV) != 0:
        var den = sf64_from_f32(ld(f, sw)) if sw >= 0 else sf64_from_int(p(q, 4))
        v = sf64_div(v, den)
    if (mode & MEAN_ROOT) != 0:
        v = sf64_sqrt(v)
    _put(f, t, p(q, 5), p(q, 6), v)


def _pct(t: Int, f: FP, q: IP):
    var n = p(q, 4)
    var v = SF64_NAN
    if ldi(f, p(q, 3) + t * n) != -1:
        v = sf64_from_f32(ld(f, p(q, 2) + t))
    _put(f, t, p(q, 5), p(q, 6), v)


def _asm(t: Int, f: FP, q: IP):
    var a = ld64(f, p(q, 2) + 2 * t)
    var b = ld64(f, p(q, 3) + 2 * t)
    var ff = p(q, 4) != 0
    var v: UInt64
    if not ff:
        if not is0(b):
            v = sf64_sub(SF64_ONE, sf64_div(a, b))
        elif is0(a):
            v = SF64_NAN
        else:
            v = SF64_INF | SF64_SIGN
    elif not is0(b) and not is0(a):
        v = sf64_sub(SF64_ONE, sf64_div(a, b))
    elif not is0(a):
        v = SF64_ZERO
    else:
        v = SF64_ONE
    st64(f, p(q, 5) + 2 * t, v)


def _avg(f: FP, q: IP):
    var D = p(q, 1)
    var src = p(q, 2)
    var mode = p(q, 3)
    var w = p(q, 4)
    var dst = p(q, 5)
    var nan_rule = p(q, 6) != 0
    var weighted = mode == AVG_CUSTOM
    if mode == AVG_VARIANCE:
        for i in range(D):
            if not is0(ld64(f, w + 2 * i)):
                weighted = True
    var s = SF64_ZERO
    var s_odd = SF64_ZERO
    if not weighted:
        for i in range(D):
            comptime if AFCL_P08:
                if i % 2 == 0:
                    s = sf64_add(s, ld64(f, src + 2 * i))
                else:
                    s_odd = sf64_add(s_odd, ld64(f, src + 2 * i))
            else:
                s = sf64_add(s, ld64(f, src + 2 * i))
        comptime if AFCL_P08:
            s = sf64_add(s, s_odd)
        st64(f, dst, sf64_div(s, sf64_from_int(D)))
        return
    var sw = SF64_ZERO
    var sw_odd = SF64_ZERO
    for i in range(D):
        var v = ld64(f, src + 2 * i)
        var wi = ld64(f, w + 2 * i)
        if nan_rule and (sf64_is_nan(v) or (is0(wi) and _is_inf(v))):
            st64(f, dst, SF64_NAN)
            return
        comptime if AFCL_P08:
            if i % 2 == 0:
                s = sf64_add(s, sf64_mul(v, wi))
                sw = sf64_add(sw, wi)
            else:
                s_odd = sf64_add(s_odd, sf64_mul(v, wi))
                sw_odd = sf64_add(sw_odd, wi)
        else:
            s = sf64_add(s, sf64_mul(v, wi))
            sw = sf64_add(sw, wi)
    comptime if AFCL_P08:
        s = sf64_add(s, s_odd)
        sw = sf64_add(sw, sw_odd)
    st64(f, dst, sf64_div(s, sw))


def _sign(t: Int, f: FP, q: IP):
    var v = f.unsafe_load(p(q, 2) + t)
    var flags = p(q, 3)
    if v < Float32(0):
        sti(f, flags, 1)
    if v <= Float32(0):
        sti(f, flags + 1, 1)


def reg_epi_unit(t: Int, f: FP, q: IP):
    """Unit t of one reg_epi stage (see the module docstring for q)."""
    var kind = p(q, 0)
    var count = p(q, 1)
    if kind == KIND_AVG:
        if t == 0:
            _avg(f, q)
        return
    if t >= count:
        return
    if kind == KIND_MEAN:
        _mean(t, f, q)
    elif kind == KIND_PCT:
        _pct(t, f, q)
    elif kind == KIND_ASM:
        _asm(t, f, q)
    elif kind == KIND_SIGN:
        _sign(t, f, q)
