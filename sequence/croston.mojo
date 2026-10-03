# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Croston's intermittent-demand forecasts, statsforecast's
(`python/statsforecast/models.py` `_croston_classic`, `_croston_optimized`,
`_croston_sba`; `src/ses.cpp` `ses_forecast`, `ses_sse`,
`golden_section_ses`): simple exponential smoothing of the positive demands
and of the intervals between non-zero observations (alpha 0.1, or the
golden-section optimum on [0.1, 0.3]), their ratio, times 0.95 for SBA;
a series with no positive demand forecasts its last value (`_naive`).
Float32, one series per element.

Difference: the golden section also stops after 200 iterations (the
reference's |b - a| >= 1e-12 is below float32 resolution near alpha, so the
float32 loop ends at the fc == fd exit or the cap)."""
from sequence.ops import FP, Args, add, fma3, ld, mul, st, sub
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_sqrt
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: lane/apple-fast-seq (2026-10-02). `op_croston` compacts the positive
#: demands and the intervals into the thread's device scratch (2 n floats
#: written, `sequence/croston.mojo` op_croston's first loop) and
#: `ses_forecast` reads them back, on a kernel of one thread per series
#: (64 series on taxi-hourly): every element is a round trip to device
#: memory with nothing to hide the latency behind. The classic and SBA
#: variants read each compacted value exactly once, in order, and the
#: smoothing is a running recurrence, so `-D MOJOLEARN_SEQ_CROSTON_REG=1`
#: folds the smoothing into the scan over y: the same operations in the
#: same order, nothing stored. The optimized variant (golden section,
#: repeated passes) keeps the stored path. FAST only; IDENTICAL compiles
#: the stored code. Default on FAST + Apple since the M3 A/B
#: (lane/apple-fast-seq 8b3f1d90e, n=1, quality identical): croston taxi-hourly
#: 3.0 -> 2.4 ms, synthetic 2.7 -> 1.8 ms. -D MOJOLEARN_SEQ_CROSTON_REG_OFF
#: restores the stored code; the old -D MOJOLEARN_SEQ_CROSTON_REG=1 is harmless.
comptime CROSTON_REG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEQ_CROSTON_REG_OFF"]()
)

comptime CROSTON_CLASSIC = 0
comptime CROSTON_OPTIMIZED = 1
comptime CROSTON_SBA = 2


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def ses_forecast(x: FP, n: Int, alpha: Float32) -> Float32:
    if n == 1:
        return ld(x, 0)
    var c = sub(Float32(1.0), alpha)
    var f = ld(x, 0)
    for i in range(1, n):
        f = fma3(alpha, ld(x, i - 1), mul(c, f))
    return fma3(alpha, ld(x, n - 1), mul(c, f))


def ses_sse(alpha: Float32, x: FP, n: Int) -> Float32:
    if n < 2:
        return Float32(0.0)
    var c = sub(Float32(1.0), alpha)
    var f = ld(x, 0)
    var sse = Float32(0.0)
    for i in range(1, n):
        f = fma3(alpha, ld(x, i - 1), mul(c, f))
        var e = sub(ld(x, i), f)
        sse = fma3(e, e, sse)
    return sse


def golden_section_ses(x: FP, n: Int, lower: Float32, upper: Float32) -> Float32:
    var gr = div(add(ftz(identical_sqrt(Float32(5.0))), Float32(1.0)), Float32(2.0))
    var a = lower
    var b = upper
    var c = sub(b, div(sub(b, a), gr))
    var d = add(a, div(sub(b, a), gr))
    var fc = ses_sse(c, x, n)
    var fd = ses_sse(d, x, n)
    var it = 0
    while abs(sub(b, a)) >= Float32(1e-12) and it < 200:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = sub(b, div(sub(b, a), gr))
            fc = ses_sse(c, x, n)
        elif fd < fc:
            a = c
            c = d
            fc = fd
            d = add(a, div(sub(b, a), gr))
            fd = ses_sse(d, x, n)
        else:
            break
        it += 1
    return div(add(b, a), Float32(2.0))


@always_inline
def _croston_reg(t: Int, a: Args):
    """`op_croston` for the classic and SBA variants with the two
    exponential smoothings folded into the scan over y
    (MOJOLEARN_SEQ_CROSTON_REG): `ses_forecast`'s step at the k-th compacted
    value is f = alpha x[k - 1] + (1 - alpha) f, so each smoothing keeps its
    state and the previous value in registers; the same operations in the
    same order as the compacted arrays, none stored."""
    var n = a.i0
    var y = a.p0 + t * n
    var alpha = Float32(0.1)
    var c = sub(Float32(1.0), alpha)
    var nd = 0
    var ni = 0
    var prev = 0
    var fd = Float32(0.0)     # ses state over the demands
    var xd = Float32(0.0)     # the last demand
    var fi = Float32(0.0)     # ses state over the intervals
    var xi = Float32(0.0)     # the last interval
    for i in range(n):
        var v = ld(y, i)
        if v > Float32(0.0):
            if nd == 0:
                fd = v
            else:
                fd = fma3(alpha, xd, mul(c, fd))
            xd = v
            nd += 1
        if v != Float32(0.0):
            var w = Float32(i + 1 - prev)
            if ni == 0:
                fi = w
            else:
                fi = fma3(alpha, xi, mul(c, fi))
            xi = w
            prev = i + 1
            ni += 1
    var mean: Float32
    if nd == 0:
        mean = ld(y, n - 1)
    else:
        var ydp = xd if nd == 1 else fma3(alpha, xd, mul(c, fd))
        var yip = xi if ni == 1 else fma3(alpha, xi, mul(c, fi))
        mean = div(ydp, yip) if yip != Float32(0.0) else ydp
        if a.i1 == CROSTON_SBA:
            mean = mul(mean, Float32(0.95))
    st(a.p1, t, mean)


def op_croston(t: Int, a: Args):
    """Series t: p0 y [B, n]; p1 mean [B] out; p2 scratch [B, 2 n].
    i0 n, i1 variant (0 classic, 1 optimized, 2 SBA)."""
    comptime if CROSTON_REG:
        if a.i1 != CROSTON_OPTIMIZED:
            _croston_reg(t, a)
            return
    var n = a.i0
    var y = a.p0 + t * n
    var dem = a.p2 + t * 2 * n
    var itv = dem + n
    var nd = 0
    var ni = 0
    var prev = 0
    for i in range(n):
        var v = ld(y, i)
        if v > Float32(0.0):
            st(dem, nd, v)
            nd += 1
        if v != Float32(0.0):
            st(itv, ni, Float32(i + 1 - prev))
            prev = i + 1
            ni += 1
    var mean: Float32
    if nd == 0:
        mean = ld(y, n - 1)
    else:
        var ad = Float32(0.1)
        var ai = Float32(0.1)
        if a.i1 == CROSTON_OPTIMIZED:
            ad = golden_section_ses(dem, nd, Float32(0.1), Float32(0.3))
            ai = golden_section_ses(itv, ni, Float32(0.1), Float32(0.3))
        var ydp = ses_forecast(dem, nd, ad)
        var yip = ses_forecast(itv, ni, ai)
        mean = div(ydp, yip) if yip != Float32(0.0) else ydp
        if a.i1 == CROSTON_SBA:
            mean = mul(mean, Float32(0.95))
    st(a.p1, t, mean)
