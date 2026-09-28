# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL float32 seams, W lanes at a time, for the CPU host paths (lane
neighbors-cpu, 2026-09-28).

HOST ONLY. Each function here is the lane-wise spelling of one scalar seam
of `checks/numerics.mojo` and returns, in every lane, the word that seam
returns for that lane's input:

  `ftz_v`        `ftz`: a zero exponent field keeps only the sign bit.
  `twiddle_v`    the order-preserving key of `select_radix`'s `twiddle_in`
                 (`core/knn_host_predict.mojo::host_twiddle_in`).
  `expf_v`       `portable_expf`: the same fma chain, the same two-step
                 power-of-two scaling, the same range and underflow
                 branches as selects.
  `logf_v`       `portable_logf`: the same reduction to [sqrt(1/2),
                 sqrt(2)) and the same Cephes core.

A vector lane performs the operations the scalar function performs, in its
order, each IEEE-exact at its rounding (fma, add, mul, floor, conversions
of in-range integers), so the lane's result IS the scalar result; the
branches become selects over lanes whose other arm is discarded (an
out-of-range lane is clamped before the arithmetic so no lane computes on
an undefined conversion). THE MEASUREMENT is `core/host_simd_identical_
check.mojo` (`pixi run check-host-simd-identical`): every one of the 2^32
float32 bit patterns through `expf_v`, `logf_v` and `ftz_v` against the
scalar seam, bit for bit, plus a sabotage arm (`-D
MOJOLEARN_HOST_SIMD_SABOTAGE`: one polynomial coefficient of `expf_v` moved
one unit) that must FAIL.

This is not a numeric row: it changes how many cells one instruction
computes, never what a cell computes.
"""
from std.math import floor
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime HOST_SIMD_SABOTAGE = is_defined["MOJOLEARN_HOST_SIMD_SABOTAGE"]()


@always_inline
def ftz_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var b = bitcast[DType.uint32, w](x)
        var sub = (b & SIMD[DType.uint32, w](0x7F800000)).eq(SIMD[DType.uint32, w](0))
        return bitcast[DType.float32, w](
            sub.select(b & SIMD[DType.uint32, w](0x80000000), b)
        )
    return x


@always_inline
def twiddle_v[w: Int](v: SIMD[DType.float32, w]) -> SIMD[DType.uint32, w]:
    """`twiddle_in(value, select_min=True)`, lane by lane: the unsigned
    order of the result is the float order of the input (-0 below +0)."""
    var b = bitcast[DType.uint32, w](v)
    var neg = (b & SIMD[DType.uint32, w](0x80000000)).ne(SIMD[DType.uint32, w](0))
    return neg.select(b ^ SIMD[DType.uint32, w](0xFFFFFFFF), b ^ SIMD[DType.uint32, w](0x80000000))


@always_inline
def untwiddle(bits: UInt32) -> Float32:
    """The inverse of `twiddle_v` for one key."""
    if (bits & UInt32(0x80000000)) != 0:
        return bitcast[DType.float32](bits ^ UInt32(0x80000000))
    return bitcast[DType.float32](bits ^ UInt32(0xFFFFFFFF))


@always_inline
def isnan_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.bool, w]:
    """`x != x`, as an integer test on the bits (an all-ones exponent and a
    nonzero mantissa). MEASURED: `SIMD.ne(x, x)` reads False on a NaN lane
    (an ordered compare), so it is not the scalar `x != x`."""
    var b = bitcast[DType.uint32, w](x) & SIMD[DType.uint32, w](0x7FFFFFFF)
    return b.gt(SIMD[DType.uint32, w](0x7F800000))


@always_inline
def _fma_v[w: Int](
    a: SIMD[DType.float32, w], b: SIMD[DType.float32, w], c: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    from std.math import fma

    return fma(a, b, c)


def expf_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`portable_expf`, lane by lane."""
    comptime V = SIMD[DType.float32, w]
    comptime U = SIMD[DType.uint32, w]
    comptime I = SIMD[DType.int32, w]
    var is_nan = isnan_v[w](x)
    var over = x.gt(V(88.722835))
    var under = x.lt(V(-87.33655))
    # Lanes a branch returns from compute on 0.0 instead (discarded below).
    var xc = (is_nan | over | under).select(V(0.0), x)
    var t = _fma_v[w](xc, V(1.4426950408889634), V(0.5))
    var zf = floor(t)
    var k = zf.cast[DType.int32]()
    var r = _fma_v[w](zf, V(-0.693359375), xc)
    r = _fma_v[w](zf, V(2.12194440e-4), r)
    var q = V(1.9875691500e-4)
    comptime if HOST_SIMD_SABOTAGE:
        q = bitcast[DType.float32, w](bitcast[DType.uint32, w](q) + U(1))
    q = _fma_v[w](q, r, V(1.3981999507e-3))
    q = _fma_v[w](q, r, V(8.3334519073e-3))
    q = _fma_v[w](q, r, V(4.1665795894e-2))
    q = _fma_v[w](q, r, V(1.6666665459e-1))
    q = _fma_v[w](q, r, V(5.0000001201e-1))
    var r2 = r * r
    var y = _fma_v[w](q, r2, r)
    y = y + V(1.0)
    var k1 = k >> I(1)
    var k2 = k - k1
    y = y * bitcast[DType.float32, w](((k1 + I(127)) << I(23)).cast[DType.uint32]())
    y = y * bitcast[DType.float32, w](((k2 + I(127)) << I(23)).cast[DType.uint32]())
    y = y.lt(V(1.1754943508222875e-38)).select(V(0.0), y)
    y = under.select(V(0.0), y)
    y = over.select(bitcast[DType.float32, w](U(0x7F800000)), y)
    return is_nan.select(x, y)


def logf_v[w: Int](x_in: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`portable_logf`, lane by lane."""
    comptime V = SIMD[DType.float32, w]
    comptime U = SIMD[DType.uint32, w]
    comptime I = SIMD[DType.int32, w]
    var is_nan = isnan_v[w](x_in)
    var x = abs(x_in).lt(V(1.1754943508222875e-38)).select(V(0.0), x_in)
    var is_zero = x.eq(V(0.0))
    var is_neg = x.lt(V(0.0))
    var inf = bitcast[DType.float32, w](U(0x7F800000))
    var is_inf = x.eq(inf)
    var special = is_nan | is_zero | is_neg | is_inf
    var xc = special.select(V(1.0), x)
    var bits = bitcast[DType.uint32, w](xc)
    var e = ((bits >> U(23)) & U(0xFF)).cast[DType.int32]() - I(126)
    var m = bitcast[DType.float32, w]((bits & U(0x007FFFFF)) | U(0x3F000000))
    var low = m.lt(V(0.7071067811865476))
    e = low.select(e - I(1), e)
    m = low.select(m + m, m)
    var t = m - V(1.0)
    var z = t * t
    var p = V(7.0376836292e-2)
    p = _fma_v[w](p, t, V(-1.1514610310e-1))
    p = _fma_v[w](p, t, V(1.1676998740e-1))
    p = _fma_v[w](p, t, V(-1.2420140846e-1))
    p = _fma_v[w](p, t, V(1.4249322787e-1))
    p = _fma_v[w](p, t, V(-1.6668057665e-1))
    p = _fma_v[w](p, t, V(2.0000714765e-1))
    p = _fma_v[w](p, t, V(-2.4999993993e-1))
    p = _fma_v[w](p, t, V(3.3333331174e-1))
    var tz = t * z
    var y = tz * p
    var ef = e.cast[DType.float32]()
    y = _fma_v[w](ef, V(-2.12194440e-4), y)
    y = _fma_v[w](V(-0.5), z, y)
    var r = t + y
    r = _fma_v[w](ef, V(0.693359375), r)
    r = is_inf.select(x, r)
    r = is_neg.select(bitcast[DType.float32, w](U(0x7FC00000)), r)
    r = is_zero.select(bitcast[DType.float32, w](U(0xFF800000)), r)
    return is_nan.select(x_in, r)
