# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Counter-based sketch draws for the kernel samplers, IDENTICAL (lane
fix-k1-neighbors, audit K1 / B8 / F7 `neighbor_sketch_rng`, 2026-10-04).

Before: SkewedChi2Sampler.fit and PolynomialCountSketch.fit drew their
random tables from numpy's legacy MT19937 stream (`_LegacyRandomState` in
`python/mojolearn/_expansion_neighbors.py`) in a Python loop on one host
thread, so a fit reproduced scikit-learn's `random_state` numbers. That
stream is sequential by definition. Old bits do not matter (owner rule,
handoff), so under `XN_IDN_SKETCH_CTR` an integer `random_state` keys a
COUNTER-BASED generator instead: draw `t` of stream `s` is a pure function
of `(seed, s, t)`, one GPU thread per draw, no host walk.

THE GENERATOR. SplitMix64's output function over a Weyl counter: `key =
mix64(seed ^ (s << 32) ^ K)`, `draw(t) = mix64(key + (t + 1) * GOLDEN)`.
Integer only (64-bit adds, multiplies that wrap, shifts, xors), so the words
are the same on NVIDIA, AMD, Apple and the host column by construction.

THE VALUES, every one exact before its single float32 rounding:
    uniform   u = (2 m + 1) * 2^-24, m = the top 23 bits: an odd integer
              below 2^24 times a power of two, exact in float32, in (0, 1)
              (never 0, so log(tan(pi/2 u)) never meets log 0)
    z         SkewedChi2 `pi/2 * u`: ONE `identical_mul` by the float32
              pi/2 (the largest z is below the float32 pi/2, so tan stays
              positive)
    offsets   SkewedChi2 `2 pi * u`, one `identical_mul`
    index     PolynomialCountSketch `indexHash_`: (hi32(draw) * nc) >> 32,
              in [0, nc) (bias at most nc / 2^32)
    sign      `bitHash_`: +1 when bit 31 of the low word is set, else -1

Same distributions as theirs, so the sketch's kernel approximation is the
same in expectation; the particular tables for a given seed differ from
scikit-learn's. A numpy RandomState or None keeps the MT19937 path (the
caller's generator is the contract there).

`-D MOJOLEARN_IDN_XN_SKETCH_CTR_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) drops the
bindings, and the glue falls back to the MT19937 draws.

Nothing here imports a GPU module, so the CPU-only host binding compiles it
(the host column's ops are at the end; the device kernels are in
`kfeat_dev.mojo`).
"""

from std.python import PythonObject
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul
from x_neighbors.items import FP, IP, skew_weights_item

comptime XN_IDN_SKETCH_CTR = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_XN_SKETCH_CTR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime KFEAT_GOLDEN: UInt64 = 0x9E3779B97F4A7C15
comptime KFEAT_KEY_SALT: UInt64 = 0x4B46454154524E47
comptime KFEAT_STREAM_Z: UInt32 = 0
comptime KFEAT_STREAM_OFF: UInt32 = 1
comptime KFEAT_STREAM_PCS: UInt32 = 2
comptime KFEAT_HALF_PI_F32 = Float32(1.57079632679489661923)
comptime KFEAT_TWO_PI_F32 = Float32(6.28318530717958647692)
comptime KFEAT_TWO_M24 = Float32(5.9604644775390625e-08)
"""2^-24, exact."""


@always_inline
def kfeat_mix64(x0: UInt64) -> UInt64:
    """SplitMix64's output function."""
    var z = x0
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def kfeat_draw(seed: UInt32, stream: UInt32, t: Int) -> UInt64:
    """Draw `t` of stream `stream` under `seed`: a pure function."""
    var key = kfeat_mix64(UInt64(seed) ^ (UInt64(stream) << 32) ^ KFEAT_KEY_SALT)
    return kfeat_mix64(key + UInt64(t + 1) * KFEAT_GOLDEN)


@always_inline
def kfeat_unit_open(h: UInt64) -> Float32:
    """(2 m + 1) * 2^-24 with m the top 23 bits of `h`: exact, in (0, 1)."""
    var m = Int((h >> 41) & UInt64(0x7FFFFF))
    return ftz(identical_mul(Float32(2 * m + 1), KFEAT_TWO_M24))


@always_inline
def kfeat_schi2_draw_item(t: Int, seed: UInt32, count: Int, z: FP, off: FP):
    """t < count: z[t] = pi/2 * u (stream 0, draw t); count <= t < count +
    nc: off[t - count] = 2 pi * u (stream 1, draw t - count)."""
    if t < count:
        var u = kfeat_unit_open(kfeat_draw(seed, KFEAT_STREAM_Z, t))
        z.unsafe_store(t, ftz(identical_mul(KFEAT_HALF_PI_F32, u)))
    else:
        var c = t - count
        var u = kfeat_unit_open(kfeat_draw(seed, KFEAT_STREAM_OFF, c))
        off.unsafe_store(c, ftz(identical_mul(KFEAT_TWO_PI_F32, u)))


@always_inline
def kfeat_pcs_draw_item(t: Int, seed: UInt32, nc: Int, idx: IP, sgn: IP):
    """`indexHash_` and `bitHash_` entry t (flat [degree, n_features] order)
    from ONE draw: the high word picks the bucket, bit 31 of the low word
    the sign."""
    var h = kfeat_draw(seed, KFEAT_STREAM_PCS, t)
    var hi = h >> 32
    var bucket = (hi * UInt64(nc)) >> 32
    idx.unsafe_store(t, Int32(Int(bucket)))
    var lo = h & UInt64(0xFFFFFFFF)
    sgn.unsafe_store(t, Int32(1) if (lo >> 31) != UInt64(0) else Int32(-1))


# ------------------------------------------------------------- host column
def op_kfeat_schi2_fit_idn_host(seed: UInt32, d: Int, nc: Int, w: FP, off: FP):
    """The host column of `x_neighbors_kfeat_schi2_fit_idn`: the same items,
    then the same `skew_weights_item` over z, in place of the device grid."""
    var count = d * nc
    var z = List[Float32](length=count, fill=Float32(0.0))
    var zp = FP(unsafe_from_address=Int(z.unsafe_ptr()))
    for t in range(count + nc):
        kfeat_schi2_draw_item(t, seed, count, zp, off)
    for t in range(count):
        skew_weights_item(t, zp, w, count)
    _ = z^


def op_kfeat_pcs_draw_idn_host(seed: UInt32, nc: Int, total: Int, idx: IP, sgn: IP):
    for t in range(total):
        kfeat_pcs_draw_item(t, seed, nc, idx, sgn)


def _kfeat_args(p: PythonObject, what: String) raises -> Tuple[UInt32, Int, Int]:
    var seed = UInt32(Int(py=p[0]) & 0xFFFFFFFF)
    var a = Int(py=p[1])
    var b = Int(py=p[2])
    if a <= 0 or b <= 0:
        raise Error("x_neighbors_" + what + ": sizes must be positive")
    if a * b > 2147483647:
        raise Error("x_neighbors_" + what + ": more than 2^31 - 1 draws")
    return (seed, a, b)


def kfeat_schi2_fit_idn_host_binding(p: PythonObject, w_out: PythonObject, off_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, d, n_components]: `random_weights_` (d x nc float32
    at w_out) and `random_offset_` (nc at off_out). Returns 0."""
    var a = _kfeat_args(p, "kfeat_schi2_fit_idn")
    var w = FP(unsafe_from_address=Int(py=w_out))
    var off = FP(unsafe_from_address=Int(py=off_out))
    if Int(w) == 0 or Int(off) == 0:
        raise Error("x_neighbors_kfeat_schi2_fit_idn: null buffer address")
    op_kfeat_schi2_fit_idn_host(a[0], a[1], a[2], w, off)
    return PythonObject(0)


def kfeat_pcs_draw_idn_host_binding(p: PythonObject, idx_out: PythonObject, sgn_out: PythonObject) raises -> PythonObject:
    """p = [seed & 2^32-1, n_components, degree * n_features]: `indexHash_`
    and `bitHash_` (int32, flat) at idx_out / sgn_out. Returns 0."""
    var a = _kfeat_args(p, "kfeat_pcs_draw_idn")
    var idx = IP(unsafe_from_address=Int(py=idx_out))
    var sgn = IP(unsafe_from_address=Int(py=sgn_out))
    if Int(idx) == 0 or Int(sgn) == 0:
        raise Error("x_neighbors_kfeat_pcs_draw_idn: null buffer address")
    op_kfeat_pcs_draw_idn_host(a[0], a[1], a[2], idx, sgn)
    return PythonObject(0)
