# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Items of lane apple-fast-gap-kapprox2 (2026-10-03), in the `items.mojo`
contract (one item = one device thread; the host driver runs the same
items):

  any_below    the input-domain checks of AdditiveChi2Sampler (X < 0) and
               SkewedChi2Sampler (X <= -skewedness) as a device flag
               instead of a serial host `X.min()` pass
               (MOJOLEARN_ACHI2_FAST_DEVCHECK). Every writer stores 1, so
               the unordered stores agree. A NaN is never below.
  schi2_draw   SkewedChi2Sampler's weights and offsets from a counter
               stream on the device (MOJOLEARN_SCHI2_FAST_DEVRNG): the same
               laws as scikit-learn's (u uniform in (0, 1), w = log(tan(pi/2
               u)) / pi, offset = 2 pi u) but not its MT19937 numbers, so a
               FAST fit draws a different (equally distributed) map.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, identical_div, identical_sin, identical_cos, identical_log
from x_neighbors.items import PI_F32

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]


def any_below_item(t: Int, x: FP, res: IP, count: Int, incl: Int, thr: Float32):
    """res[0] = 1 when x[t] < thr (incl == 0) or x[t] <= thr (incl == 1)."""
    var v = x.unsafe_load(t)
    var hit = v < thr if incl == 0 else v <= thr
    if hit:
        res.unsafe_store(0, Int32(1))


@always_inline
def _mix(z_in: UInt64) -> UInt64:
    """splitmix64's finalizer."""
    var z = z_in
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def _u01(seed: Int, t: Int) -> Float32:
    """A uniform in (0, 1): 24 bits of draw t of stream seed, centered in
    its cell (never 0 or 1)."""
    var h = _mix(_mix(UInt64(seed) ^ UInt64(0x9E3779B97F4A7C15)) + UInt64(t) * UInt64(0x9E3779B97F4A7C15))
    return (Float32(Int(h >> 40)) + Float32(0.5)) * Float32(5.9604644775390625e-08)


def schi2_draw_item(t: Int, w: FP, off: FP, d: Int, nc: Int, seed: Int):
    """t < d nc: w[t] = log(tan(pi/2 u)) / pi (skew_weights_item's
    spelling); else off[t - d nc] = 2 pi u."""
    var u = _u01(seed, t)
    if t < d * nc:
        var z = ftz(identical_mul(ftz(identical_mul(PI_F32, Float32(0.5))), u))
        var tn = ftz(identical_div(ftz(identical_sin(z)), ftz(identical_cos(z))))
        w.unsafe_store(t, ftz(identical_mul(ftz(identical_div(Float32(1), PI_F32)), ftz(identical_log(tn)))))
    else:
        off.unsafe_store(t - d * nc, ftz(identical_mul(ftz(identical_mul(Float32(2), PI_F32)), u)))
