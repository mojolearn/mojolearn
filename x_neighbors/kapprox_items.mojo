# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox (2026-10-02): the chi2 samplers' device items.

AdditiveChi2Sampler and SkewedChi2Sampler spent their clocked fit on the
host: `_f32(X)` then `X.min()` over every cell (a host pass), and the skewed
sampler's d x n_components uniforms drawn one by one in Python. These items
move that work to the device, one thread per cell:

  * `kapprox_check_item`: the samplers' refusal (x < floor, or x <= floor)
    as a device flag, no host min;
  * `kapprox_achi2_item`: the additive map with the refusal fused in;
  * `kapprox_skew_fit_item`: the skewed sampler's weights and offsets from a
    counter-based uniform (`kapprox_uniform`) instead of sklearn's serial
    MT19937 stream: the fitted parameters are i.i.d. uniforms of the same
    law, not sklearn's numbers (FAST only; IDENTICAL keeps the legacy stream);
  * `kapprox_skew_log_item`: log(x + skewedness) with the refusal fused in,
    into a device scratch the transform's product reads (no host round trip
    between the log and the product).

FAST + Apple only (x_neighbors/kapprox_dev.mojo, `-D MOJOLEARN_KAPPROX_DEVICE`).
Same spellings as x_neighbors/items.mojo; no float64 (Apple GPUs have none).
"""
from checks.numerics import (
    ftz,
    identical_mul,
    identical_div,
    identical_log,
    identical_cos,
    identical_sin,
)
from x_neighbors.items import FP, IP, PI_F32, _add, achi2_item

comptime KAPPROX_HALF_PI_F32 = Float32(1.57079632679489661923)
comptime KAPPROX_TWO_PI_F32 = Float32(6.28318530717958647692)
#: 2^-24: the top 24 bits of a hash, plus a half, scaled into (0, 1).
comptime KAPPROX_U24_F32 = Float32(5.9604644775390625e-08)


@always_inline
def _mix32(v: UInt32) -> UInt32:
    """lowbias32: a 32-bit integer bijection (xorshift-multiply, two rounds)."""
    var h = v
    h = h ^ (h >> 16)
    h = h * UInt32(0x7FEB352D)
    h = h ^ (h >> 15)
    h = h * UInt32(0x846CA68B)
    h = h ^ (h >> 16)
    return h


@always_inline
def kapprox_uniform(seed: UInt32, t: Int) -> Float32:
    """Draw t of stream `seed`, a uniform in the OPEN interval (0, 1): the
    top 24 bits of a counter hash plus one half, times 2^-24. Neither 0 nor 1
    occurs, so tan(pi/2 u) is finite and positive and its log is finite."""
    var h = _mix32(_mix32(seed) ^ UInt32(t))
    h = _mix32(h + UInt32(0x9E3779B9))
    var top = (h >> 8).cast[DType.float32]()
    return ftz(identical_mul(_add(top, Float32(0.5)), KAPPROX_U24_F32))


def kapprox_check_item(t: Int, x: FP, flag: IP, n: Int, d: Int, strict: Int, floor: Float32):
    """Cell t against a sampler's refusal: x < floor (strict == 0, the
    additive sampler's negative check) or x <= floor (strict != 0, the
    skewed sampler's -skewedness check). A NaN passes both, as the
    comparisons on sklearn's `X.min()` do. A refused cell stores 1 into
    flag[0]; every writer stores the same value, so the order is free."""
    var v = x.unsafe_load(t)
    var bad = (v <= floor) if strict != 0 else (v < floor)
    if bad:
        flag.unsafe_store(0, Int32(1))


def kapprox_achi2_item(t: Int, x: FP, res: FP, flag: IP, n: Int, d: Int, steps: Int, interval: Float32):
    """`achi2_item` for cell t, with the negative check fused in (flag[0] = 1
    on x < 0; the map is still written and the caller discards it)."""
    if x.unsafe_load(t) < Float32(0):
        flag.unsafe_store(0, Int32(1))
    achi2_item(t, x, res, n, d, steps, interval)


def kapprox_skew_fit_item(t: Int, w: FP, off: FP, d: Int, nc: Int, seed: Int):
    """Draw t of SkewedChi2Sampler.fit on stream `seed`: t < d * nc is
    random_weights_[t] = (1/pi) log(tan(pi/2 u)) (the inverse sech CDF, the
    statements of `skew_weights_item`); the next nc draws are
    random_offset_[t - d * nc] = 2 pi u."""
    var u = kapprox_uniform(UInt32(seed), t)
    if t < d * nc:
        var z = ftz(identical_mul(KAPPROX_HALF_PI_F32, u))
        var tn = ftz(identical_div(ftz(identical_sin(z)), ftz(identical_cos(z))))
        w.unsafe_store(t, ftz(identical_mul(ftz(identical_div(Float32(1), PI_F32)), ftz(identical_log(tn)))))
    else:
        off.unsafe_store(t - d * nc, ftz(identical_mul(KAPPROX_TWO_PI_F32, u)))


def kapprox_skew_log_item(t: Int, x: FP, lx: FP, flag: IP, skew: Float32):
    """lx[t] = log(x[t] + skewedness), with the refusal fused in (flag[0] = 1
    on x <= -skewedness, the log of a non-positive number is then discarded)."""
    var v = x.unsafe_load(t)
    if v <= -skew:
        flag.unsafe_store(0, Int32(1))
    lx.unsafe_store(t, ftz(identical_log(_add(v, skew))))
