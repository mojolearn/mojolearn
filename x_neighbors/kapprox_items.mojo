# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-kapprox (2026-10-02): SparseRandomProjection's device item.

  * `kapprox_sparse_rp_item`: one entry of the projection matrix from two
    counter-based uniforms (`kapprox_uniform`), one thread per entry.

The chi2 samplers' items (check, additive map, skewed fit and log) were
DROPPED-quality with `-D MOJOLEARN_KAPPROX_DEVICE`; their code is on
lane/apple-fast-kapprox @ 10d5a7970. FAST + Apple only
(x_neighbors/kapprox_dev.mojo, `-D MOJOLEARN_SPARSE_RP_DEVICE`). Same
spellings as x_neighbors/items.mojo; no float64 (Apple GPUs have none).
"""
from checks.numerics import ftz, identical_mul
from x_neighbors.items import FP, _add

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


def kapprox_sparse_rp_item(t: Int, res: FP, kc: Int, d: Int, seed: Int, dens: Float32, scale: Float32):
    """Entry t of SparseRandomProjection's kc x d matrix (Achlioptas / Li):
    +-scale with probability dens / 2 each, else 0, from two counter-based
    uniforms of stream `seed` (one for the keep test, one for the sign). One
    launch, no host draw; FAST only (IDENTICAL keeps x_decomp's Philox words)."""
    var keep = kapprox_uniform(UInt32(seed), t)
    var sgn = kapprox_uniform(UInt32(seed) ^ UInt32(0x9E3779B9), t)
    var v = Float32(0)
    if keep < dens:
        v = -scale if sgn < Float32(0.5) else scale
    res.unsafe_store(t, v)
