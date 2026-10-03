# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's gamma='scale' exact sums in Mojo, the shared arithmetic
(lane/apple-fast-py2mojo-linear). Host loop: `svm/host/scale_gamma_host.mojo`.

`python/mojolearn/_scale_gamma.py::scale_gamma` formed the EXACT sums
S1 = sum x and S2 = sum x * x of X's float32 cells as Python big integers, a
Python loop over n * d. This module forms the same two integers as limbs:

  S1 at scale 2^-149 (every float32 is an integer multiple of 2^-149):
     9 limbs of base 2^32, signed values;
  S2 at scale 2^-298 (x * x of a float32 is a multiple of 2^-298):
     18 limbs of base 2^32;
  slot 27: the count of non-finite cells (exponent field 255).

An integer sum has no fold order, so the device grid (`svm/impl/
scale_gamma_device.mojo`), this host loop and any split of the cells give
the SAME limbs' value; Python adds the 27 words into two integers and rounds
`N^2 * 2^298 / (n_features * (N * S2 - S1^2))` once, the exact rational the
old code rounded (its scales were 2^-1074 and 2^-2148; the ratio is equal),
so gamma keeps its bits on every vendor and on the host column
(`svm/host/scale_gamma_host.mojo`).

Each cell adds at most two words below 2^32 to an S1 limb and four to the
S2 limbs; `sg_normalize` carries every limb but the top one into [0, 2^32)
after at most SG_CHUNK cells, so no Int64 word overflows.
"""

comptime SG_L1 = 9
comptime SG_L2 = 18
comptime SG_LIMBS = SG_L1 + SG_L2
#: words the binding writes: the 27 limbs and the non-finite count
comptime SG_SLOTS = SG_LIMBS + 1
comptime SG_BAD = SG_LIMBS
#: cells a thread adds between two carries
comptime SG_CHUNK = 1 << 20
comptime SG_ACC = InlineArray[Int64, SG_SLOTS]

comptime _M32 = UInt64(0xFFFFFFFF)


@always_inline
def _put[base: Int, n: Int](mut acc: SG_ACC, t: Int, v: UInt64, neg: Bool):
    """Adds `v * 2^t` (v < 2^24) to the `n` limbs at `base`. Every index is
    a compile-time constant (a select per limb), so the limbs stay in
    registers on every GPU."""
    var q = t >> 5
    var w = v << UInt64(t & 31)
    var lo = Int64(w & _M32)
    var hi = Int64(w >> 32)
    if neg:
        lo = -lo
        hi = -hi
    comptime for k in range(n):
        acc[base + k] += (lo if q == k else Int64(0)) + (hi if q + 1 == k else Int64(0))


@always_inline
def sg_add_cell(mut acc: SG_ACC, bits: UInt32):
    """One float32 cell, given as its bit pattern."""
    var e = Int((bits >> 23) & 0xFF)
    if e == 255:
        acc[SG_BAD] += Int64(1)
        return
    var frac = UInt64(bits & 0x7FFFFF)
    var m = frac | (UInt64(1 << 23) if e > 0 else UInt64(0))
    if m == 0:
        return
    var s = e - 1 if e > 0 else 0
    _put[0, SG_L1](acc, s, m, (bits >> 31) != 0)
    var m2 = m * m
    _put[SG_L1, SG_L2](acc, 2 * s, m2 & 0xFFFFFF, False)
    _put[SG_L1, SG_L2](acc, 2 * s + 24, m2 >> 24, False)


@always_inline
def sg_normalize(mut acc: SG_ACC):
    """Carries every limb but each sum's top one into [0, 2^32); the value
    the limbs spell does not change."""
    comptime for i in range(SG_L1 - 1):
        var c = acc[i] >> 32
        acc[i] -= c << 32
        acc[i + 1] += c
    comptime for i in range(SG_L1, SG_LIMBS - 1):
        var c = acc[i] >> 32
        acc[i] -= c << 32
        acc[i + 1] += c


@always_inline
def sg_zero() -> SG_ACC:
    return SG_ACC(fill=Int64(0))
