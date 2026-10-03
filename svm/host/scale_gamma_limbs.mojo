# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's gamma='scale' exact sums in Mojo (lane/apple-fast-py2mojo-linear).

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
so gamma keeps its bits on every vendor and on the host column.

Each cell adds at most two words below 2^32 to an S1 limb and four to the
S2 limbs; `sg_normalize` carries every limb but the top one into [0, 2^32)
after at most SG_CHUNK cells, so no Int64 word overflows.
"""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.py2mojo_linear import py2mojo_linear_flags

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
def _put(mut acc: SG_ACC, base: Int, t: Int, v: UInt64, neg: Bool):
    """Adds `v * 2^t` (v < 2^24) to the limbs at `base`."""
    var q = t >> 5
    var w = v << UInt64(t & 31)
    var lo = Int64(w & _M32)
    var hi = Int64(w >> 32)
    if neg:
        acc[base + q] -= lo
        acc[base + q + 1] -= hi
    else:
        acc[base + q] += lo
        acc[base + q + 1] += hi


@always_inline
def sg_add_cell(mut acc: SG_ACC, bits: UInt32):
    """One float32 cell, given as its bit pattern."""
    var e = Int((bits >> 23) & 0xFF)
    if e == 255:
        acc[SG_BAD] += 1
        return
    var frac = UInt64(bits & 0x7FFFFF)
    var m = frac | (UInt64(1 << 23) if e > 0 else UInt64(0))
    if m == 0:
        return
    var s = e - 1 if e > 0 else 0
    _put(acc, 0, s, m, (bits >> 31) != 0)
    var m2 = m * m
    _put(acc, SG_L1, 2 * s, m2 & 0xFFFFFF, False)
    _put(acc, SG_L1, 2 * s + 24, m2 >> 24, False)


@always_inline
def sg_normalize(mut acc: SG_ACC):
    """Carries every limb but each sum's top one into [0, 2^32); the value
    the limbs spell does not change."""
    for i in range(SG_L1 - 1):
        var c = acc[i] >> 32
        acc[i] -= c << 32
        acc[i + 1] += c
    for i in range(SG_L1, SG_LIMBS - 1):
        var c = acc[i] >> 32
        acc[i] -= c << 32
        acc[i + 1] += c


@always_inline
def sg_zero() -> SG_ACC:
    return SG_ACC(fill=Int64(0))


def scale_gamma_limbs_host(
    x: MutPointer[UInt32, MutAnyOrigin], count: Int, dst: MutPointer[Int64, MutAnyOrigin]
):
    """THE HOST COLUMN: the same limbs from one host loop (exact, so the
    same value as the device grid)."""
    var acc = sg_zero()
    var i = 0
    while i < count:
        var end = min(count, i + SG_CHUNK)
        for j in range(i, end):
            sg_add_cell(acc, x[j])
        sg_normalize(acc)
        i = end
    for k in range(SG_SLOTS):
        dst[k] = acc[k]


def scale_gamma_limbs_host_binding(
    x_addr: PythonObject, count: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """`scale_gamma_limbs(x, count, out)` on the host column: `count` float32
    cells at x, SG_SLOTS int64 words written at out. Returns count."""
    var n = Int(py=count)
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    if n < 0 or oa == 0 or (n > 0 and xa == 0):
        raise Error("scale_gamma_limbs: null buffer or negative count")
    with GILReleased(Python()):
        scale_gamma_limbs_host(
            MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=xa),
            n,
            MutPointer[Int64, MutAnyOrigin](unsafe_from_address=oa),
        )
    return PythonObject(n)


def py2mojo_linear_flags_binding() raises -> PythonObject:
    return PythonObject(py2mojo_linear_flags())
