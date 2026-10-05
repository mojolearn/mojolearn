# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""float64 words from float32 values by integer bit work (lane apple-fast-q-clf,
2026-10-04): FAST predict_proba's float64 output (x_prep/proba64.mojo,
x_neighbors/proba64_nc.mojo) on devices without FP64 (Apple)."""
from std.memory import bitcast


@always_inline
def widen_bits(v: Float32) -> UInt64:
    """The IEEE float64 bits of float32 v, exactly (subnormals normalised)."""
    var b = UInt64(bitcast[DType.uint32](v))
    var sign = (b >> 31) << 63
    var e = Int((b >> 23) & 0xFF)
    var man = b & 0x7FFFFF
    if e == 0:
        if man == 0:
            return sign
        var e64 = 897  # 1 - 127 + 1023
        while (man & 0x800000) == 0:
            man = man << 1
            e64 -= 1
        return sign | (UInt64(e64) << 52) | ((man & 0x7FFFFF) << 29)
    if e == 255:
        return sign | (UInt64(0x7FF) << 52) | (man << 29)
    return sign | (UInt64(e - 127 + 1023) << 52) | (man << 29)


@always_inline
def one_minus_bits(c: Float32) -> UInt64:
    """The float64 bits of 1 - c for 0 <= c <= 1/2 (c a float32): 1 - c lies
    in [1/2, 1], so its fraction field is 2^52 - c * 2^53, c * 2^53 being
    c's 24-bit significand shifted (rounded to nearest when it shifts right)."""
    var b = UInt64(bitcast[DType.uint32](c))
    var e = Int((b >> 23) & 0xFF)
    var v = UInt64(0)
    if e != 0:
        var m = (b & 0x7FFFFF) | 0x800000
        var sh = e - 97  # c * 2^53 = m * 2^(e - 150 + 53)
        if sh >= 0:
            v = m << UInt64(sh)
        elif -sh < 25:
            var r = -sh
            v = (m + (UInt64(1) << UInt64(r - 1))) >> UInt64(r)
    var two52 = UInt64(1) << 52
    if v >= two52:
        return UInt64(0x3FE) << 52  # c rounds to 1/2
    return (UInt64(0x3FE) << 52) + (two52 - v)


@always_inline
def put64(f: MutPointer[Float32, MutAnyOrigin], w: Int, bits: UInt64):
    """Words w (low) and w + 1 (high) of a float64, stored raw (no flush)."""
    f.unsafe_store(w, bitcast[DType.float32](UInt32(bits & 0xFFFFFFFF)))
    f.unsafe_store(w + 1, bitcast[DType.float32](UInt32(bits >> 32)))
