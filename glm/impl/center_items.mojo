# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The arithmetic of LinearRegression's and Ridge's centering (lane
hr-small-passes, 2026-10-02), shared word for word by the device
(glm/impl/center_device.mojo) and the CPU column (glm/host/center_host.mojo).

COLUMN SUMS ARE EXACT. A float32 is `m * 2^(k - 149)` with an integer
mantissa m < 2^24 and 0 <= k <= 253, so every float32 is an integer
multiple of 2^-149 below 2^277, and a column total is held exactly in nine
32-bit digits, each kept in an Int64 so a digit can take up to 2^30
additions before its carry is propagated. Integer adds are exact in any
order: per-thread partials over fixed row blocks, then the partials folded
in any order, give the same total on every vendor and on the host. The
total is rounded ONCE to float64 (round to nearest, ties to even): the
correctly rounded column sum. NEW BITS: the old host helper
(`column_mean_f64`) summed each column in one ascending float64 chain, a
different (order-bound) number; Apple has no float64 at all, so no
on-device chain could have reproduced it.

The infinities and NaNs ride in a flag word: NaN if any NaN or both
infinities, else the infinity seen. A total of zero is +0.0.

CENTER AND SCALE are one binary32 subtraction or multiplication per cell,
operands and result flushed to signed zero when subnormal (the identity
contract's denormal policy, spelled on the bits so every column agrees,
Apple's hardware flush included). NEW BITS only where an operand or a
result is subnormal: the old host helpers kept subnormals."""
from std.bit import count_leading_zeros
from std.memory import bitcast


comptime CS_LIMBS = 9
#: limbs plus the flag word, the stride of a partial
comptime CS_WORDS = CS_LIMBS + 1
comptime CS_FLAG_NAN = Int64(1)
comptime CS_FLAG_PINF = Int64(2)
comptime CS_FLAG_NINF = Int64(4)


@always_inline
def flush32(x: Float32) -> Float32:
    """A subnormal to its signed zero, anything else unchanged, by bits."""
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x7F800000)) == UInt32(0):
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return x


@always_inline
def center_cell(x: Float32, mu: Float32) -> Float32:
    return flush32(flush32(x) - flush32(mu))


@always_inline
def scale_cell(x: Float32, w: Float32) -> Float32:
    return flush32(flush32(x) * flush32(w))


@always_inline
def exact_add(mut acc: InlineArray[Int64, CS_WORDS], v: Float32):
    """acc += v exactly (limbs 0..8), or the flag word for an infinity or a
    NaN. The limb index is compared against compile-time indices so the
    accumulator stays in registers on a GPU."""
    var b = bitcast[DType.uint32](v)
    var e = Int((b >> 23) & UInt32(0xFF))
    var neg = (b >> 31) != UInt32(0)
    var frac = b & UInt32(0x7FFFFF)
    if e == 0xFF:
        if frac != UInt32(0):
            acc[CS_LIMBS] = acc[CS_LIMBS] | CS_FLAG_NAN
        elif neg:
            acc[CS_LIMBS] = acc[CS_LIMBS] | CS_FLAG_NINF
        else:
            acc[CS_LIMBS] = acc[CS_LIMBS] | CS_FLAG_PINF
        return
    var m = UInt64(frac)
    if e != 0:
        m = m | UInt64(0x800000)
    var k = (e if e > 0 else 1) - 1
    var j = k >> 5
    var wide = m << UInt64(k & 31)
    var lo = Int64(wide & UInt64(0xFFFFFFFF))
    var hi = Int64(wide >> UInt64(32))
    if neg:
        lo = -lo
        hi = -hi
    comptime for L in range(CS_LIMBS):
        if j == L:
            acc[L] = acc[L] + lo
        if j + 1 == L:
            acc[L] = acc[L] + hi


@always_inline
def exact_merge(mut acc: InlineArray[Int64, CS_WORDS], other: InlineArray[Int64, CS_WORDS]):
    comptime for L in range(CS_LIMBS):
        acc[L] = acc[L] + other[L]
    acc[CS_LIMBS] = acc[CS_LIMBS] | other[CS_LIMBS]


@always_inline
def _bit(d: InlineArray[UInt64, CS_WORDS], i: Int) -> UInt64:
    return (d[i >> 5] >> UInt64(i & 31)) & UInt64(1)


def exact_finish(acc: InlineArray[Int64, CS_WORDS]) -> UInt64:
    """The float64 bits of the exact total, rounded once to nearest even."""
    var flags = acc[CS_LIMBS]
    if (flags & CS_FLAG_NAN) != 0 or (flags & (CS_FLAG_PINF | CS_FLAG_NINF)) == (CS_FLAG_PINF | CS_FLAG_NINF):
        return UInt64(0x7FF8000000000000)
    if (flags & CS_FLAG_PINF) != 0:
        return UInt64(0x7FF0000000000000)
    if (flags & CS_FLAG_NINF) != 0:
        return UInt64(0xFFF0000000000000)
    # carry propagation: ten base-2^32 digits, the top one signed
    var d = InlineArray[UInt64, CS_WORDS](fill=UInt64(0))
    var c = Int64(0)
    for L in range(CS_LIMBS):
        var t = acc[L] + c
        d[L] = UInt64(t) & UInt64(0xFFFFFFFF)
        c = t >> 32
    d[CS_LIMBS] = UInt64(c) & UInt64(0xFFFFFFFF)
    var neg = c < 0
    if neg:
        # two's complement negation of the 320-bit word
        var carry = UInt64(1)
        for L in range(CS_WORDS):
            var t = (d[L] ^ UInt64(0xFFFFFFFF)) + carry
            d[L] = t & UInt64(0xFFFFFFFF)
            carry = t >> UInt64(32)
    var h = CS_WORDS - 1
    while h >= 0 and d[h] == UInt64(0):
        h -= 1
    if h < 0:
        return UInt64(0)
    var p = 32 * h + 31 - Int(count_leading_zeros(UInt32(d[h])))
    var mant: UInt64
    if p <= 52:
        mant = d[0] | (d[1] << UInt64(32))
        mant = mant << UInt64(52 - p)
    else:
        var shift = p - 52
        mant = UInt64(0)
        for i in range(53):
            mant = mant | (_bit(d, shift + i) << UInt64(i))
        var rnd = _bit(d, shift - 1)
        var sticky = UInt64(0)
        var below = shift - 1
        for L in range(below >> 5):
            sticky = sticky | d[L]
        var part = below & 31
        if part > 0:
            sticky = sticky | (d[below >> 5] & ((UInt64(1) << UInt64(part)) - UInt64(1)))
        if rnd != UInt64(0) and (sticky != UInt64(0) or (mant & UInt64(1)) != UInt64(0)):
            mant += UInt64(1)
            if mant == (UInt64(1) << UInt64(53)):
                mant = mant >> UInt64(1)
                p += 1
    var biased = UInt64(p + 874)
    var bits = (biased << UInt64(52)) | (mant & UInt64(0xFFFFFFFFFFFFF))
    if neg:
        bits = bits | UInt64(0x8000000000000000)
    return bits
