# SPDX-License-Identifier: Apache-2.0
"""Shared exact binary64 OOB integer sum, callable on host and device.

T15 source extraction, preserving the existing limb/rounding contract.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""
comptime E64_LIMBS = 72
comptime E64_MAX_ROWS = 1 << 30
#: flags: [0] a non-finite term, [1] a rounded sum overflowed binary64
comptime E64_FLAGS = 4


@always_inline
def e64_add[o: MutOrigin, fo: MutOrigin, //](limbs: MutPointer[Int64, o], w: UInt64, flags: MutPointer[Int32, fo]):
    """limbs += the binary64 word w as an integer count of 2^-1074 (a NaN or
    an infinity sets flags[0] and adds nothing)."""
    var e = Int((w >> 52) & UInt64(0x7FF))
    if e == 0x7FF:
        flags[unsafe_offset=0] = 1
        return
    var m = w & UInt64(0x000FFFFFFFFFFFFF)
    var shift = 0
    if e != 0:
        m |= UInt64(1) << 52
        shift = e - 1
    if m == 0:
        return
    var s = shift % 32
    var k = shift // 32
    var lo = m << UInt64(s)
    var hi = (m >> UInt64(64 - s)) if s > 0 else UInt64(0)
    var p0 = Int64(lo & UInt64(0xFFFFFFFF))
    var p1 = Int64(lo >> 32)
    var p2 = Int64(hi)
    if (w >> 63) != 0:
        p0 = -p0
        p1 = -p1
        p2 = -p2
    limbs[unsafe_offset=k] = limbs[unsafe_offset=k] + p0
    limbs[unsafe_offset=k + 1] = limbs[unsafe_offset=k + 1] + p1
    limbs[unsafe_offset=k + 2] = limbs[unsafe_offset=k + 2] + p2


def e64_round[o: MutOrigin, fo: MutOrigin, //](l: MutPointer[Int64, o], flags: MutPointer[Int32, fo]) -> UInt64:
    """The integer sum(l[i] << 32 i) times 2^-1074 rounded once to binary64,
    nearest/even (`_scaled_integer(total, -1074)`); normalizes `l` in place.
    Overflow sets flags[1] (the Python OverflowError) and returns +inf."""
    for i in range(E64_LIMBS - 1):
        var v = l[unsafe_offset=i]
        var lo = v & Int64(0xFFFFFFFF)
        l[unsafe_offset=i] = lo
        l[unsafe_offset=i + 1] = l[unsafe_offset=i + 1] + ((v - lo) >> 32)
    var sign = UInt64(0)
    if l[unsafe_offset=E64_LIMBS - 1] < 0:
        sign = UInt64(1) << 63
        var carry = Int64(1)
        for i in range(E64_LIMBS):
            var x = ((~l[unsafe_offset=i]) & Int64(0xFFFFFFFF)) + carry
            l[unsafe_offset=i] = x & Int64(0xFFFFFFFF)
            carry = x >> 32
    var t = -1
    for i in range(E64_LIMBS):
        if l[unsafe_offset=i] != 0:
            t = i
    if t < 0:
        return UInt64(0)
    var top_limb = UInt64(l[unsafe_offset=t])
    var width = 0
    while top_limb != 0:
        top_limb >>= 1
        width += 1
    var bitlen = 32 * t + width
    if bitlen <= 53:
        # below 2^53 units of 2^-1074 the integer IS the word (subnormal or
        # the smallest normal binade)
        return sign | (UInt64(l[unsafe_offset=0]) | (UInt64(l[unsafe_offset=1]) << 32))
    var shift = bitlen - 53
    var mant = UInt64(0)
    for b in range(bitlen - 1, shift - 1, -1):
        mant = (mant << 1) | ((UInt64(l[unsafe_offset=b // 32]) >> UInt64(b % 32)) & 1)
    var half = (UInt64(l[unsafe_offset=(shift - 1) // 32]) >> UInt64((shift - 1) % 32)) & 1
    var sticky = False
    var lowbits = shift - 1
    for i in range(lowbits // 32):
        if l[unsafe_offset=i] != 0:
            sticky = True
    var rem = lowbits % 32
    if rem > 0 and (UInt64(l[unsafe_offset=lowbits // 32]) & ((UInt64(1) << UInt64(rem)) - 1)) != 0:
        sticky = True
    if half == 1 and (sticky or (mant & 1) == 1):
        mant += 1
    var top = bitlen - 1 - 1074
    if mant == (UInt64(1) << 53):
        mant >>= 1
        top += 1
    if top > 1023:
        flags[unsafe_offset=1] = 1
        return sign | UInt64(0x7FF0000000000000)
    return sign | (UInt64(top + 1023) << 52) | (mant & UInt64(0x000FFFFFFFFFFFFF))


