# SPDX-License-Identifier: Apache-2.0
"""Independent integer oracle for the native BF16 conversion contract.

All 65,536 upper halves are narrowed with discarded tails zero, below tie,
at tie and above tie. An additional 37 x 53 sample retains the former
Python test's shape and 13-bit-significand/eight-binade distribution, using
a native deterministic generator rather than importing NumPy's RNG.
Every raw BF16 word is widened too, including NaNs and subnormals.
"""

comptime BF16_NARROW_EXHAUSTIVE = 65536 * 4
comptime BF16_NARROW_COUNT = BF16_NARROW_EXHAUSTIVE + 37 * 53
comptime BF16_WIDEN_COUNT = 65536
comptime BF16_CHUNK = 1024
comptime BF16_NARROW_UNITS = (BF16_NARROW_COUNT + BF16_CHUNK - 1) // BF16_CHUNK
comptime BF16_WIDEN_UNITS = BF16_WIDEN_COUNT // BF16_CHUNK
comptime BF16_UNITS = BF16_NARROW_UNITS + BF16_WIDEN_UNITS


def input_word(index: Int) -> UInt32:
    if index < BF16_NARROW_EXHAUSTIVE:
        var tail_id = index % 4
        var tail = UInt32(0)
        if tail_id == 1:
            tail = 0x7FFF
        elif tail_id == 2:
            tail = 0x8000
        elif tail_id == 3:
            tail = 0xFFFF
        return UInt32(index // 4) << 16 | tail
    var h = UInt64(index - BF16_NARROW_EXHAUSTIVE + 1) * UInt64(0x9E3779B97F4A7C15)
    h ^= h >> 30
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 27
    var mantissa = UInt32(h & 0xFFF) << 11
    var exponent = (UInt32((h >> 12) & 7) + 123) << 23
    var sign = UInt32((h >> 19) & 1) << 31
    return sign | exponent | mantissa


def expected_narrow(word: UInt32) -> UInt16:
    # Integer quotient/remainder rounding, independent of the product's
    # add-bias implementation. Native contract flushes binary32 subnormals
    # before rounding and preserves/quietens the high NaN payload bits.
    var exponent = (word >> 23) & 255
    var fraction = word & 0x007FFFFF
    if exponent == 0:
        return UInt16((word >> 16) & 0x8000)
    var upper = word >> 16
    if exponent == 255 and fraction != 0:
        return UInt16(upper | 0x0040)
    var remainder = word & 0xFFFF
    if remainder > 0x8000 or (remainder == 0x8000 and (upper & 1) != 0):
        upper += 1
    return UInt16(upper)


def expected_widen(word: UInt16) -> UInt32:
    return UInt32(word) << 16


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)
