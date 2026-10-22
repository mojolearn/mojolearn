# SPDX-License-Identifier: Apache-2.0
# Native float32 output conversions. Separate from the checked CastIn input
# conversions: array-protocol output casts have unsafe NumPy semantics.
from std.memory import bitcast
from std.sys.info import CompilationTarget
from core.cast_in import _top_bit, f32_to_f64_bits
from core.sequence_array_format import sequence_write_text

comptime SEQ_CAST_OVERFLOW = 1
comptime SEQ_CAST_UNDERFLOW = 2
comptime SEQ_CAST_INVALID = 4


def sequence_store_word(destination: Int, word: UInt64, size: Int, little: Bool):
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for byte in range(size):
        out[byte if little else size - 1 - byte] = UInt8(word >> UInt64(8 * byte))


def sequence_wide_word(bits: UInt32) -> UInt64:
    # Widen NaNs without losing their sign or payload; CastIn intentionally
    # canonicalizes them, which is a different interface contract.
    var raw = UInt64(bits)
    if (bits & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000):
        return ((raw >> 31) << 63) | UInt64(0x7FF8000000000000) | ((raw & UInt64(0x7FFFFF)) << 29)
    return f32_to_f64_bits(raw)


def sequence_signaling(bits: UInt32) -> Bool:
    return (bits & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000) and (bits & UInt32(0x00400000)) == 0


def sequence_half_word(bits: UInt32, mut status: Int) -> UInt64:
    var magnitude = bits & UInt32(0x7FFFFFFF)
    var sign = UInt64(bits >> 31) << 15
    if magnitude >= UInt32(0x7F800000):
        if magnitude == UInt32(0x7F800000):
            return sign | UInt64(0x7C00)
        if sequence_signaling(bits):
            status |= SEQ_CAST_INVALID
        return sign | UInt64(0x7E00) | UInt64((bits & UInt32(0x7FFFFF)) >> 13)
    if magnitude >= UInt32(0x477FF000):
        status |= SEQ_CAST_OVERFLOW
        return sign | UInt64(0x7C00)
    if magnitude == 0:
        return sign
    var exponent = Int(magnitude >> 23)
    var mantissa = UInt64((magnitude & UInt32(0x7FFFFF)) | UInt32(0x800000))
    if exponent < 102:
        status |= SEQ_CAST_UNDERFLOW
        return sign
    var shift = 13 if exponent >= 113 else 126 - exponent
    var quotient = mantissa >> UInt64(shift)
    var remainder = mantissa & ((UInt64(1) << UInt64(shift)) - 1)
    var halfway = UInt64(1) << UInt64(shift - 1)
    if remainder > halfway or (remainder == halfway and (quotient & 1) != 0):
        quotient += 1
    if exponent < 113:
        if remainder != 0:
            status |= SEQ_CAST_UNDERFLOW
        return sign | quotient
    return sign | (UInt64(exponent - 113) << 10) + quotient


def sequence_integer_word(bits: UInt32, width: Int, unsigned: Bool, date: Bool, mut status: Int) -> UInt64:
    var magnitude = bits & UInt32(0x7FFFFFFF)
    var negative = (bits >> 31) != 0
    var nan = magnitude > UInt32(0x7F800000)
    # datetime/timedelta explicitly use NaT for nonfinite values. Their finite
    # conversion is signed int64; the unit remains dtype metadata.
    if date and nan:
        return UInt64(1) << 63
    var target = max(width, 4)
    var use_unsigned = unsigned and width >= 4
    var exponent = Int(magnitude >> 23) - 127
    var significant = UInt64((magnitude & UInt32(0x7FFFFF)) | UInt32(0x800000))
    var limit = target * 8 - (0 if use_unsigned else 1)
    var invalid = nan or exponent >= limit or (use_unsigned and negative and exponent >= 0)
    if not use_unsigned and negative and exponent == limit and (magnitude & UInt32(0x7FFFFF)) == 0:
        invalid = False
    if invalid:
        status |= SEQ_CAST_INVALID
        # Out-of-range unsafe casts are platform-defined in NumPy. Preserve
        # the ARM host ABI used by the retained baseline, and the x86 signed
        # indefinite value. Narrow integer outputs truncate the int32 result.
        comptime if CompilationTarget.has_neon():
            if nan:
                return 0
            if use_unsigned:
                return 0 if negative else (UInt64.MAX if target == 8 else UInt64(0xFFFFFFFF))
            var top = UInt64(1) << UInt64(target * 8 - 1)
            return top if negative else top - 1
        else:
            return (UInt64(1) << UInt64(target * 8 - 1)) if not use_unsigned else 0
    if exponent < 0:
        return 0
    var word = significant << UInt64(exponent - 23) if exponent >= 23 else significant >> UInt64(23 - exponent)
    return UInt64(0) - word if negative else word


def sequence_long_float(bits: UInt32, destination: Int, width: Int, little: Bool):
    # Both supported extended native layouts can represent every binary32
    # exactly. x86 uses x87's 80 meaningful bits in a 16-byte slot; AArch64
    # Linux uses IEEE binary128. Padding is initialized, never read as data.
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(width):
        out[i] = 0
    var magnitude = bits & UInt32(0x7FFFFFFF)
    var exponent = Int(magnitude >> 23)
    var fraction = UInt64(magnitude & UInt32(0x7FFFFF))
    var sign = UInt64(bits >> 31)
    var significand = fraction | UInt64(0x800000)
    var biased = exponent + 16256
    if exponent == 0:
        if fraction == 0:
            biased = 0
            significand = 0
        else:
            var lead = _top_bit(fraction)
            biased = lead - 149 + 16383
            significand = fraction << UInt64(23 - lead)
    elif exponent == 255:
        biased = 32767
        if fraction != 0:
            significand |= UInt64(0x400000)
    comptime if CompilationTarget.is_x86():
        sequence_store_word(destination, significand << 40, 8, little)
        sequence_store_word(destination + 8, (sign << 15) | UInt64(biased), 2, little)
    else:
        var high = (sign << 63) | (UInt64(biased) << 48) | ((significand & UInt64(0x7FFFFF)) << 25)
        sequence_store_word(destination + (8 if little else 0), high, 8, little)


def sequence_cast_leaf(source: Int, destination: Int, count: Int, stride: Int, offset: Int,
                       kind: String, width: Int, little: Bool) raises -> Int:
    var input = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=source)
    var status = 0
    for index in range(count):
        var bits = input[index]
        var address = destination + index * stride + offset
        if kind == "U" or kind == "S":
            sequence_write_text(bits, address, width, kind == "U", little)
        elif kind == "V":
            var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
            for i in range(width):
                target[i] = UInt8(bits >> UInt32(8 * i)) if i < 4 else UInt8(0)
        elif kind == "b":
            sequence_store_word(address, UInt64((bits & UInt32(0x7FFFFFFF)) != 0), 1, True)
        elif kind == "i" or kind == "u" or kind == "m" or kind == "M":
            sequence_store_word(address, sequence_integer_word(bits, width, kind == "u", kind == "m" or kind == "M", status), width, little)
        else:
            var size = width // 2 if kind == "c" else width
            if size == 2:
                sequence_store_word(address, sequence_half_word(bits, status), 2, little)
            elif size == 4:
                sequence_store_word(address, UInt64(bits), 4, little)
            elif size == 8:
                sequence_store_word(address, sequence_wide_word(bits), 8, little)
                if sequence_signaling(bits):
                    status |= SEQ_CAST_INVALID
            else:
                sequence_long_float(bits, address, size, little)
                if sequence_signaling(bits):
                    status |= SEQ_CAST_INVALID
            if kind == "c":
                var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address + size)
                for i in range(size):
                    target[i] = 0
    return status
