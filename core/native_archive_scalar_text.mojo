# SPDX-License-Identifier: Apache-2.0
"""Native scalar spelling for the existing archive metadata coercion contract.

Normal saved metadata is Unicode. Historical archives can also carry numeric
or byte fields; their standard scalar spelling is retained without constructing
Python/NumPy scalar objects. This is formatting only, never model arithmetic.
"""
from std.ffi import c_char, c_int, c_size_t, external_call
from std.memory import bitcast
from core.native_archive_span import ArchiveSpan
from core.sequence_array_format import SequenceNumericLocale, sequence_float_text


def archive_data_word(raw: ArchiveSpan, at: Int, width: Int) raises -> UInt64:
    var value = UInt64(0)
    var bytes = raw.slice(at, width)
    for i in range(width):
        value |= UInt64(bytes.byte(i)) << UInt64(8 * i)
    return value


def archive_bytes_repr(raw: ArchiveSpan) raises -> String:
    var single = False
    var double = False
    for i in range(raw.size):
        single = single or raw.byte(i) == 39
        double = double or raw.byte(i) == 34
    var quote = 34 if single and not double else 39
    var result = "b" + chr(quote)
    var digits = String("0123456789abcdef")
    for i in range(raw.size):
        var byte = raw.byte(i)
        if byte == quote or byte == 92:
            result += "\\" + chr(byte)
        elif byte == 9:
            result += "\\t"
        elif byte == 10:
            result += "\\n"
        elif byte == 13:
            result += "\\r"
        elif byte >= 32 and byte < 127:
            result += chr(byte)
        else:
            result += "\\x" + chr(Int(digits[byte=byte >> 4])) + chr(Int(digits[byte=byte & 15]))
    return result + chr(quote)


def archive_float_text(bits: UInt64, width: Int, print_mode: Int) raises -> String:
    if width == 4:
        return sequence_float_text(UInt32(bits), print_mode)
    var half = width == 2
    var sign_shift = 15 if half else 63
    var infinity = UInt64(0x7C00) if half else UInt64(0x7FF0000000000000)
    var magnitude = bits & (UInt64(0x7FFF) if half else UInt64(0x7FFFFFFFFFFFFFFF))
    if magnitude > infinity:
        return "nan"
    if magnitude == infinity:
        return "-inf" if bits >> UInt64(sign_shift) else "inf"
    if magnitude == 0:
        return "-0.0" if bits >> UInt64(sign_shift) else "0.0"
    var value = Float64(bitcast[DType.float16](UInt16(bits))) if half else bitcast[DType.float64](bits)
    var locale = SequenceNumericLocale()
    var buffer = Array[c_char, 64](uninitialized=True)
    var general = String("%.*g")
    var significant = 5 if half else 17
    if print_mode <= 113:
        # NumPy's half legacy path delegates to float formatting (six digits),
        # while binary64 uses DOUBLEPREC_STR=12; HALFPREC_STR is not that path.
        var used = external_call["snprintf", c_int, num_fixed_args=3](buffer.unsafe_ptr(), c_size_t(64), general.as_c_string_slice().unsafe_ptr(), c_int(6 if half else 12), value)
        if used < 0 or used >= 64:
            raise Error("native archive scalar formatting failed")
        var result = String(unsafe_from_utf8_ptr=buffer.unsafe_ptr())
        if result.find(".") < 0 and result.find("e") < 0:
            result += ".0"
        _ = locale^
        return result
    for precision in range(1, significant + 1):
        var used = external_call["snprintf", c_int, num_fixed_args=3](buffer.unsafe_ptr(), c_size_t(64), general.as_c_string_slice().unsafe_ptr(), c_int(precision), value)
        if used < 0 or used >= 64:
            raise Error("native archive scalar formatting failed")
        # Public strtod preserves subnormal round trips (errno may be ERANGE
        # for a representable subnormal; only the resulting bits matter).
        var candidate = external_call["strtod", Float64](buffer.unsafe_ptr(), Int(0))
        var equal = UInt64(bitcast[DType.uint16](Float16(candidate))) == bits if half else bitcast[DType.uint64](candidate) == bits
        if equal:
            significant = precision
            break
    # NumPy2.3 chooses 1e3 for half; earlier/legacy<=2.2 and binary64
    # retain 1e16. Match the unrounded scalar's notation decision.
    var maximum_fixed = Float64(1e3) if half and print_mode > 202 else Float64(1e16)
    var scientific = abs(value) < Float64(1e-4) or abs(value) >= maximum_fixed
    var exponent_format = String("%.*e")
    var used = external_call["snprintf", c_int, num_fixed_args=3](buffer.unsafe_ptr(), c_size_t(64), exponent_format.as_c_string_slice().unsafe_ptr(), c_int(significant - 1), value)
    if used < 0 or used >= 64:
        raise Error("native archive scalar formatting failed")
    var exponential = String(unsafe_from_utf8_ptr=buffer.unsafe_ptr())
    if scientific:
        _ = locale^
        return exponential
    # Expanding the shortest candidate avoids inventing low integral digits.
    var marker = exponential.find("e")
    var exponent = Int(String(exponential[byte=marker + 1:]))
    var negative = exponential.startswith("-")
    var digits = String("")
    for i in range(1 if negative else 0, marker):
        if exponential[byte=i] != 46:
            digits += String(exponential[byte=i:i + 1])
    var point = exponent + 1
    var result = String("-") if negative else String("")
    if point <= 0:
        result += "0."
        for i in range(-point):
            result += "0"
        result += digits
    else:
        for i in range(max(point, digits.byte_length())):
            if i == point:
                result += "."
            result += String(digits[byte=i:i + 1]) if i < digits.byte_length() else String("0")
        if point >= digits.byte_length():
            result += ".0"
    _ = locale^
    return result


def archive_scalar_text(raw: ArchiveSpan, dtype: String, print_mode: Int) raises -> String:
    if dtype == "<f2" or dtype == "<f4" or dtype == "<f8":
        var width = 2 if dtype == "<f2" else (4 if dtype == "<f4" else 8)
        return archive_float_text(archive_data_word(raw, 0, width), width, print_mode)
    if dtype == "<i8":
        return String(bitcast[DType.int64](archive_data_word(raw, 0, 8)))
    if dtype == "<i4":
        return String(bitcast[DType.int32](UInt32(archive_data_word(raw, 0, 4))))
    if dtype == "<u4":
        return String(UInt32(archive_data_word(raw, 0, 4)))
    return String(raw.byte(0))
