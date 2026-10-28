# SPDX-License-Identifier: Apache-2.0
# Prophet input storage conversion. Metadata is captured at the provider;
# these loops consume only native dimensions, strides and typed bytes.
from std.memory import bitcast
from core.cast_in import cast_f32_bits, cast_w64_bits, CAST_KIND_FLOAT, CAST_KIND_SIGNED, cast_in_kind
from core.native_scalar_decode import NativeRealBits, native_scalar_real

comptime _U8 = MutPointer[UInt8, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64 = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64 = MutPointer[Int64, MutUntrackedOrigin]


def _raw(base: Int, offset: Int, item: Int, native_order: Bool) -> UInt64:
    var bytes = _U8(unsafe_from_address=base + offset)
    var raw = UInt64(0)
    for j in range(item):
        raw |= UInt64(bytes[j]) << UInt64(8 * (j if native_order else item - 1 - j))
    return raw


def _real32(value: NativeRealBits) -> UInt32:
    var sign = UInt32(value.negative) << 31
    if value.special == 1:
        return sign | UInt32(0x7F800000)
    if value.special == 2:
        return sign | UInt32(0x7FC00000)
    var highest = value.highest()
    if highest < 0:
        return sign
    var exponent = highest + value.scale
    if exponent > 127:
        return sign | UInt32(0x7F800000)
    if exponent < -126:
        return sign | UInt32(value._rounded(-149 - value.scale))
    var significand = value._rounded(highest - 23)
    if significand >= UInt64(1 << 24):
        significand >>= 1
        exponent += 1
    if exponent > 127:
        return sign | UInt32(0x7F800000)
    return sign | (UInt32(exponent + 127) << 23) | UInt32(significand & UInt64(0x7FFFFF))


def prophet_numeric_copy(base: Int, dims: List[Int], strides: List[Int], item: Int,
                         kind: Int, native_order: Bool, dest: Int, wide: Bool) raises:
    var count = 1
    for dimension in dims:
        count *= dimension
    var indices = List[Int](length=len(dims), fill=0)
    var offset = 0
    for i in range(count):
        if kind < 0:
            # Extended and complex primitive arrays retain the native real
            # component conversion. The provider emits the complex warning.
            var code = String("f") if kind == -2 else (String("d") if kind == -3 else String("g"))
            var width = item if kind == -1 else item // 2
            var real = native_scalar_real(_U8(unsafe_from_address=base), offset, code, width, native_order)
            if wide:
                _F64(unsafe_from_address=dest)[i] = bitcast[DType.float64](real.float64_bits())
            else:
                _F32(unsafe_from_address=dest)[i] = bitcast[DType.float32](_real32(real))
        elif wide:
            var raw = _raw(base, offset, item, native_order)
            var word = cast_w64_bits(raw, kind, item)
            var number = bitcast[DType.float64](word) if kind == CAST_KIND_FLOAT else (
                Float64(bitcast[DType.int64](word)) if kind == CAST_KIND_SIGNED else Float64(word))
            _F64(unsafe_from_address=dest)[i] = number
        else:
            var raw = _raw(base, offset, item, native_order)
            _F32(unsafe_from_address=dest)[i] = bitcast[DType.float32](cast_f32_bits(raw, kind, item))
        var axis = len(dims) - 1
        while axis >= 0:
            indices[axis] += 1
            offset += strides[axis]
            if indices[axis] < dims[axis]:
                break
            offset -= strides[axis] * dims[axis]
            indices[axis] = 0
            axis -= 1


def _calendar_days(year: Int, month: Int) -> Int:
    var prior = year - 1
    var days = 365 * (year - 1970) + (prior // 4 - 1969 // 4) - (prior // 100 - 1969 // 100) + (prior // 400 - 1969 // 400)
    var month_days: List[Int] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    days += month_days[month]
    if month >= 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days += 1
    return days


def _nanoseconds(raw: UInt64, unit: String, multiplier: Int) raises -> Int64:
    if raw == UInt64(0x8000000000000000):
        return bitcast[DType.int64](raw)
    var original = Int(bitcast[DType.int64](raw))
    if unit == "ps" or unit == "fs" or unit == "as":
        var divisor = 1000 if unit == "ps" else (1000000 if unit == "fs" else 1000000000)
        # Divide before multiplying: a fine-unit dtype multiplier need not
        # fit its intermediate ticks into int64 when the nanoseconds do.
        return bitcast[DType.int64](UInt64(original // divisor) * UInt64(multiplier)
            + UInt64((original % divisor) * multiplier // divisor))
    var ticks = Int(bitcast[DType.int64](raw * UInt64(multiplier)))
    var factor = UInt64(1)
    if unit == "Y":
        ticks = _calendar_days(ticks + 1970, 0)
        factor = 86400000000000
    elif unit == "M":
        ticks = _calendar_days(ticks // 12 + 1970, ticks % 12)
        factor = 86400000000000
    elif unit == "W":
        factor = 604800000000000
    elif unit == "D":
        factor = 86400000000000
    elif unit == "h":
        factor = 3600000000000
    elif unit == "m":
        factor = 60000000000
    elif unit == "s":
        factor = 1000000000
    elif unit == "ms":
        factor = 1000000
    elif unit == "us":
        factor = 1000
    elif unit == "ps":
        ticks //= 1000
    elif unit == "fs":
        ticks //= 1000000
    elif unit == "as":
        ticks //= 1000000000
    elif unit != "ns":
        raise Error("ValueError: ProphetForecaster: datetime64 requires an explicit supported time unit")
    return bitcast[DType.int64](UInt64(ticks) * factor)


def _gcd(var a: Int, var b: Int) -> Int:
    while b != 0:
        var remainder = a % b
        a = b
        b = remainder
    return a


def _common_ticks(raw: UInt64, unit: String, multiplier: Int, target: String, target_multiplier: Int) raises -> Int64:
    if raw == UInt64(0x8000000000000000):
        return bitcast[DType.int64](raw)
    var source = unit
    var word = raw
    var numerator = UInt64(multiplier)
    var denominator = target_multiplier
    if source == "Y" or source == "M":
        if target == "Y" or target == "M":
            numerator *= UInt64(12 if source == "Y" and target == "M" else 1)
            return bitcast[DType.int64](word * (numerator // UInt64(denominator)))
        var ticks = Int(bitcast[DType.int64](word * numerator))
        word = UInt64(_calendar_days(ticks + 1970, 0) if source == "Y" else _calendar_days(ticks // 12 + 1970, ticks % 12))
        source = "D"
        numerator = 1
    var units: List[String] = ["W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"]
    var factors: List[Int] = [7, 24, 60, 60, 1000, 1000, 1000, 1000, 1000, 1000]
    var start = -1
    var stop = -1
    for i in range(len(units)):
        if units[i] == source:
            start = i
        if units[i] == target:
            stop = i
    if start < 0 or stop < start or denominator < 1:
        raise Error("ValueError: unsupported common datetime unit metadata")
    var cancel = _gcd(Int(numerator), denominator)
    numerator //= UInt64(cancel)
    denominator //= cancel
    for i in range(start, stop):
        cancel = _gcd(factors[i], denominator)
        numerator *= UInt64(factors[i] // cancel)
        denominator //= cancel
    # A common temporal dtype is an exact integer refinement of each input.
    # Cancel its divisor before the wrapped product, including ps/fs/as units.
    if denominator != 1:
        raise Error("ValueError: common datetime unit is not an integral refinement")
    return bitcast[DType.int64](word * numerator)


def prophet_datetime_copy(base: Int, dims: List[Int], strides: List[Int], native_order: Bool,
                          unit: String, multiplier: Int, dest: Int,
                          target: String = "ns", target_multiplier: Int = 1, common: Bool = False) raises:
    var count = 1
    for dimension in dims:
        count *= dimension
    var indices = List[Int](length=len(dims), fill=0)
    var offset = 0
    for i in range(count):
        var raw = _raw(base, offset, 8, native_order)
        _I64(unsafe_from_address=dest)[i] = (_common_ticks(raw, unit, multiplier, target, target_multiplier)
            if common else _nanoseconds(raw, unit, multiplier))
        var axis = len(dims) - 1
        while axis >= 0:
            indices[axis] += 1
            offset += strides[axis]
            if indices[axis] < dims[axis]:
                break
            offset -= strides[axis] * dims[axis]
            indices[axis] = 0
            axis -= 1
