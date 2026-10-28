# SPDX-License-Identifier: Apache-2.0
# Native category/range decisions for NumPy temporal scalar unboxing.
# Source contract: numpy/_core/src/multiarray/datetime.c,
# convert_datetime_to_pyobject / convert_timedelta_to_pyobject. No date object,
# NumPy .item(), numeric conversion slot or Python calendar helper executes.


def temporal_floor(value: Int64, divisor: UInt64) -> Int64:
    if value >= 0:
        return Int64(UInt64(value) // divisor)
    var magnitude = UInt64(0) - UInt64(value)
    var quotient = magnitude // divisor
    if magnitude % divisor != 0:
        quotient += 1
    return Int64(UInt64(0) - quotient)


def temporal_object_kind(word: Int64, layout: String, delta: Bool) raises -> Int:
    # 0 original integer, 1 None, 2 date, 3 datetime, 4 timedelta.
    if UInt64(word) == UInt64(1) << UInt64(63):
        return 1
    var unit = String("")
    var multiple = String("")
    var bracket = False
    for i in range(layout.byte_length()):  # small-loop(dtype: native temporal unit and optional multiplier)
        var byte = UInt8(layout.unsafe_ptr()[i])
        if byte == 91:
            bracket = True
        elif bracket and byte != 93:
            if byte >= 48 and byte <= 57:
                multiple += String(layout[byte=i:i+1])
            else:
                unit += String(layout[byte=i:i+1])
    if unit == "ns" or unit == "ps" or unit == "fs" or unit == "as" or (delta and (unit == "" or unit == "Y" or unit == "M")):
        return 0
    if unit == "":
        return 1
    var multiplier = Int(multiple) if multiple.byte_length() > 0 else 1
    var scaled = Int64(UInt64(word) * UInt64(multiplier))
    var days = scaled
    var divisor = UInt64(1)
    if unit == "W":
        days = Int64(UInt64(scaled) * UInt64(7))
    elif unit == "h":
        divisor = 24
    elif unit == "m":
        divisor = 1440
    elif unit == "s":
        divisor = 86400
    elif unit == "ms":
        divisor = 86400000
    elif unit == "us":
        divisor = 86400000000
    if divisor > 1:
        days = temporal_floor(scaled, divisor)
    if delta:
        # npy_timedeltastruct.day is int64; the original integer is returned
        # outside Python's +/-999999999-day range, before any narrowing.
        return 0 if days < -999999999 or days > 999999999 else 4
    var outside = days < -719162 or days >= 2932897
    if unit == "Y":
        var year = Int64(UInt64(scaled) + UInt64(1970))
        outside = year < 1 or year > 9999
    elif unit == "M":
        var year = temporal_floor(scaled, UInt64(12))
        year = Int64(UInt64(year) + UInt64(1970))
        outside = year < 1 or year > 9999
    if outside:
        return 0
    return 3 if divisor > 1 else 2
