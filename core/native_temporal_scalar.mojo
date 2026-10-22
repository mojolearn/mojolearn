# SPDX-License-Identifier: Apache-2.0
# Native temporal scalar boxing policy, without constructing datetime objects.
# Calendar/timedelta bounds are Python's supported range. Fixed-width wrap and
# the pre-NumPy2.2 day narrowing preserve the historical conversion contract.
# Sources: numpy/core/src/multiarray/datetime.c (v1.26.4) and
# numpy/_core/src/multiarray/datetime.c (v2.2.0), convert_*_to_pyobject.
from std.memory import bitcast


def temporal_wrap_product(value: Int, multiplier: Int) -> Int:
    return Int(bitcast[DType.int64](UInt64(value) * UInt64(multiplier)))


def temporal_result_kind(ticks: Int, datetime: Bool, unit: String,
                        multiplier: Int, old_day_width: Bool) -> Int:
    # 0 raw integer, 1 None, 2 date, 3 datetime, 4 timedelta.
    if UInt64(ticks) == (UInt64(1) << 63) or (datetime and unit == "generic"):
        return 1
    if unit == "ns" or unit == "ps" or unit == "fs" or unit == "as":
        return 0
    if not datetime and (unit == "generic" or unit == "Y" or unit == "M"):
        return 0
    var value = temporal_wrap_product(ticks, multiplier)
    if datetime and unit == "Y":
        var year = Int(bitcast[DType.int64](UInt64(value) + UInt64(1970)))
        return 2 if year >= 1 and year <= 9999 else 0
    if datetime and unit == "M":
        var year = value // 12 + 1970
        return 2 if year >= 1 and year <= 9999 else 0
    var days = value
    if unit == "W":
        days = temporal_wrap_product(value, 7)
    elif unit == "h":
        days = value // 24
    elif unit == "m":
        days = value // 1440
    elif unit == "s":
        days = value // 86400
    elif unit == "ms":
        days = value // 86400000
    elif unit == "us":
        days = value // 86400000000
    if datetime:
        # Days since 1970 for 0001-01-01 and 9999-12-31. Subday values
        # divide with floor, including negative ticks before the epoch.
        if days < -719162 or days > 2932896:
            return 0
        return 2 if unit == "W" or unit == "D" else 3
    if old_day_width:
        days = Int(bitcast[DType.int32](UInt32(UInt64(days) & UInt64(4294967295))))
    return 4 if days >= -999999999 and days <= 999999999 else 0
