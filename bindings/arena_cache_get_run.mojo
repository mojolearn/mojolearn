# SPDX-License-Identifier: Apache-2.0
"""RUN of the arena cache key reader (bindings/arena_cache_get.mojo): the
typed integer arithmetic of `_integer`'s datetime/timedelta branch. No Python
import (tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12):
moved word for word; one scalar per call, nothing to split across cores."""


def _floor_quotient(value: Int64, divisor: UInt64) -> Int64:
    # Use unsigned magnitude so the result is independent of signed division
    # rounding and does not negate the minimum signed storage value.
    if value >= 0:
        return Int64(UInt64(value)//divisor)
    var magnitude = UInt64(0)-UInt64(value)
    var quotient = magnitude//divisor
    if magnitude%divisor != 0:
        quotient += 1
    return Int64(UInt64(0)-quotient)


def _temporal_integer_name(word: Int64, kind: String, layout: String) raises -> String:
    """`_integer`'s decision for a datetime ("M") or timedelta ("m") value
    with storage word `word` and dtype string `layout`: "" when `int()` returns
    the raw word, else the Python type name `int()` refuses."""
    var name = String("NoneType")
    if UInt64(word) != UInt64(1)<<63:
        var unit = String("")
        var multiple = String("")
        var bracket = False
        for i in range(layout.byte_length()):  # driver-loop: parse native dtype unit and multiplier metadata
            var byte = UInt8(layout.unsafe_ptr()[i])
            if byte == 91:
                bracket = True
            elif bracket and byte != 93:
                if byte >= 48 and byte <= 57:
                    multiple += String(layout[byte=i:i+1])
                else:
                    unit += String(layout[byte=i:i+1])
        if unit == "ns" or unit == "ps" or unit == "fs" or unit == "as" or (kind == "m" and (unit == "" or unit == "Y" or unit == "M")):
            return String("")
        if kind == "M" and unit == "":
            raise Error("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'")
        var multiplier = Int(multiple) if multiple.byte_length() > 0 else 1
        var scaled = Int64(UInt64(word)*UInt64(multiplier))
        # NumPy's public datetime scalar conversion returns the original
        # integer outside Python date/timedelta's representable range.
        # Reproduce that range decision from storage, without constructing
        # dates or invoking NumPy's item()/number conversion slots.
        # Source: numpy/core/src/multiarray/datetime.c,
        # convert_datetime_to_pyobject / convert_timedelta_to_pyobject.
        var days = scaled
        var divisor = 1
        if unit == "W":
            days = Int64(UInt64(scaled)*7)
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
            days = _floor_quotient(scaled,UInt64(divisor))
        if kind == "m":
            # npy_timedeltastruct.day is int64. Check Python's timedelta
            # range before narrowing; a large value must return the raw
            # integer rather than wrap into an apparently valid date.
            if days < -999999999 or days > 999999999:
                return String("")
            name = "datetime.timedelta"
        else:
            var outside = days < -719162 or days >= 2932897
            if unit == "Y":
                var year = Int64(UInt64(scaled)+1970)
                outside = year < 1 or year > 9999
            elif unit == "M":
                var year = _floor_quotient(scaled,12)
                year = Int64(UInt64(year)+1970)
                outside = year < 1 or year > 9999
            if outside:
                return String("")
            name = String("datetime.date") if unit == "Y" or unit == "M" or unit == "W" or unit == "D" else String("datetime.datetime")
    return name
