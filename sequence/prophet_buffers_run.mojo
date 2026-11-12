# SPDX-License-Identifier: Apache-2.0
"""RUN of `providers/python/prophet_buffers_pyapi.mojo` (wave 8 run split):
the unit and multiplier of a datetime64/timedelta64 dtype string, typed Mojo
only. Imports no Python (tools/python_free_dirs.sh). The statements are the
former `_time_unit` parse unchanged; the door keeps the `dtype.str` read and
packs the tuple (lane rs-sequence-tokenizer, 2026-11-12)."""


def prophet_time_unit(text: String) raises -> Tuple[String, Int]:
    """`("generic", 1)` for a unitless dtype, else the bracketed unit and its
    leading multiplier (`"<M8[10s]"` -> `("s", 10)`)."""
    var parts = text.split("[")
    var unit = String("generic")
    var multiplier = 1
    if len(parts) == 2:
        var tail = String(parts[1])
        var stop = tail.byte_length() - 1
        var start = 0
        while start < stop:  # driver-loop: native datetime dtype unit metadata
            var digit = String(tail[byte=start:start + 1])
            if digit < "0" or digit > "9":
                break
            start += 1
        if start > 0:
            multiplier = Int(String(tail[byte=0:start]))
        unit = String(tail[byte=start:stop])
    return (unit^, multiplier)
