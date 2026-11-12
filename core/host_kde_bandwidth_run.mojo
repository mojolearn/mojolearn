# SPDX-License-Identifier: Apache-2.0
"""RUN of the KDE temporal bandwidth read (bindings/host_kde_bandwidth_native.mojo,
`kde_temporal_bandwidth`): the dtype unit text parse and the decode of the
eight exported bytes into the float the door returns. Typed Mojo only, no
Python import (tools/python_free_dirs.sh). The door reads the dtype text,
exports the buffer and reads the NumPy release (UNPACK) and boxes the
float (PACK). Lane rs-boosting-trees (2026-11-12): moved word for word.
One scalar per call: nothing to split across cores."""
from std.memory import bitcast
from core.native_scalar_decode import native_scalar_real
from core.native_temporal_scalar import temporal_result_kind


def kde_temporal_unit_run(text: String, mut unit: String, mut multiplier: Int) raises:
    """The `[<multiplier><unit>]` tail of a temporal dtype's `str`."""
    var parts = text.split("[")
    if len(parts) == 2:
        var tail = String(parts[1])
        var length = tail.byte_length() - 1
        var start = 0
        while start < length:  # driver-loop: parse exact native dtype unit metadata
            var digit = String(tail[byte=start:start + 1])
            if digit < "0" or digit > "9":
                break
            start += 1
        if start > 0:
            multiplier = Int(String(tail[byte=0:start]))
        unit = String(tail[byte=start:length])


def kde_temporal_decode_run(
    bytes: List[UInt8],
    data: MutPointer[UInt8, MutUntrackedOrigin],
    little: Bool,
    code: String,
    unit: String,
    multiplier: Int,
    release: Int,
) raises -> Float64:
    """The exported signed temporal word, as `float()` reads it, or the
    refusal `float()` raises for a date, datetime or timedelta result."""
    var word = UInt64(0)
    for i in range(8):  # driver-loop: one signed native temporal word, either byte order
        word |= UInt64(bytes[i]) << UInt64(8 * (i if little else 7 - i))
    var kind = temporal_result_kind(Int(bitcast[DType.int64](word)), code == "M", unit,
                                   multiplier, release < 202, release >= 205)
    var number = native_scalar_real(data, 0, String("q"), 8, little)
    if kind == 5:
        raise Error("OverflowError: Overflow when converting between datetime64 units")
    if kind != 0:
        var name = String("NoneType") if kind == 1 else String("datetime.date") if kind == 2 else String("datetime.datetime") if kind == 3 else String("datetime.timedelta")
        raise Error("TypeError: float() argument must be a string or a real number, not '" + name + "'")
    return bitcast[DType.float64](number.float64_bits())
