# SPDX-License-Identifier: Apache-2.0
# Float32 output formatting for the sequence tensor's native array protocol.
# This is output preparation, not an optimizer/training kernel.
from std.ffi import c_char, c_int, c_size_t, external_call
from std.memory import bitcast
from std.math import floor, log10


struct SequenceNumericLocale(Movable):
    # POSIX newlocale(0, "C", null) creates an independent C locale. uselocale
    # is thread-local; the caller's locale is restored even on formatting errors.
    var locale: Optional[MutPointer[UInt8, MutUntrackedOrigin]]
    var previous: Optional[MutPointer[UInt8, MutUntrackedOrigin]]

    def __init__(out self) raises:
        var name = String("C")
        var created = external_call["newlocale", Optional[MutPointer[UInt8, MutUntrackedOrigin]]](
            c_int(0), name.as_c_string_slice().unsafe_ptr(), Optional[MutPointer[UInt8, MutUntrackedOrigin]](None))
        if not created:
            raise Error("RuntimeError: native float formatting could not create a C locale")
        var previous = external_call["uselocale", Optional[MutPointer[UInt8, MutUntrackedOrigin]]](created.value())
        if not previous:
            # Construction has not published either owner field. Release the
            # successful allocation explicitly if selecting it failed.
            external_call["freelocale", NoneType](created.value())
            raise Error("RuntimeError: native float formatting could not select its C locale")
        self.locale = created^
        self.previous = previous^

    def __del__(deinit self):
        if self.previous:
            _ = external_call["uselocale", Optional[MutPointer[UInt8, MutUntrackedOrigin]]](self.previous.value())
        if self.locale:
            external_call["freelocale", NoneType](self.locale.value())


def sequence_float_text(bits: UInt32) raises -> String:
    """Shortest decimal that round-trips to this binary32, in NumPy's
    scalar notation: scientific below 1e-4 or from 1e6, fixed otherwise.

    Each binary32 has a round-tripping decimal with at most nine significant
    digits. libc rounds the exact widened value; the first round-tripping
    candidate therefore gives the shortest precision. The final notation is
    chosen from the unrounded value, including values just below 1e-4.
    """
    var magnitude = bits & UInt32(0x7FFFFFFF)
    if magnitude > UInt32(0x7F800000):
        return "nan"
    if magnitude == UInt32(0x7F800000):
        return "-inf" if (bits >> 31) != 0 else "inf"
    if magnitude == 0:
        return "-0.0" if (bits >> 31) != 0 else "0.0"
    var numeric_locale = SequenceNumericLocale()
    var value = Float64(bitcast[DType.float32](bits))
    var buffer = Array[c_char, 64](uninitialized=True)
    var general = String("%.*g")
    var significant = 9
    for precision in range(1, 10):
        var used = external_call["snprintf", c_int, num_fixed_args=3](
            buffer.unsafe_ptr(), c_size_t(64), general.as_c_string_slice().unsafe_ptr(),
            c_int(precision), value)
        if used < 0 or used >= 64:
            raise Error("RuntimeError: native float formatting failed")
        var candidate = String(unsafe_from_utf8_ptr=buffer.unsafe_ptr())
        if bitcast[DType.uint32](Float32(Float64(candidate))) == bits:
            significant = precision
            break
    var scientific = abs(value) < Float64(1e-4) or abs(value) >= Float64(1e6)
    var precision = significant - 1
    if not scientific:
        precision = max(0, significant - 1 - Int(floor(log10(abs(value)))))
    var format = String("%.*e") if scientific else String("%.*f")
    var used = external_call["snprintf", c_int, num_fixed_args=3](
        buffer.unsafe_ptr(), c_size_t(64), format.as_c_string_slice().unsafe_ptr(),
        c_int(precision), value)
    if used < 0 or used >= 64:
        raise Error("RuntimeError: native float formatting failed")
    var result = String(unsafe_from_utf8_ptr=buffer.unsafe_ptr())
    if not scientific:
        if result.find(".") >= 0:
            while result.endswith("0"):
                var shortened = String(result[byte=0:result.byte_length() - 1])
                result = shortened^
            if result.endswith("."):
                result += "0"
        else:
            result += ".0"
    _ = numeric_locale^
    return result


def sequence_write_text(bits: UInt32, destination: Int, width: Int, unicode: Bool, little: Bool) raises:
    var text = sequence_float_text(bits)
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    var count = min(text.byte_length(), width // 4 if unicode else width)
    for index in range(width):
        out[index] = 0
    for index in range(count):
        out[index * 4 + (0 if little else 3) if unicode else index] = UInt8(text.unsafe_ptr()[index])
