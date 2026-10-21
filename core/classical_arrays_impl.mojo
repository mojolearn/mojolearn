# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling moved word for word from bindings/classical_arrays.mojo."""
from std.memory import bitcast
from bindings.classical_cast import classical_f32_bits


def classical_cast_rows(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, name: String,
    borrowed: Bool, out: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    for r in range(rows):  # native input marshaling: input rows, never a model or training loop
        for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for k in range(item):
                raw |= UInt64(bytes[k]) << UInt64(k * 8)
            var bits = classical_f32_bits(raw, kind, item)
            if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                raise Error("Input " + name + " contains NaN or infinity")
            if not borrowed:
                out[r * cols + c] = bitcast[DType.float32](bits)


def classical_check_weights(
    input: MutPointer[Float32, MutUntrackedOrigin], output: MutPointer[Float32, MutUntrackedOrigin],
    rows: Int, scalar: Bool,
) raises -> Bool:
    var any_positive = False
    for i in range(rows):  # native input marshaling: weight validation and scalar expansion
        var v = input[0 if scalar else i]
        if v < 0:
            raise Error("Negative values in data passed to `sample_weight`")
        any_positive = any_positive or v > 0
        if scalar:
            output[i] = v
    return any_positive
