# SPDX-License-Identifier: Apache-2.0
"""Typed pandas storage reads moved word for word from bindings/classical_pandas.mojo."""
from std.memory import bitcast
from core.cast_in import cast_w64_bits, f64_to_f32_bits
from bindings.classical_cast import classical_f32_bits


def pd_raw_at(base: Int, offset: Int, size: Int, native: Bool) -> UInt64:
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base + offset)
    var raw = UInt64(0)
    for k in range(size):
        raw |= UInt64(pointer[k if native else size - 1 - k]) << UInt64(8 * k)
    return raw


def _pd_double(raw: UInt64, kind: Int, size: Int) -> Float64:
    var word = cast_w64_bits(raw, kind, size)
    if kind == 2:
        return bitcast[DType.float64](word)
    if kind == 1:
        return Float64(bitcast[DType.int64](word))
    return Float64(word)


def pd_mask_check(base: Int, rows: Int, stride: Int, size: Int, native: Bool) raises:
    for row in range(rows):  # native input marshaling: refuse missing values before numeric preparation
        if pd_raw_at(base, row * stride, size, native) != 0:
            raise Error("TypeError: pandas input contains a missing nullable value")


def pd_scatter_numeric(
    base: Int, size: Int, native: Bool, kind: Int, row_stride: Int, col_stride: Int,
    col: Int, dest: Int, rows: Int, cols: Int, double_round: Bool,
    output: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    for row in range(rows):  # native input marshaling: numeric conversion preserves pandas promotion before float32
        var raw = pd_raw_at(base, row * row_stride + col * col_stride, size, native)
        var bits = f64_to_f32_bits(bitcast[DType.uint64](_pd_double(raw, kind, size))) if double_round else classical_f32_bits(raw, kind, size)
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("Input contains NaN or infinity")
        output[row * cols + dest] = bitcast[DType.float32](bits)
