# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling of the classical family owners
(bindings/family_common.mojo): the `_buffer.as_f32_c` cast (the same
classical_f32_bits conversion as core/classical_arrays_impl.mojo, with no
finiteness refusal, as as_f32_c made none), and the minimum a chi2 kernel's
negative-input refusal reads."""
from std.memory import bitcast
from bindings.classical_cast import classical_f32_bits
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, cast_w64_bits


def family_cast_rows(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    for r in range(rows):  # native input marshaling: input rows, never a model or training loop
        for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for k in range(item):
                raw |= UInt64(bytes[k]) << UInt64(k * 8)
            dst[r * cols + c] = bitcast[DType.float32](classical_f32_bits(raw, kind, item))


def family_min_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The minimum of `count` float32 words (count >= 1); a NaN never wins,
    as `Array.min` after a NaN-free cast."""
    var lo = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked input
        if src[i] < lo:
            lo = src[i]
    return lo


def family_cast_rows_f64(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int,
    dst: MutPointer[Float64, MutUntrackedOrigin],
):
    """`_buffer.as_f64_c`'s cast: a float widens exactly, an integer or a
    bool converts to the nearest binary64."""
    for r in range(rows):  # native input marshaling: input rows, never a model or training loop
        for c in range(cols):  # native input marshaling: numeric buffer items
            var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
            var raw = UInt64(0)
            for k in range(item):
                raw |= UInt64(bytes[k]) << UInt64(k * 8)
            var word = cast_w64_bits(raw, kind, item)
            var value: Float64
            if kind == CAST_KIND_SIGNED:
                value = Float64(bitcast[DType.int64](word))
            elif kind == CAST_KIND_UNSIGNED or kind == CAST_KIND_BOOL:
                value = Float64(word)
            else:
                value = bitcast[DType.float64](word)
            dst[r * cols + c] = value


def family_all_finite_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Bool:
    """True when none of `count` float32 words is a NaN or an infinity
    (`_buffer.all_finite`)."""
    for i in range(count):  # native input marshaling: one finiteness scan over the checked input
        var bits = bitcast[DType.uint32](src[i])
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return False
    return True


def family_max_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The maximum of `count` float32 words (count >= 1)."""
    var hi = src[0]
    for i in range(1, count):  # native input marshaling: one reduction over the checked input
        if src[i] > hi:
            hi = src[i]
    return hi


def family_fill_i32(out_buf: MutPointer[Int32, MutUntrackedOrigin], n: Int, value: Int32):
    for i in range(n):  # native output marshaling: one constant label per row
        out_buf[i] = value
