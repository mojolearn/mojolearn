# SPDX-License-Identifier: Apache-2.0
"""Typed input marshaling of the classical family owners
(bindings/family_common.mojo): the `_buffer.as_f32_c` cast (the same
classical_f32_bits conversion as core/classical_arrays_impl.mojo, with no
finiteness refusal, as as_f32_c made none), and the minimum a chi2 kernel's
negative-input refusal reads."""
from std.memory import bitcast
from core.classical_cast import classical_f32_bits
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, cast_w64_bits
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


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


def family_cast_rows_finite(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int,
    dst: MutPointer[Float32, MutUntrackedOrigin],
) -> Bool:
    """`family_cast_rows` followed by `family_all_finite_f32` of its output,
    in one pass over the input (rows N352, N353 of
    docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md): the same
    classical_f32_bits word per item, written to the same cell, and True
    when none of the written words is a NaN or an infinity. Rows are
    independent (each writes only its own cells and its task's flag), so
    they are split into contiguous row tasks over the host's cores
    (core/host_predict_threads.mojo, with its grain floor); the output words
    do not depend on the split, and the flag is the AND of the tasks'."""
    if rows <= 0 or cols <= 0:
        return True
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm kind, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var finite = True
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
                var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + r * s0 + c * s1)
                var raw = UInt64(0)
                for k in range(item):
                    raw |= UInt64(bytes[k]) << UInt64(k * 8)
                var bits = classical_f32_bits(raw, kind, item)
                dst.unsafe_store(r * cols + c, bitcast[DType.float32](bits))
                if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    finite = False
        if not finite:
            flag_ptr.unsafe_store(t, UInt8(0))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var all_finite = True
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            all_finite = False
    _ = flags^
    return all_finite


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
