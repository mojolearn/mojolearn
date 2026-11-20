# SPDX-License-Identifier: Apache-2.0
"""Typed pandas storage reads moved word for word from bindings/classical_pandas.mojo.

Lane a5-admit-core-followup (2026-11-08; docs/plans/REGRESSIONS_0866_VS_0855.md
row R4): a native-order item is ONE typed load (`admitted_raw_word`), not a
byte assembly, and the column cast and the mask check run in contiguous row
tasks across the host's cores (core/host_predict_threads.mojo grain floor:
a short column stays on the calling thread). Every row writes only its own
cell and its task's flag, so the words written and the refusals are the
serial loop's for any split: no output bit changes.
"""
from std.memory import bitcast
from core.cast_in import cast_w64_bits, f64_to_f32_bits
from core.classical_cast import classical_f32_bits
from core.admitted_input import admitted_raw_word
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _PD_U8P = MutPointer[UInt8, MutUntrackedOrigin]


def _pd_any_flag(flags: List[UInt8], tasks: Int) -> Bool:
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[t] != UInt8(0):
            return True
    return False


def pd_raw_at(base: Int, offset: Int, size: Int, native: Bool) -> UInt64:
    if native:
        # One typed little-endian load: the word the byte assembly below
        # builds for a native-order item (R4).
        return admitted_raw_word(base + offset, size)
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
    """Refuse a pandas nullable column whose mask marks any row missing."""
    # parallel: mask test over row tasks across host cores (R4, serial per row)
    if rows <= 0:
        return
    var tasks = host_predict_task_count_work(rows, rows * max(size, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[_PD_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm base, imm rows, imm stride, imm size, imm native, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for row in range(lo, hi):  # native input marshaling: refuse missing values before numeric preparation
            if pd_raw_at(base, row * stride, size, native) != 0:
                flag_ptr.unsafe_store(t, UInt8(1))
                return

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var missing = _pd_any_flag(flags, tasks)
    _ = flags^
    if missing:
        raise Error("TypeError: pandas input contains a missing nullable value")


def pd_scatter_numeric(
    base: Int, size: Int, native: Bool, kind: Int, row_stride: Int, col_stride: Int,
    col: Int, dest: Int, rows: Int, cols: Int, double_round: Bool,
    output: MutPointer[Float32, MutUntrackedOrigin], allow_nonfinite: Bool = False,
) raises:
    """Column `col` of a pandas block (`rows` items of `size` bytes, byte
    strides `row_stride` between rows and `col_stride` between columns) cast
    to float32 into column `dest` of the C-order rows x cols `output`, with
    pandas' promotion (`double_round`: through float64 first). Refuses NaN
    or infinity unless `allow_nonfinite`. Every row is written before a
    refusal; `output` is the caller's fresh result, discarded on one."""
    # parallel: typed pandas column cast over row tasks across host cores (R4, one thread with byte assembly)
    if rows <= 0:
        return
    var tasks = host_predict_task_count_work(rows, rows * max(size, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[_PD_U8P](flags.unsafe_ptr())
    var column = base + col * col_stride

    def _rows(t: Int) {imm column, imm size, imm native, imm kind, imm row_stride, imm dest, imm rows, imm cols, imm double_round, imm output, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var bad = False
        for row in range(lo, hi):  # native input marshaling: numeric conversion preserves pandas promotion before float32
            var raw = pd_raw_at(column, row * row_stride, size, native)
            var bits = f64_to_f32_bits(bitcast[DType.uint64](_pd_double(raw, kind, size))) if double_round else classical_f32_bits(raw, kind, size)
            if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                bad = True
            output.unsafe_store(row * cols + dest, bitcast[DType.float32](bits))
        if bad:
            flag_ptr.unsafe_store(t, UInt8(1))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var nonfinite = _pd_any_flag(flags, tasks)
    _ = flags^
    if nonfinite and not allow_nonfinite:
        raise Error("Input contains NaN or infinity")
