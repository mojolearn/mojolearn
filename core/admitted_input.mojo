# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host cast and check of admitted input words (no Python import).

Plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2, lane A0
(consol-a0-admission, 2026-11-08). Every public door that turns a caller's
X, y or sample_weight into float32 (or float64) words reaches these through
`bindings/admitted_input.mojo` (`admit_x`, `admit_y`, `admit_weights`); no
family keeps its own cast loop or finiteness scan.

    admitted_raw_word(address, item)          one little-endian item, typed load
    admitted_cast_f32(addr, s0, s1, rows, cols, item, kind, dst) -> Bool
                                              the classical_f32_bits cast of a
                                              rows x cols view (byte strides
                                              s0, s1) into C-order `dst`, row
                                              tasks across the host's cores;
                                              True when every word is finite
    admitted_cast_f64(...)                    the same view widened to float64
    admitted_check_f32(..., name, borrowed, dst)
                                              admitted_cast_f32, or for a
                                              borrowed float32 C-order view the
                                              parallel finiteness scan only;
                                              refuses "Input <name> contains
                                              NaN or infinity"
    admitted_check_weights_f32(input, output, rows, scalar) -> Bool
                                              the sign test (and scalar fill)
                                              of admitted float32 weights, row
                                              tasks across the host's cores;
                                              True when any weight is positive

Numerics: none of these is arithmetic on the values. The cast writes the same
classical_f32_bits word per item that the serial walks wrote, to the same
cell; the finiteness and sign verdicts are ANDs / ORs over tasks, so neither
the split nor the read order can change an output bit or a refusal.

Destinations may be uninitialised: every function that writes `dst` writes
all rows * cols cells before it returns (the caller discards `dst` on a
refusal).

Moved here word for word (no bit change): `family_raw_word`,
`family_cast_rows_finite`, `family_cast_rows`, `family_cast_rows_f64`
(core/family_cast_impl.mojo) and `classical_cast_rows`,
`classical_check_weights` (core/classical_arrays_impl.mojo, deleted). The
weight check was a serial loop; it now runs in row tasks with the same
verdicts.
"""
from std.memory import bitcast
from core.classical_cast import classical_f32_bits
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, cast_w64_bits
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.host_finite_scan import host_all_finite_f32_strided

# TOMBSTONE: classical_cast_rows and classical_check_weights (core/classical_arrays_impl.mojo, merged into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. They are
# `admitted_check_f32` and `admitted_check_weights_f32` below.

comptime _AI_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _AI_F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _AI_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _AI_EXP = UInt32(0x7F800000)


@always_inline
def admitted_raw_word(address: Int, item: Int) -> UInt64:
    """The little-endian item of `item` bytes at `address`, zero-extended
    to 64 bits: one typed unaligned load for a 1, 2, 4 or 8-byte item (the
    word the byte-by-byte assembly built, as the inputs are checked to be in
    native little-endian order), the byte assembly otherwise."""
    if item == 4:
        return UInt64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load[alignment=1](0))
    if item == 8:
        return MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load[alignment=1](0)
    if item == 2:
        return UInt64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load[alignment=1](0))
    if item == 1:
        return UInt64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address).unsafe_load(0))
    var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    var raw = UInt64(0)
    for k in range(item):  # native input marshaling: bytes of one item of an unusual width
        raw |= UInt64(bytes[k]) << UInt64(k * 8)
    return raw


def admitted_cast_f32(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, dst: _AI_F32P,
) -> Bool:
    """The classical_f32_bits word of every item of the rows x cols view at
    byte address `addr` (byte strides `s0`, `s1`), written to `dst` in C
    order, and True when none of the written words is a NaN or an infinity.
    Rows are independent (each writes only its own cells and its task's
    flag), so they are split into contiguous row tasks over the host's cores
    (core/host_predict_threads.mojo, with its grain floor); the output words
    do not depend on the split, and the flag is the AND of the tasks'.
    Each item is one typed load (`admitted_raw_word`), not a byte assembly.
    A caller that does not refuse non-finite input ignores the flag."""
    # parallel: cast and finiteness test over row tasks across host cores (0.8.66 regression C1, 13 ns/element serial)
    if rows <= 0 or cols <= 0:
        return True
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AI_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm kind, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var finite = True
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
                var bits = classical_f32_bits(admitted_raw_word(addr + r * s0 + c * s1, item), kind, item)
                dst.unsafe_store(r * cols + c, bitcast[DType.float32](bits))
                if (bits & _AI_EXP) == _AI_EXP:
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


def admitted_cast_f64(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, dst: _AI_F64P,
):
    """The view widened to float64 in C order (`_buffer.as_f64_c`): a float
    widens exactly, an integer or a bool converts to the nearest binary64.
    Rows are split into contiguous row tasks over the host's cores, as
    `admitted_cast_f32`; the output words do not depend on the split."""
    # parallel: cast over row tasks across host cores (0.8.66 regression C1, 13 ns/element serial)
    if rows <= 0 or cols <= 0:
        return
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm kind, imm dst, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            for c in range(cols):  # native input marshaling: numeric buffer items
                var word = cast_w64_bits(admitted_raw_word(addr + r * s0 + c * s1, item), kind, item)
                var value: Float64
                if kind == CAST_KIND_SIGNED:
                    value = Float64(bitcast[DType.int64](word))
                elif kind == CAST_KIND_UNSIGNED or kind == CAST_KIND_BOOL:
                    value = Float64(word)
                else:
                    value = bitcast[DType.float64](word)
                dst.unsafe_store(r * cols + c, value)

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)


def admitted_check_f32(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, name: String,
    borrowed: Bool, dst: _AI_F32P,
) raises:
    """Refuse NaN/infinity in the rows x cols input at `addr` (byte strides
    s0, s1) and, unless `borrowed` (float32 in C order, used as is), write
    its classical_f32_bits words to `dst` in C order.
    A borrowed input is not rebuilt: its words are only tested, by the
    shared SIMD scan split across the host's cores
    (`core/host_finite_scan.mojo`). Any other input goes through
    `admitted_cast_f32` (row tasks across the host's cores). The refusal and
    its wording are the serial walk's; `dst` is the caller's fresh result,
    discarded on a refusal, so writing every cell before refusing changes no
    output."""
    # parallel: finiteness scan / cast across host cores (0.8.66 regression C1, 13 ns/element serial)
    var finite: Bool
    if borrowed:
        finite = host_all_finite_f32_strided(addr, s0, s1, rows, cols)
    else:
        finite = admitted_cast_f32(addr, s0, s1, rows, cols, item, kind, dst)
    if not finite:
        raise Error("Input " + name + " contains NaN or infinity")


def admitted_check_weights_f32(input: _AI_F32P, output: _AI_F32P, rows: Int, scalar: Bool) raises -> Bool:
    """The sign test of `rows` admitted (finite) float32 weights: refuses a
    negative weight with scikit-learn's wording and returns True when any
    weight is positive. A `scalar` weight is `input[0]`, written to every
    one of the `rows` cells of `output` (a vector weight is not copied:
    `output` is then not written). Rows are split into contiguous tasks over
    the host's cores; the verdicts are an OR over tasks, so they are the
    serial walk's verdicts for any split."""
    # parallel: weight sign test and scalar fill over row tasks across host cores
    if rows <= 0:
        return False
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    # Two flags per task: [2t] a negative weight, [2t + 1] a positive one.
    var flags = List[UInt8](length=2 * tasks, fill=UInt8(0))
    var flag_ptr = rebind[_AI_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm input, imm output, imm rows, imm scalar, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var negative = False
        var positive = False
        for i in range(lo, hi):  # native input marshaling: weight validation and scalar expansion
            var v = input.unsafe_load(0 if scalar else i)
            if v < 0:
                negative = True
            if v > 0:
                positive = True
            if scalar:
                output.unsafe_store(i, v)
        if negative:
            flag_ptr.unsafe_store(2 * t, UInt8(1))
        if positive:
            flag_ptr.unsafe_store(2 * t + 1, UInt8(1))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var any_negative = False
    var any_positive = False
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[2 * t] != UInt8(0):
            any_negative = True
        if flags[2 * t + 1] != UInt8(0):
            any_positive = True
    _ = flags^
    if any_negative:
        raise Error("Negative values in data passed to `sample_weight`")
    return any_positive
