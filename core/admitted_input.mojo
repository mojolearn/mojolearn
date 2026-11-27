# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host cast and check of admitted input words (no Python import).

Plan plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) section 3.2, lane A0
(consol-a0-admission, 2026-11-08). Every public door that turns a caller's
X, y or sample_weight into float32 (or float64) words reaches these through
`bindings/admitted_input_unpack.mojo` (`admit_x`, `admit_y`, `admit_weights`); no
family keeps its own cast loop or finiteness scan.

    admitted_raw_word(address, item)          one little-endian item, typed load
    admitted_cast_f32(addr, s0, s1, rows, cols, item, kind, dst) -> Bool
                                              the classical_f32_bits cast of a
                                              rows x cols view (byte strides
                                              s0, s1) into C-order `dst`, row
                                              tasks across the host's cores;
                                              True when every word is finite
    admitted_cast_f32_words(..., transposed) -> Int
                                              the same cast, its verdict as a
                                              status: ADMITTED_ALL_FINITE,
                                              ADMITTED_NONFINITE, or the
                                              smallest column holding a
                                              finite float64 that rounds to an
                                              infinity (never raises)
    admitted_f64_overflow_refuse(column)      THE refusal of such a column
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
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, CAST_KIND_FLOAT, cast_w64_bits
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.host_finite_scan import host_all_finite_f32_strided
from core.admitted_rows_gather import _arg_copy_row

# TOMBSTONE: classical_cast_rows and classical_check_weights (core/classical_arrays_impl.mojo, merged into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. They are
# `admitted_check_f32` and `admitted_check_weights_f32` below.

comptime _AI_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _AI_F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _AI_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _AI_U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _AI_EXP = UInt32(0x7F800000)
comptime _AI_EXP64 = UInt64(0x7FF0000000000000)
comptime _AI_INTP = MutPointer[Int, MutUntrackedOrigin]
comptime _AI_NO_COLUMN = 9223372036854775807

# The status words of `admitted_cast_f32_words` (a status >= 0 is a column).
comptime ADMITTED_ALL_FINITE = -1
comptime ADMITTED_NONFINITE = -2

# float64 input (Andrew, 2026-11-27). An estimator that computes in float32
# converts a float64 input ONCE, here, at the native boundary, with one rule
# on every device and side: IEEE round-to-nearest-even (`f64_to_f32_bits`,
# core/cast_in.mojo, the soft narrowing every column already runs), a NaN
# stays a NaN (the quiet NaN 0x7FC00000), an infinity stays that infinity
# (the door then treats it as it treats a float32 infinity), and a FINITE
# value whose rounding is an infinity (magnitude at or above float32's
# largest finite value plus half an ulp, so every value that rounds to
# 3.4028235e38 or below is kept) is refused, naming its column, never turned
# into an infinity. The test rides in the cast pass and runs only on a word
# the cast already found non-finite: a finite input pays no extra work, and a
# float32 C-order input is still borrowed with no copy.
comptime ADMITTED_F32_MAX_TEXT = "3.4028234663852886e+38"


def admitted_f64_overflow_refuse(column: Int) raises:
    """THE refusal of a float64 input whose column `column` holds a finite
    value outside float32's range (every float32-computing door raises this
    one text)."""
    raise Error("ValueError: column " + String(column) + " holds a float64 value whose magnitude exceeds"
                + " float32's largest finite value (" + ADMITTED_F32_MAX_TEXT + "); float64 input is"
                + " converted to float32 once, with round-to-nearest-even, and this value would become"
                + " infinity, so it is refused")


@always_inline
def admitted_f64_rounds_to_inf(raw: UInt64, bits: UInt32, wide: Bool) -> Bool:
    """`raw` is a finite float64 item (`wide`: the item is float64) whose
    float32 word `bits` is an infinity."""
    return wide and (raw & _AI_EXP64) != _AI_EXP64 and (bits & UInt32(0x7FFFFFFF)) == _AI_EXP


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
    transposed: Bool = False,
) raises -> Bool:
    """`admitted_cast_f32_words`, refusing a float64 column that holds a
    finite value outside float32's range (`admitted_f64_overflow_refuse`);
    True when every written word is finite. `transposed`: the view is the
    transpose of the caller's matrix (an F-order cast), so the caller's
    column is the view's row."""
    var status = admitted_cast_f32_words(addr, s0, s1, rows, cols, item, kind, dst, transposed)
    if status >= 0:
        admitted_f64_overflow_refuse(status)
    return status == ADMITTED_ALL_FINITE


def admitted_cast_f32_words(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, dst: _AI_F32P,
    transposed: Bool = False,
) -> Int:
    """The classical_f32_bits word of every item of the rows x cols view at
    byte address `addr` (byte strides `s0`, `s1`), written to `dst` in C
    order, with its finiteness verdict as a status word (below).
    Rows are independent (each writes only its own cells and its task's
    flag), so they are split into contiguous row tasks over the host's cores
    (core/host_predict_threads.mojo, with its grain floor); the output words
    do not depend on the split, and the flag is the AND of the tasks'.
    Each item is one typed load (`admitted_raw_word`), not a byte assembly.
    Returns ADMITTED_ALL_FINITE, ADMITTED_NONFINITE, or (a float64 view) the
    smallest column (the smallest row when `transposed`) holding a finite
    value whose float32 word is an infinity; that test runs only on a word
    already found non-finite, so a finite input pays nothing for it. Each
    task keeps its own smallest column and the result is their minimum, so
    the split cannot change it."""
    # parallel: cast and finiteness test over row tasks across host cores (0.8.66 regression C1, 13 ns/element serial)
    if rows <= 0 or cols <= 0:
        return ADMITTED_ALL_FINITE
    # Lane fx-rows-rest (2026-11-18): a float32 view whose rows are runs of
    # adjacent words (unit column stride: a column slice X[:, a:b] or a row
    # slice X[::k] of a C-order block, a row-padded block) needs no cast,
    # only its rows packed. Each row is then ONE SIMD copy that tests the
    # exponent bits as it goes (`_arg_copy_row`, the fold gather's row copy),
    # in place of cols typed loads, kind dispatches and stores (the general
    # walk below; the two column-slice operands of a PLS/CCA transform, 1.1 M
    # items, are cast per call: call profile nvc-prof-20261118T182616Z-11b404).
    # Same words to the same cells (cast_f32_bits of a float32 item is the
    # item) and the same verdict (the exponent-bits test on the same words):
    # no bit moves. The grain floor counts one operation per word on this
    # path (as admitted_gather_rows_f32), bytes on the general walk as
    # before. Not keyed on any shape: any rows, cols and row stride. Owed
    # re-time beside the board row: a 1,000,000 x 10 column slice and a
    # non-board 300,000 x 7 slice of random float32.
    var row_runs = kind == CAST_KIND_FLOAT and item == 4 and (s1 == 4 or cols == 1)
    var tasks = host_predict_task_count_work(rows, rows * cols * (1 if row_runs else max(item, 1)))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AI_U8P](flags.unsafe_ptr())
    var firsts = List[Int](length=tasks, fill=_AI_NO_COLUMN)
    var first_ptr = rebind[_AI_INTP](firsts.unsafe_ptr())
    var wide = kind == CAST_KIND_FLOAT and item == 8
    var dst_addr = Int(dst)

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm kind, imm dst, imm chunk, imm flag_ptr, imm row_runs, imm dst_addr, imm first_ptr, imm wide, imm transposed}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var finite = True
        var first = _AI_NO_COLUMN
        if row_runs:
            for r in range(lo, hi):  # native input marshaling: this task's input rows, one run each
                if not _arg_copy_row(_AI_U32P(unsafe_from_address=addr + r * s0),
                                     _AI_U32P(unsafe_from_address=dst_addr + r * cols * 4), cols):
                    finite = False
            if not finite:
                flag_ptr.unsafe_store(t, UInt8(0))
            return
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
                var raw = admitted_raw_word(addr + r * s0 + c * s1, item)
                var bits = classical_f32_bits(raw, kind, item)
                dst.unsafe_store(r * cols + c, bitcast[DType.float32](bits))
                if (bits & _AI_EXP) == _AI_EXP:
                    finite = False
                    if admitted_f64_rounds_to_inf(raw, bits, wide):
                        var at = r if transposed else c
                        if at < first:
                            first = at
        if not finite:
            flag_ptr.unsafe_store(t, UInt8(0))
        if first < _AI_NO_COLUMN:
            first_ptr.unsafe_store(t, first)

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var all_finite = True
    var column = _AI_NO_COLUMN
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            all_finite = False
        if firsts[t] < column:
            column = firsts[t]
    _ = flags^
    _ = firsts^
    if column < _AI_NO_COLUMN:
        return column
    return ADMITTED_ALL_FINITE if all_finite else ADMITTED_NONFINITE


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
    borrowed: Bool, dst: _AI_F32P, transposed: Bool = False,
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
        finite = admitted_cast_f32(addr, s0, s1, rows, cols, item, kind, dst, transposed)
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
