# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host read of admitted integer ids and labels (no Python import).

Plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.3, lane A3
(a3-admit-decomp-prep-metrics, 2026-11-08): `admit_ids`. Every public door
that turns a caller's integer labels or ids (classification metric labels,
approximate-neighbor candidate ids) into int64 or int32 words reads these;
no door keeps its own byte-assembled walk.

    admitted_ids_i64(addr, s0, s1, rows, cols, item, signed, swap, dst) -> Bool
        the rows x cols integer view at byte address `addr` (byte strides
        s0, s1; items of `item` bytes, sign-extended when `signed`, byte
        reversed when `swap`) as C-order int64; False when an unsigned
        64-bit id does not fit int64 (`_buffer.as_i64_c` refuses it)
    admitted_ids_i32(addr, s0, s1, rows, cols, item, signed, swap, dst) -> Bool
        the same view narrowed to C-order int32 in the same pass; False when
        any id lies outside int32 (the `c.min() < -(1 << 31) or c.max() >=
        (1 << 31)` refusal and `_as_i32_1d`'s range rule); such an id is
        written saturated. The one int32 id cast: lane A4's copy in
        core/admitted_input_nd.mojo (the Byte-LM token ids) merged in here
    admitted_narrow_i32(src, count, dst) -> Bool
        `count` int64 words narrowed to int32; False when one lies outside

Rows are independent (each writes only its own cells and its task's flag),
so they are split into contiguous row tasks over the host's cores
(core/host_predict_threads.mojo, with its grain floor: a small input runs on
the calling thread). Each item is one typed load (`admitted_raw_word`), not
a byte assembly. The verdict is the AND of the tasks' verdicts, so neither
the split nor the read order changes a word or a refusal.

A destination may be uninitialised; on a False verdict its words are
unspecified and the caller refuses and discards it (every caller does).

Moved here (no value change): `mcls_read_i64` and `mcls_narrow_i32`
(core/mcls_impl.mojo) and `ann_candidates_i32` / `_ann_item`
(core/neighbors_ann_impl.mojo), which were serial byte walks; the ANN check
and its narrowing copy were two passes and are one now.
"""
from std.memory import bitcast
from core.admitted_input import admitted_raw_word
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _AID_I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _AID_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _AID_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _AID_I32_MIN = Int64(-2147483648)
comptime _AID_I32_MAX = Int64(2147483647)


@always_inline
def _aid_raw(address: Int, item: Int, swap: Bool) -> UInt64:
    """The item of `item` bytes at `address`, zero-extended: one typed load
    in native (little-endian) order, the reversed bytes when `swap`."""
    if not swap:
        return admitted_raw_word(address, item)
    var bytes = _AID_U8P(unsafe_from_address=address)
    var raw = UInt64(0)
    for k in range(item):  # small-loop(bytes of one item): at most 8, a non-native exporter only
        raw |= UInt64(bytes[item - 1 - k]) << UInt64(k * 8)
    return raw


@always_inline
def _aid_signed(raw: UInt64, item: Int) -> Int64:
    """`raw` (an item of `item` bytes) sign-extended to int64."""
    var word = raw
    if item < 8:
        var top = UInt64(1) << UInt64(item * 8 - 1)
        if (word & top) != 0:
            word |= ~((top << 1) - 1)
    return bitcast[DType.int64](word)


def _aid_flags_all(flags: List[UInt8], tasks: Int) -> Bool:
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[t] == UInt8(0):
            return False
    return True


def admitted_ids_i64(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, signed: Bool, swap: Bool, dst: _AID_I64P,
) -> Bool:
    """The rows x cols integer view as C-order int64 words in `dst`; False
    when an unsigned 64-bit id does not fit int64."""
    # parallel: id read over row tasks across host cores (was a serial byte walk)
    if rows <= 0 or cols <= 0:
        return True
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AID_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm signed, imm swap, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var fits = True
        for r in range(lo, hi):  # admitted ids: this task's rows
            for c in range(cols):  # admitted ids: the ids of one row
                var raw = _aid_raw(addr + r * s0 + c * s1, item, swap)
                var value: Int64
                if signed:
                    value = _aid_signed(raw, item)
                else:
                    if item == 8 and raw > UInt64(9223372036854775807):
                        fits = False
                    value = bitcast[DType.int64](raw)
                dst.unsafe_store(r * cols + c, value)
        if not fits:
            flag_ptr.unsafe_store(t, UInt8(0))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var verdict = _aid_flags_all(flags, tasks)
    _ = flags^
    return verdict


def admitted_ids_i32(
    addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, signed: Bool, swap: Bool, dst: _AID_I32P,
) -> Bool:
    """The rows x cols integer (or bool) view narrowed to C-order int32
    words in `dst`; False when any id lies outside int32. An id outside int32
    is written saturated (-2^31 or 2^31 - 1, `cast_i32_bits`' rule), so a
    caller that keeps the words (the Byte-LM token ids, whose embedding check
    refuses both saturated values by position) reads the same words as the
    former core/admitted_input_nd.mojo copy."""
    # parallel: id read and int32 range rule over row tasks across host cores (was two serial byte walks)
    if rows <= 0 or cols <= 0:
        return True
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AID_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm signed, imm swap, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var fits = True
        for r in range(lo, hi):  # admitted ids: this task's rows
            for c in range(cols):  # admitted ids: the ids of one row
                var raw = _aid_raw(addr + r * s0 + c * s1, item, swap)
                var value = Int64(0)
                if signed:
                    value = _aid_signed(raw, item)
                    if value < _AID_I32_MIN:
                        fits = False
                        value = _AID_I32_MIN
                    elif value > _AID_I32_MAX:
                        fits = False
                        value = _AID_I32_MAX
                elif raw > UInt64(2147483647):
                    fits = False
                    value = _AID_I32_MAX
                else:
                    value = bitcast[DType.int64](raw)
                dst.unsafe_store(r * cols + c, Int32(value))
        if not fits:
            flag_ptr.unsafe_store(t, UInt8(0))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var verdict = _aid_flags_all(flags, tasks)
    _ = flags^
    return verdict


def admitted_narrow_i32(src: _AID_I64P, count: Int, dst: _AID_I32P) -> Bool:
    """`count` int64 words narrowed to int32 (`a.astype("<i4")` after the
    int32 range rule); False when one lies outside int32."""
    # parallel: int32 range rule over contiguous tasks across host cores (was a serial walk)
    if count <= 0:
        return True
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AID_U8P](flags.unsafe_ptr())

    def _span(t: Int) {imm src, imm count, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, count)
        var fits = True
        for i in range(lo, hi):  # admitted ids: this task's words
            var v = src.unsafe_load(i)
            if v < _AID_I32_MIN or v > _AID_I32_MAX:
                fits = False
                v = 0
            dst.unsafe_store(i, Int32(v))
        if not fits:
            flag_ptr.unsafe_store(t, UInt8(0))

    if tasks <= 1:
        _span(0)
    else:
        host_parallelize(_span, tasks)
    var verdict = _aid_flags_all(flags, tasks)
    _ = flags^
    return verdict
