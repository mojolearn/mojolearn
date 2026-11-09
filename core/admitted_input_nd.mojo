# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one host cast of admitted N-d input words (no Python import): the
rank-N and integer-id members of the admission path of
`core/admitted_input.mojo`.

Plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 step 2
("`family_cast_rows_finite`'s shape generalised to N-d strides, so
`xseq_cast_nd` folds in"), lane A4 a4-admit-sequence-tsa-neural, 2026-11-08.

    admitted_cast_nd_f32(base, dims, steps, item, kind, dst) -> Bool
        the float32 word of every item of the strided rank-N array at byte
        address `base` (byte `steps` per axis), written to `dst` in C order;
        True when none of the written words is a NaN or an infinity.
        Integers take the one admission rule, `Array.astype`'s integer ->
        double -> float32 route (`classical_f32_bits`, as `admit_x`).
    admitted_ids_i32(addr, s0, s1, rows, cols, item, kind, dst)
        a strided (rows, cols) integer or bool array as C-order int32 ids
        (`cast_i32_bits`: exact, saturated outside int32).

Numerics: no arithmetic on values. Each item is one typed load
(`admitted_raw_word`) and one bit function, written to its own cell; the
finiteness verdict is an AND over tasks, so neither the split nor the read
order can change an output bit. Bit change (allowed between versions, owner
rule 2026-11-08): `xseq_cast_nd` had two integer rules; the doors that used
NumPy's direct integer -> float32 rounding (`cast_f32_bits`: xseq_f32 by
default, the VAR / forecast matrices) now take the one admission rule
(`classical_f32_bits`). The two differ only for an integer input of
magnitude above 2^53 (int64 / uint64); float and bool inputs, and every
float32 board input, keep their words. The host column runs this same
code, so NVIDIA, AMD, Apple and the host column change together.

Parallel plan: the array is read as rows x cols, cols the last axis (a rank-1
array is one element per row, so its elements split too). Rows are
independent (each writes only its own cells and its task's flag), so they are
split into contiguous row tasks over the host's cores with the grain floor
(core/host_predict_threads.mojo). Destinations may be uninitialised: every
cell is written before return.
"""
from std.memory import bitcast
from core.classical_cast import classical_f32_bits
from core.cast_in import cast_i32_bits
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.admitted_input import admitted_raw_word

comptime _AN_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _AN_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _AN_INTP = MutPointer[Int, MutUntrackedOrigin]
comptime _AN_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _AN_EXP = UInt32(0x7F800000)


def admitted_cast_nd_f32(
    base: Int, dims: List[Int], steps: List[Int], item: Int, kind: Int, dst: _AN_F32P,
) -> Bool:
    """`np.array(a, dtype=float32, order="C")` of the strided array at `base`
    into `dst` (element i in C order at `dst[i]`); True when every written
    word is finite. A caller that does not refuse non-finite input ignores
    the flag."""
    # parallel: N-d cast and finiteness test over row tasks across host cores (was the serial odometer xseq_cast_nd)
    var rank = len(dims)
    var total = 1
    for k in range(rank):  # driver-loop: over the array's rank metadata
        total *= dims[k]
    if total <= 0:
        return True
    var cols = 1 if rank <= 1 else dims[rank - 1]
    var col_step = 0 if rank <= 1 else steps[rank - 1]
    var rows = total // cols
    # The axes a row index decomposes over: all but the last (rank >= 2), the
    # one axis (rank 1), none (rank 0: one row at `base`).
    var lead = rank - 1 if rank >= 2 else rank
    var lead_dims = List[Int](length=max(lead, 1), fill=1)
    var lead_steps = List[Int](length=max(lead, 1), fill=0)
    for k in range(lead):  # driver-loop: over the array's rank metadata
        lead_dims[k] = dims[k]
        lead_steps[k] = steps[k]
    var dims_ptr = rebind[_AN_INTP](lead_dims.unsafe_ptr())
    var steps_ptr = rebind[_AN_INTP](lead_steps.unsafe_ptr())
    var tasks = host_predict_task_count_work(rows, total * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)
    var flags = List[UInt8](length=tasks, fill=UInt8(1))
    var flag_ptr = rebind[_AN_U8P](flags.unsafe_ptr())

    def _rows(t: Int) {imm base, imm item, imm kind, imm dst, imm cols, imm col_step, imm rows, imm lead, imm dims_ptr, imm steps_ptr, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var finite = True
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            var offset = 0
            var rest = r
            var axis = lead - 1
            while axis >= 0:  # driver-loop: the row's coordinate on each leading axis (rank metadata)
                var size = dims_ptr.unsafe_load(axis)
                offset += (rest % size) * steps_ptr.unsafe_load(axis)
                rest //= size
                axis -= 1
            for c in range(cols):  # native input marshaling: numeric buffer items preserve exact cast bits
                var raw = admitted_raw_word(base + offset + c * col_step, item)
                var bits = classical_f32_bits(raw, kind, item)
                dst.unsafe_store(r * cols + c, bitcast[DType.float32](bits))
                if (bits & _AN_EXP) == _AN_EXP:
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
    _ = lead_dims^
    _ = lead_steps^
    return all_finite


def admitted_ids_i32(addr: Int, s0: Int, s1: Int, rows: Int, cols: Int, item: Int, kind: Int, dst: _AN_I32P):
    """A strided (rows, cols) integer or bool array (byte strides `s0`, `s1`)
    as C-order int32 words in `dst` (`cast_i32_bits`: exact, saturated
    outside int32)."""
    # parallel: id cast over row tasks across host cores (was the serial neural_cast_i32_2d)
    if rows <= 0 or cols <= 0:
        return
    var tasks = host_predict_task_count_work(rows, rows * cols * max(item, 1))
    var chunk = host_predict_chunk(rows, tasks)

    def _rows(t: Int) {imm addr, imm s0, imm s1, imm rows, imm cols, imm item, imm kind, imm dst, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        for r in range(lo, hi):  # native input marshaling: this task's input rows
            for c in range(cols):  # native input marshaling: one id per position
                var bits = cast_i32_bits(admitted_raw_word(addr + r * s0 + c * s1, item), kind, item)
                dst.unsafe_store(r * cols + c, bitcast[DType.int32](bits))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
