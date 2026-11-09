# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host gather of a model-selection fold's rows, with the finite test
in the same pass (lane a5-admit-core-followup, 2026-11-08; plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 step 4;
docs/plans/REGRESSIONS_0866_VS_0855.md row R13).

A `_arena_io.DeviceRows` fold is rows `indices` (int64) of a float32 C-order
base that model selection holds on the host (and resident on the device).
Its X used to reach a host-address kernel as: device gather out of the
resident base, DOWNLOAD, host finite scan, kernel re-upload. A door whose
kernel reads host words now takes the rows straight out of the host base:

    admitted_gather_rows_f32(base, base_rows, row_words, indices, rows, dst) -> Int
        dst[r, :] = base[indices[r], :] for every r, row tasks across the
        host's cores (grain floor), each row one SIMD copy that tests the
        exponent bits as it goes. Returns 0 (all finite), 1 (a NaN or an
        infinity) or 2 (an index outside [0, base_rows)).

No device round trip and no second pass: the copy is the pass. The words
are copied, never converted, so `dst` holds the bytes the device gather
held (the same rows in the same order): no output bit changes.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _ARG_U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _ARG_I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _ARG_U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _ARG_W = 16
comptime ARG_FINITE = 0
comptime ARG_NONFINITE = 1
comptime ARG_OUT_OF_RANGE = 2


@always_inline
def _arg_copy_row(src: _ARG_U32P, dst: _ARG_U32P, n: Int) -> Bool:
    """Copy `n` words and return True when every one is finite."""
    var exp1 = UInt32(0x7F800000)
    var exp = SIMD[DType.uint32, _ARG_W](exp1)
    var bad = SIMD[DType.bool, _ARG_W](fill=False)
    var i = 0
    var body = n - n % _ARG_W
    while i < body:
        var words = src.unsafe_load[width=_ARG_W, alignment=1](i)
        dst.unsafe_store[width=_ARG_W, alignment=1](i, words)
        bad = bad | (words & exp).eq(exp)
        i += _ARG_W
    var finite = not bad.reduce_or()
    while i < n:
        var w = src.unsafe_load[alignment=1](i)
        dst.unsafe_store(i, w)
        if (w & exp1) == exp1:
            finite = False
        i += 1
    return finite


def admitted_gather_rows_f32(
    base: Int, base_rows: Int, row_words: Int, indices: Int, rows: Int, dst: Int,
) -> Int:
    """Rows `indices[0:rows]` (int64 words at `indices`) of the C-order
    float32 base at `base` (`base_rows` x `row_words`) into the C-order
    destination at `dst` (uninitialised is fine: every cell is written
    unless an index is out of range, which the caller refuses). Rows are
    independent, so they go in contiguous row tasks over the host's cores;
    the flags are ORs over tasks, so no split changes a verdict."""
    # parallel: fold row gather and finiteness test over row tasks across host cores (R13: was a device gather, download and host scan)
    if rows <= 0 or row_words <= 0:
        return ARG_FINITE
    var tasks = host_predict_task_count_work(rows, rows * row_words)
    var chunk = host_predict_chunk(rows, tasks)
    # Two flags per task: [2t] a non-finite word, [2t + 1] an index out of range.
    var flags = List[UInt8](length=2 * tasks, fill=UInt8(0))
    var flag_ptr = rebind[_ARG_U8P](flags.unsafe_ptr())
    var index_ptr = _ARG_I64P(unsafe_from_address=indices)

    def _rows(t: Int) {imm base, imm base_rows, imm row_words, imm index_ptr, imm rows, imm dst, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, rows)
        var finite = True
        for r in range(lo, hi):  # native input marshaling: this task's fold rows
            var k = Int(index_ptr.unsafe_load(r))
            if k < 0 or k >= base_rows:
                flag_ptr.unsafe_store(2 * t + 1, UInt8(1))
                return
            var src = _ARG_U32P(unsafe_from_address=base + k * row_words * 4)
            var out = _ARG_U32P(unsafe_from_address=dst + r * row_words * 4)
            if not _arg_copy_row(src, out, row_words):
                finite = False
        if not finite:
            flag_ptr.unsafe_store(2 * t, UInt8(1))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    var status = ARG_FINITE
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        if flags[2 * t + 1] != UInt8(0):
            status = ARG_OUT_OF_RANGE
        elif flags[2 * t] != UInt8(0) and status == ARG_FINITE:
            status = ARG_NONFINITE
    _ = flags^
    return status
