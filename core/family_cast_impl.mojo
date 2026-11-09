# SPDX-License-Identifier: Apache-2.0
"""Small reductions of the classical family owners
(bindings/family_common.mojo): the minimum a chi2 kernel's negative-input
refusal reads, the maximum, and a constant label fill. The casts and the
finiteness scan that lived here are the one admission path
(core/admitted_input.mojo, core/host_finite_scan.mojo)."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


# TOMBSTONE: family_raw_word (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_raw_word`.


# TOMBSTONE: family_cast_rows (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


# TOMBSTONE: family_cast_rows_finite (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


comptime _FCI_F32P = MutPointer[Float32, MutUntrackedOrigin]


def _family_extreme_f32[maximum: Bool](src: _FCI_F32P, count: Int) -> Float32:
    """The serial walk's minimum (or maximum) of `count` >= 1 float32 words,
    split into contiguous tasks across the host's cores. The serial rule
    starts at src[0] and never lets a NaN win; task 0 keeps that rule word
    for word (a NaN src[0] stays the answer), every later task reduces its
    non-NaN words (a NaN task result is skipped), and the tasks combine in
    order with the serial comparison, so the answer is the serial one."""
    # parallel: one reduction over contiguous tasks across host cores (audit R12, was serial)
    var tasks = host_predict_task_count_work(count, count)
    if tasks <= 1:
        var v = src[0]
        for i in range(1, count):  # admitted input: one reduction on the calling thread (below the grain floor)
            var x = src[i]
            comptime if maximum:
                if x > v:
                    v = x
            else:
                if x < v:
                    v = x
        return v
    var chunk = host_predict_chunk(count, tasks)
    var parts = List[Float32](length=tasks, fill=Float32(0))
    var parts_ptr = rebind[_FCI_F32P](parts.unsafe_ptr())

    def _span(t: Int) {imm src, imm count, imm chunk, imm parts_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, count)
        if lo >= hi:
            # An empty trailing task (the ceiling split): a NaN, skipped below.
            parts_ptr.unsafe_store(t, bitcast[DType.float32](UInt32(0x7FC00000)))
            return
        var v = src.unsafe_load(lo)
        for i in range(lo + 1, hi):  # admitted input: this task's words
            var x = src.unsafe_load(i)
            var take: Bool
            comptime if maximum:
                take = x > v
            else:
                take = x < v
            if take or (t > 0 and v != v):
                v = x
        parts_ptr.unsafe_store(t, v)

    host_parallelize(_span, tasks)
    var v = parts[0]
    for t in range(1, tasks):  # driver-loop: over the reduction tasks' results (at most one per core)
        var x = parts[t]
        comptime if maximum:
            if x > v:
                v = x
        else:
            if x < v:
                v = x
    _ = parts^
    return v


def family_min_f32(src: _FCI_F32P, count: Int) -> Float32:
    """The minimum of `count` float32 words (count >= 1); a NaN never wins,
    as `Array.min` after a NaN-free cast."""
    return _family_extreme_f32[False](src, count)


# TOMBSTONE: family_cast_rows_f64 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f64`.


# TOMBSTONE: family_all_finite_f32 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/host_finite_scan.mojo `host_all_finite_f32`.


def family_max_f32(src: _FCI_F32P, count: Int) -> Float32:
    """The maximum of `count` float32 words (count >= 1)."""
    return _family_extreme_f32[True](src, count)


def family_fill_i32(out_buf: MutPointer[Int32, MutUntrackedOrigin], n: Int, value: Int32):
    for i in range(n):  # native output marshaling: one constant label per row
        out_buf[i] = value
