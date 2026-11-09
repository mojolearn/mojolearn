# SPDX-License-Identifier: Apache-2.0
"""Small reductions of the classical family owners
(bindings/family_common.mojo): the minimum a chi2 kernel's negative-input
refusal reads, the maximum, and a constant label fill. The casts and the
finiteness scan that lived here are the one admission path
(core/admitted_input.mojo, core/host_finite_scan.mojo)."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


# TOMBSTONE: family_raw_word (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_raw_word`.


# TOMBSTONE: family_cast_rows (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


# TOMBSTONE: family_cast_rows_finite (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f32`.


# integ/pre-0867-w2: lane a3-admit-decomp-prep-metrics (0cb550c337) wrote a second
# parallel min / max (_family_extreme_f32, the same serial-walk answer, NaN never
# wins); one implementation kept, this one (lane a2-admit-family). A3's copy never
# reached main: recoverable at 0cb550c337.
def _family_extreme[minimum: Bool](src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    # The serial walk's verdict (`lo = src[0]`, then `if v < lo: lo = v`, or
    # `>` for the maximum) split into contiguous row tasks across the host's
    # cores (lane a2-admit-family, regression audit R12): task 0 starts at
    # src[0] exactly as the walk did; a later task starts at its first word
    # that is not a NaN (the walk's running value is never a NaN there unless
    # src[0] was, and then nothing replaces it); the partials combine in task
    # order with the same comparison. A NaN never wins, as before; the result
    # is the walk's for any split (min and max are exact, no rounding).
    # parallel: min / max over row tasks across host cores
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var parts = List[Float32](length=tasks, fill=Float32(0))
    var valid = List[UInt8](length=tasks, fill=UInt8(0))
    var part_ptr = rebind[MutPointer[Float32, MutUntrackedOrigin]](parts.unsafe_ptr())
    var valid_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](valid.unsafe_ptr())

    def _task(t: Int) {imm src, imm count, imm chunk, imm part_ptr, imm valid_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, count)
        if lo >= hi:
            return
        var i = lo
        var best = src[i]
        if t > 0:
            while i < hi and best != best:  # native input marshaling: this task's leading NaN words
                i += 1
                if i < hi:
                    best = src[i]
            if i >= hi:
                return
        for j in range(i + 1, hi):  # native input marshaling: this task's words
            var v = src[j]
            comptime if minimum:
                if v < best:
                    best = v
            else:
                if v > best:
                    best = v
        part_ptr.unsafe_store(t, best)
        valid_ptr.unsafe_store(t, UInt8(1))

    if tasks <= 1:
        _task(0)
    else:
        host_parallelize(_task, tasks)
    var result = parts[0]
    for t in range(1, tasks):  # driver-loop: over the row tasks' partials (at most one per core)
        if valid[t] != UInt8(0):
            comptime if minimum:
                if parts[t] < result:
                    result = parts[t]
            else:
                if parts[t] > result:
                    result = parts[t]
    _ = parts^
    _ = valid^
    return result


def family_min_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The minimum of `count` float32 words (count >= 1); a NaN never wins,
    as `Array.min` after a NaN-free cast. Row tasks across the host's cores."""
    return _family_extreme[True](src, count)


# TOMBSTONE: family_cast_rows_f64 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/admitted_input.mojo `admitted_cast_f64`.


# TOMBSTONE: family_all_finite_f32 (moved into the one admission path) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch. Now core/host_finite_scan.mojo `host_all_finite_f32`.


def family_max_f32(src: MutPointer[Float32, MutUntrackedOrigin], count: Int) -> Float32:
    """The maximum of `count` float32 words (count >= 1). Row tasks across
    the host's cores."""
    return _family_extreme[False](src, count)


# TOMBSTONE: family_fill_i32 (a serial constant-label fill) deleted 2026-11-09 by lane a6-device-x-family-classical; code recoverable at f59d913b0b.
# Restore: experiments/removed/A6_FAMILY_FILL_I32.patch. core/admitted_weights.mojo
# admitted_fill[DType.int32] fills in row tasks across the host's cores.
