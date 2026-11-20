# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The weight status pass of admitted sample weights (no Python import).

Plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 3.2 step 5,
lane A2 admit-family (2026-11-08). `bindings/admitted_family.mojo`
`admit_family_weights` reaches these; they replace the serial walks of
core/family_weights_impl.mojo (deleted with this lane).

    admitted_fill[dtype](dst, n, value)       a scalar weight into every row
    admitted_weight_status[dtype](w, n, allow_negative) -> Int
                                              WEIGHTS_OK or the first refusal
                                              in prepare_targets' order:
                                              non-finite, then negative (when
                                              not allowed), then all zero

Both split the rows into contiguous tasks across the host's cores
(core/host_predict_threads.mojo, with its grain floor). The status is an
OR of per-task flags evaluated in the serial order (any non-finite word
first, then any negative, then "no positive and no negative"), so it is the
serial walk's verdict for any split: the serial walk's minimum and maximum
were only compared against zero. Numerics: none (a fill and predicates).
"""
from std.math import isfinite

# TOMBSTONE: family_weight_status, family_weight_status_f64, family_fill_f32, family_fill_f64 (core/family_weights_impl.mojo, serial weight marshaling) deleted 2026-11-08 by lane a2-admit-family; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A2_FAMILY_ADMISSION_HELPERS.patch. They are
# admitted_weight_status and admitted_fill below, in row tasks.
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime WEIGHTS_OK = 0
comptime WEIGHTS_NONFINITE = -3
comptime WEIGHTS_NEGATIVE = -4
comptime WEIGHTS_ALL_ZERO = -5

comptime _AW_U8P = MutPointer[UInt8, MutUntrackedOrigin]


def admitted_fill[dtype: DType](dst: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, value: Scalar[dtype]):
    """`value` into each of the `n` words at `dst`, row tasks across the
    host's cores."""
    # parallel: scalar weight fill over row tasks across host cores
    if n <= 0:
        return
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _fill(t: Int) {imm dst, imm n, imm value, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):  # native input marshaling: this task's weight rows
            dst.unsafe_store(i, value)

    if tasks <= 1:
        _fill(0)
    else:
        host_parallelize(_fill, tasks)


def admitted_weight_status[dtype: DType](w: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int,
                                         allow_negative: Bool) -> Int:
    """prepare_targets' weight status of the `n` words at `w`: WEIGHTS_OK,
    or WEIGHTS_NONFINITE, WEIGHTS_NEGATIVE (only when not `allow_negative`)
    or WEIGHTS_ALL_ZERO, in that order of precedence."""
    # parallel: weight finiteness / sign test over row tasks across host cores
    if n <= 0:
        return WEIGHTS_ALL_ZERO
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    # Three flags per task: [3t] a non-finite word, [3t + 1] a negative one,
    # [3t + 2] a positive one.
    var flags = List[UInt8](length=3 * tasks, fill=UInt8(0))
    var flag_ptr = rebind[_AW_U8P](flags.unsafe_ptr())

    def _scan(t: Int) {imm w, imm n, imm chunk, imm flag_ptr}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        var nonfinite = False
        var negative = False
        var positive = False
        for i in range(lo, hi):  # native input marshaling: this task's weight rows
            var v = w.unsafe_load(i)
            if not isfinite(v):
                nonfinite = True
            elif v < 0:
                negative = True
            elif v > 0:
                positive = True
        if nonfinite:
            flag_ptr.unsafe_store(3 * t, UInt8(1))
        if negative:
            flag_ptr.unsafe_store(3 * t + 1, UInt8(1))
        if positive:
            flag_ptr.unsafe_store(3 * t + 2, UInt8(1))

    if tasks <= 1:
        _scan(0)
    else:
        host_parallelize(_scan, tasks)
    var any_nonfinite = False
    var any_negative = False
    var any_positive = False
    for t in range(tasks):  # driver-loop: over the row tasks' flags (at most one per core)
        any_nonfinite = any_nonfinite or flags[3 * t] != UInt8(0)
        any_negative = any_negative or flags[3 * t + 1] != UInt8(0)
        any_positive = any_positive or flags[3 * t + 2] != UInt8(0)
    _ = flags^
    if any_nonfinite:
        return WEIGHTS_NONFINITE
    if any_negative and not allow_negative:
        return WEIGHTS_NEGATIVE
    if not (any_positive or any_negative):
        return WEIGHTS_ALL_ZERO
    return WEIGHTS_OK
