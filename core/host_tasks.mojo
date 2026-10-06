# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU-only copy and row scheduling policy.

These implementations are moved unchanged from host_lanes. GPU staging
allocation belongs in host_storage; GPU kernels must not call these policies.
Thresholds express per-task work and copy amortization, not benchmark shapes.
"""
from std.math import max, min
from std.memory import unsafe_memcpy
from core.host_storage import HostF32Ptr
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count


#: Floats below which a copy is one memcpy on the calling thread.
comptime HOST_COPY_TASK_MIN = 1 << 20


def host_f32_copy(dst: HostF32Ptr, src: HostF32Ptr, n: Int):
    """`memcpy` of `n` floats, in chunks over host tasks when `n` is large
    (lane neural-pass8): a copy moves no bit, so the task count is a
    schedule knob. The 80 MB registry copies of the byte LM host step took
    a fifth of its wall on one thread of a 64-core host."""
    if n <= 0:
        return
    var tasks = 1
    if n >= 2 * HOST_COPY_TASK_MIN:
        tasks = max(1, min(host_predict_task_count(n // HOST_COPY_TASK_MIN), n // HOST_COPY_TASK_MIN))
    if tasks <= 1:
        unsafe_memcpy(dest=dst, src=src, count=n)
        return
    var chunk = (n + tasks - 1) // tasks
    def _copy(t: Int) {imm dst, imm src, imm n, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        if hi > lo:
            unsafe_memcpy(dest=dst.unsafe_offset(lo), src=src.unsafe_offset(lo), count=hi - lo)
    host_parallelize(_copy, tasks)


#: Scalar operations below which a row split is not worth a thread fork.
comptime HOST_ROW_TASK_MIN_WORK = 1 << 16


def host_row_tasks(rows: Int, work_per_row: Int) -> Int:
    """Tasks for a split of `rows` independent rows of about `work_per_row`
    operations each: the host thread policy (`core/host_predict_threads.mojo`)
    capped so every task gets at least HOST_ROW_TASK_MIN_WORK operations. A
    schedule knob: it moves no bit."""
    var work = rows * max(work_per_row, 1)
    if rows <= 1 or work < 2 * HOST_ROW_TASK_MIN_WORK:
        return 1
    return max(1, min(host_predict_task_count(rows), work // HOST_ROW_TASK_MIN_WORK))
