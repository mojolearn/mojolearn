# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's host runner: the same PLANNED program as
x_metrics/device.mojo (x_metrics/plan.mojo), each stage's units run over
the caller's arena plus the plan's scratch. The one schedule difference: a
sort merge pass runs as two-pointer merges of MERGE_SPAN-output spans of
each run pair (`sort_merge_span_unit`) instead of one binary search per
element; both write the unique merged order. No accelerator import, so the
CPU-only host binding compiles it.

THREADS (lane/metrics phase 5, 2026-09-27). A stage is the device's one
launch: its units are independent (the device runs them all at once, so no
unit reads a word another unit of the same stage writes), and each unit is
the same statements in the same order whoever runs it. The host therefore
splits every stage's units into contiguous ranges across the host pool
(`core/host_predict_threads.mojo`, `MOJOLEARN_CPU_THREADS`; unset: one task
per physical core) and joins before the next stage, exactly as the device's
stream orders its launches. Which thread runs a unit moves no bit: the arena
is word for word the one thread's at every thread count (the seam gate's
`check_parallel_schedules` runs 1, 2, 3 and 8 tasks against the sequential
units). A stage fans out only when it holds at least HOST_TASK_WORK rows of
work per task (`_unit_work`: roughly the rows one unit touches), so a
fixture of five rows never wakes the pool. The sequential Float32 prefixes
(DEVIATION 6107) stay one unit per problem.
"""
from std.memory import memcpy
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count, host_predict_chunk
from x_metrics.common import FP, IP, STAGE_INTS, LEAF
from x_metrics.units import N_OPS, run_unit
from x_metrics.plan import (
    plan_program, N_USER_OPS, OP_SORT_MERGE, OP_CS_HIST, OP_CS_SCAN_ROWS, OP_CS_PLACE,
    OP_FOLD_LEAF, OP_SORT_RUNS, OP_WPCT_SELECT, CS_CHUNK, OP_CM_CHUNK, CM_CHUNK,
)
from x_metrics.par import sort_merge_span_unit, merge_span_units, MERGE_SPAN, RUN

#: The least work (rows touched, `_unit_work`) one host task is given.
comptime HOST_TASK_WORK = 16384


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_host_ptr(FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages)


def run_program_host_ptr(f: FP, arena_len: Int, qbase: IP, stages: Int, legacy: Bool = False,
                         threads: Int = 0) raises:
    """`legacy` runs the caller's stages unplanned (the seam gate's
    reference for the planned schedules); every binding call plans.
    `threads` > 0 is the task ceiling (the seam gate's thread arms); 0 reads
    MOJOLEARN_CPU_THREADS (`core/host_predict_threads.mojo`)."""
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_USER_OPS:
            raise Error(String("x_metrics: unknown op ", op))
    if legacy:
        _run(f, qbase, stages, 1)
        return
    var pl = plan_program(qbase, stages, arena_len)
    var q = pl.rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if pl.size == arena_len:
        _run(f, q, pl.stages, threads)
    else:
        var work = List[Float32](length=pl.size, fill=Float32(0))
        var wp = work.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if arena_len > 0:
            memcpy(dest=wp, src=f, count=arena_len)
        _run(wp, q, pl.stages, threads)
        if arena_len > 0:
            memcpy(dest=f, src=wp, count=arena_len)
        _ = len(work)
    _ = len(pl.rows)


@always_inline
def _unit_work(op: Int) -> Int:
    """Roughly the rows one unit of `op` touches (the fan-out threshold
    only; never a numeric input)."""
    if op == OP_CS_HIST or op == OP_CS_PLACE:
        return CS_CHUNK
    if op == OP_CS_SCAN_ROWS:
        return 64
    if op == OP_FOLD_LEAF:
        return LEAF
    if op == OP_SORT_RUNS:
        return 4 * RUN
    if op == OP_SORT_MERGE:
        return MERGE_SPAN
    if op == OP_WPCT_SELECT:
        return 64
    if op == OP_CM_CHUNK:
        return CM_CHUNK
    if op < N_USER_OPS and op != 2 and op != 3 and op != 8 and op != 9:
        return HOST_TASK_WORK      # a caller's whole-column unit
    return 1


def _tasks(units: Int, op: Int, threads: Int) -> Int:
    var tasks = host_predict_task_count(units)
    if threads > 0:
        tasks = min(threads, units)
    var by_work = (units * _unit_work(op)) // HOST_TASK_WORK
    if tasks > by_work:
        tasks = by_work
    return max(tasks, 1)


def _run(f: FP, qbase: IP, stages: Int, threads: Int):
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        var total = Int(qbase.unsafe_load(s * STAGE_INTS + 1))
        var q = qbase + (s * STAGE_INTS + 2)
        var units = total
        if op == OP_SORT_MERGE:
            var n = Int(q.unsafe_load(0))
            units = merge_span_units(n, Int(q.unsafe_load(1)), total // n)
        if units <= 0:
            continue
        var tasks = _tasks(units, op, threads)
        var chunk = host_predict_chunk(units, tasks)

        def _range(c: Int) {imm f, imm q, imm op, imm units, imm chunk}:
            var lo = c * chunk
            var hi = min(lo + chunk, units)
            if op == OP_SORT_MERGE:
                for t in range(lo, hi):
                    sort_merge_span_unit(t, f, q)
                return
            comptime for k in range(N_OPS):
                if op == k:
                    for t in range(lo, hi):
                        run_unit[k](t, f, q)

        if tasks == 1:
            _range(0)
        else:
            host_parallelize(_range, tasks)
