# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's host runner: the same program as x_prep/device.mojo, on
the caller's arena, in place. No accelerator import, so the CPU-only host
binding compiles it.

THE THREAD SPLIT (lane prep-cpu, 2026-09-28). The device launches one thread
per unit of a stage, all at once, and joins before the next stage; a unit
therefore reads only what earlier stages wrote and writes only its own
outputs (x_prep/common.mojo: every loop INSIDE a unit is its reduction
order). The host runs the same contract: each stage's units are cut into
contiguous ranges of `t`, one task per range, run by `host_parallelize` in
the calling thread's floating-point environment (DEVIATION 5900), joined
before the next stage. Which thread runs a unit never changes what the unit
computes, so the bits are the serial walk's at every task count
(MOJOLEARN_CPU_THREADS = 1, 3 or the default; `core/host_predict_threads.mojo`
holds the one count policy). THIS IS NOT A NUMERIC ROW."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.host.sort import sort_cols_host_unit

#: op 0 of x_prep/units.mojo, `sort_cols`: the host runs x_prep/host/sort.mojo,
#: the same output words (that file says why).
comptime OP_SORT_COLS = 0


@always_inline
def _host_unit[K: Int](t: Int, f: FP, q: IP):
    comptime if K == OP_SORT_COLS:
        sort_cols_host_unit(t, f, q)
    else:
        run_unit[K](t, f, q)


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_host_ptr(FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages)


def _run_stage[K: Int](total: Int, f: FP, q: IP):
    """Units [0, total) of one stage: serially on the calling thread when one
    task covers them, else contiguous ranges across the pool."""
    var tasks = host_predict_task_count(total)
    if tasks <= 1:
        for t in range(total):
            _host_unit[K](t, f, q)
        return
    var chunk = host_predict_chunk(total, tasks)

    def body(c: Int) {imm f, imm q, imm chunk, imm total}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > total:
            hi = total
        for t in range(lo, hi):
            _host_unit[K](t, f, q)

    host_parallelize(body, tasks)


def run_program_host_ptr(f: FP, arena_len: Int, qbase: IP, stages: Int) raises:
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_OPS:
            raise Error(String("x_prep: unknown op ", op))
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        var total = Int(qbase.unsafe_load(s * STAGE_INTS + 1))
        var q = qbase + (s * STAGE_INTS + 2)
        comptime for k in range(N_OPS):
            if op == k:
                _run_stage[k](total, f, q)
