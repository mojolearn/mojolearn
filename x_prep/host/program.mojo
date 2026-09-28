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
from x_prep.host.power import pt_fit_host_unit
from x_prep.host.target import te_enc_host_groups, te_enc_host_group
from x_prep.host.mutual_info import mi_cc_host_stage, mi_cd_host_stage, mi_dc_host_stage

#: Ops of x_prep/units.mojo the host runs through its own spelling of the
#: SAME words (each file says why): 0 `sort_cols` (x_prep/host/sort.mojo),
#: 44 `pt_fit` (x_prep/host/power.mojo); and ops whose units the host runs
#: GROUPED, one task item per group of units: 21 `te_enc`
#: (x_prep/host/target.mojo, one group per fold, feature and target column);
#: and ops the host runs as a WHOLE STAGE (an index per column, then the
#: points across the pool): 68 `mi_cc`, 69 `mi_cd`, 94 `mi_dc`
#: (x_prep/host/mutual_info.mojo).
comptime OP_SORT_COLS = 0
comptime OP_TE_ENC = 21
comptime OP_PT_FIT = 44
comptime OP_MI_CC = 68
comptime OP_MI_CD = 69
comptime OP_MI_DC = 94


@always_inline
def _host_unit[K: Int](t: Int, f: FP, q: IP):
    comptime if K == OP_SORT_COLS:
        sort_cols_host_unit(t, f, q)
    elif K == OP_PT_FIT:
        pt_fit_host_unit(t, f, q)
    else:
        run_unit[K](t, f, q)


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_host_ptr(FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages)


def _items[K: Int](total: Int, q: IP) -> Int:
    """The task items of a stage of `total` units: its groups for a grouped
    op, else its units."""
    comptime if K == OP_TE_ENC:
        return te_enc_host_groups(total, q)
    else:
        return total


@always_inline
def _host_item[K: Int](t: Int, f: FP, q: IP):
    comptime if K == OP_TE_ENC:
        te_enc_host_group(t, f, q)
    else:
        _host_unit[K](t, f, q)


def _run_stage[K: Int](total: Int, f: FP, q: IP):
    """Items [0, items) of one stage: serially on the calling thread when one
    task covers them, else contiguous ranges across the pool."""
    comptime if K == OP_MI_CC:
        mi_cc_host_stage(total, f, q)
        return
    elif K == OP_MI_CD:
        mi_cd_host_stage(total, f, q)
        return
    elif K == OP_MI_DC:
        mi_dc_host_stage(total, f, q)
        return
    var items = _items[K](total, q)
    var tasks = host_predict_task_count(items)
    if tasks <= 1:
        for t in range(items):
            _host_item[K](t, f, q)
        return
    var chunk = host_predict_chunk(items, tasks)

    def body(c: Int) {imm f, imm q, imm chunk, imm items}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > items:
            hi = items
        for t in range(lo, hi):
            _host_item[K](t, f, q)

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
