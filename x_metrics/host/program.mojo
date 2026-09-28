# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's host runner: the same PLANNED program as
x_metrics/device.mojo (x_metrics/plan.mojo), each stage's units run in
ascending t over the caller's arena plus the plan's scratch. The one
schedule difference: a sort merge pass runs as a two-pointer merge per run
pair (`sort_merge_pair_unit`) instead of one binary search per element;
both write the unique merged order. No accelerator import, so the CPU-only
host binding compiles it."""
from std.memory import memcpy
from x_metrics.common import FP, IP, STAGE_INTS
from x_metrics.units import N_OPS, run_unit
from x_metrics.plan import plan_program, N_USER_OPS, OP_SORT_MERGE
from x_metrics.par import sort_merge_pair_unit


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_host_ptr(FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages)


def run_program_host_ptr(f: FP, arena_len: Int, qbase: IP, stages: Int, legacy: Bool = False) raises:
    """`legacy` runs the caller's stages unplanned (the seam gate's
    reference for the planned schedules); every binding call plans."""
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_USER_OPS:
            raise Error(String("x_metrics: unknown op ", op))
    if legacy:
        _run(f, qbase, stages)
        return
    var pl = plan_program(qbase, stages, arena_len)
    var q = pl.rows.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if pl.size == arena_len:
        _run(f, q, pl.stages)
    else:
        var work = List[Float32](length=pl.size, fill=Float32(0))
        var wp = work.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if arena_len > 0:
            memcpy(dest=wp, src=f, count=arena_len)
        _run(wp, q, pl.stages)
        if arena_len > 0:
            memcpy(dest=f, src=wp, count=arena_len)
        _ = len(work)
    _ = len(pl.rows)


def _run(f: FP, qbase: IP, stages: Int):
    for s in range(stages):
        var op = Int(qbase.unsafe_load(s * STAGE_INTS))
        var total = Int(qbase.unsafe_load(s * STAGE_INTS + 1))
        var q = qbase + (s * STAGE_INTS + 2)
        if op == OP_SORT_MERGE:
            var n = Int(q.unsafe_load(0))
            var w = Int(q.unsafe_load(1))
            var pairs = (total // n) * ((n + 2 * w - 1) // (2 * w))
            for t in range(pairs):
                sort_merge_pair_unit(t, f, q)
            continue
        comptime for k in range(N_OPS):
            if op == k:
                for t in range(total):
                    run_unit[k](t, f, q)
