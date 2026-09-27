# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's host runner: the same program as x_prep/device.mojo, each
stage's units run in ascending t on the caller's arena, in place. No
accelerator import, so the CPU-only host binding compiles it."""
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit


def run_program_host(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    var f = FP(unsafe_from_address=arena_addr)
    var qbase = IP(unsafe_from_address=prog_addr)
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
                for t in range(total):
                    run_unit[k](t, f, q)
