# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Per-phase wall time for DBSCAN fits, printed as machine-parseable lines.

THE MEASUREMENT MAIN FOR THE 50,000 DIP. `bench/results/LANE_rbc-build_
2026-08-19.md` SS0.3 established that a whole RBC fit at n = 50,000 moves
1.4-2.0x with the batch SIZE while the ball cover itself is flat across the
same shapes, and named `weak_cc_batched` as the suspect it could not confirm
because the confirmation needs a timer inside the runner. This is that timer.

Every fit here runs with `phase_timing = True`, which makes `dbscan_fit`
print one line per phase per batch (the format and its mapping onto cuML's
nvtx ranges are documented on `dbscan_fit` in
`dbscan/impl/dbscan/runner.mojo`):

    PHASE budget mbytes <mb> batch <b>
    PHASE plan n_rows <N> batch <b> n_batches <nb> method <rbc|brute>
    PHASE mask.vertexdeg batch <i>/<n> <ms>
    PHASE mask.corepoints batch <i>/<n> <ms>
    PHASE label.vertexdeg batch <i>/<n> <ms>
    PHASE label.adjgraph batch <i>/<n> <ms>          (brute arm only)
    PHASE label.weak_cc batch <i>/<n> <ms> passes <p>
    PHASE label.merge_labels batch <i>/<n> <ms>      (batches > 0)
    PHASE final_relabel batch 1/1 <ms>
    ARM dbscan_phase@<n> <total ms>

The fixture is the DBSCAN scaling fixture VERBATIM (`bench/scaling_main.mojo`:
d = 8, coordinates `_u01(i, f, 4) * 4.0`, eps = 0.30, min_pts = 5), so the
per-phase numbers decompose the same fits every DBSCAN row in
`bench/results/` was taken on.

Usage:

    phase_probe [n] [max_mbytes_per_batch] [rbc|brute] [reps]

defaults: n = 50000, max_mbytes_per_batch = 0 (cuML's 80%-of-device
estimate), rbc, reps = 3. Only arms interleaved inside ONE process compare;
to compare two budgets, run one process per budget is WRONG on this box --
run the same binary twice within a repeat script only if the repeats
alternate, or better, compare batches WITHIN one fit, which is what the
per-batch lines are for.
"""

from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from dbscan.impl.dbscan.dbscan import dbscan_fit_impl
from dbscan.impl.dbscan.runner import EPS_NN_BRUTE_FORCE, EPS_NN_RBC


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    """`bench/scaling_main.mojo::_u01`, copied so the fixture is identical."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _parse_int(s: String) raises -> Int:
    var v = 0
    var seen = False
    for cp in s.codepoints():
        var c = Int(cp)
        if c < 48 or c > 57:
            raise Error("expected a number, got: " + s)
        v = v * 10 + (c - 48)
        seen = True
    if not seen:
        raise Error("expected a number, got an empty argument")
    return v


def main() raises:
    var n = 50000
    var mb = 0
    var method = EPS_NN_RBC
    var reps = 3

    var args = argv()
    if len(args) > 1:
        n = _parse_int(String(args[1]))
    if len(args) > 2:
        mb = _parse_int(String(args[2]))
    if len(args) > 3:
        var m = String(args[3])
        if m == "rbc":
            method = EPS_NN_RBC
        elif m == "brute":
            method = EPS_NN_BRUTE_FORCE
        else:
            raise Error("method must be rbc or brute, got: " + m)
    if len(args) > 4:
        reps = _parse_int(String(args[4]))

    var ctx = DeviceContext()
    var d = 8
    var eps = 0.30

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var lab = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var h = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            h.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 4) * 4.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    for _r in range(reps):
        var t0 = perf_counter_ns()
        _ = dbscan_fit_impl(
            ctx, x, lab, n, d, eps, 5, mb, 200, method, True
        )
        ctx.synchronize()
        print(
            "ARM dbscan_phase@" + String(n) + " "
            + String(Float64(perf_counter_ns() - t0) / 1.0e6)
        )
