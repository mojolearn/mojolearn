# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The DIMENSIONALITY axis, which every scoreboard row so far has avoided.

The standing scaling curves run DBSCAN at d=8 and k-NN at d=32 -- the exact
regime where scikit-learn's kd-tree is strongest. Tree indexes degrade toward
brute force as d grows and CPU caches stop helping; a GPU's arithmetic does
not care. So the win region may be drawn along d, not n, and nothing has
measured that. Fixed n = 200,000, d sweeps.

DBSCAN eps scales as 0.30 * sqrt(d/8): on the same uniform fixture the
expected inter-point distance grows with sqrt(d), so this holds the
neighborhood's RELATIVE scale constant. Distance concentration still shifts
the degree distribution with d -- that is a property of high-dimensional
data itself, disclosed rather than tuned away. Both sides get the identical
fixture and the identical eps.

Prints `ARM <name>_d<D>@<n> <ms>`; pair with dsweep_sklearn.py via
run_bench.py for same-invocation verdicts.
"""

from std.math import sqrt
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from dbscan.derived.dbscan.dbscan import dbscan_fit_impl
from dbscan.derived.dbscan.runner import EPS_NN_RBC
from neighbors.derived.neighbors.detail.knn_brute_force import (
    brute_force_knn_impl,
    compute_norms,
)


comptime REPEATS = 3


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _knn_d(ctx: DeviceContext, n_index: Int, n_queries: Int, d: Int) raises:
    var k = 10
    var tile = 256
    var buf_len = max(n_index // 8, k)

    var index = ctx.enqueue_create_buffer[DType.float32](n_index * d)
    var queries = ctx.enqueue_create_buffer[DType.float32](n_queries * d)
    var inorm = ctx.enqueue_create_buffer[DType.float32](n_index)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    var dist = ctx.enqueue_create_buffer[DType.float32](tile * n_index)
    var bv = ctx.enqueue_create_buffer[DType.float32](tile * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](tile * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](n_index * d)
    for i in range(n_index):
        for f in range(d):
            hi.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 1)))
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_queries * d)
    for i in range(n_queries):
        for f in range(d):
            hq.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 2)))
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        compute_norms(ctx, index, inorm, n_index, d, False)
        compute_norms(ctx, queries, qnorm, n_queries, d, False)
        brute_force_knn_impl(
            ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
            n_queries, n_index, d, k, tile, buf_len, False,
        )
        ctx.synchronize()
        print(
            "ARM knn_d" + String(d) + "@" + String(n_index) + " "
            + String(Float64(perf_counter_ns() - t0) / 1.0e6)
        )


def _dbscan_d(ctx: DeviceContext, n: Int, d: Int) raises:
    var eps = 0.30 * sqrt(Float64(d) / 8.0)

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var lab = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    var h = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            h.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 4) * 4.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        _ = dbscan_fit_impl(ctx, x, lab, n, d, eps, 5, 0, 200, EPS_NN_RBC)
        ctx.synchronize()
        print(
            "ARM dbscan_d" + String(d) + "@" + String(n) + " "
            + String(Float64(perf_counter_ns() - t0) / 1.0e6)
        )


def main() raises:
    var ctx = DeviceContext()
    _knn_d(ctx, 200000, 2000, 32)
    _knn_d(ctx, 200000, 2000, 64)
    _knn_d(ctx, 200000, 2000, 128)
    # LARGE-n AT HIGH d, added 2026-08-20. Every d>=32 cell on the board ran
    # at n = 200,000, and every large-n cell ran at d = 8. Those are the two
    # halves of the claim measured separately: "we win at width" and "we lose
    # at size in their best corner". Nothing had ever run at both at once, so
    # whether the width win SURVIVES scale was unmeasured and was being
    # assumed. These rows measure it.
    _knn_d(ctx, 1000000, 2000, 32)
    _dbscan_d(ctx, 200000, 8)
    _dbscan_d(ctx, 200000, 32)
    _dbscan_d(ctx, 200000, 64)
    _dbscan_d(ctx, 400000, 32)
    _dbscan_d(ctx, 800000, 32)
