# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-identical-neural (2026-09-27): k-means||'s incremental
candidate assignment (`_assign_to_candidates_fast`, KMEANS_FAST_INCR_INIT)
against the full reassignment (`_assign_to_candidates`), bit for bit.

The candidate set grows over several rounds, as in
`init_scalable_kmeans_plus_plus`; after every round the incremental
(min_dist, labels) must equal a full reassignment against every candidate so
far, word for word. Planted duplicates (a later candidate identical to an
earlier one, and candidates identical to data rows) make exact distance
ties, where the full argmin keeps the LOWER candidate index and the fold's
strict `<` must too. The existing k-means and IVF gates never reach this
path (their fixtures take one round), which is why it has its own gate.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        cluster/checks/kmeans_incr_init_check.mojo
"""

from std.memory import bitcast
from max.gpu.host import DeviceContext
from cluster.impl.detail.kmeans import (
    KMEANS_FAST_INCR_INIT,
    _assign_to_candidates,
    _assign_to_candidates_fast,
)
from cluster.impl.detail.min_cluster_distance_compute import compute_centroid_norms
from cluster.impl.kmeans_params import KMeansParams


def _mix(x_in: UInt32) -> UInt32:
    var x = x_in
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


def _val(i: Int, f: Int, seed: Int) -> Float32:
    var h = _mix(UInt32(i * 7919 + f * 104729 + seed * 131 + 7))
    # a coarse grid (multiples of 1/8) so exact distance ties are common
    return Float32(Int(h % UInt32(64)) - 32) * Float32(0.125)


def main() raises:
    comptime assert KMEANS_FAST_INCR_INIT, "build on a column where the incremental path is on"
    var ctx = DeviceContext()
    var params = KMeansParams.default()
    var total_bad = 0
    var total_cells = 0
    var rounds_run = 0
    for seed in range(3):
        var n = 4000 + 997 * seed
        var d = 5 + 3 * seed
        var hx = List[Float32](length=n * d, fill=0)
        for i in range(n):
            for f in range(d):
                hx[i * d + f] = _val(i, f, seed)
        # candidates arrive in rounds of these sizes
        var sizes: List[Int] = [7, 40, 3, 120, 1, 64]
        var cmax = 0
        for s in sizes:
            cmax += s
        var hc = List[Float32](length=cmax * d, fill=0)
        for c in range(cmax):
            for f in range(d):
                if c % 9 == 4 and c >= 9:
                    hc[c * d + f] = hc[(c - 9) * d + f]  # duplicate of an earlier candidate
                elif c % 11 == 3:
                    hc[c * d + f] = hx[((c * 37) % n) * d + f]  # a data row
                else:
                    hc[c * d + f] = _val(c + 100000, f, seed)
        var dx = ctx.enqueue_create_buffer[DType.float32](n * d)
        var dc = ctx.enqueue_create_buffer[DType.float32](cmax * d)
        var xn = ctx.enqueue_create_buffer[DType.float32](n)
        var dist = ctx.enqueue_create_buffer[DType.float32](n * cmax)
        var l_inc = ctx.enqueue_create_buffer[DType.uint32](n)
        var m_inc = ctx.enqueue_create_buffer[DType.float32](n)
        var l_new = ctx.enqueue_create_buffer[DType.uint32](n)
        var m_new = ctx.enqueue_create_buffer[DType.float32](n)
        var l_full = ctx.enqueue_create_buffer[DType.uint32](n)
        var m_full = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dx, hx.unsafe_ptr())
        ctx.enqueue_copy(dc, hc.unsafe_ptr())
        ctx.synchronize()
        compute_centroid_norms(ctx, dx, xn, n, d, params.metric)
        ctx.synchronize()
        var assigned = 0
        var count = 0
        for s in sizes:
            count += s
            var cand = dc.create_sub_buffer[DType.float32](0, count * d)
            assigned = _assign_to_candidates_fast(
                ctx, dx, xn, cand, dist, l_inc, m_inc, l_new, m_new,
                params, n, d, assigned, count,
            )
            _assign_to_candidates(
                ctx, dx, xn, cand, dist, l_full, m_full, params, n, d, count,
            )
            ctx.synchronize()
            var a1 = List[UInt32](length=n, fill=0)
            var a2 = List[UInt32](length=n, fill=0)
            var b1 = List[Float32](length=n, fill=0)
            var b2 = List[Float32](length=n, fill=0)
            ctx.enqueue_copy(a1.unsafe_ptr(), l_inc)
            ctx.enqueue_copy(a2.unsafe_ptr(), l_full)
            ctx.enqueue_copy(b1.unsafe_ptr(), m_inc)
            ctx.enqueue_copy(b2.unsafe_ptr(), m_full)
            ctx.synchronize()
            for i in range(n):
                total_cells += 1
                if a1[i] != a2[i] or bitcast[DType.uint32](b1[i]) != bitcast[DType.uint32](b2[i]):
                    if total_bad < 5:
                        print("MISMATCH seed", seed, "candidates", count, "row", i,
                              "incr", a1[i], b1[i], "full", a2[i], b2[i])
                    total_bad += 1
            rounds_run += 1
            _ = cand^
    print("KMEANS_INCR_INIT_CHECK rounds", rounds_run, "row checks", total_cells, "mismatches", total_bad)
    if total_bad != 0:
        raise Error("incremental k-means|| assignment differs from the full one")
    print("== cluster/checks/kmeans_incr_init_check.mojo PASSED ==")
