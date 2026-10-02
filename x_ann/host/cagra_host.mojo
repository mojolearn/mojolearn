# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the host: `x_ann/cagra_device.mojo`'s launches as loops over the
SAME cells, in the same order (lane ann-cpu, 2026-09-28: the rows split over
tasks by `ann_rows`, the exact k-NN's distances scored eight candidates per
vector step; every value and every tie is the cell's)."""

from checks.numerics import ftz
from x_ann.cagra_core import cagra_prune, cagra_reverse_merge, cg_search_row
from x_ann.host.ann_host_cells import ann_span, ann_task_count, ann_tasks, ftz_v, mul_add_v
from x_ann.host.ivf_pq_host import fp, ip
from x_ann.ivf_pq_core import F32P, I32P

#: Candidate rows scored per vector step of the exact k-NN.
comptime KNN_W = 8
#: Query rows that share one pass over the transposed candidates.
comptime KNN_R = 4


def knn_rows_host(x: F32P, n: Int, d: Int, nn: Int, nn_d: F32P, nn_i: I32P):
    """`tsne_core.ts_knn_cell` for every row, on the caller's arrays.

    Row i's squared distance to row j is `ts_sqdist`'s chain (c ascending,
    `ftz(ftz(x_i) - ftz(x_j))`, one fused step, flushed), computed in one
    vector lane per candidate j against the flushed rows stored column-major
    (`xt[c * n_pad + j]`), for KNN_R query rows at a time; then the cell's
    insertion, j ascending, under (distance, index). A lane is one pair's
    scalar chain, so every distance and every neighbor is the cell's."""
    comptime W = KNN_W
    comptime R = KNN_R
    var n_pad = ((n + W - 1) // W) * W
    var xt = List[Float32](length=d * n_pad, fill=Float32(0.0))
    for j in range(n):
        for c in range(d):
            xt[c * n_pad + j] = ftz(x.unsafe_load(j * d + c))
    var tp = fp(xt)
    var n_blocks = (n + R - 1) // R
    var tasks = ann_task_count(n_blocks, R * n * d)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, n_blocks)
        var dist = List[Float32](length=R * n_pad, fill=Float32(0.0))
        var dp = fp(dist)
        for blk in range(span[0], span[1]):
            var i0 = blk * R
            var rows = min(R, n - i0)
            var jb = 0
            while jb < n_pad:
                var acc = InlineArray[SIMD[DType.float32, W], R](fill=SIMD[DType.float32, W](0.0))
                for c in range(d):
                    var b = tp.load[width=W](c * n_pad + jb)
                    comptime for r in range(R):
                        if r < rows:
                            var a = SIMD[DType.float32, W](ftz(x.unsafe_load((i0 + r) * d + c)))
                            var diff = ftz_v[W](a - b)
                            acc[r] = ftz_v[W](mul_add_v[W](diff, diff, acc[r]))
                comptime for r in range(R):
                    dp.store(r * n_pad + jb, acc[r])
                jb += W
            for r in range(rows):
                var i = i0 + r
                var base = i * nn
                var filled = 0
                for j in range(n):
                    if j == i:
                        continue
                    var dd = dp.unsafe_load(r * n_pad + j)
                    if filled == nn:
                        var ld = nn_d.unsafe_load(base + nn - 1)
                        var li = Int(nn_i.unsafe_load(base + nn - 1))
                        if not (dd < ld or (dd == ld and j < li)):
                            continue
                    else:
                        filled += 1
                    var s = filled - 1
                    while s > 0:
                        var pd = nn_d.unsafe_load(base + s - 1)
                        var pi = Int(nn_i.unsafe_load(base + s - 1))
                        if dd < pd or (dd == pd and j < pi):
                            nn_d.unsafe_store(base + s, pd)
                            nn_i.unsafe_store(base + s, Int32(pi))
                            s -= 1
                        else:
                            break
                    nn_d.unsafe_store(base + s, dd)
                    nn_i.unsafe_store(base + s, Int32(j))
        _ = dist^

    ann_tasks(task, tasks)
    _ = xt^


def cagra_build_host(x: F32P, n: Int, d: Int, kdeg: Int, deg: Int) raises -> List[Int32]:
    var nd = List[Float32](length=n * kdeg, fill=Float32(0.0))
    var ni = List[Int32](length=n * kdeg, fill=Int32(0))
    knn_rows_host(x, n, d, kdeg, fp(nd), ip(ni))
    var pruned = cagra_prune(n, kdeg, ni, deg)
    _ = nd^
    return cagra_reverse_merge(n, deg, pruned)


def cagra_search_host(
    x: F32P, n: Int, d: Int, graph: I32P, deg: Int, queries: F32P, m: Int,
    k: Int, L: Int, width: Int, max_iter: Int, n_seeds: Int, out_d: F32P, out_i: I32P, rs: Int = 0,
):
    """`cg_search_cell` per query on the caller's arrays; each task keeps one
    itopk buffer and one visited bitset (`cg_search_row` at offset 0)."""
    var words = (n + 31) // 32
    var tasks = ann_task_count(m, (n_seeds + max_iter * width * deg) * d)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, m)
        var bd = List[Float32](length=L, fill=Float32(0.0))
        var bi = List[Int32](length=L, fill=Int32(0))
        var bx = List[Int32](length=L, fill=Int32(0))
        var vis = List[Int32](length=words, fill=Int32(0))
        for q in range(span[0], span[1]):
            cg_search_row(q, queries, x, n, d, graph, deg, k, L, width, max_iter, n_seeds,
                          fp(bd), ip(bi), ip(bx), 0, ip(vis), 0, words, out_d, out_i, rs)
        _ = bd^
        _ = bi^
        _ = bx^
        _ = vis^

    ann_tasks(task, tasks)
