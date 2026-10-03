# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CAGRA on the host: `x_ann/cagra_device.mojo`'s launches as loops over the
SAME cells, in the same order (lane ann-cpu, 2026-09-28: the rows split over
tasks by `ann_rows`, the exact k-NN's distances scored eight candidates per
vector step; every value and every tie is the cell's)."""

from checks.numerics import ftz
from x_ann.cagra_core import cg_search_row
from x_ann.host.ann_host_cells import ann_span, ann_task_count, ann_tasks, ftz_v, mul_add_v
from x_ann.host.ivf_pq_host import fp, ip
from x_ann.ivf_pq_core import F32P, I32P

#: Candidate rows scored per vector step of the exact k-NN.
comptime KNN_W = 8
#: Query rows that share one pass over the transposed candidates.
comptime KNN_R = 4


def cagra_prune(n: Int, kdeg: Int, knn: List[Int32], deg: Int) raises -> List[Int32]:
    """`kern_prune`: per node, detour counts over its k-NN list, then the
    `deg` edges of smallest (count, rank) (DEVIATION 5820). Nodes are
    independent (node a reads the k-NN graph and writes only its own output
    row), so they are split over host tasks (`ann_rows`'s split, lane
    ann-cpu step 3); integer work, the same integers at every thread count."""
    var out = List[Int32](length=n * deg, fill=Int32(0))
    var short = List[Int32](length=n, fill=Int32(0))
    var kp = knn.unsafe_ptr()
    var op = out.unsafe_ptr()
    var sp = short.unsafe_ptr()
    var tasks = ann_task_count(n, kdeg * kdeg)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, n)
        var cnt = List[Int](length=kdeg, fill=0)
        # rank[v] = v's position in row a, valid while stamp[v] == a (lane
        # ann-cpu, 2026-09-28): when row a names kdeg DISTINCT rows in [0, n)
        # (every exact k-NN row does), "the first kab > kad with knn[a, kab] ==
        # cand" is "rank[cand], if it is > kad", so the detour count is the
        # same integer in O(kdeg^2) per node instead of O(kdeg^3). A row with a
        # repeat or an out-of-range entry walks the original search.
        var rank = List[Int32](length=n, fill=Int32(0))
        var stamp = List[Int32](length=n, fill=Int32(-1))
        for a in range(span[0], span[1]):
            for k in range(kdeg):
                cnt[k] = kdeg if Int(kp[a * kdeg + k]) == a else 0
            var distinct = True
            for k in range(kdeg):
                var v = Int(kp[a * kdeg + k])
                if v < 0 or v >= n or Int(stamp[v]) == a:
                    distinct = False
                    break
                stamp[v] = Int32(a)
                rank[v] = Int32(k)
            if distinct:
                for kad in range(kdeg - 1):
                    var d = Int(kp[a * kdeg + kad])
                    for kdb in range(kdeg):
                        var cand = Int(kp[d * kdeg + kdb])
                        if cand >= 0 and cand < n and Int(stamp[cand]) == a:
                            var kab = Int(rank[cand])
                            if kab > kad:
                                cnt[kab] += 1
            else:
                for k in range(kdeg):
                    var v = Int(kp[a * kdeg + k])
                    if v >= 0 and v < n:
                        stamp[v] = Int32(-1)
                for kad in range(kdeg - 1):
                    var d = Int(kp[a * kdeg + kad])
                    if d < 0 or d >= n:
                        continue  # no row to walk (prune_kernel skips it too)
                    for kdb in range(kdeg):
                        var cand = Int(kp[d * kdeg + kdb])
                        for kab in range(kad + 1, kdeg):
                            if Int(kp[a * kdeg + kab]) == cand:
                                cnt[kab] += 1
                                break
            for i in range(deg):
                var best = -1
                for k in range(kdeg):
                    if cnt[k] < 0xFFFF and (best < 0 or cnt[k] < cnt[best]):
                        best = k
                if best < 0:
                    sp[a] = Int32(1)
                    break
                var sel = kp[a * kdeg + best]
                for k in range(kdeg):
                    if kp[a * kdeg + k] == sel:
                        cnt[k] = 0xFFFF
                op[a * deg + i] = sel
        _ = cnt^
        _ = rank^
        _ = stamp^

    ann_tasks(task, tasks)
    for a in range(n):
        if short[a] != 0:
            raise Error("CAGRA: the k-NN graph has too few distinct neighbors for graph_degree")
    _ = short^
    return out^


def cagra_reverse_merge(n: Int, deg: Int, pruned: List[Int32]) -> List[Int32]:
    """`kern_make_rev_graph` + the merge, with the reverse list of each node
    in (rank, source id) order and capped at `deg`, as their count is
    (DEVIATION 5821)."""
    var rev = List[Int32](length=n * deg, fill=Int32(-1))
    var rcount = List[Int](length=n, fill=0)
    for k in range(deg):
        for src in range(n):
            var dst = Int(pruned[src * deg + k])
            if rcount[dst] < deg:
                rev[dst * deg + rcount[dst]] = Int32(src)
            rcount[dst] += 1
    var out = pruned.copy()
    var protected = deg // 2
    if protected == deg:
        return out^
    # lane ann-apple2: each node's merge reads only its own row and its own
    # reverse list and writes only its own row, so the nodes are split over
    # host tasks (`ann_rows`'s split, as cagra_prune): the same integers
    var op = out.unsafe_ptr()
    var rp = rev.unsafe_ptr()
    var cp = rcount.unsafe_ptr()
    var tasks = ann_task_count(n, deg * deg)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, n)
        var row = List[Int32](length=deg, fill=Int32(0))
        for nid in range(span[0], span[1]):
            for i in range(deg):
                row[i] = op[nid * deg + i]
            var kr = cp[nid] if cp[nid] < deg else deg
            while kr > 0:
                kr -= 1
                var v = rp[nid * deg + kr]
                if v < 0:
                    continue
                var pos = deg
                for i in range(deg):
                    if row[i] == v:
                        pos = i
                        break
                if pos < protected:
                    continue
                var num_shift = pos - protected
                if pos >= deg:
                    num_shift = deg - protected - 1
                var s = protected + num_shift
                while s > protected:
                    row[s] = row[s - 1]
                    s -= 1
                row[protected] = v
            for i in range(deg):
                op[nid * deg + i] = row[i]
        _ = row^

    ann_tasks(task, tasks)
    _ = rev^
    _ = rcount^
    return out^



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
