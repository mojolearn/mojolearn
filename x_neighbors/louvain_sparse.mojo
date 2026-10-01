# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`louvain_item` (x_neighbors/items.mojo, DEVIATION 5204) on the host as a
SPARSE walk: the same statements, the same folds in the same order, over the
graph's edges instead of every cell of an n x n matrix (lane neural-pass14).

The board's Louvain race is a 20,000-node kNN graph (`GRAPH_SMALL`), which the
dense item walked as two 1.6 GB matrices, 4e8 cells per pass over the nodes,
for 60 s where networkx takes 1 to 2 s on the sparse adjacency. Every float
fold in the dense item is a serial ascending chain seeded `+0.0` over the
cells of a row (or a row-major half), and a cell holding `+0.0` (or `-0.0`)
is bitwise inert in such a chain: `ftz(ftz(x) + ftz(0.0))` is `x` for every
`x` the chains hold (`+0.0` seeded, nonnegative weights, so never `-0.0`).
So walking only the nonzero cells, in the same ascending column order, gives
every fold the same bits. Every other statement (the candidate scan in
ascending community id with a strictly larger gain to move, the renumbering
in ascending old id, the aggregation in row-major order over each edge once)
is reproduced with its order kept, on sparse structures. The dense item stays
the device kernel's body and the reference; `x_neighbors/checks/
louvain_sparse_check.mojo` holds this one to `o_louvain` and to the dense
item, labels, modularity and level count bit for bit.
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul
from x_neighbors.items import FP, IP, _add, _sub


struct _Csr(Movable):
    """A symmetric weighted graph on `nn` nodes, rows ascending, each row's
    columns ascending, nonzero weights only (a `+0.0` cell is no edge)."""

    var nn: Int
    var indptr: List[Int]
    var cols: List[Int32]
    var vals: List[Float32]

    def __init__(out self, nn: Int):
        self.nn = nn
        self.indptr = List[Int](length=nn + 1, fill=0)
        self.cols = List[Int32]()
        self.vals = List[Float32]()


def _csr_of_dense(a: FP, n: Int) -> _Csr:
    """The nonzero cells of the row-major `n x n` matrix at `a`."""
    var g = _Csr(n)
    for u in range(n):
        for v in range(n):
            var wv = a.unsafe_load(u * n + v)
            if wv != Float32(0):
                g.cols.append(Int32(v))
                g.vals.append(wv)
        g.indptr[u + 1] = len(g.cols)
    return g^


def _modularity_sparse(g: _Csr, comm: List[Int32], m: Float32, resolution: Float32) -> Float32:
    """`_louvain_modularity` over the sparse graph: the same per-row chains
    over the row's nonzero cells in ascending column order, then the fold
    over communities in ascending id."""
    var nn = g.nn
    var tot = List[Float32](length=nn, fill=Float32(0))
    var inner = List[Float32](length=nn, fill=Float32(0))
    for u in range(nn):
        var cu = Int(comm[u])
        var deg = Float32(0)
        for e in range(g.indptr[u], g.indptr[u + 1]):
            var v = Int(g.cols[e])
            var wv = g.vals[e]
            deg = _add(deg, wv)
            if v == u:
                deg = _add(deg, wv)
                inner[cu] = _add(inner[cu], wv)
            elif v > u and Int(comm[v]) == cu:
                inner[cu] = _add(inner[cu], wv)
        tot[cu] = _add(tot[cu], deg)
    var q = Float32(0)
    var two_m = ftz(identical_mul(Float32(2), m))
    for c in range(nn):
        var lc = ftz(identical_div(inner[c], m))
        var fr = ftz(identical_div(tot[c], two_m))
        q = _add(q, _sub(lc, ftz(identical_mul(resolution, ftz(identical_mul(fr, fr))))))
    return q


def _insert_sorted(mut xs: List[Int], v: Int):
    """Insert `v` into the ascending list `xs` unless present."""
    var i = len(xs)
    while i > 0 and xs[i - 1] > v:
        i -= 1
    if i > 0 and xs[i - 1] == v:
        return
    xs.append(0)
    var j = len(xs) - 1
    while j > i:
        xs[j] = xs[j - 1]
        j -= 1
    xs[i] = v


def louvain_item_sparse(
    a: FP, labels: IP, info: FP, n: Int, max_level: Int, resolution: Float32, threshold: Float32,
) raises:
    """`louvain_item` on the host over the nonzero cells of `a` (module note).
    `labels[u]` ends as original node `u`'s community, numbered by first
    appearance; `info = [modularity, levels]`."""
    var g0 = _csr_of_dense(a, n)
    # m = total edge weight, each undirected edge once, self-loops once:
    # the dense item's row-major chain over u <= v, zeros inert.
    var m = Float32(0)
    for u in range(n):
        for e in range(g0.indptr[u], g0.indptr[u + 1]):
            if Int(g0.cols[e]) >= u:
                m = _add(m, g0.vals[e])
    var two_m2 = ftz(identical_mul(Float32(2), ftz(identical_mul(m, m))))
    for u in range(n):
        labels.unsafe_store(u, Int32(u))
    var g = g0^
    var nn = n
    var comm = List[Int32](length=nn, fill=Int32(0))
    for u in range(nn):
        comm[u] = Int32(u)
    var levels = 0
    var mod = _modularity_sparse(g, comm, m, resolution)
    while max_level <= 0 or levels < max_level:
        # ---- networkx `_one_level` on the current graph (nn nodes) --------
        var deg = List[Float32](length=nn, fill=Float32(0))
        var stot = List[Float32](length=nn, fill=Float32(0))
        for u in range(nn):
            comm[u] = Int32(u)
            var dg = Float32(0)
            var self_w = Float32(0)
            for e in range(g.indptr[u], g.indptr[u + 1]):
                var wv = g.vals[e]
                dg = _add(dg, wv)
                if Int(g.cols[e]) == u:
                    self_w = wv
            dg = _add(dg, self_w)
            deg[u] = dg
            stot[u] = dg
        var k2c = List[Float32](length=nn, fill=Float32(0))
        var touched = List[Int]()
        var improvement = False
        var moves = 1
        while moves > 0:
            moves = 0
            for u in range(nn):
                var cu = Int(comm[u])
                # k2c over u's neighbors in ascending column order (the dense
                # item's v loop, zero cells skipped there too)
                touched.clear()
                for e in range(g.indptr[u], g.indptr[u + 1]):
                    var v = Int(g.cols[e])
                    if v != u:
                        var wv = g.vals[e]
                        if wv != Float32(0):
                            var cv = Int(comm[v])
                            k2c[cv] = _add(k2c[cv], wv)
                            _insert_sorted(touched, cv)
                var du = deg[u]
                stot[cu] = _sub(stot[cu], du)
                var remove_cost = _add(
                    -ftz(identical_div(k2c[cu], m)),
                    ftz(identical_div(ftz(identical_mul(resolution, ftz(identical_mul(stot[cu], du)))), two_m2)),
                )
                var best = cu
                var best_gain = Float32(0)
                # the candidate scan in ascending community id; a community
                # whose k2c is zero (untouched, or a flushed sum) is skipped,
                # as the dense scan skips it
                for ti in range(len(touched)):
                    var c = touched[ti]
                    var kc = k2c[c]
                    if kc == Float32(0):
                        continue
                    var gain = _sub(
                        _add(remove_cost, ftz(identical_div(kc, m))),
                        ftz(identical_div(ftz(identical_mul(resolution, ftz(identical_mul(stot[c], du)))), two_m2)),
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best = c
                stot[best] = _add(stot[best], du)
                if best != cu:
                    comm[u] = Int32(best)
                    moves += 1
                    improvement = True
                for ti in range(len(touched)):
                    k2c[touched[ti]] = Float32(0)
        if levels > 0 and not improvement:
            break
        # renumber by ascending old community id
        var used = List[Bool](length=nn, fill=False)
        for u in range(nn):
            used[Int(comm[u])] = True
        var node_of = List[Int32](length=nn, fill=Int32(-1))
        var nc = 0
        for c in range(nn):
            if used[c]:
                node_of[c] = Int32(nc)
                nc += 1
        for u in range(nn):
            comm[u] = node_of[Int(comm[u])]
        for u in range(n):
            labels.unsafe_store(u, comm[Int(labels.unsafe_load(u))])
        levels += 1
        var new_mod = _modularity_sparse(g, comm, m, resolution)
        if not (_sub(new_mod, mod) > threshold):
            break
        mod = new_mod
        # aggregate (their _gen_graph): W'[c,d] = sum of w[u,v] over u in c,
        # v in d, each undirected edge once, in the dense item's row-major
        # order over u <= v; an edge inside c becomes c's self-loop. The
        # accumulator is a dense nc x nc block when that is small, else a
        # dictionary keyed by the cell; either way each cell's chain is the
        # same terms in the same order.
        var g2 = _Csr(nc)
        if nc * nc <= (1 << 24):
            var w2 = List[Float32](length=nc * nc, fill=Float32(0))
            for u in range(nn):
                var cu2 = Int(comm[u])
                for e in range(g.indptr[u], g.indptr[u + 1]):
                    var v = Int(g.cols[e])
                    if v < u:
                        continue
                    var wv = g.vals[e]
                    if wv == Float32(0):
                        continue
                    var cv2 = Int(comm[v])
                    w2[cu2 * nc + cv2] = _add(w2[cu2 * nc + cv2], wv)
                    if cu2 != cv2:
                        w2[cv2 * nc + cu2] = _add(w2[cv2 * nc + cu2], wv)
            for c in range(nc):
                for d in range(nc):
                    var wv = w2[c * nc + d]
                    if wv != Float32(0):
                        g2.cols.append(Int32(d))
                        g2.vals.append(wv)
                g2.indptr[c + 1] = len(g2.cols)
        else:
            # per row, the ascending column list and the cells' chains
            var rows = List[List[Int]]()
            for _ in range(nc):
                rows.append(List[Int]())
            var cells = Dict[Int, Float32]()
            for u in range(nn):
                var cu2 = Int(comm[u])
                for e in range(g.indptr[u], g.indptr[u + 1]):
                    var v = Int(g.cols[e])
                    if v < u:
                        continue
                    var wv = g.vals[e]
                    if wv == Float32(0):
                        continue
                    var cv2 = Int(comm[v])
                    var key = cu2 * nc + cv2
                    if key in cells:
                        cells[key] = _add(cells[key], wv)
                    else:
                        cells[key] = _add(Float32(0), wv)
                        _insert_sorted(rows[cu2], cv2)
                    if cu2 != cv2:
                        var key2 = cv2 * nc + cu2
                        if key2 in cells:
                            cells[key2] = _add(cells[key2], wv)
                        else:
                            cells[key2] = _add(Float32(0), wv)
                            _insert_sorted(rows[cv2], cu2)
            for c in range(nc):
                for di in range(len(rows[c])):
                    var d = rows[c][di]
                    var wv = cells[c * nc + d]
                    if wv != Float32(0):
                        g2.cols.append(Int32(d))
                        g2.vals.append(wv)
                g2.indptr[c + 1] = len(g2.cols)
        g = g2^
        nn = nc
        comm = List[Int32](length=nn, fill=Int32(0))
        for u in range(nn):
            comm[u] = Int32(u)
    # the final report: modularity of the original graph under `labels`
    var final_labels = List[Int32](length=n, fill=Int32(0))
    for u in range(n):
        final_labels[u] = labels.unsafe_load(u)
    var g_final = _csr_of_dense(a, n)
    info.unsafe_store(0, _modularity_sparse(g_final, final_labels, m, resolution))
    info.unsafe_store(1, Float32(levels))
