# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column's shortest-path rows on a heap (lane decomp-cpu,
2026-09-28). Host only: compiled into the CPU host binding, never a GPU one.

`dijkstra_row` (x_decomp/cells.mojo, DEVIATION 5314) scans all n nodes to
pick the next one and all n to relax it: O(n^2) per source, O(n^3) for
Isomap's all pairs, which is the GPU's shape (one source per thread) and
minutes on a CPU at a few thousand points. Here the undirected edges are
listed once (the same weight rule: the smaller nonzero of W[u, v] and
W[v, u], each through ftz) and each source runs Dijkstra on a binary heap
keyed (distance, node):
  * the node settled next is the unsettled reached node of least distance,
    ties to the LOWER index, exactly the dense scan's choice;
  * a relaxation writes `add(best, w)` (the cell's flushed add) when the
    node is unreached or the new sum is strictly smaller, exactly the
    dense relaxation, over the same edge set (an absent edge relaxes
    nothing in either spelling).
So every distance is the same sum formed the same way (and 5314's point
stands: the minimum does not depend on the visiting order). Each source
writes only its own row; the rows are independent tasks.

Proof: x_decomp/checks/graph_check.mojo holds HostExec.dijkstra_rows to
the independent oracle (xd_oracles.oracle_dijkstra) and to the device
column, distances and reached counts; arm 5314_host_edge_weight.patch.
"""
from checks.numerics import ftz
from x_decomp.cells import F32Ptr, add


struct EdgeList(Movable):
    """Compressed rows of the undirected graph: node u's edges are
    (nbr[e], wt[e]) for e in [off[u], off[u + 1]), neighbors ascending."""

    var off: List[Int]
    var nbr: List[Int]
    var wt: List[Float32]

    def __init__(out self, W: F32Ptr, n: Int):
        self.off = List[Int](capacity=n + 1)
        self.nbr = List[Int]()
        self.wt = List[Float32]()
        self.off.append(0)
        for u in range(n):
            for v in range(n):
                if v == u:
                    continue
                var a = ftz(W.unsafe_load(u * n + v))
                var b = ftz(W.unsafe_load(v * n + u))
                var w = a
                if w == Float32(0) or (b != Float32(0) and b < w):
                    w = b
                if w == Float32(0):
                    continue
                self.nbr.append(v)
                self.wt.append(w)
            self.off.append(len(self.nbr))


@always_inline
def _before(da: Float32, a: Int, db: Float32, b: Int) -> Bool:
    """Heap order: less distance first, then the lower node."""
    return da < db or (not (db < da) and a < b)


def dijkstra_heap_row(g: EdgeList, dist: F32Ptr, i: Int, n: Int) -> Float32:
    """Row i of the all-pairs distances (-1 = unreachable); returns the
    number of nodes reached, as dijkstra_row does."""
    var base = i * n
    var done = List[Bool](length=n, fill=False)
    for v in range(n):
        dist.unsafe_store(base + v, Float32(-1))
    dist.unsafe_store(base + i, Float32(0))
    var hd = List[Float32]()
    var hn = List[Int]()
    hd.append(Float32(0))
    hn.append(i)
    var reached = 0
    while len(hn) > 0:
        # pop the root
        var best = hd[0]
        var u = hn[0]
        var last = len(hn) - 1
        hd[0] = hd[last]
        hn[0] = hn[last]
        _ = hd.pop()
        _ = hn.pop()
        var sz = len(hn)
        var p = 0
        while True:
            var l = 2 * p + 1
            if l >= sz:
                break
            var c = l
            if l + 1 < sz and _before(hd[l + 1], hn[l + 1], hd[l], hn[l]):
                c = l + 1
            if not _before(hd[c], hn[c], hd[p], hn[p]):
                break
            var td = hd[p]
            var tn = hn[p]
            hd[p] = hd[c]
            hn[p] = hn[c]
            hd[c] = td
            hn[c] = tn
            p = c
        # a stale entry: settled already, or superseded by a smaller distance;
        # a negative distance is never selected by the dense scan (it reads
        # as unreached there), so it is not settled here either
        if done[u] or dist.unsafe_load(base + u) != best or best < Float32(0):
            continue
        done[u] = True
        reached += 1
        for e in range(g.off[u], g.off[u + 1]):
            var v = g.nbr[e]
            if done[v]:
                continue
            var nd = add(best, g.wt[e])
            var dv = dist.unsafe_load(base + v)
            if dv < Float32(0) or nd < dv:
                dist.unsafe_store(base + v, nd)
                # push (nd, v)
                hd.append(nd)
                hn.append(v)
                var q = len(hn) - 1
                while q > 0:
                    var pa = (q - 1) // 2
                    if not _before(hd[q], hn[q], hd[pa], hn[pa]):
                        break
                    var td = hd[pa]
                    var tn = hn[pa]
                    hd[pa] = hd[q]
                    hn[pa] = hn[q]
                    hd[q] = td
                    hn[q] = tn
                    q = pa
    return Float32(reached)
