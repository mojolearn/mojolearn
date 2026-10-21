# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HOST ONLY (the CPU binding): the graph cells of x_decomp/graph_cells.mojo
run row by row on the host, under the address contract of the other
x_decomp host entries (lane hr2-graph-embed, 2026-10-02)."""
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.host import HostExec
from x_decomp.graph_cells import (
    knn_select_row,
    knn_dense_row,
    radius_cell,
    radius_geo_cell,
    lle_iw_row,
    rowbest_row,
    members_comp,
    count_comp,
    join_pair,
)


def host_components(W: F32Ptr, n: Int, comp: F32Ptr) -> Int:
    """The min-label fixed point by union-find (each root is the lowest node
    of its set: the larger root links under the smaller), then the
    components numbered by their lowest node. The fixed point is unique, so
    it is the device's hooking result word for word."""
    var par = List[Int](length=max(n, 1), fill=0)
    for v in range(n):
        par[v] = v

    for u in range(n):
        for v in range(n):
            if W.unsafe_load(u * n + v) != Float32(0):
                var ru = u
                while par[ru] != ru:
                    ru = par[ru]
                var rv = v
                while par[rv] != rv:
                    rv = par[rv]
                if ru != rv:
                    if ru < rv:
                        par[rv] = ru
                    else:
                        par[ru] = rv
                # compress u's and v's paths
                var x = u
                while par[x] != x:
                    var nx = par[x]
                    par[x] = min(ru, rv)
                    x = nx
                x = v
                while par[x] != x:
                    var nx = par[x]
                    par[x] = min(ru, rv)
                    x = nx
    var rank = List[Int](length=max(n, 1), fill=0)
    var c = 0
    for v in range(n):
        var r = v
        while par[r] != r:
            r = par[r]
        par[v] = r
        if r == v:
            rank[v] = c
            c += 1
    for v in range(n):
        comp.unsafe_store(v, Float32(rank[par[v]]))
    return c


def graph_knn_run(pd: F32Ptr, pi: F32Ptr, po: F32Ptr, n: Int, m: Int, nn: Int, ex: Int) raises:
    """`graph_knn_py`'s typed body (p = [n, m, nn, excl])."""
    var excl = ex != 0
    var take = nn + (1 if excl else 0)
    if take > m or m >= 1 << 24:
        raise Error("x_decomp: knn asks for more neighbors than columns")
    var sv = List[Float32](length=max(n * take, 1), fill=Float32(0))
    var si = List[Float32](length=max(n * take, 1), fill=Float32(0))
    var psv = F32Ptr(unsafe_from_address=Int(sv.unsafe_ptr()))
    var psi = F32Ptr(unsafe_from_address=Int(si.unsafe_ptr()))
    for i in range(n):
        knn_select_row(pd, i, m, take, excl, nn, psv, psi, pi, po)
    _ = sv^
    _ = si^


def graph_knn_dense_run(pi: F32Ptr, pw: F32Ptr, po: F32Ptr, n: Int, nn: Int) raises:
    """`graph_knn_dense_py`'s typed body; out (n x n) is written whole."""
    for t in range(n * n):
        po.unsafe_store(t, Float32(0))
    for i in range(n):
        knn_dense_row(i, pi, pw, n, nn, po)


def graph_radius_run(pd: F32Ptr, po: F32Ptr, n: Int, r: Float64) raises:
    """`graph_radius_py`'s typed body."""
    var rr = Float32(r)
    for t in range(n * n):
        radius_cell(t, pd, n, rr, po)


def graph_radius_geo_run(pq: F32Ptr, pd: F32Ptr, pg: F32Ptr, nq: Int, n: Int, r: Float64) raises:
    """`graph_radius_geo_py`'s typed body."""
    var rr = Float32(r)
    for t in range(nq * n):
        radius_geo_cell(t, pq, pd, n, rr, pg)


def graph_lle_iw_run(pi: F32Ptr, pw: F32Ptr, po: F32Ptr, n: Int, nn: Int) raises:
    """`graph_lle_iw_py`'s typed body."""
    for t in range(n * n):
        po.unsafe_store(t, Float32(0))
    for i in range(n):
        lle_iw_row(i, pi, pw, n, nn, po)


def graph_join_run(pw: F32Ptr, pd: F32Ptr, pc: F32Ptr, n: Int, C: Int) raises:
    """`graph_join_py`'s typed body (p = [n, C]); w (n x n) gains the joining
    edges in place."""
    var rv = List[Float32](length=max(n * C, 1), fill=Float32(0))
    var rj = List[Float32](length=max(n * C, 1), fill=Float32(0))
    var cnt = List[Int32](length=C + 1, fill=Int32(0))
    var start = List[Int32](length=C + 1, fill=Int32(0))
    var mem = List[Int32](length=max(n, 1), fill=Int32(0))
    var prv = F32Ptr(unsafe_from_address=Int(rv.unsafe_ptr()))
    var prj = F32Ptr(unsafe_from_address=Int(rj.unsafe_ptr()))
    var pcnt = I32Ptr(unsafe_from_address=Int(cnt.unsafe_ptr()))
    var pst = I32Ptr(unsafe_from_address=Int(start.unsafe_ptr()))
    var pm = I32Ptr(unsafe_from_address=Int(mem.unsafe_ptr()))
    for i in range(n):
        rowbest_row(i, pd, pc, n, C, prv, prj)
    for b in range(C):
        count_comp(b, pc, n, pcnt)
    var acc = Int32(0)
    for b in range(C + 1):
        pst.unsafe_store(b, acc)
        if b < C:
            acc += pcnt.unsafe_load(b)
    for b in range(C):
        members_comp(b, pc, n, pst, pm)
    for a in range(C):
        for b in range(a + 1, C):
            join_pair(a, b, n, C, pst, pm, prv, prj, pw)
    _ = rv^
    _ = rj^
    _ = cnt^
    _ = start^
    _ = mem^


def graph_dijkstra_run(pw: F32Ptr, pd: F32Ptr, n: Int) raises:
    """`graph_dijkstra_py`'s typed body."""
    var reached = List[Float32](length=max(n, 1), fill=Float32(0))
    var pr = F32Ptr(unsafe_from_address=Int(reached.unsafe_ptr()))
    HostExec.dijkstra_rows(pw, pd, pr, n)
    _ = reached^
