# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbor-graph builds of Isomap and LocallyLinearEmbedding as cells
(lane hr2-graph-embed, 2026-10-02).

Before this lane the Python kit did these steps in Python on the host: the
k smallest of every distance row by `heapq.nsmallest` (10,000 x 10,000
Python key calls at the board's Isomap / LLE shape), the dense kNN /
radius graph and LLE's I - W written one cell at a time, the connected
component walk and the closest-pair joins of `_fix_connected_components`,
and the compressed arcs of the shortest-path rows (a serial n x n scan in
DevExec). Each is now a cell: the GPU binding launches it on device
resident matrices (x_decomp/graph_device.mojo), the host binding runs the
SAME cells in a loop (the `graph_*_py` entries below), so the words agree
on every column.

Compares read operands through `ftz` (Metal flushes compare operands), on
every column alike.

  knn_select_row   the `take` smallest of row i of D (m columns), ties to the
                   LOWER column, kept in insertion order; `excl` drops
                   column i; the first nn are written (index as an exact
                   float, the raw distance).
  knn_dense_row    W[i, idx[i, a]] = w[i, a] (GRAPH_TINY for a zero: scipy
                   drops explicit zeros, the edge must stay); W zero first.
  radius_cell      W[i, j] = D[i, j] (GRAPH_TINY for 0) when j != i and
                   D[i, j] <= r, else 0.
  lle_iw_row       I - W on the kNN lists: IW[i, i] = 1, IW[i, j] = 0 - w.
  components       min-label fixed point (every node labelled by the lowest
                   node of its weak component, W nonzero either way), then
                   the components numbered by their lowest node ascending.
  join             sklearn `_fix_connected_components`: for every pair of
                   components a < b, the closest (i in a, j in b) by D, ties
                   to the lower i then the lower j, gets the edge D[i, j].
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from checks.numerics import ftz
from x_decomp.cells import F32Ptr, I32Ptr, sub
from x_decomp.host import HostExec

comptime GRAPH_TINY = Float32(1.0e-10)


@always_inline
def knn_select_row(
    d: F32Ptr, i: Int, m: Int, take: Int, excl: Bool, nn: Int, sv: F32Ptr, si: F32Ptr, idx: F32Ptr, dst: F32Ptr
):
    var base = i * take
    var row = i * m
    var size = 0
    var worst = Float32(0)
    for j in range(m):
        var v = d.unsafe_load(row + j)
        var key = ftz(v)
        if size == take and not (key < worst):
            continue
        var pos = size if size < take else take - 1
        while pos > 0 and key < ftz(sv.unsafe_load(base + pos - 1)):
            sv.unsafe_store(base + pos, sv.unsafe_load(base + pos - 1))
            si.unsafe_store(base + pos, si.unsafe_load(base + pos - 1))
            pos -= 1
        sv.unsafe_store(base + pos, v)
        si.unsafe_store(base + pos, Float32(j))
        if size < take:
            size += 1
        if size == take:
            worst = ftz(sv.unsafe_load(base + take - 1))
    var a = 0
    for t in range(size):
        var j = Int(si.unsafe_load(base + t))
        if excl and j == i:
            continue
        if a < nn:
            idx.unsafe_store(i * nn + a, Float32(j))
            dst.unsafe_store(i * nn + a, sv.unsafe_load(base + t))
            a += 1
    while a < nn:
        idx.unsafe_store(i * nn + a, Float32(-1))
        dst.unsafe_store(i * nn + a, Float32(0))
        a += 1


@always_inline
def knn_dense_row(i: Int, idx: F32Ptr, w: F32Ptr, n: Int, nn: Int, W: F32Ptr):
    for a in range(nn):
        var j = Int(idx.unsafe_load(i * nn + a))
        if j < 0:
            continue
        var v = w.unsafe_load(i * nn + a)
        W.unsafe_store(i * n + j, v if ftz(v) != Float32(0) else GRAPH_TINY)


@always_inline
def radius_cell(t: Int, D: F32Ptr, n: Int, r: Float32, W: F32Ptr):
    var i = t // n
    var j = t - i * n
    var v = D.unsafe_load(t)
    var out = Float32(0)
    if j != i and ftz(v) <= r:
        out = v if ftz(v) != Float32(0) else GRAPH_TINY
    W.unsafe_store(t, out)


@always_inline
def lle_iw_row(i: Int, idx: F32Ptr, wb: F32Ptr, n: Int, nn: Int, IW: F32Ptr):
    IW.unsafe_store(i * n + i, Float32(1))
    for a in range(nn):
        var j = Int(idx.unsafe_load(i * nn + a))
        if j < 0:
            continue
        IW.unsafe_store(i * n + j, sub(IW.unsafe_load(i * n + j), wb.unsafe_load(i * nn + a)))


@always_inline
def rowbest_row(i: Int, D: F32Ptr, comp: F32Ptr, n: Int, C: Int, rv: F32Ptr, rj: F32Ptr):
    """Per component b > comp(i): the closest j in b to row i (ties to the
    lower j); rj = -1 when b holds no such j (never for a real component)."""
    var a = Int(comp.unsafe_load(i))
    for b in range(C):
        rv.unsafe_store(i * C + b, Float32(0))
        rj.unsafe_store(i * C + b, Float32(-1))
    for j in range(n):
        var b = Int(comp.unsafe_load(j))
        if b <= a:
            continue
        var v = D.unsafe_load(i * n + j)
        var cur = Int(rj.unsafe_load(i * C + b))
        if cur < 0 or ftz(v) < ftz(rv.unsafe_load(i * C + b)):
            rv.unsafe_store(i * C + b, v)
            rj.unsafe_store(i * C + b, Float32(j))


@always_inline
def members_comp(b: Int, comp: F32Ptr, n: Int, start: I32Ptr, mem: I32Ptr):
    """Component b's nodes ascending at mem[start[b] ..)."""
    var w = Int(start.unsafe_load(b))
    for v in range(n):
        if Int(comp.unsafe_load(v)) == b:
            mem.unsafe_store(w, Int32(v))
            w += 1


@always_inline
def count_comp(b: Int, comp: F32Ptr, n: Int, cnt: I32Ptr):
    var c = 0
    for v in range(n):
        if Int(comp.unsafe_load(v)) == b:
            c += 1
    cnt.unsafe_store(b, Int32(c))


@always_inline
def join_pair(a: Int, b: Int, n: Int, C: Int, start: I32Ptr, mem: I32Ptr, rv: F32Ptr, rj: F32Ptr, W: F32Ptr):
    var bi = -1
    var bj = -1
    var bv = Float32(0)
    for t in range(Int(start.unsafe_load(a)), Int(start.unsafe_load(a + 1))):
        var i = Int(mem.unsafe_load(t))
        var j = Int(rj.unsafe_load(i * C + b))
        if j < 0:
            continue
        var v = rv.unsafe_load(i * C + b)
        if bi < 0 or ftz(v) < ftz(bv):
            bi = i
            bj = j
            bv = v
    if bi < 0:
        return
    var e = bv if ftz(bv) != Float32(0) else GRAPH_TINY
    W.unsafe_store(bi * n + bj, e)
    W.unsafe_store(bj * n + bi, e)


@always_inline
def arc_count_row(u: Int, W: F32Ptr, n: Int, cnt: I32Ptr):
    """`dijkstra_arc_count` for one node: v with W[u, v] or W[v, u] nonzero."""
    var c = 0
    for v in range(n):
        if W.unsafe_load(u * n + v) != Float32(0) or W.unsafe_load(v * n + u) != Float32(0):
            c += 1
    cnt.unsafe_store(u, Int32(c))


@always_inline
def arc_fill_row(u: Int, W: F32Ptr, n: Int, rp: I32Ptr, adj: I32Ptr, wa: F32Ptr, wb: F32Ptr):
    """`dijkstra_arcs` for one node: its arcs ascending from rp[u]."""
    var e = Int(rp.unsafe_load(u))
    for v in range(n):
        var x = W.unsafe_load(u * n + v)
        var y = W.unsafe_load(v * n + u)
        if x != Float32(0) or y != Float32(0):
            adj.unsafe_store(e, Int32(v))
            wa.unsafe_store(e, x)
            wb.unsafe_store(e, y)
            e += 1


# ---- the host column: the same cells, rows in a loop ----------------------


def _f(o: PythonObject) raises -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(py=o))


def _ni(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: graph dimension out of range")
    return v


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


def graph_knn_py(d: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, m, nn, excl]
    var n = _ni(p, 0)
    var m = _ni(p, 1)
    var nn = _ni(p, 2)
    var excl = Int(py=p[3]) != 0
    var take = nn + (1 if excl else 0)
    if take > m or m >= 1 << 24:
        raise Error("x_decomp: knn asks for more neighbors than columns")
    var pd = _f(d)
    var pi = _f(idx)
    var po = _f(dst)
    var sv = List[Float32](length=max(n * take, 1), fill=Float32(0))
    var si = List[Float32](length=max(n * take, 1), fill=Float32(0))
    var psv = F32Ptr(unsafe_from_address=Int(sv.unsafe_ptr()))
    var psi = F32Ptr(unsafe_from_address=Int(si.unsafe_ptr()))
    with GILReleased(Python()):
        for i in range(n):
            knn_select_row(pd, i, m, take, excl, nn, psv, psi, pi, po)
    _ = sv^
    _ = si^
    return PythonObject(n)


def graph_knn_dense_py(idx: PythonObject, w: PythonObject, out: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, nn]; out (n x n) is written whole
    var n = _ni(p, 0)
    var nn = _ni(p, 1)
    var pi = _f(idx)
    var pw = _f(w)
    var po = _f(out)
    with GILReleased(Python()):
        for t in range(n * n):
            po.unsafe_store(t, Float32(0))
        for i in range(n):
            knn_dense_row(i, pi, pw, n, nn, po)
    return PythonObject(n)


def graph_radius_py(d: PythonObject, out: PythonObject, p: PythonObject, r: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var rr = Float32(Float64(py=r))
    var pd = _f(d)
    var po = _f(out)
    with GILReleased(Python()):
        for t in range(n * n):
            radius_cell(t, pd, n, rr, po)
    return PythonObject(n)


def graph_lle_iw_py(idx: PythonObject, wb: PythonObject, out: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var nn = _ni(p, 1)
    var pi = _f(idx)
    var pw = _f(wb)
    var po = _f(out)
    with GILReleased(Python()):
        for t in range(n * n):
            po.unsafe_store(t, Float32(0))
        for i in range(n):
            lle_iw_row(i, pi, pw, n, nn, po)
    return PythonObject(n)


def graph_components_py(w: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var pw = _f(w)
    var pc = _f(comp)
    var c = 0
    with GILReleased(Python()):
        c = host_components(pw, n, pc)
    return PythonObject(c)


def graph_join_py(w: PythonObject, d: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, C]; w (n x n) gains the joining edges in place
    var n = _ni(p, 0)
    var C = _ni(p, 1)
    var pw = _f(w)
    var pd = _f(d)
    var pc = _f(comp)
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
    with GILReleased(Python()):
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
    return PythonObject(C)


def graph_dijkstra_py(w: PythonObject, dist: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var pw = _f(w)
    var pd = _f(dist)
    var reached = List[Float32](length=max(n, 1), fill=Float32(0))
    var pr = F32Ptr(unsafe_from_address=Int(reached.unsafe_ptr()))
    with GILReleased(Python()):
        HostExec.dijkstra_rows(pw, pd, pr, n)
    _ = reached^
    return PythonObject(n)
