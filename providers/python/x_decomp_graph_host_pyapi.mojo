# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/graph_host.mojo: the host graph doors the CPU
binding registers. Each unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/graph_host.mojo; no data work happens
here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.graph_host import (
    graph_dijkstra_run,
    graph_join_run,
    graph_knn_dense_run,
    graph_knn_run,
    graph_lle_iw_run,
    graph_radius_geo_run,
    graph_radius_run,
    host_components,
)


def _f(o: PythonObject) raises -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(py=o))


def _ni(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: graph dimension out of range")
    return v


def graph_knn_py(d: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, m, nn, excl]
    var n = _ni(p, 0)
    var m = _ni(p, 1)
    var nn = _ni(p, 2)
    var ex = Int(py=p[3])
    var pd = _f(d)
    var pi = _f(idx)
    var po = _f(dst)
    with GILReleased(Python()):
        graph_knn_run(pd, pi, po, n, m, nn, ex)
    return PythonObject(n)


def graph_knn_dense_py(idx: PythonObject, w: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, nn]; out (n x n) is written whole
    var n = _ni(p, 0)
    var nn = _ni(p, 1)
    var pi = _f(idx)
    var pw = _f(w)
    var po = _f(wout)
    with GILReleased(Python()):
        graph_knn_dense_run(pi, pw, po, n, nn)
    return PythonObject(n)


def graph_radius_py(d: PythonObject, wout: PythonObject, p: PythonObject, r: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var rr = Float64(py=r)
    var pd = _f(d)
    var po = _f(wout)
    with GILReleased(Python()):
        graph_radius_run(pd, po, n, rr)
    return PythonObject(n)


def graph_radius_geo_py(
    dq: PythonObject, d: PythonObject, g: PythonObject, p: PythonObject, r: PythonObject
) raises -> PythonObject:
    var nq = _ni(p, 0)
    var n = _ni(p, 1)
    var rr = Float64(py=r)
    var pq = _f(dq)
    var pd = _f(d)
    var pg = _f(g)
    with GILReleased(Python()):
        graph_radius_geo_run(pq, pd, pg, nq, n, rr)
    return PythonObject(nq)


def graph_lle_iw_py(idx: PythonObject, wb: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var nn = _ni(p, 1)
    var pi = _f(idx)
    var pw = _f(wb)
    var po = _f(wout)
    with GILReleased(Python()):
        graph_lle_iw_run(pi, pw, po, n, nn)
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
    with GILReleased(Python()):
        graph_join_run(pw, pd, pc, n, C)
    return PythonObject(C)


def graph_dijkstra_py(w: PythonObject, dist: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _ni(p, 0)
    var pw = _f(w)
    var pd = _f(dist)
    with GILReleased(Python()):
        graph_dijkstra_run(pw, pd, n)
    return PythonObject(n)
