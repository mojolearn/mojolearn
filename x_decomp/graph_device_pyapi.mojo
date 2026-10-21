# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/graph_device.mojo: the device graph doors
(resident ids) the GPU binding registers. Each unpacks pool ids, floats and
integers and calls one typed `*_run` entry in x_decomp/graph_device.mojo; no
data work happens here."""
from std.python import PythonObject

from x_decomp.graph_device import (
    dev_graph_components_run,
    dev_graph_dijkstra_run,
    dev_graph_join_run,
    dev_graph_knn_dense_run,
    dev_graph_knn_run,
    dev_graph_lle_iw_run,
    dev_graph_radius_geo_run,
    dev_graph_radius_run,
)
from x_decomp.resident_pyapi import _id, _n


def dev_graph_knn_py(d: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """idx / dst (n x nn): the nn nearest columns of every row of the
    resident D (n x m), `knn_select_row`. p = [n, m, nn, excl]."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nn = _n(p, 2)
    var ex = Int(py=p[3])
    var i_d = _id(d)
    var i_idx = _id(idx)
    var i_dst = _id(dst)
    return PythonObject(dev_graph_knn_run(i_d, i_idx, i_dst, n, m, nn, ex))


def dev_graph_knn_dense_py(idx: PythonObject, w: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nn = _n(p, 1)
    var i_idx = _id(idx)
    var i_w = _id(w)
    var i_wout = _id(wout)
    return PythonObject(dev_graph_knn_dense_run(i_idx, i_w, i_wout, n, nn))


def dev_graph_radius_py(d: PythonObject, wout: PythonObject, p: PythonObject, r: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var i_d = _id(d)
    var i_wout = _id(wout)
    var f_r = Float64(py=r)
    return PythonObject(dev_graph_radius_run(i_d, i_wout, f_r, n))


def dev_graph_radius_geo_py(
    dq: PythonObject, d: PythonObject, g: PythonObject, p: PythonObject, r: PythonObject
) raises -> PythonObject:
    """Isomap's radius transform (`radius_geo_cell`), one thread a cell of
    the nq x n result."""
    var nq = _n(p, 0)
    var n = _n(p, 1)
    var i_dq = _id(dq)
    var i_d = _id(d)
    var i_g = _id(g)
    var f_r = Float64(py=r)
    return PythonObject(dev_graph_radius_geo_run(i_dq, i_d, i_g, f_r, nq, n))


def dev_graph_lle_iw_py(idx: PythonObject, wb: PythonObject, wout: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nn = _n(p, 1)
    var i_idx = _id(idx)
    var i_wb = _id(wb)
    var i_wout = _id(wout)
    return PythonObject(dev_graph_lle_iw_run(i_idx, i_wb, i_wout, n, nn))


def dev_graph_components_py(w: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    """comp (n floats): every node's component, numbered by lowest node;
    returns the number of components."""
    var n = _n(p, 0)
    var i_w = _id(w)
    var i_comp = _id(comp)
    return PythonObject(dev_graph_components_run(i_w, i_comp, n))


def dev_graph_join_py(w: PythonObject, d: PythonObject, comp: PythonObject, p: PythonObject) raises -> PythonObject:
    """`graph_join_py` on resident W (in place), D and comp; p = [n, C]."""
    var n = _n(p, 0)
    var C = _n(p, 1)
    var i_w = _id(w)
    var i_d = _id(d)
    var i_comp = _id(comp)
    return PythonObject(dev_graph_join_run(i_w, i_d, i_comp, n, C))


def dev_graph_dijkstra_py(w: PythonObject, dist: PythonObject, p: PythonObject) raises -> PythonObject:
    """dist (n x n, resident): the shortest paths of the resident W, the
    arcs compressed on the device (count, scan, fill) and the rows of
    `dijkstra_kernel`, DIJKSTRA_ROWS a launch."""
    var n = _n(p, 0)
    var i_w = _id(w)
    var i_dist = _id(dist)
    return PythonObject(dev_graph_dijkstra_run(i_w, i_dist, n))
