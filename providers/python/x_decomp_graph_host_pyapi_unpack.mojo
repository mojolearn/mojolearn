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
