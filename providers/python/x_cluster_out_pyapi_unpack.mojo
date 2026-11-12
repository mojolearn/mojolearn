# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python boundary both cluster bindings share (lane/algos-cluster):
`ClusterOut` to Python arrays and Python sequences to typed lists. Plain
copies, no arithmetic on data."""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call
from x_cluster.out import ClusterOut


def py_ints(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for t in range(len(obj)):  # small-loop(argument words)
        out.append(Int(py=obj[t]))
    return out^


def py_floats(obj: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for t in range(len(obj)):  # small-loop(argument words)
        out.append(Float64(py=obj[t]))
    return out^
