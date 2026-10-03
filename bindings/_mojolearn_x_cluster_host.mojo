# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_cluster` (lane/algos-cluster). HOST ONLY: the
GPU binding's export names and contract, over `x_cluster.host.host_ops.HostOps`
(the same `x_cluster/bodies.mojo` bodies in plain loops). The sabotage arm is
`X_CLUSTER_HOST_SABOTAGE` (`-D MOJOLEARN_HOST_SABOTAGE=1`)."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import read_f32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_cluster.entries import run_entry
from x_cluster.host.host_ops import HostOps, X_CLUSTER_HOST_SABOTAGE
from x_cluster.out import ClusterOut, py_floats, py_ints
from x_cluster.tree_cut import PY2MOJO_CLUSTER


def call_binding(
    which: PythonObject, x_addr: PythonObject, x_len: PythonObject, a_addr: PythonObject,
    a_len: PythonObject, ip: PythonObject, fp: PythonObject,
) raises -> PythonObject:
    var w = Int(py=which)
    var nx = Int(py=x_len)
    var na = Int(py=a_len)
    var x = read_f32(Int(py=x_addr), nx) if nx > 0 else List[Float32]()
    var a = read_f32(Int(py=a_addr), na) if na > 0 else List[Float32]()
    var ints = py_ints(ip)
    var floats = py_floats(fp)
    var res = ClusterOut()
    with GILReleased(Python()):
        var ops = HostOps()
        res = run_entry(ops, w, x, a, ints, floats)
    return res.to_py()


def x_cluster_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_cluster_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_cluster_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_cluster host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_cluster_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_CLUSTER_HOST_SABOTAGE)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def py2mojo_binding() raises -> PythonObject:
    """1: the steps `_expansion_cluster.py` / `_hierarchy_impl.py` ran in
    Python run here (lane apple-fast-py2mojo-cluster); 0 under
    `-D MOJOLEARN_PY2MOJO_cluster_OFF`, and Python takes its old path."""
    return PythonObject(1 if PY2MOJO_CLUSTER else 0)


def vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


@export
def PyInit__mojolearn_x_cluster_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_cluster_host")
        m.def_function[x_cluster_host_numeric_mode_binding]("x_cluster_host_numeric_mode")
        m.def_function[x_cluster_host_vendor_binding]("x_cluster_host_vendor")
        m.def_function[x_cluster_host_column_binding]("x_cluster_host_column")
        m.def_function[x_cluster_host_sabotage_binding]("x_cluster_host_sabotage")
        m.def_function[call_binding]("x_cluster_call")
        m.def_function[numeric_mode_binding]("x_cluster_numeric_mode")
        m.def_function[vendor_binding]("x_cluster_vendor")
        m.def_function[py2mojo_binding]("x_cluster_py2mojo")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_cluster_host: ", e))
