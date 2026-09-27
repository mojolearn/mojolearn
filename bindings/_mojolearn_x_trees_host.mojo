# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding of the trees expansion lane (`x_trees` family).

HOST ONLY: no DeviceContext, no kernel. It exports the GPU binding's names
(`xtrees/api.mojo::register`, the SAME functions the GPU binding registers)
plus `x_trees_numeric_mode` (1) and `x_trees_vendor` ("cpu"), and the four
host read-backs `x_trees_host_{numeric_mode,vendor,column,sabotage}`, so
`_expansion_trees.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES`."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from xtrees.api import register


def x_trees_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_trees_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "x_trees host: this binding compiles the CPU column only; pass -D MOJOLEARN_COLUMN_CPU"
        " (bindings/build_x_trees_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def x_trees_host_sabotage_binding() raises -> PythonObject:
    """No host sabotage define in pass 1: the host and GPU columns share one
    spelling (xtrees/ops.mojo)."""
    return PythonObject(False)


@export
def PyInit__mojolearn_x_trees_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_trees_host")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_host_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_host_vendor")
        m.def_function[x_trees_host_column_binding]("x_trees_host_column")
        m.def_function[x_trees_host_sabotage_binding]("x_trees_host_sabotage")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_vendor")
        register(m)
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees_host: ", e))
