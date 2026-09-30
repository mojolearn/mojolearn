# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees expansion lane's binding (algorithm expansion
lane 7): the ensemble glue of `xtrees/ops.mojo`, registered by
`xtrees/api.mojo::register`. The trees themselves are fitted through the
existing `_mojolearn_rf` / `_mojolearn_gbdt` entry points."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from xtrees.api import register


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_trees() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_trees")
        register(m)
        m.def_function[numeric_mode_binding]("x_trees_numeric_mode")
        m.def_function[vendor_binding]("x_trees_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees: ", e))
