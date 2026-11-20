# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the forest fits (lane calltable-forest, 2026-11-11).

`forest_fit_args_py` turns a forest fit Python entry's arguments (the X and
y address words, the `params` list and the criterion) into the native
`ForestFitArgs` (core/forest_fit_request.mojo) once, at the boundary; the
fit bodies of the four forest bindings read plain values. Each parameter
keeps both readings the former bodies made of it (`gbdt_fit_params_py`:
an integer where `Int(py=)` succeeds and a float where `Float64(py=)`
succeeds, with a flag for a reading that failed), so a body's read of that
slot refuses as it did. Used by the forest bindings' Python entries (kept
for the tests, `host_surface` and the unconverted boost-door session and
AdaBoost fits) and by the forest door, which then calls the fit natively.
"""
# The PACK half of the former `bindings/forest_fit_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/forest_fit_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.gbdt_fit_boundary_unpack import gbdt_fit_params_py
from bindings.py_balanced import py_call, py_item
from core.forest_fit_request import ForestFitArgs, forest_set_trees


def forest_request_words_py(args: ForestFitArgs) raises -> PythonObject:
    """PACK of a request's parameter words for the tree-range shard RPC
    (lane forest-typed-fit): `(criterion, ints, floats, flags)`, plain ints
    and floats the native codec carries. The X / y / weight addresses are
    the worker's own and are not sent."""
    var ints = Python.list()
    var floats = Python.list()
    var flags = Python.list()
    for k in range(len(args)):  # small-loop(the fit's parameter slots): request words, not row data
        _ = py_call(ints.append, PythonObject(args.ints[k]))
        _ = py_call(floats.append, PythonObject(args.floats[k]))
        _ = py_call(flags.append, PythonObject(Int(args.bad[k])))
    return Python.tuple(PythonObject(args.criterion), ints, floats, flags)
