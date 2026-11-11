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
from std.python import PythonObject
from bindings.gbdt_fit_boundary import gbdt_fit_params_py
from core.forest_fit_request import ForestFitArgs


def forest_fit_args_py(x_addr: PythonObject, y_addr: PythonObject, params: PythonObject,
                       criterion: PythonObject) raises -> ForestFitArgs:
    """The request of one fit; `criterion` None for the ET fits (whose
    criterion is params[21])."""
    var ints = List[Int]()
    var floats = List[Float64]()
    var bad = List[UInt8]()
    gbdt_fit_params_py(params, ints, floats, bad)
    var crit = -1 if criterion is None else Int(py=criterion)
    return ForestFitArgs(Int(py=x_addr), Int(py=y_addr), ints^, floats^, bad^, crit)


def forest_params_py(params: PythonObject) raises -> ForestFitArgs:
    """A params list alone as native views (the session, DART and AdaBoost
    Python entries, whose other arguments they read themselves)."""
    var ints = List[Int]()
    var floats = List[Float64]()
    var bad = List[UInt8]()
    gbdt_fit_params_py(params, ints, floats, bad)
    return ForestFitArgs(0, 0, ints^, floats^, bad^, -1)


def forest_fit_request_py(x: Int, y: Int, params: PythonObject, criterion: PythonObject) raises -> ForestFitArgs:
    """The forest door's request: X and y addresses it already holds, its
    params list (UNPACK) and the criterion (None for the ET fits)."""
    var ints = List[Int]()
    var floats = List[Float64]()
    var bad = List[UInt8]()
    gbdt_fit_params_py(params, ints, floats, bad)
    var crit = -1 if criterion is None else Int(py=criterion)
    return ForestFitArgs(x, y, ints^, floats^, bad^, crit)
