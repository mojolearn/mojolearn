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
from std.python import Python, PythonObject
from bindings.gbdt_fit_boundary import gbdt_fit_params_py
from bindings.py_balanced import py_call, py_item
from core.forest_fit_request import ForestFitArgs, forest_set_trees


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


def forest_request_from_words_py(words: PythonObject, x: Int, y: Int, trees: Int) raises -> ForestFitArgs:
    """UNPACK of `forest_request_words_py` in the shard worker: the request
    on the worker's X and y addresses, with the shard's tree count in the
    tree-count slot (XF_P_TREES)."""
    var crit = Int(py=py_item(words, PythonObject(0)))
    var ints_py = py_item(words, PythonObject(1))
    var floats_py = py_item(words, PythonObject(2))
    var flags_py = py_item(words, PythonObject(3))
    var n = len(ints_py)
    if len(floats_py) != n or len(flags_py) != n:
        raise Error("forest shard: parameter views differ in length")
    var ints = List[Int](capacity=n)
    var floats = List[Float64](capacity=n)
    var bad = List[UInt8](capacity=n)
    for k in range(n):  # small-loop(the fit's parameter slots): request words, not row data
        ints.append(Int(py=py_item(ints_py, PythonObject(k))))
        floats.append(Float64(py=py_item(floats_py, PythonObject(k))))
        bad.append(UInt8(Int(py=py_item(flags_py, PythonObject(k)))))
    var args = ForestFitArgs(x, y, ints^, floats^, bad^, crit)
    forest_set_trees(args, trees)
    return args^
