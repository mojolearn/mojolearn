# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of `gbdt_fit` (lane calltable-gbdt-forest-dart, 2026-11-11).

`gbdt_fit_args_py` turns the `gbdt_fit` Python entries' arguments (six
address words, the `params` list and the `strs` list) into the native
`GbdtFitArgs` (gbdt/fit_request.mojo) once, at the boundary; the fit itself
(`gbdt_fit_native` in each gbdt binding) reads plain values. Each parameter
keeps both readings the former body made of it: an integer where
`Int(py=)` succeeds and a float where `Float64(py=)` succeeds, with a flag
for a reading that failed, so the body's read of that slot refuses as it
did. `gbdt_fit_outcome_py` packs the outcome as the former result list.
Used by both gbdt bindings' Python entries and by the GBDT door
(bindings/trees_public_gbdt_fit.mojo), which then calls the fit natively.
"""
from std.python import Python, PythonObject
from bindings.py_lists import py_list_f64
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome, GBDT_ARG_NO_INT, GBDT_ARG_NO_FLOAT


def gbdt_fit_params_py(params: PythonObject, mut ints: List[Int], mut floats: List[Float64],
                       mut bad: List[UInt8]) raises:
    """The params list's two views and flags (UNPACK)."""
    var n = len(params)
    for k in range(n):  # small-loop(params: 35 fixed slots, the class weights and the short tails): Python parameter list, not row data
        var item = params[k]
        var flag = UInt8(0)
        var iv = 0
        var fv = Float64(0)
        try:
            iv = Int(py=item)
        except:
            flag |= UInt8(GBDT_ARG_NO_INT)
        try:
            fv = Float64(py=item)
        except:
            flag |= UInt8(GBDT_ARG_NO_FLOAT)
        ints.append(iv)
        floats.append(fv)
        bad.append(flag)


def gbdt_fit_strs_py(strs: PythonObject) raises -> List[String]:
    var out = List[String]()
    for k in range(len(strs)):  # small-loop(strs: four to eight option names)
        out.append(String(py=strs[k]))
    return out^


def gbdt_fit_args_py(x_addr: PythonObject, y_addr: PythonObject, weights_addr: PythonObject,
                     cat_flags_addr: PythonObject, eval_x_addr: PythonObject, eval_y_addr: PythonObject,
                     params: PythonObject, strs: PythonObject) raises -> GbdtFitArgs:
    var ints = List[Int]()
    var floats = List[Float64]()
    var bad = List[UInt8]()
    gbdt_fit_params_py(params, ints, floats, bad)
    return GbdtFitArgs(Int(py=x_addr), Int(py=y_addr), Int(py=weights_addr), Int(py=cat_flags_addr),
                       Int(py=eval_x_addr), Int(py=eval_y_addr), ints^, floats^, bad^, gbdt_fit_strs_py(strs))


def gbdt_fit_outcome_py(outcome: GbdtFitOutcome) raises -> PythonObject:
    """`[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]` (PACK): one PyList_New per curve (bindings/py_lists.mojo)."""
    var out = Python.list()
    out.append(PythonObject(outcome.text))
    out.append(PythonObject(outcome.best_iteration))
    out.append(PythonObject(outcome.stopped_early))
    out.append(py_list_f64(outcome.learn))
    out.append(py_list_f64(outcome.test))
    return out
