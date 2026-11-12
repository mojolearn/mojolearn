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
# The PACK half of the former `bindings/gbdt_fit_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/gbdt_fit_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_lists import py_list_f64
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome, GBDT_ARG_NO_INT, GBDT_ARG_NO_FLOAT


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
