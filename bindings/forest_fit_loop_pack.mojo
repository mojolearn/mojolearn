# SPDX-License-Identifier: Apache-2.0
"""PACK of a forest fit's loop record as the estimator's `fit_loop_`
attribute (lane alm-forest, 2026-11-24; AGENTS.md "Candidate evaluation":
an A/B arm is decided on loop time per iteration). RandomForest*,
ExtraTrees* (bindings/trees_public_forest_door.mojo) and IsolationForest
(bindings/trees_public_iforest_door.mojo) store through this.

The fit's image wrote two Int words (`core/forest_fit_request.mojo`
`fit_loop_put`): the iterations its tree loop ran and the loop region's
nanoseconds (`core/fit_loop_clock.mojo`). The caller prefills them with -1;
a fit that recorded no loop leaves them so, and the attribute is removed
(a stale one from an earlier fit never survives). The attribute lives in
the instance dictionary like the forest's other fitted attributes and is
not part of the saved archive, so a loaded model has none.

SERIAL BY NATURE: one tuple per fit, GIL held."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from core.fit_loop_clock import FitLoop


def forest_fit_loop_store(fields: PythonObject, iterations: Int, loop_ns: Int) raises:
    """`fit_loop_ = (iterations, milliseconds)` in the instance dictionary
    `fields`, or no `fit_loop_` when the fit reported none (`iterations`
    negative)."""
    if iterations < 0:
        _ = py_call(fields.pop, "fit_loop_", PythonObject(None))
        return
    var loop = FitLoop(iterations, loop_ns, True)
    py_setitem(fields, PythonObject("fit_loop_"),
               Python.tuple(PythonObject(loop.iterations), PythonObject(loop.milliseconds())))
