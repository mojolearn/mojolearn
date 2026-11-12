# SPDX-License-Identifier: Apache-2.0
# One native `score` for the linear expansion regressors
# (python/mojolearn/_expansion_linear.py `_LinearRegressorMixin.score`):
# the prediction link(X @ coef^T + intercept) on the estimator's x_linear
# binding (`x_linear_decision`, the kernel `predict` runs) and R^2 on the
# process-default metrics binding (`r2_score`, the kernel
# `metrics.r2_score` runs), in one call. The same kernels in the same order
# as the Python path it replaces, so the score keeps its bits.
# The PACK half of the former `bindings/linear_score_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/linear_score_native_unpack.mojo`.
from std.math import isfinite
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.classical_arrays_door import classical_address, classical_array
from bindings.admitted_input import admit_x
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_exceptions_door import classical_not_fitted_class

def _linear_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator's numeric mode, or the process default."""
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # ASCII strip/lower natively; anything else takes Python's Unicode-aware
    # strip/lower (same accepted spellings, same refusal).
    var mode_name = String(String(String(mode).strip()).lower())
    if mode_name != "identical" and mode_name != "fast":
        mode_name = String(py_call(py_call(mode.strip).lower))
    if mode_name != "identical" and mode_name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return PythonObject(mode_name)


def _linear_entries(table: PythonObject, mode: PythonObject) raises -> PythonObject:
    var entries = table[mode]
    if py_type(entries) is py_type(PythonObject("")):
        raise Error("ImportError: " + String(entries))
    return entries
