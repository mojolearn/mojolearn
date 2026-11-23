# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PACK and the public read of a gradient-boosting model's
`feature_importances_` (lane mf-imp6, 2026-11-23).

The fit computes the importances in Mojo on its image's column
(`gbdt/fit_calls.mojo::xg_importances`: core/gbdt_importance.mojo on the
host, gbdt/gpu_importance.mojo on a GPU) and its PACK stores them here as
the private fitted field `_feature_importances_` (float64, the float32
results widened exactly). The public property is ONE native call
(`gbdt_importances_read`): a fresh float64 array of shape (n_features,), as
scikit-learn's property returns a new array per read, or scikit-learn's
NotFittedError words before a fit. A leaf module: the fit doors, the member
PACK, the archive loader and the host-tree door all import it, and it
imports none of them.
"""
from std.memory import unsafe_memcpy
from std.python import PythonObject
from bindings.py_balanced import py_get, py_setitem, py_type
from bindings.admitted_input_unpack import admitted_vector
from bindings.classical_arrays_door import classical_address
from bindings.parallel_public_result_door import parallel_public_result

comptime GBDT_IMPORTANCES_FIELD = "_feature_importances_"
#: the `save` archive member that carries them (python/mojolearn/ensemble.py)
comptime GBDT_IMPORTANCES_MEMBER = "feature_importances"


def gbdt_importances_store(fields: PythonObject, values: List[Float32], context: PythonObject) raises:
    """PACK: `_feature_importances_` = the float32 results widened exactly
    to a float64 array (one word per input column)."""
    var n = len(values)
    var stored = admitted_vector(n, "<f8", context)
    var into = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(stored)))
    for c in range(n):  # small-loop(one word per input column: the PACK widening)
        into[c] = Float64(values[c])
    py_setitem(fields, PythonObject(GBDT_IMPORTANCES_FIELD), stored)


def gbdt_importances_clear(fields: PythonObject) raises:
    """The unfitted state (a constructor or `set_params` rebuild)."""
    py_setitem(fields, PythonObject(GBDT_IMPORTANCES_FIELD), PythonObject(None))


def gbdt_importances_read(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """`feature_importances_`: a fresh float64 copy of the stored vector, or
    scikit-learn's NotFittedError (`check_is_fitted`'s words) when the
    estimator holds none."""
    var held = py_get(fields, PythonObject(GBDT_IMPORTANCES_FIELD))
    if held is None:
        raise Error("NotFittedError: This " + String(py_type(model).__name__) + " instance is not fitted yet. "
                    + "Call 'fit' with appropriate arguments before using this estimator.")
    var n = Int(py=held.size)
    var fresh = admitted_vector(n, "<f8", context)
    if n > 0:
        unsafe_memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(fresh))),
                      src=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(held))),
                      count=n)
    _ = held  # keep the stored vector alive through the native copy above
    # scikit-learn returns an ndarray: a NumPy view of the fresh float64
    # vector when NumPy is loaded, the estimators' one result rule (as
    # predict's; no widening, it is float64 already).
    return parallel_public_result(fresh, "predict", PythonObject(None), context)
