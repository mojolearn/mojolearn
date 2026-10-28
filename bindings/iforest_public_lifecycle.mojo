# SPDX-License-Identifier: Apache-2.0
# Complete public IsolationForest lifecycle. The original trees_iforest_*
# entries remain internal operations for CV/parallel owners; only the public
# door replaces the former _mode validation/output decorators.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _class_fields
from bindings.classical_attributes import _type_mro
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_width, classical_estimator_name, classical_view
from bindings.classical_labels import labels_shape_of
from bindings.classical_exceptions import classical_require_fitted
from bindings.host_inference_scalars import host_inference_scalar
from bindings.family_common import family_cast_view_f64
from bindings.native_array_view import native_array_ndarray_view
from bindings.trees_public_iforest import iforest_operation_with_source
from bindings.trees_public_state import _release_tree_values
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal


def _admit(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = classical_family_admit(model)
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no post-fit Python callback at the native public boundary
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: native IsolationForest cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native IsolationForest cannot invoke Python fit callbacks")
    var scalars = Python.tuple("numeric_mode", "n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "verbose", "_n_estimators", "_max_samples_mode", "_max_samples_int", "_max_samples_frac", "_max_depth", "_max_features_mode", "_max_features_int", "_max_features_frac", "_contamination_auto", "_contamination", "_bootstrap", "_seed", "offset_", "max_samples_", "_if_token", "n_features_in_", "input_copied_")
    for i in range(len(scalars)):  # driver-loop: consumed scalar state before any native conversion slot
        if py_contains(fields, scalars[i]):
            host_inference_scalar(fields[scalars[i]], context, True)
    var b = native_loaded_module("builtins")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: native fit depth must be an exact integer")
    if py_contains(fields, "_x"):
        _ = classical_view(fields["_x"], context, "fitted X")
    if py_contains(fields, "feature_names_in_"):
        classical_admit_value(fields["feature_names_in_"], context)
    # Both fit and resident prediction may replace the native forest owner.
    # Admit only displaced fields, leaving unrelated caller state untouched.
    var replaced = Python.list()
    for i in range(len(scalars)):  # driver-loop: overwritten scalar arrays may retain non-native base owners
        var name = String(scalars[i])
        if (name.startswith("_") or name.endswith("_")) and py_contains(fields, scalars[i]):
            _ = py_call(replaced.append, fields[scalars[i]])
    for key in ["_if_lifetime", "_x", "feature_names_in_"]:  # driver-loop: native storage released by this operation
        if py_contains(fields, key):
            _ = py_call(replaced.append, fields[key])
    _release_tree_values(replaced, Python.tuple(), context)
    return fields


def iforest_public_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    var fields = _admit(model, context)
    var method = String(operation[byte=String("trees_public_iforest_").byte_length():])
    var fitting = method == "fit" or method == "fit_predict"
    var nested = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0))) > 0
    if not fitting:
        classical_require_fitted(model, False, method, context)
    var name = classical_estimator_name(model)
    var source = args[0]
    var x = classical_check_x(source, name, context)
    var shape = labels_shape_of(x, context)
    var rows = shape[0]
    var columns = shape[1]
    if not nested:
        if fitting:
            _target(args[1], rows, name, context)
        else:
            classical_check_width(columns, Int(py=fields["n_features_in_"]), name)
            estimator_feature_check(model, fields, source, context)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        result = iforest_operation_with_source(model, "trees_iforest_" + method, Python.tuple(x), context, source)
    except error:
        var refused = _refusal(error, fitting, nested, rows, columns, name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, source, False, context)
    if py_type(result) is context["array_class"]:
        # Preserve internal Float32 arrays; public outlier scores widen
        # even when the optional NumPy output view is unavailable.
        if outside and (method == "score_samples" or method == "decision_function"):
            result = family_cast_view_f64(result, "scores", context)
        if outside and py_get(context, "ndarray") is not None:
            return native_array_ndarray_view(result, context)
    return result
