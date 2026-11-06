# SPDX-License-Identifier: Apache-2.0
# Complete public IsolationForest lifecycle. The original trees_iforest_*
# entries remain internal operations for CV/parallel owners; only the public
# door replaces the former _mode validation/output decorators.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _class_fields
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit
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
from bindings.trees_side_records import trees_side_get, trees_side_put, trees_side_drop, trees_key_same, trees_key_shallow
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal


comptime _IFOREST_CLASS_SLOT = "iforest_public_class_v1"
comptime _IFOREST_FITTED_SLOT = "iforest_public_fitted_inputs_v1"


def _fitted_inputs_key(fields: PythonObject) raises -> PythonObject:
    """The identities the fitted-X view and the feature-name admission read:
    each field's presence, its object and, for an exact list/tuple/dict, its
    items (trees_key_shallow)."""
    var mods = NativeModules()
    var key = Python.list()
    var names = Python.tuple("_x", "feature_names_in_")
    for i in range(2):  # small-loop(names): the two fitted input fields
        if py_contains(fields, names[i]):
            _ = py_call(key.append, names[i])
            trees_key_shallow(mods, key, fields[names[i]], 0)
        else:
            _ = py_call(key.append, PythonObject(None))
    return key


def _admit(model: PythonObject, context: PythonObject, fitting: Bool) raises -> PythonObject:
    var fields = classical_family_admit(model)
    # The class callback refusal depends only on the class rows: once per
    # class state (native_class_admission).
    var mods = NativeModules()
    var cls = py_type(model)
    if class_record(mods, _IFOREST_CLASS_SLOT, cls) is None:
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: no post-fit Python callback at the native public boundary
            if py_contains(_class_fields(mods, ancestry[i]), "_mojolearn_after_fit"):
                raise Error("TypeError: native IsolationForest cannot invoke Python fit callbacks")
        class_admit(mods, _IFOREST_CLASS_SLOT, cls, PythonObject(True))
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
    # The fitted X view and the feature-name admission depend only on those
    # two stored objects, which only a fit replaces: a query reuses the
    # previous acceptance while their identity key matches (lane perf2-trees,
    # 2026-11-06; both ran on every score_samples/decision_function/predict).
    # The record lives in the native side table, never in `__dict__`.
    var inputs_key = _fitted_inputs_key(fields)
    var inputs_known = False
    if fitting:
        # Pass 2 H7: a fit replaces both fields, so the old record (which
        # holds the previous training X) is dropped now, not at the next query.
        trees_side_drop(_IFOREST_FITTED_SLOT, model)
    else:
        var held = trees_side_get(_IFOREST_FITTED_SLOT, model)
        inputs_known = held is not None and trees_key_same(held, inputs_key)
    if not inputs_known:
        if py_contains(fields, "_x"):
            _ = classical_view(fields["_x"], context, "fitted X")
        if py_contains(fields, "feature_names_in_"):
            classical_admit_value(fields["feature_names_in_"], context)
        if not fitting:
            _ = trees_side_put(_IFOREST_FITTED_SLOT, model, inputs_key)
    # Both fit and resident prediction may replace the native forest owner.
    # Admit only displaced fields, leaving unrelated caller state untouched.
    # A query (score_samples/decision_function/predict, trees_public_iforest
    # `_run`) replaces only offset_, max_samples_, _if_token and, when the
    # resident token changes, _if_lifetime; the fitted X, feature names and
    # the remaining learned scalars stay in place, so only fit admits them.
    var replaced = Python.list()
    if fitting:
        for i in range(len(scalars)):  # driver-loop: overwritten scalar arrays may retain non-native base owners
            var name = String(scalars[i])
            if (name.startswith("_") or name.endswith("_")) and py_contains(fields, scalars[i]):
                _ = py_call(replaced.append, fields[scalars[i]])
        for key in ["_if_lifetime", "_x", "feature_names_in_"]:  # driver-loop: native storage released by this operation
            if py_contains(fields, key):
                _ = py_call(replaced.append, fields[key])
    else:
        for key in ["offset_", "max_samples_", "_if_token", "_if_lifetime"]:  # driver-loop: fields a query replaces
            if py_contains(fields, key):
                _ = py_call(replaced.append, fields[key])
    _release_tree_values(replaced, Python.tuple(), context)
    return fields


def iforest_public_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    var method = String(operation[byte=String("trees_public_iforest_").byte_length():])
    var fitting = method == "fit" or method == "fit_predict"
    var fields = _admit(model, context, fitting)
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
