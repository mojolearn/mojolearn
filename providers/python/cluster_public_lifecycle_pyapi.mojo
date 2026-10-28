# SPDX-License-Identifier: Apache-2.0
# Public cluster lifecycle formerly implemented by _mode's Python decorators.
# Validation, fitted tracking, feature metadata and final output wrapping stay
# inside this one native operation. The existing algorithm owners are retained;
# their legacy extension dispatch remains separately tracked migration debt.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.public_attributes_native import _plan, _lookup, _string_keys
from bindings.classical_attributes import _type_mro, classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_kind
from bindings.classical_validate import classical_check_x, classical_view, classical_estimator_name, classical_check_width
from bindings.classical_labels import labels_shape_of
from bindings.classical_warnings import classical_warn_column
from bindings.classical_exceptions import classical_require_fitted
from bindings.native_array_view import native_array_ndarray_view
from bindings.native_owner_lifetime import native_release_admit
from bindings.host_inference_scalars import host_inference_scalar
from bindings.family_common import family_f32
from bindings.cluster_fit_a_native import cluster_fit_a_operation, _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING
from bindings.cluster_native import cluster_operation
from bindings.mixture_native import mixture_operation
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_constructor_pyapi import cluster_initialize_py
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal


def _scalar_state(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 8:
        raise Error("TypeError: cluster scalar state exceeds its native nesting depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.tuple or kind is b.list:
        for i in range(len(value)):  # driver-loop: native partial-fit seed and scalar state words
            _scalar_state(value[i], context, depth + 1)
    else:
        host_inference_scalar(value, context)


def _admit_model(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var module_fields = py_fields(native_loaded_module("mojolearn._native_public_attributes"))
    _string_keys(module_fields)
    var attributes = py_get(module_fields, "_CONTEXT")
    if py_type(attributes) is not b.dict:
        raise Error("TypeError: native cluster attribute metadata requires its connected dictionary")
    _string_keys(attributes)
    var plan = _plan(model, attributes)
    var fields = _fields(model)
    var ancestry = _type_mro(py_type(model))
    var parameters = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: passive subclasses share their connected native owner
        if parameters is None:
            parameters = py_get(context["cluster_public_parameters"], ancestry[i])
    if parameters is None:
        raise Error("TypeError: cluster execution requires a connected native estimator class")
    var setter_seen = False
    for i in range(len(ancestry)):  # driver-loop: admit native field writes before entering algorithm owners
        var names = _class_fields(ancestry[i])
        if not setter_seen and py_contains(names, "__setattr__"):
            if names["__setattr__"] is not b.object.__dict__["__setattr__"]:
                raise Error("TypeError: native cluster execution cannot invoke Python field setters")
            setter_seen = True
        if py_contains(names, "_mojolearn_after_fit"):
            raise Error("TypeError: native cluster execution cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native cluster execution cannot invoke Python fit callbacks")
    for i in range(len(parameters)):  # driver-loop: fixed model parameter metadata, never a Python property
        var name = parameters[i]
        var value = _lookup(model, name, fields, plan, attributes)
        if String(name) == "init":
            ref cpy = Python().cpython()
            if cpy.lib.get_function[Int32]("PyCallable_Check")(value._obj_ptr) != 0:
                raise Error("TypeError: Python init callbacks are not accepted; pass explicit initial centers or use 'k-means++' or 'random'")
        classical_admit_value(value, context)
        if String(name) == "metric_params" and value is not None:
            # OPTICS consumes only p from this native mapping. An admitted
            # object ndarray still must not invoke element conversion hooks.
            var p = py_call(value.get, "p", PythonObject(None))
            if p is not None:
                host_inference_scalar(p, context)
        if (not py_contains(context["cluster_public_array_parameters"], name)
                or (py_type(value) is py_get(context, "ndarray") and Int(py=value.ndim) == 0)):
            # Exact ndarray admission alone would still permit an object-dtype
            # scalar to execute Python __float__/__int__ during old-owner casts.
            host_inference_scalar(value, context, True)
    var learned = Python.tuple("cluster_centers_", "labels_", "n_features_in_", "counts_", "_partial_init", "_partial_batch", "_partial_state", "_tree_nodes", "_tree_centers", "weights_", "means_", "precisions_cholesky_", "log_det_chol_", "_log_consts", "_full_pchol", "_warm", "n_steps_", "lower_bound_")
    for i in range(len(learned)):  # driver-loop: admit only fitted storage the cluster owners actually consume
        if py_contains(fields, learned[i]):
            var value = _lookup(model, learned[i], fields, plan, attributes)
            classical_admit_value(value, context)
            var name = String(learned[i])
            if name == "n_features_in_" or name == "_partial_batch" or name == "n_steps_" or name == "lower_bound_":
                host_inference_scalar(value, context)
            elif name == "_partial_init" or name == "_partial_state":
                _scalar_state(value, context)
            elif name == "_warm" and value is not None:
                if (py_type(value) is not b.tuple and py_type(value) is not b.list) or len(value) != 2:
                    raise Error("TypeError: cluster warm state requires its native key and array pair")
                _scalar_state(value[0], context)
    return fields


def _admit_replaced_state(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var ancestry = _type_mro(py_type(model))
    var names = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: use the connected owner's fit-state schema
        if names is None:
            names = py_get(context["cluster_public_replaced_fields"], ancestry[i])
    if py_type(names) is not b.tuple:
        raise Error("TypeError: cluster refit requires its native replaced-field schema")
    for i in range(len(names)):  # driver-loop: prior fitted owners before any numerical/state mutation
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: cluster fitted-field names require exact strings")
        if py_contains(fields, names[i]):
            native_release_admit(fields[names[i]], context)
    for name in ["_mojolearn_fitted", "_mojolearn_fitting", "feature_names_in_"]:  # driver-loop: shared lifecycle writes/removals
        if py_contains(fields, name):
            native_release_admit(fields[name], context)


def _method(operation: String) -> String:
    if operation.endswith("partial_fit"):
        return String("partial_fit")
    if operation.endswith("fit_predict"):
        return String("fit_predict")
    if operation.endswith("fit_transform"):
        return String("fit_transform")
    if operation.endswith("predict_proba"):
        return String("predict_proba")
    if operation.endswith("score_samples"):
        return String("score_samples")
    if operation.endswith("transform"):
        return String("transform")
    if operation.endswith("predict"):
        return String("predict")
    return String("fit") if operation.endswith("fit") else String("score")


def cluster_public_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    if operation == "family_xc_initialize":
        return cluster_initialize_py(model, args[1], context)
    classical_connect_numpy(context)
    var fields = _admit_model(model, context)
    var method = _method(operation)
    var fitting = method == "fit" or method == "partial_fit" or method == "fit_predict" or method == "fit_transform"
    if fitting:
        _admit_replaced_state(model, fields, context)
    var fitted_flag = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted_flag) is not py_type(PythonObject(True)) and py_type(fitted_flag) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    var depth_value = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fit depth must be an exact integer")
    var nested = Int(py=depth_value) > 0
    if not fitting:
        classical_require_fitted(model, False, method, context)
    var name = classical_estimator_name(model)
    var source = args[1]
    var x = classical_check_x(source, name, context)
    var shape = x.shape
    var rows = Int(py=shape[0])
    var columns = Int(py=shape[1])
    if not nested:
        if fitting:
            # All six classes retain their y argument, including fit_predict.
            # It follows the old owner arguments in the public call envelope.
            _target(args[len(args) - 1], rows, name, context)
        if not fitting or method == "partial_fit":
            var precomputed = False
            for key in ["metric", "affinity"]:  # driver-loop: two supported precomputed-mode settings
                var value = py_get(fields, key)
                precomputed = precomputed or (py_type(value) is py_type(PythonObject("")) and String(value) == "precomputed")
            if not precomputed:
                var width = py_get(fields, "n_features_in_")
                if py_type(width) is py_type(PythonObject(0)):
                    classical_check_width(columns, Int(py=width), name)
                estimator_feature_check(model, fields, source, context)
    var values = Python.list(PythonObject(None), x)
    # The trailing y belongs to lifecycle validation, not the algorithm ABI.
    var stop = len(args) - 1 if fitting else len(args)
    for i in range(2, stop):  # driver-loop: fixed public argument positions
        classical_admit_value(args[i], context)
        _ = py_call(values.append, args[i])
    var prepared = estimator_arguments(model, "_mojolearn_x_cluster", values, context)
    var outside = estimator_outside()
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        if operation.startswith("family_cfa_"):
            result = cluster_fit_a_operation(model, operation, prepared, context)
        elif operation.startswith("family_xc_"):
            result = cluster_operation(model, operation, prepared, context)
        else:
            result = mixture_operation(model, operation, prepared, context)
    except error:
        var refused = _refusal(error, fitting, nested, rows, columns, name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, source, False, context)
    if outside and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
