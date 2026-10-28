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
from bindings.host_inference_scalars import host_inference_scalar
from bindings.family_common import family_f32
from bindings.cluster_fit_a_native import cluster_fit_a_operation, _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING
from bindings.cluster_native import cluster_operation
from bindings.mixture_native import mixture_operation
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.cluster_constructor_pyapi import cluster_initialize_py


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


def _target(source: PythonObject, rows: Int, name: String, context: PythonObject) raises:
    if source is None:
        return
    var b = native_loaded_module("builtins")
    var value = classical_view(source, context, "y")
    var kind = classical_kind(value, context)
    if kind == "c":
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var shape = labels_shape_of(source, context)
    if len(shape) > 0 and shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(shape[0]) + "]")
    if len(shape) == 2 and shape[1] == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview:
        classical_warn_column(context)
    if kind == "f":
        # The native cast owns the finite-value scan; y is accepted/ignored by
        # these unsupervised algorithms, exactly as their public signatures say.
        _ = family_f32(source, -1, "y", context)


def _refusal(error: Error, fitting: Bool, nested: Bool, rows: Int, columns: Int, name: String) raises -> Error:
    var text = String(error)
    var pending = text == PYTHON_ERROR_PENDING
    var b = native_loaded_module("builtins")
    var original = _pending(0) if pending else PythonObject(None)
    # Only standard native exception classes may have their message read here.
    # A foreign exception retains the original pending exception unchanged.
    if pending and original is not b.Exception and original is not b.ValueError and original is not b.ZeroDivisionError:
        return Error(text)
    if pending:
        text = _pending_text(error)
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # driver-loop: fixed device failure spellings
        if text.find(fault) >= 0:
            return Error(String(error))
    var message = String("")
    if not nested and pending and original is b.Exception:
        for spelling in ["NaN", "infinity", "Infinity", "infinite", "Infinite", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # driver-loop: former public nonfinite error classification
            if text.find(spelling) >= 0:
                message = (String("Input") if fitting else String("Input X")) + " contains NaN or infinity (" + name + ": " + text + ")"
                break
    if message.byte_length() == 0 and fitting and not nested:
        var value_error = original is b.ValueError or (not pending and not text.startswith("TypeError:") and not text.startswith("RuntimeError:") and not text.startswith("ImportError:") and not text.startswith("AttributeError:") and not text.startswith("OverflowError:") and not text.startswith("IndexError:"))
        if rows == 1 and (value_error or original is b.Exception or original is b.ZeroDivisionError):
            message = name + " cannot be fit on n_samples=1: " + text
        elif columns == 1 and value_error:
            message = name + " cannot be fit on n_features=1: " + text
    if message.byte_length() == 0:
        return Error(String(error))
    if pending:
        _drop_pending()
    return Error("ValueError: " + message)


def cluster_public_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    if operation == "family_xc_initialize":
        return cluster_initialize_py(model, args[1], context)
    classical_connect_numpy(context)
    var fields = _admit_model(model, context)
    var method = _method(operation)
    var fitting = method == "fit" or method == "partial_fit" or method == "fit_predict" or method == "fit_transform"
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
