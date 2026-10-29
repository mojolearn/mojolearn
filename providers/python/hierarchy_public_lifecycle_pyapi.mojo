# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering's public lifecycle, setting admission and captured X.
# Numerical backend extension dispatch remains explicit migration debt in the
# dedicated owner; moving this envelope does not make those calls descriptor ABI.
from std.math import isfinite, floor
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules, native_loaded_module, _loaded_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields, _fitted
from bindings.public_attributes_native import _plan, _lookup
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.classical_warnings import classical_warn_column
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_array_view import native_array_ndarray_view
from bindings.family_common import FAMILY_F32P, family_f32_snapshot
from bindings.metrics_native_support import metric_float, metric_is_real, metric_is_integral
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.cluster_fit_a_native import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.estimator_lifecycle_pyapi import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)


struct HierarchyPublicCall(Movable):
    var fields: PythonObject
    var settings: PythonObject
    var source: PythonObject
    var x: ArrayMetadata
    var fitting: Bool
    var outside: Bool
    var depth: Int
    var rows: Int
    var columns: Int

    def __init__(out self, fields: PythonObject, settings: PythonObject, source: PythonObject,
                 var x: ArrayMetadata, fitting: Bool, outside: Bool, depth: Int):
        self.fields = fields
        self.settings = settings
        self.source = source
        self.rows = x.shape[0]
        self.columns = x.shape[1]
        self.x = x^
        self.fitting = fitting
        self.outside = outside
        self.depth = depth


def hierarchy_settings(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native hierarchy operations refuse Python metaclass hooks")
    var fields = _fields(model)
    var attributes = py_get(_loaded_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    var modules = NativeModules()
    var plan = _plan(modules, model, attributes)
    var ancestry = _type_mro(py_type(model))
    var setter_seen = False
    for i in range(len(ancestry)):  # driver-loop: passive subclass storage admission
        var names = _class_fields(ancestry[i])
        if not setter_seen and py_contains(names, "__setattr__"):
            if names["__setattr__"] is not b.object.__dict__["__setattr__"]:
                raise Error("TypeError: native hierarchy execution cannot invoke Python field setters")
            setter_seen = True
        if py_contains(names, "_mojolearn_after_fit"):
            raise Error("TypeError: native hierarchy execution cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native hierarchy execution cannot invoke Python fit callbacks")
    var settings = Python.dict()
    var mode = py_get(fields, "numeric_mode")
    host_inference_scalar(mode, context)
    py_setitem(settings, "numeric_mode", mode)
    var parameters = Python.tuple("n_clusters", "metric", "connectivity", "linkage", "c", "memory", "compute_full_tree", "distance_threshold", "compute_distances", "prediction_data")
    for i in range(len(parameters)):  # small-loop(fixed hierarchy parameter schema)
        var name = parameters[i]
        var value = _lookup(model, name, fields, plan, attributes)
        if String(name) == "connectivity":
            ref cpy = Python().cpython()
            if cpy.lib.get_function[Int32]("PyCallable_Check")(value._obj_ptr) != 0:
                raise Error("TypeError: connectivity callbacks cannot run inside native fit; pass the connectivity matrix explicitly")
            if host_sparse_format(value).byte_length() == 0:
                classical_admit_value(value, context)
        elif String(name) == "memory":
            if value is not None:
                raise Error("NotImplementedError: mojolearn AgglomerativeClustering: memory is refused; there is no host tree cache")
        else:
            host_inference_scalar(value, context, True)
        py_setitem(settings, name, value)
    return settings


def hierarchy_target(source: PythonObject, rows: Int, name: String, classifier: Bool, required: Bool,
            multioutput: Bool, context: PythonObject) raises -> PythonObject:
    if source is None:
        if required:
            raise Error(name + " requires y to be passed, but the target y is None.")
        return source
    var b = native_loaded_module("builtins")
    if host_sparse_format(source).byte_length() > 0:
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int:
            raise Error("TypeError: sparse target shape requires native dimensions")
        if Int(py=shape[0]) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=shape[0])) + "]")
        return source
    classical_admit_value(source, context)
    if py_type(source) is py_get(context, "ndarray") and String(source.dtype.kind) == "c":
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var shape = labels_shape_of(source, context)
    if len(shape) > 0 and shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(shape[0]) + "]")
    var flat = _flatten_labels(source, context)
    classical_admit_value(flat, context)
    for i in range(len(flat)):  # driver-loop: admitted label words, never Python element protocols
        if py_type(flat[i]) is b.complex:
            raise Error("Complex data not supported: " + name + " got a complex y.")
        host_inference_scalar(flat[i], context)
        if metric_is_real(flat[i], context) and not metric_is_integral(flat[i], context):
            var number = metric_float(flat[i], context)
            if not isfinite(number):
                raise Error("Input y contains NaN or infinity (" + name + ").")
            if classifier and floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
    if (not multioutput and len(shape) == 2 and shape[1] == 1 and py_type(source) is not b.list
            and py_type(source) is not b.tuple and py_type(source) is not b.memoryview):
        classical_warn_column(context)
        return flat
    return source


def hierarchy_public_begin(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> HierarchyPublicCall:
    classical_connect_numpy(context)
    var settings = hierarchy_settings(model, context)
    var fields = _fields(model)
    var fitting = operation != "family_cfb_agglo_predict"
    var b = native_loaded_module("builtins")
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(depth) is not b.int or (py_type(fitted) is not b.bool and py_type(fitted) is not b.int):
        raise Error("TypeError: native hierarchy fit state requires exact integer and boolean flags")
    if not fitting and not Bool(py=_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This AgglomerativeClustering instance is not fitted yet. Call 'fit' with appropriate arguments before using predict.")
    var source = args[1]
    if host_sparse_format(source).byte_length() > 0:
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: sparse X is refused; pass a dense matrix")
    var x = family_f32_snapshot(source, 2, "X", context)
    var values = FAMILY_F32P(unsafe_from_address=x.source.addr)
    for i in range(x.count):  # driver-loop: one public finite-input scan over retained storage
        if not isfinite(values[i]):
            raise Error("Input X contains NaN or infinity (AgglomerativeClustering).")
    if Int(py=depth) == 0:
        if fitting:
            _ = hierarchy_target(args[2], x.shape[0], "AgglomerativeClustering", False, False, False, context)
        else:
            var width = py_get(fields, "n_features_in_")
            host_inference_scalar(width, context)
            classical_check_width(x.shape[1], Int(py=width), "AgglomerativeClustering")
            estimator_feature_check(model, fields, source, context)
    var previous = estimator_fit_begin(fields) if fitting else Int(py=depth)
    return HierarchyPublicCall(fields, settings, source, x^, fitting, estimator_outside(), previous)


def hierarchy_public_end(model: PythonObject, call: HierarchyPublicCall) raises:
    if call.fitting:
        estimator_fit_end(model, call.depth)


def hierarchy_public_finish(model: PythonObject, result: PythonObject,
                            call: HierarchyPublicCall, context: PythonObject) raises -> PythonObject:
    if call.fitting:
        estimator_fit_complete(model, call.source, False, context)
    if call.outside and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result


def hierarchy_public_refusal(error: Error, call: HierarchyPublicCall, model: PythonObject) raises -> Error:
    var text = String(error)
    var pending = text == PYTHON_ERROR_PENDING
    var b = native_loaded_module("builtins")
    var original = _pending(0) if pending else PythonObject(None)
    if pending and original is not b.Exception and original is not b.ValueError and original is not b.ZeroDivisionError:
        return Error(text)
    if pending:
        text = _pending_text(error)
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop(device errors remain native failures)
        if text.find(fault) >= 0:
            return Error(String(error))
    var message = String("")
    var name = classical_estimator_name(model)
    if call.depth == 0 and pending and original is b.Exception:
        for spelling in ["NaN", "infinity", "Infinity", "infinite", "Infinite", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop(public nonfinite refusal spellings)
            if text.find(spelling) >= 0:
                message = (String("Input") if call.fitting else String("Input X")) + " contains NaN or infinity (" + name + ": " + text + ")"
                break
    if message.byte_length() == 0 and call.fitting and call.depth == 0:
        var value_error = original is b.ValueError or (not pending and not text.startswith("TypeError:") and not text.startswith("RuntimeError:") and not text.startswith("ImportError:") and not text.startswith("AttributeError:") and not text.startswith("OverflowError:") and not text.startswith("IndexError:"))
        if call.rows == 1 and (value_error or original is b.Exception or original is b.ZeroDivisionError):
            message = name + " cannot be fit on n_samples=1: " + text
        elif call.columns == 1 and value_error:
            message = name + " cannot be fit on n_features=1: " + text
    if message.byte_length() == 0:
        return Error(String(error))
    if pending:
        _drop_pending()
    return Error("ValueError: " + message)
