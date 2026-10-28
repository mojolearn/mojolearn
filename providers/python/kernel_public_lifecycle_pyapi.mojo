# SPDX-License-Identifier: Apache-2.0
# Kernel public lifecycle formerly owned by _mode's Python decorators. The
# numerical owner retains operation-local input descriptors through completion.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.mode_metadata_native import _fields
from bindings.public_attributes_native import _plan, _lookup, _string_keys
from bindings.classical_attributes import _type_mro, classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_require_fitted
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import _ordinary_torch
from bindings.cluster_fit_a_native import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING
from bindings.classical_arrays import classical_address
from bindings.native_array_view import native_array_ndarray_view
from bindings.family_common import family_f32_snapshot, family_array
from core.family_cast_impl import family_all_finite_f32, family_cast_rows_f64
from providers.python.array_metadata_pyapi import ArrayMetadata
from providers.python.estimator_lifecycle_pyapi import estimator_feature_check


def kernel_public_admit(model: PythonObject, method: String, fitting: Bool,
                        source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = _fields(model)
    var attributes = py_get(py_fields(native_loaded_module("mojolearn._native_public_attributes")), "_CONTEXT")
    if py_type(attributes) is not py_type(Python.dict()):
        raise Error("TypeError: kernel methods require their native attribute metadata")
    _string_keys(attributes)
    var plan = _plan(model, attributes)
    var ancestry = _type_mro(py_type(model))
    var parameters = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: passive subclasses inherit the connected schema
        if parameters is None:
            parameters = py_get(context["family_km_public_parameters"], ancestry[i])
    if parameters is None:
        raise Error("TypeError: kernel execution requires a connected estimator class")
    for i in range(len(parameters)):  # driver-loop: fixed constructor parameter names
        var value = _lookup(model, parameters[i], fields, plan, attributes)
        classical_admit_value(value, context)
    var learned = Python.tuple("n_features_in_", "n_targets_", "_kernel_params", "_params", "_squeeze",
                               "X_fit_", "dual_coef_", "info_", "components_", "component_indices_",
                               "normalization_", "eigenvalues_", "eigenvectors_", "sweeps_",
                               "random_weights_", "random_offset_", "sigma_", "scale_", "feature_names_in_")
    for i in range(len(learned)):  # driver-loop: only fitted storage consumed by kernel owners
        if py_contains(fields, learned[i]):
            classical_admit_value(_lookup(model, learned[i], fields, plan, attributes), context)
    var flag = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(flag) is not py_type(PythonObject(True)) and py_type(flag) is not py_type(PythonObject(0)):
        raise Error("TypeError: kernel fitted flags must be native booleans or integers")
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not py_type(PythonObject(0)):
        raise Error("TypeError: kernel fit depth must be a native integer")
    if not fitting:
        classical_require_fitted(model, False, method, context)
        var kernel = py_get(fields, "kernel")
        var precomputed = py_type(kernel) is py_type(PythonObject("")) and String(kernel) == "precomputed"
        if Int(py=depth) == 0 and not precomputed:
            estimator_feature_check(model, fields, source, context)
    return fields


def kernel_public_finite(value: ArrayMetadata, name: String) raises:
    if not family_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=value.source.addr), value.count):
        raise Error("ValueError: Input " + name + " contains NaN or infinity")
    _ = value


def kernel_public_column_target(source: PythonObject, value: ArrayMetadata,
                                context: PythonObject) raises -> Bool:
    var cls = py_type(source)
    var reshape = cls is py_get(context, "ndarray") or cls is context["array_class"] or _ordinary_torch(cls, context)
    if len(value.shape) == 2 and value.source.cols == 1 and reshape:
        classical_warn_column(context)
        return True
    return False


def kernel_public_ignored_target(source: PythonObject, rows: Int, context: PythonObject) raises:
    if source is None:
        return
    var value = family_f32_snapshot(source, -1, "y", context)
    if len(value.shape) > 0 and value.source.rows != rows:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples: ["
                    + String(rows) + ", " + String(value.source.rows) + "]")
    _ = kernel_public_column_target(source, value, context)
    kernel_public_finite(value, "y")
    _ = value^


def kernel_public_result(result: PythonObject, prediction: Bool, outside: Bool,
                         context: PythonObject) raises -> PythonObject:
    if not outside or py_get(context, "ndarray") is None or py_type(result) is not context["array_class"]:
        return result
    if not prediction:
        return native_array_ndarray_view(result, context)
    # The established regressor public output is float64; internal predictions
    # retain float32. Widen the finished result in native code before publishing.
    var value = family_f32_snapshot(result, -1, "prediction", context)
    var wide = family_array(value.python_shape(), "<f8", context)
    var source = value.source
    family_cast_rows_f64(source.addr, source.s0, source.s1, source.rows, source.cols,
                         source.itemsize, source.kind,
                         MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(wide))))
    _ = value^
    return native_array_ndarray_view(wide, context)


def kernel_public_refusal(error: Error, fitting: Bool, nested: Bool, rows: Int,
                          columns: Int, model: PythonObject) raises -> Error:
    # Preserve the old public decorator's nonfinite/single-shape error mapping.
    var text = String(error)
    var pending = text == PYTHON_ERROR_PENDING
    var b = native_loaded_module("builtins")
    var kind = _pending(0) if pending else PythonObject(None)
    if pending and kind is not b.Exception and kind is not b.ValueError and kind is not b.ZeroDivisionError:
        return Error(text)
    if pending:
        text = _pending_text(error)
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop: prior device-error spellings
        if text.find(fault) >= 0:
            return Error(String(error))
    var message = String("")
    var name = String(py_type(model).__name__)
    if not nested and pending and kind is b.Exception:
        for spelling in ["NaN", "infinity", "Infinity", "infinite", "Infinite", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop: prior nonfinite spellings
            if text.find(spelling) >= 0:
                message = (String("Input") if fitting else String("Input X")) + " contains NaN or infinity (" + name + ": " + text + ")"
                break
    if message.byte_length() == 0 and fitting and not nested:
        var value_error = kind is b.ValueError or (not pending and not text.startswith("TypeError:") and not text.startswith("RuntimeError:") and not text.startswith("ImportError:") and not text.startswith("AttributeError:") and not text.startswith("OverflowError:") and not text.startswith("IndexError:") and not text.startswith("DataConversionWarning:"))
        if rows == 1 and (value_error or kind is b.Exception or kind is b.ZeroDivisionError):
            message = name + " cannot be fit on n_samples=1: " + text
        elif columns == 1 and value_error:
            message = name + " cannot be fit on n_features=1: " + text
    if message.byte_length() == 0:
        return Error(String(error))
    if pending:
        _drop_pending()
    return Error("ValueError: " + message)
