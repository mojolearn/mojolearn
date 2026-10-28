# SPDX-License-Identifier: Apache-2.0
# The native owner of QNRegressor's public fit, predict and score. Each is one
# call into qn_c, which converts and checks the inputs, selects the numeric-mode
# binding, runs the existing qn_fit / qn_decision_function entries (the same
# arguments the Python shell passed, so the same bits) and stores the fitted
# attributes. score predicts and scores in the same call: the prediction stays
# a native buffer handed straight to the R^2 program.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_owner_lifetime import native_release_admit
from bindings.host_inference_scalars import host_inference_scalar
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_require_fitted
from bindings.classical_attributes import _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.family_raw_attr import family_raw_get_or
from bindings.classical_labels import labels_cast
from providers.python.classical_family_admission_pyapi import classical_family_admit
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete, estimator_feature_check
from providers.python.estimator_tabular_boundary_pyapi import _refusal
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_host_f32, classical_dtype_str, classical_c_order
from bindings.classical_validate import classical_view, classical_check_weights_input, classical_check_x, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_score import classical_r2_score

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def _qn_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # The estimator's numeric mode (None: the process mode), its connected
    # estimators entries, checked against the selected vendor and mode.
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["qn_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    _entry_table(entries, False)
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: QN native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: QN native mode differs from selected execution")
    return entries


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def qn_regressor_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var entries = _qn_entries(fields, context)
    var x = classical_host_f32(args[0], 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var view = classical_view(args[1], context, "y")
    if Int(py=view.ndim) != 1:
        raise Error("mojolearn QNRegressor requires a 1-D y")
    var target = classical_host_f32(view, 1, "y", context)
    if Int(py=target.shape[0]) != rows:
        raise Error("mojolearn QNRegressor X and y lengths differ")
    var weights = classical_check_weights_input(args[2], rows, context)
    var tol = _float(fields["tol"])
    var delta = fields["delta"]
    var change_tol = _float(delta) if delta is not None else tol * 0.01
    var loss = py_get(context["qn_losses"], fields["loss"])
    if loss is None:
        raise Error("mojolearn QNRegressor: unknown loss " + String(py_call(native_loaded_module("builtins").repr, fields["loss"])))
    var intercept = Bool(py=fields["fit_intercept"])
    var w = classical_typed(Python.tuple(cols + (1 if intercept else 0)), "<f4", context["output_context"])
    var info = classical_typed(Python.tuple(2), "<f4", context["output_context"])
    var has_sw = weights is not None
    var params = Python.list(rows, cols, 1, _float(fields["l1_strength"]), _float(fields["l2_strength"]),
                             tol, change_tol, _int(fields["max_iter"]), _int(fields["linesearch_max_iter"]),
                             _int(fields["lbfgs_memory"]), 1 if intercept else 0,
                             1 if Bool(py=fields["penalty_normalized"]) else 0, 1 if has_sw else 0,
                             _int(loss), Float64(0.0))
    if has_sw:
        _ = py_call(params.append, classical_address(weights))
        _ = weights  # keep the Python object alive through the native call above
    var n_iter = py_call_typed(entries["qn_fit"], classical_address(x), classical_address(target),
                               classical_address(w), classical_address(info), params)
    _ = target  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    var coef = classical_typed(Python.tuple(cols), "<f4", context["output_context"])
    var source = F32P(unsafe_from_address=Int(py=classical_address(w)))
    _ = w  # keep the Python object alive through the native call above
    var dest = F32P(unsafe_from_address=Int(py=classical_address(coef)))
    _ = coef  # keep the Python object alive through the native call above
    for j in range(cols):  # driver-loop: over the fitted coefficients (model structure)
        dest[j] = source[j]
        _ = coef  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
    var iterations = classical_typed(Python.tuple(1), "<i8", context["output_context"])
    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(iterations)))[0] = Int64(_int(n_iter))
    _ = iterations  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = target  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var words = F32P(unsafe_from_address=Int(py=classical_address(info)))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "input_copied_", PythonObject(not (x is args[0])))
    py_setitem(fields, "_w", w)
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", PythonObject(Float64(source[cols])) if intercept else PythonObject(0.0))
    _ = w  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", iterations)
    py_setitem(fields, "objective_", PythonObject(Float64(words[0])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "retcode_", PythonObject(Int(words[1])))
    _ = info  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", cols)
    _ = x^
    _ = target^
    _ = weights^
    _ = info^
    return model


def qn_regressor_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "_w"):
        raise Error("mojolearn QNRegressor: call fit before predict")
    var entries = _qn_entries(fields, context)
    var x = classical_host_f32(source, 2, "X", context)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if cols != Int(py=fields["n_features_in_"]):
        raise Error("mojolearn QNRegressor feature count differs from fit")
    var weights = classical_view(fields["_w"], context, "fitted coefficients")
    var expected = cols + (1 if Bool(py=fields["fit_intercept"]) else 0)
    if classical_dtype_str(weights, context) != "<f4" or not classical_c_order(weights, context) or Int(py=weights.size) != expected:
        raise Error("TypeError: QN fitted coefficients require contiguous float32 storage matching the fitted width")
    var out = classical_typed(Python.tuple(rows), "<f4", context["output_context"])
    _ = py_call_typed(entries["qn_decision_function"], classical_address(x), classical_address(weights),
                      classical_address(out), Python.list(rows, cols, 1 if Bool(py=fields["fit_intercept"]) else 0))
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = x^
    _ = weights^
    return out


def qn_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        _ = classical_family_admit(model)
        var result = qn_public_operation(model, operation, args, context)
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:") or String(error).startswith("DataConversionWarning:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 3)
                var warning = String(error).startswith("DataConversionWarning:")
                var cls = causal_borrowed_item(context, "warning_category" if warning else "active_not_fitted_class")
                return causal_set_exception(error_without_class(error, "DataConversionWarning" if warning else "NotFittedError"), cls)
            except nested:
                return raise_python_exception(nested)
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


# Native public lifecycle, replacing NumericModeMixin's Python decorators.
def _entry_table(entries: PythonObject, metrics: Bool) raises:
    var b = native_loaded_module("builtins")
    if py_type(entries) is not b.dict:
        raise Error("TypeError: QN providers require their native entry dictionaries")
    var keys = py_call(b.list, entries)
    for i in range(len(keys)):  # driver-loop: fixed provider metadata keys
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: QN provider names require exact strings")
    var module = entries["_module"]
    for i in range(len(keys)):  # driver-loop: every called entry belongs to the retained compiled provider
        var name = String(keys[i])
        if name != "_module" and entries[keys[i]] is not neural_descriptor_entry(module, name):
            raise Error("TypeError: QN provider entry changed after connection")


def _qn_schema(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = context["qn_parameters"]
    if py_type(names) is not b.tuple:
        raise Error("TypeError: QN parameter schema requires its native tuple")
    for i in range(len(names)):  # driver-loop: exact constructor parameter names
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: QN parameter names require exact strings")
    return names


def _qn_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    var names = _qn_schema(context)
    if len(names) != len(args):
        raise Error("TypeError: QN constructor metadata differs")
    var values = Python.dict()
    for i in range(len(names)):  # driver-loop: scalar metadata admission precedes every conversion/comparison
        host_inference_scalar(args[i], context, True)
        py_setitem(values, names[i], args[i])
    if not py_contains(context["qn_losses"], values["loss"]):
        raise Error("Expected loss to be one of " + String(py_call(b.repr, py_call(b.list, context["qn_losses"]))) + ", got " + String(py_call(b.repr, values["loss"])))
    if Bool(py=values["warm_start"]):
        raise Error("NotImplementedError: mojolearn QNRegressor: warm_start is not implemented (the solver starts from w0 = 0)")
    if Bool(py=py_call(op.lt, values["l1_strength"], 0)) or Bool(py=py_call(op.lt, values["l2_strength"], 0)):
        raise Error("mojolearn QNRegressor: l1_strength and l2_strength must be non-negative, got " + String(py_call(b.str, values["l1_strength"])) + " and " + String(py_call(b.str, values["l2_strength"])))
    if values["delta"] is not None and Bool(py=py_call(op.lt, values["delta"], 0)):
        raise Error("mojolearn QNRegressor: delta must be non-negative, got " + String(py_call(b.str, values["delta"])))
    if not Bool(py=py_call(op.gt, values["tol"], 0)):
        raise Error("mojolearn QNRegressor: tol must be positive, got " + String(py_call(b.str, values["tol"])))
    for name in ["max_iter", "linesearch_max_iter", "lbfgs_memory"]:  # driver-loop: three solver settings
        var value = values[name]
        if Bool(py=py_call(op.ne, py_call(b.int, value), value)) or Bool(py=py_call(op.lt, value, 1)):
            raise Error("mojolearn QNRegressor: " + name + " must be a positive integer, got " + String(py_call(b.str, value)))
    py_setitem(values, "warm_start", PythonObject(False))
    var fields = py_fields(model)
    if values["numeric_mode"] is None:
        py_setitem(values, "numeric_mode", family_raw_get_or(model, "numeric_mode", PythonObject(None)))
    for i in range(len(names)):  # driver-loop: full release admission before publishing constructor values
        if py_contains(fields, names[i]) and fields[names[i]] is not values[names[i]]:
            native_release_admit(fields[names[i]], context)
    for i in range(len(names)):  # driver-loop: preserve verbatim public parameter values
        py_setitem(fields, names[i], values[names[i]])
    return PythonObject(None)


def _qn_admit(model: PythonObject, context: PythonObject, fitting: Bool) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: never invoke a Python post-fit hook
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: QNRegressor cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: QNRegressor cannot invoke Python fit callbacks")
    var names = _qn_schema(context)
    for i in range(len(names)):  # driver-loop: fields used by the numerical owner
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context, True)
    var scalars = Python.tuple("n_features_in_", "input_copied_", "intercept_", "objective_", "retcode_", "_mojolearn_fitted", "_mojolearn_fitting")
    var arrays = Python.tuple("_w", "coef_", "n_iter_", "feature_names_in_")
    for i in range(len(scalars)):  # driver-loop: scalar fitted metadata before native conversion
        if py_contains(fields, scalars[i]):
            host_inference_scalar(fields[scalars[i]], context)
            if fitting:
                native_release_admit(fields[scalars[i]], context)
    for i in range(len(arrays)):  # driver-loop: retained fitted buffers and released prior owners
        if py_contains(fields, arrays[i]):
            classical_admit_value(fields[arrays[i]], context)
            if fitting:
                native_release_admit(fields[arrays[i]], context)
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags require exact booleans or integers")
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    return fields


def qn_public_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: QN arguments require a native tuple")
    var ancestry = _type_mro(py_type(model))
    var known = False
    for i in range(len(ancestry)):  # driver-loop: preserve passive subclasses of the connected owner
        known = known or ancestry[i] is context["qn_public_class"]
    if not known:
        raise Error("TypeError: QNRegressor requires its connected native class")
    if operation == "initialize":
        return _qn_initialize(model, args, context)
    var fitting = operation == "fit"
    var fields = _qn_admit(model, context, fitting)
    if not fitting:
        classical_require_fitted(model, False, operation, context)
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    if not fitting:
        classical_check_width(columns, Int(py=fields["n_features_in_"]), name)
        estimator_feature_check(model, fields, args[0], context)
    var y = classical_check_y(args[1], rows, name, context) if fitting else PythonObject(None)
    var depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        if fitting:
            result = qn_regressor_fit(model, Python.tuple(x, y, args[2]), context)
        elif operation == "predict":
            result = qn_regressor_predict(model, x, context)
        elif operation == "score":
            var metric = context["classical_metrics"][context["backend_fields"]["_DEFAULT_MODE"]]
            if py_type(metric) is not b.str:
                _entry_table(metric, True)
            result = classical_r2_score(qn_regressor_predict(model, x, context), args[1], context)
        else:
            raise Error("RuntimeError: unknown native QN operation")
    except problem:
        var refused = _refusal(problem, fitting, depth > 0, rows, columns, name)
        raise refused^
    finally:
        if fitting:
            estimator_fit_end(model, depth)
    if fitting:
        estimator_fit_complete(model, args[0], False, context)
    if operation == "predict" and estimator_outside() and py_get(context, "ndarray") is not None:
        var wide = classical_typed(result.shape, "<f8", context)
        labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8", Int(py=result.size), context)
        _ = result^
        return wide
    return result
