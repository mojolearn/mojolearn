# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_description import sequential_fields
from bindings.classical_ridge import classical_ridge_fit, classical_linear_predict, classical_linear_predict_public
from bindings.classical_ols import classical_ols_fit
from bindings.classical_score import classical_linear_score
from bindings.classical_metadata import classical_initialize, classical_get_params, classical_set_params, classical_is_fitted
from bindings.classical_archive import classical_archive_save
from bindings.classical_archive_read import classical_archive_load
from bindings.classical_exceptions import classical_require_fitted, classical_error_restore, classical_error_reduce
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.classical_pandas import classical_admit_value


def classical_entries(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var mode = fields.get("numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if Python.import_module("builtins").type(mode) is not Python.import_module("builtins").str:
        raise Error("TypeError: numeric_mode must be identical, fast or None")
    mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be identical, fast or None")
    var entries = context["classical_bindings"][mode]
    if Python.import_module("builtins").type(entries) is Python.import_module("builtins").str:
        raise Error("ImportError: " + String(entries))
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: classical native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: classical native mode differs from selected execution")
    return entries


def classical_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var b = Python.import_module("builtins")
        if operation == "restore_not_fitted":
            var error = classical_error_restore(args, context)
            return error.steal_data()
        if operation == "reduce_not_fitted":
            var reduced = classical_error_reduce(model, context)
            return reduced.steal_data()
        var loading = operation == "load"
        var ridge = model is context["ridge_class"] if loading else b.type(model) is context["ridge_class"]
        var linear = model is context.get("linear_class") if loading else b.type(model) is context.get("linear_class")
        if not ridge and not linear:
            raise Error("TypeError: the private classical owner requires an exact Ridge or LinearRegression instance")
        if loading:
            var loaded = classical_archive_load(model, args[0], ridge, context)
            return loaded.steal_data()
        if operation != "save":
            classical_admit_value(args, context)
        var fields = sequential_fields(model)
        var names = context["classical_fields"]
        for k in range(len(names)):  # driver-loop: over connected estimator metadata fields
            if Bool(py=fields.__contains__(names[k])):
                sequential_admit_value(fields[names[k]], context)
        var result = PythonObject(None)
        if operation == "initialize":
            result = classical_initialize(model, args, ridge)
            return result.steal_data()
        if operation == "get_params":
            result = classical_get_params(model, ridge)
            return result.steal_data()
        if operation == "set_params":
            result = classical_set_params(model, args[0], ridge)
            return result.steal_data()
        if operation == "is_fitted":
            result = classical_is_fitted(model)
            return result.steal_data()
        if operation == "save":
            result = classical_archive_save(model, args[0], ridge, context)
            return result.steal_data()
        if operation == "predict_public" or operation == "score_public":
            classical_require_fitted(model, ridge, "predict" if operation == "predict_public" else "score", context)
        var entries = classical_entries(model, context)
        if operation == "ridge_fit":
            if not ridge:
                raise Error("TypeError: ridge_fit requires Ridge")
            result = classical_ridge_fit(model, args, entries, context)
        elif operation == "ols_fit":
            if not linear:
                raise Error("TypeError: ols_fit requires LinearRegression")
            result = classical_ols_fit(model, args, entries, context)
        elif operation == "predict":
            result = classical_linear_predict(model, args[0], entries, context)
        elif operation == "predict_public":
            result = classical_linear_predict_public(model, args[0], entries, context)
        elif operation == "score" or operation == "score_public":
            result = classical_linear_score(model, args, entries, context)
        elif operation == "numeric_mode_used":
            result = PythonObject("identical" if Int(py=entries["estimators_numeric_mode"]()) == 1 else "fast")
        elif operation == "vendor_used":
            result = entries["estimators_vendor"]()
        else:
            raise Error("unknown private classical operation")
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 3)
                var cls = causal_borrowed_item(context, "active_not_fitted_class")
                var message = String(error)
                return causal_set_exception(Error(String(message[byte=16:])), cls)
            except nested:
                return raise_python_exception(nested)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error, ExceptionType.ValueError)
