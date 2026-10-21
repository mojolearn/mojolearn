# SPDX-License-Identifier: Apache-2.0
# Incremental native classical owner. Public activation waits for full API
# migration; this private entry is exercised directly by strict audit tests.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_description import sequential_fields
from bindings.classical_ridge import classical_ridge_fit, classical_linear_predict
from bindings.classical_ols import classical_ols_fit


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
        var ridge = b.type(model) is context["ridge_class"]
        var linear = b.type(model) is context.get("linear_class")
        if not ridge and not linear:
            raise Error("TypeError: the private classical owner requires an exact Ridge or LinearRegression instance")
        sequential_admit_value(args, context)
        var fields = sequential_fields(model)
        var names = context["classical_fields"]
        for k in range(len(names)):  # driver-loop: over connected estimator metadata fields
            if Bool(py=fields.__contains__(names[k])):
                sequential_admit_value(fields[names[k]], context)
        var entries = classical_entries(model, context)
        var result = PythonObject(None)
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
        else:
            raise Error("unknown private classical operation")
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error, ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error, ExceptionType.ValueError)
