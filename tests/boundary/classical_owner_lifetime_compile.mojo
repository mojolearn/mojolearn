# SPDX-License-Identifier: Apache-2.0
# Focused compile of actual constructor/parameter/refit owners; no execution.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import raise_python_exception
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_metadata import classical_initialize, classical_set_params
from bindings.classical_ridge import classical_ridge_fit
from bindings.classical_ols import classical_ols_fit
from bindings.logistic_native import logistic_c

@export
def classical_lifetime_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=packed)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        classical_check_layout(model, context)
        var result = PythonObject(None)
        if operation == "ridge_initialize":
            result = classical_initialize(model, args, True, context)
        elif operation == "linear_initialize":
            result = classical_initialize(model, args, False, context)
        elif operation == "ridge_set_params":
            result = classical_set_params(model, args[0], True, context)
        elif operation == "linear_set_params":
            result = classical_set_params(model, args[0], False, context)
        elif operation == "ridge_fit":
            result = classical_ridge_fit(model, args, call[4], context)
        elif operation == "linear_fit":
            result = classical_ols_fit(model, args, call[4], context)
        else:
            raise Error("unknown focused classical operation")
        return result.steal_data()
    except error:
        return raise_python_exception(error)

@export
def logistic_lifetime_compile(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    return logistic_c(py_self, packed)
