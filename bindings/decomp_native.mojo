# SPDX-License-Identifier: Apache-2.0
# The native owner door of the decomposition expansion's estimators
# (FastICA, ...): every public operation of those classes is one call into
# decomp_c, which admits the model by its connected class and dispatches the
# operation to the class's native file (bindings/ica_native.mojo, ...).
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type
from bindings.classical_attributes import classical_connect_numpy
from bindings.ica_native import ica_fit, ica_project


def _family(model: PythonObject, context: PythonObject) raises -> String:
    """The connected class the model's MRO holds ('' for none)."""
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return ""
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var classes = context["decomp_classes"]
    var names = py_call(b.list, classes.keys())
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        for j in range(len(names)):  # driver-loop: over the connected classes
            if mro[k] is classes[names[j]]:
                return String(names[j])
    return ""


def decomp_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: decomp_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple:
            raise Error("TypeError: native decomposition operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if py_type(context) is not b.dict:
            raise Error("TypeError: native decomposition requires connected context")
        var family = _family(model, context)
        if family == "":
            raise Error("TypeError: the native decomposition owner requires a connected estimator")
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if family == "FastICA":
            if operation == "fit":
                result = ica_fit(model, args[0], False, context)
            elif operation == "fit_transform":
                result = ica_fit(model, args[0], True, context)
            elif operation == "transform" or operation == "inverse_transform":
                result = ica_project(model, args[0], operation == "inverse_transform", context)
            else:
                raise Error("unknown private FastICA operation")
        else:
            raise Error("unknown private decomposition estimator")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
