# SPDX-License-Identifier: Apache-2.0
# Shared C-entry exception mapping. Each native owner compiles this boundary
# with the same pending-error storage as its own callees; no raw Mojo error
# or pending-error object crosses between development shared libraries.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.py_balanced import py_type, py_get, py_contains


def _classical_error_class(args_ptr: PyObjectPtr, key: String) raises -> PyObjectPtr:
    ref cpy = Python().cpython()
    var root = cpy.PyTuple_GetItem(args_ptr, 3)
    var context = PythonObject(from_borrowed=root)
    var host = py_get(context, "classical_host_context")
    if host is not None:
        var model = PythonObject(from_borrowed=cpy.PyTuple_GetItem(args_ptr, 0))
        var b = Python.import_module("builtins")
        var cls = model if py_type(model) is b.type else py_type(model)
        if py_contains(host["classical_kinds"], cls):
            root = causal_borrowed_item(root, "classical_host_context")
    # The active call owns its root/host dictionaries and the returned class.
    return causal_borrowed_item(root, key)


def classical_python_error(var error: Error, args_ptr: PyObjectPtr) -> PyObjectPtr:
    if String(error) == PYTHON_ERROR_PENDING:
        return optimizer_python_error(error^)
    if String(error).startswith("HostIForestKernelError:"):
        return raise_python_exception(error_without_class(error, "HostIForestKernelError"))
    if String(error).startswith("HostKDEKernelError:"):
        return raise_python_exception(error_without_class(error, "HostKDEKernelError"))
    if String(error).startswith("DataConversionWarning:"):
        try:
            var cls = _classical_error_class(args_ptr, "warning_category")
            var message = String(error)
            return causal_set_exception(Error(String(message[byte=23:])), cls)
        except nested:
            return raise_python_exception(nested)
    if String(error).startswith("NotFittedError:"):
        try:
            var cls = _classical_error_class(args_ptr, "active_not_fitted_class")
            var message = String(error)
            return causal_set_exception(Error(String(message[byte=16:])), cls)
        except nested:
            return raise_python_exception(nested)
    if String(error).startswith("HostInferenceWarning:"):
        try:
            ref cpy = Python().cpython()
            # The active ABI args tuple owns the context and admitted
            # category throughout PyErr_SetString; borrow without transfer.
            var context = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 3), "host_inference_context")
            var category = causal_borrowed_item(context, "warning_category")
            return causal_set_exception(error_without_class(error, "HostInferenceWarning"), category)
        except nested:
            return raise_python_exception(nested)
    if String(error).startswith("TypeError:"):
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
    if String(error).startswith("ZeroDivisionError:"):
        return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
    if String(error).startswith("OverflowError:"):
        return raise_python_exception(error_without_class(error, "OverflowError"), ExceptionType("PyExc_OverflowError"))
    if String(error).startswith("NotImplementedError:"):
        return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
    if String(error).startswith("ImportError:"):
        return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
    if String(error).startswith("IndexError:"):
        return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
    if String(error).startswith("AttributeError:"):
        return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
    if String(error).startswith("RuntimeError:"):
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
    if String(error).startswith("ZeroDivisionError:"):
        return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
    return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
