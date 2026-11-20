# SPDX-License-Identifier: Apache-2.0
# Legacy tokenizer entry only. Its import graph must not reach Sequential's
# implementation; import setup retains the independently compiled native entry.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_type


def runtime_sequential_forward_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var context = call[3]
        # The binding context is an exact native dict. Refuse protocol objects
        # before looking up either retained entry or its import-time type.
        if py_type(context) is not py_type(Python.dict()):
            raise Error("TypeError: Sequential requires its native binding context")
        var entry = context["sequential_owner_entry"]
        if py_type(entry) is not context["sequential_builtin_type"]:
            raise Error("TypeError: Sequential requires its compiled native entry")
        # Forward the existing tuple directly. A NULL result keeps the native
        # owner's exact Python exception; no reboxing or callback router runs.
        ref cpy = Python().cpython()
        return cpy.PyObject_Call(entry._obj_ptr, args_ptr, PyObjectPtr())
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
