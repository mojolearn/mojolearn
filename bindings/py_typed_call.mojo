# SPDX-License-Identifier: Apache-2.0
# A balanced Python call that keeps the class of a builtin exception the
# callee raises (another native entry's ValueError/TypeError/... refusal), so
# the caller's door re-raises that class instead of a bare Exception. Kept out
# of bindings/py_balanced.mojo so only the doors that use it rebuild.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call


def typed_python_error() -> Error:
    """The pending Python exception as an Error that keeps its class: a
    builtin ValueError, TypeError, IndexError, AttributeError, RuntimeError,
    NotImplementedError, ZeroDivisionError, OverflowError, ArithmeticError,
    ImportError or FileNotFoundError becomes `"<Class>: <message>"`, the
    prefix every native door maps back to that class. Any other exception
    (subclasses included) keeps its bare message. Clears the indicator.
    The caller MUST know an exception is pending."""
    ref cpy = Python().cpython()
    var kind = cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")()
    if not kind:
        return Error("mojolearn: a Python call failed without an exception")
    # PyErr_Occurred borrows the class; keep it across the fetch below.
    cpy.Py_IncRef(kind)
    var error = cpy.unsafe_get_error()
    var cls = PythonObject(from_owned=kind)
    try:
        var name = String(py=cls.__name__)
        if (name == "ValueError" or name == "TypeError" or name == "IndexError" or name == "AttributeError"
                or name == "RuntimeError" or name == "NotImplementedError" or name == "ZeroDivisionError"
                or name == "OverflowError" or name == "ArithmeticError" or name == "ImportError"
                or name == "FileNotFoundError"):
            var b = Python.import_module("builtins")
            if py_call(b.getattr, b, PythonObject(name)) is cls:
                return Error(name + ": " + String(error))
    except:
        pass
    return error^


def py_call_typed(callable: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`py_call` for a callee whose refusal class matters (another native
    entry, a builtin conversion): a builtin exception it raises reaches the
    caller's door as `"<Class>: <message>"` (`typed_python_error`), so the
    door re-raises the same class instead of a bare `Exception`."""
    ref cpy = Python().cpython()
    var size = len(args)
    var envelope = cpy.PyTuple_New(size)
    if not envelope:
        raise typed_python_error()
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        raise typed_python_error()
    return PythonObject(from_owned=result)
