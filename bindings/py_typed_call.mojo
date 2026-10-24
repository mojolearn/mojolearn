# SPDX-License-Identifier: Apache-2.0
# A balanced Python call that keeps the class of a builtin exception the
# callee raises (another native entry's ValueError/TypeError/... refusal), so
# the caller's door re-raises that class instead of a bare Exception. Kept out
# of bindings/py_balanced.mojo so only the doors that use it rebuild.
from bindings.native_loaded_module import _loaded_lookup
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def _connected_builtin(name: String) raises -> PythonObject:
    """The builtin export retained at import time in
    `mojolearn._native_loaded_modules` (None when absent). Reading the
    retained identity, not the live builtins module, keeps the class of a
    pending refusal even when that refusal is a replaced builtins export:
    `native_loaded_module("builtins")` would raise again here and the door
    would fall back to a bare Exception."""
    ref cpy = Python().cpython()
    var modules = cpy.lib.get_function[PyObjectPtr]("PyImport_GetModuleDict")()
    if not modules:
        return PythonObject(None)
    var registry = _loaded_lookup(modules, "mojolearn._native_loaded_modules")
    if not registry:
        return PythonObject(None)
    var fields = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(registry)
    if not fields:
        cpy.lib.get_function[NoneType]("PyErr_Clear")()
        return PythonObject(None)
    var raw = _loaded_lookup(fields, "CONNECTIONS")
    if not raw:
        return PythonObject(None)
    var connections = PythonObject(from_owned=cpy.Py_NewRef(raw))
    # driver-loop: over the fixed import-time module inventory
    for i in range(len(connections)):  # small-loop(connections: registered modules): selects the builtins record
        var record = connections[i]
        if String(py=record[0]) != "builtins":
            continue
        var exports = record[2]
        # driver-loop: over retained builtin export names
        for j in range(len(exports)):  # small-loop(exports: retained builtin names): finds one setup-time class
            var item = exports[j]
            if String(py=item[0]) == name:
                return item[1]
        break
    return PythonObject(None)


def typed_python_error() -> Error:
    """The pending Python exception as an Error that keeps its class: a
    builtin ValueError, TypeError, IndexError, AttributeError, RuntimeError,
    NotImplementedError, ZeroDivisionError, OverflowError, ArithmeticError,
    ImportError or FileNotFoundError becomes `"<Class>: <message>"`, the
    prefix every native door maps back to that class. Any other exception
    (subclasses included) keeps its bare message. Clears the indicator.
    The caller MUST know an exception is pending."""
    ref cpy = Python().cpython()
    var kind: PyObjectPtr
    try:
        kind = cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")()
    except e:
        return e^
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
            if _connected_builtin(name) is cls:
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
