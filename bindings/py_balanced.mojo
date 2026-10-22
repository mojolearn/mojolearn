# SPDX-License-Identifier: Apache-2.0
# Python door helpers with balanced reference ownership.
#
# Mojo 1.0's `PythonObject.__call__` keeps one extra reference on every
# positional argument, and `__setattr__`/`__setitem__` keep one on the value
# (docs/MODULAR_ASKS.md, "PythonObject attribute assignment and calls retain
# one extra reference"). A native owner that calls those on a caller's model or
# arrays never lets them be freed. These helpers use the documented CPython
# calls directly, with their documented ownership: every argument is borrowed,
# the call envelope we create is released once, and each result is a new
# reference owned by the returned PythonObject. Nothing here adds or removes a
# reference to compensate for the standard library; when Modular ships the
# balanced operations, call sites can go back to the ordinary syntax.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


def py_call(callable: PythonObject, *args: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """`callable(*args, **kwargs)` without retaining the arguments."""
    ref cpy = Python().cpython()
    var size = len(args)
    var envelope = cpy.PyTuple_New(size)
    if not envelope:
        raise cpy.unsafe_get_error()
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        # PyTuple_SetItem steals the new reference the tuple now owns; the
        # caller keeps its own.
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var keywords = PyObjectPtr()
    if len(kwargs) > 0:
        # Python.dict(**) borrows each value into the dictionary it owns.
        keywords = Python.dict(**kwargs^).steal_data()
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, keywords)
    cpy.Py_DecRef(envelope)
    if keywords:
        cpy.Py_DecRef(keywords)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


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


def py_type(value: PythonObject) raises -> PythonObject:
    """`type(value)`: a new reference to the value's class."""
    ref cpy = Python().cpython()
    return PythonObject(from_owned=cpy.PyObject_Type(value._obj_ptr))


def py_setitem(container: PythonObject, key: PythonObject, value: PythonObject) raises:
    """`container[key] = value`; the container takes its own references."""
    ref cpy = Python().cpython()
    if cpy.PyObject_SetItem(container._obj_ptr, key._obj_ptr, value._obj_ptr) == -1:
        raise cpy.unsafe_get_error()


def py_setattr(owner: PythonObject, var name: String, value: PythonObject) raises:
    """`setattr(owner, name, value)`; the owner takes its own reference."""
    ref cpy = Python().cpython()
    if cpy.PyObject_SetAttrString(owner._obj_ptr, name^, value._obj_ptr) == -1:
        raise cpy.unsafe_get_error()


def py_get(mapping: PythonObject, key: PythonObject) raises -> PythonObject:
    """`mapping.get(key)`."""
    return py_call(mapping.get, key)


def py_get(mapping: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    """`mapping.get(key, default)`."""
    return py_call(mapping.get, key, default)


def py_contains(container: PythonObject, key: PythonObject) raises -> Bool:
    """`key in container`."""
    ref cpy = Python().cpython()
    var found = cpy.PySequence_Contains(container._obj_ptr, key._obj_ptr)
    if found == -1:
        raise cpy.unsafe_get_error()
    return found == 1


def py_fields(value: PythonObject) raises -> PythonObject:
    """`object.__getattribute__(value, "__dict__")`: the instance dictionary
    through object's native lookup, without class or instance hooks."""
    return py_call(Python.import_module("builtins").object.__getattribute__, value, "__dict__")
