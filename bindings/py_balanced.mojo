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
from bindings.native_loaded_module import native_loaded_module
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
    return py_call(native_loaded_module("builtins").object.__getattribute__, value, "__dict__")
