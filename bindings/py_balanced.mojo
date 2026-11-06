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
from std.ffi import _Global
from bindings.native_loaded_module import native_loaded_module
from bindings.cpython_table import cpython_table
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr


struct _ExactDictType(Defaultable, Movable):
    # Address of CPython's dict type (a static, immortal type), 0 until read.
    var address: Int

    def __init__(out self):
        self.address = 0


comptime _EXACT_DICT_TYPE = _Global[
    StorageType=_ExactDictType, name="MojolearnPyBalancedDictTypeV1",
    init_fn=_ExactDictType.__init__,
]


def _is_exact_dict(value: PythonObject) raises -> Bool:
    # Exact dict only: a subclass may override get(), so it keeps the call.
    ref cpy = Python().cpython()
    var g = _EXACT_DICT_TYPE.get_or_create_ptr()
    if g[].address == 0:
        var sample = Python.dict()
        var dict_type = cpy.PyObject_Type(sample._obj_ptr)
        g[].address = Int(dict_type)
        cpy.Py_DecRef(dict_type)
    var cls = cpy.PyObject_Type(value._obj_ptr)
    var same = Int(cls) == g[].address
    cpy.Py_DecRef(cls)
    return same


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
    return py_get(mapping, key, PythonObject(None))


def py_get(mapping: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    """`mapping.get(key, default)`."""
    if _is_exact_dict(mapping):
        # dict.get's own C lookup: no bound method, argument tuple or call
        # per read. Same key hashing/equality and the same errors (an
        # unhashable key raises TypeError).
        ref cpy = Python().cpython()
        var found = cpy.PyDict_GetItemWithError(mapping._obj_ptr, key._obj_ptr)
        if not found:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            return default
        return PythonObject(from_borrowed=found)
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
    ref cpy = Python().cpython()
    var t = cpython_table()
    var cls = cpy.PyObject_Type(value._obj_ptr)
    var flags = t[].PyType_GetFlags(cls)
    cpy.Py_DecRef(cls)
    # Py_TPFLAGS_TYPE_SUBCLASS (1 << 31): a class's __dict__ is its
    # mappingproxy (type's data descriptor), not the raw namespace; keep
    # object.__getattribute__ for classes.
    if (flags & UInt64(1 << 31)) == 0:
        # An instance: object.__getattribute__(value, "__dict__") resolves
        # the class's `__dict__` getset, whose getter is this same generic
        # dictionary read (one C call instead of two attribute lookups and a
        # Python call).
        var raw = t[].PyObject_GenericGetDict(value._obj_ptr, PyObjectPtr())
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)
    return py_call(native_loaded_module("builtins").object.__getattribute__, value, "__dict__")
