# SPDX-License-Identifier: Apache-2.0
# Admit NumPy's native dtype constructor before any live-module field call.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem


def native_numpy_module_fields(module: PythonObject) raises -> PythonObject:
    # The caller admits the exact native module class. The public C API
    # returns its actual dictionary without looking up mutable object or
    # invoking a module attribute hook.
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_borrowed=pointer)
    var keys = py_call(py_type(Python.list()),fields)
    var string_type = py_type(PythonObject(""))
    for i in range(len(keys)):  # driver-loop: over dictionary lookup must not invoke a custom key's equality
        if py_type(keys[i]) is not string_type:
            raise Error("TypeError: NumPy native module fields require native string keys")
    return fields


def native_numpy_type(name: String, context: PythonObject) raises -> PythonObject:
    # Obtain type from a C-created scalar, independent of builtins.type.
    var type_class = py_type(py_type(PythonObject(0)))
    var modules = context["modules"]
    var module_type = py_type(native_loaded_module("sys"))
    var numpy = py_get(modules, "numpy")
    var owner = py_get(modules, "numpy._core._multiarray_umath")
    if owner is None:
        owner = py_get(modules, "numpy.core._multiarray_umath")
    if numpy is None or owner is None or py_type(numpy) is not module_type or py_type(owner) is not module_type:
        raise Error("TypeError: ndarray dtype requires NumPy's native extension metadata")
    var dtype = py_get(native_numpy_module_fields(owner), PythonObject(name))
    # The type descriptor itself refuses non-types without invoking them.
    # Python-created types/metaclasses carry HEAPTYPE; the supported NumPy
    # dtype constructor and its metaclass are static extension types.
    var flags = type_class.__dict__["__flags__"]
    if dtype is None or (Int(py=py_call(flags.__get__, dtype)) & 512) != 0 or (Int(py=py_call(flags.__get__, py_type(dtype))) & 512) != 0:
        raise Error("TypeError: np." + name + " must be NumPy's native type")
    if String(py_call(type_class.__getattribute__, dtype, "__name__")) != name or String(py_call(type_class.__getattribute__, dtype, "__module__")) != "numpy":
        raise Error("TypeError: np." + name + " must be owned by NumPy's native extension")
    var key = PythonObject("native_numpy_" + name)
    var retained = py_get(context, key)
    if retained is not None and retained is not dtype:
        raise Error("TypeError: NumPy's native " + name + " type has been replaced")
    if py_get(native_numpy_module_fields(numpy), PythonObject(name)) is not dtype:
        raise Error("TypeError: np." + name + " must remain NumPy's native type; Python callbacks cannot execute")
    py_setitem(context, key, dtype)
    return dtype


def native_numpy_dtype(context: PythonObject) raises -> PythonObject:
    return native_numpy_type("dtype", context)
