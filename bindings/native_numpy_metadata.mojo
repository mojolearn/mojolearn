# SPDX-License-Identifier: Apache-2.0
# Admit NumPy's native dtype constructor before any live-module field call.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem


def native_numpy_dtype(context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var modules = context["modules"]
    var module_type = py_type(Python.import_module("sys"))
    var numpy = py_get(modules, "numpy")
    var owner = py_get(modules, "numpy._core._multiarray_umath")
    if owner is None:
        owner = py_get(modules, "numpy.core._multiarray_umath")
    if numpy is None or owner is None or py_type(numpy) is not module_type or py_type(owner) is not module_type:
        raise Error("TypeError: ndarray dtype requires NumPy's native extension metadata")
    var dtype = py_get(py_fields(owner), "dtype")
    # The type descriptor itself refuses non-types without invoking them.
    # Python-created types/metaclasses carry HEAPTYPE; the supported NumPy
    # dtype constructor and its metaclass are static extension types.
    var flags = b.type.__dict__["__flags__"]
    if dtype is None or (Int(py=py_call(flags.__get__, dtype)) & 512) != 0 or (Int(py=py_call(flags.__get__, py_type(dtype))) & 512) != 0:
        raise Error("TypeError: np.dtype must be NumPy's native dtype constructor")
    if String(py_call(b.type.__getattribute__, dtype, "__name__")) != "dtype" or String(py_call(b.type.__getattribute__, dtype, "__module__")) != "numpy":
        raise Error("TypeError: np.dtype must be owned by NumPy's native extension")
    var retained = py_get(context, "sequence_native_dtype")
    if retained is not None and retained is not dtype:
        raise Error("TypeError: NumPy's native dtype constructor has been replaced")
    if py_get(py_fields(numpy), "dtype") is not dtype:
        raise Error("TypeError: np.dtype must remain NumPy's native dtype constructor; Python dtype callbacks cannot execute")
    py_setitem(context, "sequence_native_dtype", dtype)
    return dtype
