# SPDX-License-Identifier: Apache-2.0
# Model attribute reads for the family owners without the class's Python
# `__getattribute__` (PublicOutputsMixin's NumPy publication hook).
#
# The family owners run inside a mojolearn method, so the hook's own rule
# (`_mode.PublicOutputsMixin.__getattribute__`: internal readers get the
# stored value) always answered with `object.__getattribute__`'s result.
# CPython's generic lookup (`PyObject_GenericGetAttr`, the C body of
# `object.__getattribute__`) gives that same value -- data descriptors,
# then the instance dictionary, then class attributes -- and raises the same
# AttributeError, without executing the Python hook or a `__getattr__`
# fallback.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import stash_python_error


def family_raw_get(model: PythonObject, name: String) raises -> PythonObject:
    """`object.__getattribute__(model, name)`; its AttributeError is restored
    unchanged at the door."""
    ref cpy = Python().cpython()
    var key = PythonObject(name)
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(model._obj_ptr, key._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_raw_get_or(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)` through the generic lookup: only an
    AttributeError selects `default`; any other exception propagates."""
    ref cpy = Python().cpython()
    var key = PythonObject(name)
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetAttr")(model._obj_ptr, key._obj_ptr)
    if not result:
        var attribute_error = native_loaded_module("builtins").AttributeError
        if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(attribute_error._obj_ptr) != 0:
            cpy.lib.get_function[NoneType]("PyErr_Clear")()
            return default
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_raw_has(model: PythonObject, name: String) raises -> Bool:
    """`hasattr(model, name)` through the generic lookup."""
    var missing = Python.list()  # a fresh sentinel no stored attribute can be
    return family_raw_get_or(model, name, missing) is not missing
