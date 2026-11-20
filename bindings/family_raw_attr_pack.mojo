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
# The PACK half of the former `bindings/family_raw_attr.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/family_raw_attr_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_class_admission_door import static_builtin_type
from bindings.optimizer_public_words_unpack import stash_python_error


def family_raw_get(model: PythonObject, name: String) raises -> PythonObject:
    """`object.__getattribute__(model, name)`; its AttributeError is restored
    unchanged at the door."""
    ref cpy = Python().cpython()
    var key = PythonObject(name)
    var result = cpython_table()[].PyObject_GenericGetAttr(model._obj_ptr, key._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_raw_get_or(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)` through the generic lookup: only an
    AttributeError selects `default`; any other exception propagates."""
    ref cpy = Python().cpython()
    var key = PythonObject(name)
    # Resolve the exception type before lookup can set CPython's error indicator.
    # The registry validation performs PythonObject operations; doing that with
    # a pending lookup error consumes it, losing both missing-field defaults and
    # non-AttributeError exceptions (BGMM first fit, extended GMM, GMM reload).
    # Reproduction: tests/native_family_raw_attr_probe.mojo and its Python test.
    var attribute_error = static_builtin_type("builtins", "AttributeError")
    var result = cpython_table()[].PyObject_GenericGetAttr(model._obj_ptr, key._obj_ptr)
    if not result:
        if cpython_table()[].PyErr_ExceptionMatches(attribute_error._obj_ptr) != 0:
            cpython_table()[].PyErr_Clear()
            return default
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)
