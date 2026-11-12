# SPDX-License-Identifier: Apache-2.0
# Shared door helpers of the classical family owners (bindings/family_native.mojo,
# lane migrate-classical-b): the `_buffer.as_f32_c` and
# `_sample_weight.check_sample_weight` refusals with their wording, mojolearn
# Array outputs, and calls into the compiled entries of the binding a public
# shell hands over. A Python exception a callee raises (a kernel's refusal,
# a CPU-only install's by-name refusal, a user callable's own exception) is
# stashed and restored unchanged at the door (`stash_python_error`), so the
# caller sees the class and message it saw before the migration.
# The PACK half of the former `bindings/family_common.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/family_common_unpack.mojo`.
from providers.python.array_metadata_pyapi_door import ArrayMetadata
from bindings.native_array_storage_door import native_array_zeros
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays_door import classical_address, classical_wrap_array, _store_code
from core.family_cast_impl import family_min_f32
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.native_class_admission_door import static_builtin_type

def family_call(callable: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`callable(*args)` without retaining the arguments; a Python exception
    it raises is restored unchanged at the door."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_call_kw(callable: PythonObject, keywords: PythonObject, *args: PythonObject) raises -> PythonObject:
    """`callable(*args, **keywords)` (`keywords` a dict) without retaining
    the arguments; a Python exception it raises is restored unchanged."""
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # small-loop(args: positional arguments of one call): bounded by the call site
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var result = cpy.PyObject_Call(callable._obj_ptr, envelope, keywords._obj_ptr)
    cpy.Py_DecRef(envelope)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def family_entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The compiled entry `name` of the binding the shell resolved for the
    model's numeric mode. A by-name refusal of the binding (a tier that is
    not built, an entry a CPU-only install does not have) is raised
    unchanged; anything other than a compiled entry is refused."""
    # Perf (lane perf2-cluster): CPython's own attribute lookup, the one
    # `getattr(binding, name)` makes (a module `__getattr__` included), not a
    # call of `builtins.getattr` through the builtins connection (a namespace
    # stamp walk per entry, several entries per public call). A failure is
    # stashed and restored unchanged at the door, as family_call did.
    ref cpy = Python().cpython()
    var raw = cpy.PyObject_GetAttrString(binding._obj_ptr, name.copy())
    if not raw:
        var failure = stash_python_error()
        raise failure^
    var entry = PythonObject(from_owned=raw)
    if py_type(entry) is not context["family_builtin_type"]:
        raise Error("TypeError: mojolearn: " + name + " must be a compiled entry")
    return entry


def family_array(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """A new zero-filled C-order mojolearn Array (`_buffer.empty`/`zeros`),
    with or without NumPy: the outputs the Python shells always returned."""
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        width = 4
    elif typestr == "<u1" or typestr == "<i1":
        width = 1
    var store = native_array_zeros(String(_store_code(typestr)), count)
    return classical_wrap_array(store, shape, typestr, "C", context)


def family_field(fields: PythonObject, name: String) raises -> PythonObject:
    """`fields[name]`, refused as the attribute lookup refused it."""
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is None and not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value
