# SPDX-License-Identifier: Apache-2.0
# Shared door helpers of the classical family owners (bindings/family_native.mojo,
# lane migrate-classical-b): the `_buffer.as_f32_c` and
# `_sample_weight.check_sample_weight` refusals with their wording, mojolearn
# Array outputs, and calls into the compiled entries of the binding a public
# shell hands over. A Python exception a callee raises (a kernel's refusal,
# a CPU-only install's by-name refusal, a user callable's own exception) is
# stashed and restored unchanged at the door (`stash_python_error`), so the
# caller sees the class and message it saw before the migration.
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.native_array_storage import native_array_zeros
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_arrays import classical_address, classical_wrap_array, _store_code
from core.family_cast_impl import family_min_f32
from bindings.classical_validate import classical_view, RESHAPE_HINT
from bindings.optimizer_public_words import stash_python_error
from bindings.native_class_admission import static_builtin_type

comptime FAMILY_F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime FAMILY_F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime FAMILY_I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime FAMILY_I64P = MutPointer[Int64, MutUntrackedOrigin]



def family_repr(value: PythonObject) raises -> String:
    """`repr(value)`."""
    return String(py_call(native_loaded_module("builtins").repr, value))


def family_str(value: PythonObject) raises -> String:
    """`str(value)`."""
    return String(py_call(native_loaded_module("builtins").str, value))


def family_isinstance(value: PythonObject, classes: PythonObject) raises -> Bool:
    """`isinstance(value, classes)`."""
    return Bool(py=py_call(native_loaded_module("builtins").isinstance, value, classes))


# Perf (lane perf2-cluster): the parameter tests below run several times per
# public call. For a value whose class is exactly str, int, float or bool the
# answer is known without a call: `isinstance` returns at its exact-type check
# before any `__instancecheck__`, and bool is a final class. Those classes are
# static builtin types, resolved once per process (static_builtin_type), so
# the common case needs no builtins namespace stamp; any other value takes
# the `isinstance` calls as before.
def family_is_str(value: PythonObject) raises -> Bool:
    if py_type(value) is static_builtin_type("builtins", "str"):
        return True
    return family_isinstance(value, native_loaded_module("builtins").str)


def family_is_int(value: PythonObject) raises -> Bool:
    """An int that is not a bool (`isinstance(v, int) and not isinstance(v, bool)`)."""
    var cls = py_type(value)
    if cls is static_builtin_type("builtins", "int"):
        return True
    if cls is static_builtin_type("builtins", "bool"):
        return False
    var b = native_loaded_module("builtins")
    return family_isinstance(value, b.int) and not family_isinstance(value, b.bool)


def family_is_real(value: PythonObject) raises -> Bool:
    """An int or float that is not a bool."""
    var cls = py_type(value)
    if cls is static_builtin_type("builtins", "int") or cls is static_builtin_type("builtins", "float"):
        return True
    if cls is static_builtin_type("builtins", "bool"):
        return False
    var b = native_loaded_module("builtins")
    return family_isinstance(value, Python.tuple(b.int, b.float)) and not family_isinstance(value, b.bool)


def family_type_name(value: PythonObject) raises -> String:
    """`type(value).__name__`."""
    return String(py_type(value).__name__)


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


def family_shape_repr(view: PythonObject) raises -> String:
    """`repr(tuple(a.shape))`."""
    var b = native_loaded_module("builtins")
    return family_repr(py_call(b.tuple, view.shape))


def family_view(source: PythonObject, ndim: Int, name: String, context: PythonObject) raises -> PythonObject:
    """The admitted input with `_buffer._as_typed`'s rank and emptiness
    refusals, in its wording; `ndim` < 0 accepts any rank."""
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    if ndim >= 0 and nd != ndim:
        var hint = String("")
        if ndim == 2 and nd < 2:
            hint = ". " + String(RESHAPE_HINT)
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(nd)
                    + "-D shape " + family_shape_repr(view) + hint)
    if Int(py=view.size) == 0:
        var what = String("sample(s)")
        if nd == 2 and Int(py=view.shape[0]) != 0:
            what = "feature(s)"
        var shape = family_shape_repr(view)
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return view


# TOMBSTONE: family_f32, family_cast_f32, family_input_snapshot, family_snapshot_f32, family_f32_snapshot, family_as_f32, family_as_f32_dense, family_cast_view, family_cast_view_f64, family_owned, family_owned_snapshot, family_sample_weight, family_sample_weight_snapshot, _family_weight_scalar, _family_weight_buffer (family admission helpers, moved onto the one admission path) deleted 2026-11-08 by lane a2-admit-family; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A2_FAMILY_ADMISSION_HELPERS.patch. They are
# admit_family / admit_family_meta / admit_family_f64 / admit_family_owned /
# admit_family_weights in bindings/admitted_family.mojo.


def family_snapshot_min(value: ArrayMetadata) raises -> Float32:
    return family_min_f32(FAMILY_F32P(unsafe_from_address=value.source.addr), value.count)


def family_min(value: PythonObject) raises -> Float32:
    """The minimum of a non-empty C-order float32 array."""
    return family_min_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), Int(py=value.size))


def family_nonfinite(error: Error, name: String) -> Bool:
    """True for the cast's NaN-or-infinity refusal of `name`."""
    return String(error) == "Input " + name + " contains NaN or infinity"


def family_field(fields: PythonObject, name: String) raises -> PythonObject:
    """`fields[name]`, refused as the attribute lookup refused it."""
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is None and not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value
