# SPDX-License-Identifier: Apache-2.0
# Shared door helpers of the classical family owners (bindings/family_native.mojo,
# lane migrate-classical-b): the `_buffer.as_f32_c` and
# `_sample_weight.check_sample_weight` refusals with their wording, mojolearn
# Array outputs, and calls into the compiled entries of the binding a public
# shell hands over. A Python exception a callee raises (a kernel's refusal,
# a CPU-only install's by-name refusal, a user callable's own exception) is
# stashed and restored unchanged at the door (`stash_python_error`), so the
# caller sees the class and message it saw before the migration.
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


def family_shape_repr(view: PythonObject) raises -> String:
    """`repr(tuple(a.shape))`."""
    var b = native_loaded_module("builtins")
    return family_repr(py_call(b.tuple, view.shape))


# TOMBSTONE: family_view (the rank / emptiness refusals on a view before a second admission) deleted 2026-11-09 by lane a6-device-x-family-classical; code recoverable at f59d913b0b.
# Restore: experiments/removed/A6_FAMILY_VIEW.patch. Its callers read
# admit_family / admit_family_input (+ admit_family_cast) or admit_typed, which
# refuse with the same wording on the one metadata capture.


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
