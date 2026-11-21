# SPDX-License-Identifier: Apache-2.0
# Shared pieces of the decomposition expansion's native owners
# (bindings/decomp_native.mojo and the per-class files it dispatches to):
# instance fields with Python's AttributeError, mojolearn Array outputs and
# scalar reads. (The estimator's x_decomp image is resolved by
# bindings/decomp_kit_native_door.mojo `decomp_entries_routed`.)
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type, py_contains, py_setitem
from bindings.classical_arrays_door import classical_typed

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def decomp_attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` for an instance field, with Python's AttributeError."""
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def decomp_int(value: PythonObject) raises -> Int:
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def decomp_float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def decomp_lt(a: PythonObject, b: PythonObject) raises -> Bool:
    """`a < b` with Python's comparison."""
    return Bool(py=py_call(native_loaded_module("operator").lt, a, b))


def decomp_le(a: PythonObject, b: PythonObject) raises -> Bool:
    return Bool(py=py_call(native_loaded_module("operator").le, a, b))


def decomp_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    return Bool(py=py_call(native_loaded_module("operator").eq, a, b))


def decomp_repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def decomp_set_mode(fields: PythonObject, context: PythonObject) raises:
    # `self.numeric_mode_ = _mode(self.numeric_mode)`: the process mode when
    # None, else the value as given.
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    py_setitem(fields, "numeric_mode_", mode)


def decomp_f32(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """A zero-filled C-order float32 mojolearn Array (`_M.out`)."""
    return classical_typed(shape, "<f4", context["output_context"])


def decomp_check(model: PythonObject, fields: PythonObject, attr: String) raises:
    """`_Base._check`: RuntimeError until `attr` is fitted."""
    if not py_contains(fields, attr):
        raise Error("RuntimeError: " + String(py_type(model).__name__) + " is not fitted; call fit first")


def decomp_seed(random_state: PythonObject) raises -> Int:
    """`_seed_of`: None is 0; a bool or a non-int refused."""
    var b = native_loaded_module("builtins")
    if random_state is None:
        return 0
    if py_type(random_state) is b.bool or not Bool(py=py_call(b.isinstance, random_state, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    # mod 2^32 with Python integer semantics: every consumer masks it so
    return Int(py=py_call(b.int.__dict__["__and__"], random_state, PythonObject(0xFFFFFFFF)))
