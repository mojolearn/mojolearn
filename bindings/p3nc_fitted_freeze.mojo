# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fitted arrays behind a resident device copy are read-only
(p3-neighbors-cluster, 2026-11-06; the orchestrator's hazard-class-A scheme).

A resident copy (a device upload, a prepared handle) of a fitted array is
keyed by the array's identity, address and shape, never its values. So the
fitted array the model exposes is made read-only at the buffer level when the
resident copy is taken: an exact mojolearn `Array`'s own read-only flag, which
NumPy honours through `__array_interface__` (a view of it is not writeable and
cannot be made writeable), so `np.asarray(model.attr_)[:] = ...` raises NumPy's
normal read-only error instead of silently diverging from the device copy.
Replacing the attribute (`model.attr_ = new`) is still allowed; the new object
changes the key and is uploaded again. Any other storage (a caller-assigned
ndarray) cannot be frozen this way, so callers keep the per-call upload for it.
No content hash anywhere. Not noticed: writes through a NumPy view that was
taken BEFORE the first resident copy (that view keeps its writeable flag)."""
from std.python import PythonObject
from bindings.py_balanced import py_type, py_setattr, py_get
from bindings.native_loaded_module import native_loaded_module


def fitted_is_freezable(value: PythonObject, context: PythonObject) raises -> Bool:
    """True for an exact mojolearn Array (the only storage this module can
    make read-only for good)."""
    return py_type(value) is context["array_class"]


def fitted_freeze(value: PythonObject, context: PythonObject) raises -> Bool:
    """Mark an exact mojolearn Array read-only; False (nothing done) for any
    other storage."""
    if py_type(value) is not context["array_class"]:
        return False
    py_setattr(value, "_readonly", PythonObject(True))
    return True


def _array_class() raises -> PythonObject:
    var module = py_get(native_loaded_module("sys").modules, "mojolearn._array")
    if module is None:
        return PythonObject(None)
    return module.Array


def fitted_is_freezable_any(value: PythonObject) raises -> Bool:
    """`fitted_is_freezable` where no connected context is at hand: the Array
    class is read from the loaded `mojolearn._array` module."""
    var cls = _array_class()
    return cls is not None and py_type(value) is cls


def fitted_freeze_any(value: PythonObject) raises -> Bool:
    """`fitted_freeze` without a connected context."""
    if not fitted_is_freezable_any(value):
        return False
    py_setattr(value, "_readonly", PythonObject(True))
    return True
