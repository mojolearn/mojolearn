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


def fitted_freeze_foreign(value: PythonObject) raises -> Bool:
    """The read-only rule for a foreign buffer read in place (no copy) by the
    kernels: an exact `numpy.ndarray` gets its writeable flag cleared (NumPy
    then refuses writes through it and through views taken from it later);
    True only when the flag reads back cleared. An exact mojolearn Array is
    frozen as in `fitted_freeze_any`. Any other exporter (array.array,
    memoryview, bytearray, an ndarray subclass) cannot be frozen: False,
    and the caller keeps its per-call copy. Same caveat as above: a view
    taken before the freeze keeps its own writeable flag."""
    if fitted_freeze_any(value):
        return True
    var numpy = py_get(native_loaded_module("sys").modules, "numpy")
    if numpy is None or py_type(value) is not numpy.ndarray:
        return False
    var flags = value.flags
    try:
        py_setattr(flags, "writeable", PythonObject(False))
    except:
        return False
    return not Bool(py=flags.writeable)


def fitted_freeze_sole_owner(value: PythonObject) raises -> Bool:
    """The read-only rule for a caller's training matrix that a fit keeps
    in place instead of copying (lane fix-c3-c5-no-copies, 0.8.66 regression
    C3: 0.8.55 kept such a matrix as is). True, with the buffer frozen as
    `fitted_freeze_foreign` freezes it, only when `value` owns its memory:
    an exact mojolearn Array with no `_base` and no `_pin` (not a view of
    another Array or of a foreign buffer), or an exact `numpy.ndarray` whose
    `base` is None. A view could still be written through its writable
    owner, so it is never frozen: False, and the caller keeps its copy.
    Same caveat as the module note: a view taken from `value` before the
    freeze keeps its own writeable flag."""
    var array_cls = _array_class()
    if array_cls is not None and py_type(value) is array_cls:
        if value._base is not None or value._pin is not None:
            return False
        return fitted_freeze_any(value)
    var numpy = py_get(native_loaded_module("sys").modules, "numpy")
    if numpy is None or py_type(value) is not numpy.ndarray:
        return False
    if value.base is not None:
        return False
    return fitted_freeze_foreign(value)
