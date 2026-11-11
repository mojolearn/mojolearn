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
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_setattr, py_get
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.native_array_storage_admission import array_storage_admit, array_storage_context
from bindings.native_owner_lifetime import native_release_admit
from bindings.cpython_table import cpython_table


def _fitted_class(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(context) is not b.dict:
        raise Error("TypeError: fitted Array context requires an exact dictionary")
    if not dict_keys_admitted(context):
        if not native_namespace_string_keys(context._obj_ptr, b.str):
            raise Error("TypeError: fitted Array context requires native string keys")
        dict_keys_admit(context)
    return context["array_class"]


def _fitted_admit(value: PythonObject, context: PythonObject) raises -> Bool:
    if py_type(value) is not _fitted_class(context):
        return False
    var storage = array_storage_context()
    _ = array_storage_admit(py_type(value), storage["array_class"], storage["from_buffer_slots"], "fitted Array")
    return True


def _fitted_slot(value: PythonObject, name: String) raises -> PythonObject:
    # Caller admitted the complete retained member-descriptor layout. Generic
    # access therefore reaches only the native slot, never a rebound property.
    var key = PythonObject(name)
    var raw = cpython_table()[].PyObject_GenericGetAttr(value._obj_ptr, key._obj_ptr)
    _ = value
    if not raw:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _fitted_mark(value: PythonObject, context: PythonObject) raises:
    # Replacing an arbitrary previous flag could run its finalizer. Retain and
    # admit that old owner before the native slot setter decrements anything.
    var previous = _fitted_slot(value, "_readonly")
    native_release_admit(previous, context)
    # No GIL release occurs here. Recheck the watched full descriptor layout
    # after old-owner admission, immediately before the generic native setter.
    _ = _fitted_admit(value, context)
    var key = PythonObject("_readonly")
    var flag = PythonObject(True)
    var status = cpython_table()[].PyObject_GenericSetAttr(value._obj_ptr, key._obj_ptr, flag._obj_ptr)
    _ = previous
    _ = value
    if status < 0:
        ref cpy = Python().cpython()
        raise cpy.unsafe_get_error()


def fitted_is_freezable(value: PythonObject, context: PythonObject) raises -> Bool:
    return _fitted_admit(value, context)


def fitted_freeze(value: PythonObject, context: PythonObject) raises -> Bool:
    if not _fitted_admit(value, context):
        return False
    _fitted_mark(value, context)
    return True


def fitted_is_freezable_any(value: PythonObject) raises -> Bool:
    return _fitted_admit(value, array_storage_context())


def fitted_owns_store_any(value: PythonObject) raises -> Bool:
    """Exact connected Array with an admitted native `_base is None` slot."""
    if not fitted_is_freezable_any(value):
        return False
    return _fitted_slot(value, "_base") is None


def fitted_freeze_any(value: PythonObject) raises -> Bool:
    var context = array_storage_context()
    if not _fitted_admit(value, context):
        return False
    _fitted_mark(value, context)
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

