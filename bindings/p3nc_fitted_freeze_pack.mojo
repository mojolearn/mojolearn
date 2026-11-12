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
# The PACK half of the former `bindings/p3nc_fitted_freeze.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/p3nc_fitted_freeze_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_setattr, py_get
from bindings.native_loaded_module_door import native_loaded_module, native_namespace_string_keys
from bindings.native_class_admission_door import dict_keys_admitted, dict_keys_admit
from bindings.native_array_storage_admission_door import array_storage_admit, array_storage_context
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.py_cpython_table import cpython_table


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
