# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public door's half of the device-resident fitted state (lane
ne1-a-gp-svm, workstream A): one helper for every family whose binding keeps
fitted arrays in a `core/fitted_resident_store.mojo` store (the Gaussian
processes, the kernel SVMs).

`fitted_resident_lease(model, slot, binding, open_entry, mirrors)` returns the
lease the binding's resident predict entries take, or 0 (the entry then
uploads the model for that call only, as before). The lease is taken once
per fitted state and kept in the model's prepared-owner record
(bindings/native_prepared_owner.mojo): a native registry keyed by a weak
reference to the model, outside its `__dict__`, so pickles and copies never
carry it and the model's death drops it. The record holds:

    (owner, lease, binding, generation, mirrors)

- `owner` is the binding's native lease object; dropping the record drops
  it, and that frees the device copies (no Python code runs).
- `generation` is the model's generation word
  (bindings/native_model_records.mojo `model_generation`): `fit`,
  `set_params`, `__setstate__` and the family's setters bump it, so a record
  of an older state is never reused.
- `mirrors` are the fitted array OBJECTS the copies mirror. The record keeps
  them alive (so their memory cannot be reused under the same address) and a
  replaced attribute fails the identity check: a new lease, a new upload.

THE READ-ONLY RULE (bindings/p3nc_fitted_freeze.mojo, decided 2026-11-06):
a lease is taken only when every mirrored array is an exact mojolearn Array
owning its store; each is then made read-only, so an in-place write raises
NumPy's read-only error instead of leaving the device copy stale. Any other
storage keeps the per-call upload (lease 0). No content hash anywhere.

Nothing here computes on data.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules, native_loaded_module
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.native_model_records import model_generation
from bindings.p3nc_fitted_freeze import fitted_is_freezable_any, fitted_freeze_any
from bindings.py_balanced import py_call, py_type
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry


def _owns_store(value: PythonObject) raises -> Bool:
    """An exact mojolearn Array that owns its store (a view's base could
    still be written, so a view is never frozen in its place)."""
    return fitted_is_freezable_any(value) and value._base is None


def _record_matches(record: PythonObject, binding: PythonObject, generation: Int,
                    mirrors: PythonObject) raises -> Int:
    """The record's lease when it was taken for exactly this binding, this
    generation and these array objects; else 0. Never raises on a record of
    another shape (the caller replaces it)."""
    var b = native_loaded_module("builtins")
    if py_type(record) is not b.tuple or len(record) != 5:
        return 0
    if record[2] is not binding or py_type(record[1]) is not b.int or py_type(record[3]) is not b.int:
        return 0
    if Int(py=record[3]) != generation:
        return 0
    var held = record[4]
    if py_type(held) is not b.tuple or len(held) != len(mirrors):
        return 0
    for i in range(len(mirrors)):  # small-loop(the model's mirrored fitted arrays: a handful)
        if held[i] is not mirrors[i]:
            return 0
    return Int(py=record[1])


def fitted_resident_lease(model: PythonObject, slot: String, binding: PythonObject, open_entry: String,
                          mirrors: PythonObject) raises -> Int:
    """The lease of `model`'s device copies of `mirrors` (a Python list of
    the fitted array objects, in the binding's slot order) in `binding`'s
    store, taken through `binding.<open_entry>()` when none is held for this
    fitted state; 0 when the binding has no such entry or an array cannot
    be made read-only (the predict then uploads per call)."""
    var mods = NativeModules()
    var generation = model_generation(model)
    var record = prepared_get(mods, slot, model)
    if record is not None:
        var held = _record_matches(record, binding, generation, mirrors)
        if held != 0:
            return held
    var opener = neural_descriptor_entry(binding, open_entry, False)
    if opener is None:
        if record is not None:
            prepared_drop(mods, slot, model)
        return 0
    for i in range(len(mirrors)):  # small-loop(the model's mirrored fitted arrays: a handful)
        if not _owns_store(mirrors[i]):
            if record is not None:
                prepared_drop(mods, slot, model)
            return 0
    for i in range(len(mirrors)):  # small-loop(the model's mirrored fitted arrays: a handful)
        _ = fitted_freeze_any(mirrors[i])
    var b = native_loaded_module("builtins")
    var opened = py_call(opener)
    var lease = Int(py=opened[1])
    var kept = Python.tuple(opened[0], opened[1], binding, PythonObject(generation), py_call(b.tuple, mirrors))
    # Replacing the record drops the previous lease object (its copies go).
    # Where the model takes no weak reference the record is not kept: the
    # lease object dies with this call's locals and the predict entry falls
    # back to per-call buffers for a lease that is not live.
    prepared_put(mods, slot, model, kept)
    _ = opened^
    return lease


def fitted_resident_drop(model: PythonObject, slot: String) raises:
    """Drop `model`'s record in `slot` (and so its device copies), if any:
    the fit doors call it before writing new fitted arrays."""
    var mods = NativeModules()
    prepared_drop(mods, slot, model)
