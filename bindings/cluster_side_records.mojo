# SPDX-License-Identifier: Apache-2.0
# Native records of the cluster family doors (lane perf2-cluster,
# 2026-11-06): KMeans and the other family-lifecycle estimators,
# GaussianMixture, HDBSCAN, DBSCAN and KernelDensity.
#
# Their public queries (predict, transform, score_samples, approximate_predict,
# membership_vector, ...) used to redo, on every call, work whose result
# depends only on objects the model or its class already holds:
#
# - the class-ancestry walk refusing a `_mojolearn_after_fit` hook (one
#   `type.__dict__` read and key admission per ancestor);
# - the connected-class check (an MRO walk comparing every ancestor);
# - the admission of every constructor parameter and fitted field (each a
#   full value admission with its type-vector construction);
# - views of the fitted arrays a query hands to its kernel.
#
# Class facts are kept per class state (bindings/native_class_admission.mojo:
# a type watcher invalidates them on any modification of the class or an
# ancestor). Model facts are kept per model in the CLUSTER slot of the one
# record table (bindings/native_model_records.mojo, lane ne1-e-records):
# keyed by a weak reference to the model, outside its `__dict__`, so a query
# leaves `__dict__` unchanged, pickles and copies carry no record, and a
# record dies with its model. A model record is valid while its key matches:
# the model's class and the identities of every field the cached work read
# (exact containers item by item, with their lengths), so `set_params`, a
# refit, an assignment or an in-place edit of a stored list, tuple or dict
# changes the key and the full admission runs again, with its refusals in
# their original order. A record is written only after the full path
# succeeded, and never holds its own model.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.mode_metadata_native import _class_fields
from bindings.classical_attributes import _type_mro
from bindings.py_balanced import py_call, py_type, py_contains, py_get
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_address
from bindings.trees_side_records import trees_key_same, trees_key_shallow
from bindings.native_model_records import records_get, records_put, records_drop, records_key_append, RECORDS_CLUSTER

# PyType_IsSubtype comes from the one CPython table (ClusterCPython,
# MojolearnClusterCPythonV1, folded into it by lane ne1-e-table, 2026-11-06).

# Py_TPFLAGS_TYPE_SUBCLASS: set on `type` and every metaclass.
comptime _TYPE_SUBCLASS_FLAG = UInt64(1) << 31


def cluster_is_subclass(cls: PythonObject, owner: PythonObject) raises -> Bool:
    """`owner` is in `cls.__mro__` (`cls` a class): CPython's own MRO test,
    with no Python-level tuple walk. An `owner` that is not a class is
    compared against the MRO entries one by one, as before."""
    var owner_type = py_type(owner)
    if (cpython_table()[].PyType_GetFlags(owner_type._obj_ptr) & _TYPE_SUBCLASS_FLAG) == UInt64(0):
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: over class ancestry (an owner that is not a class)
            if ancestry[i] is owner:
                return True
        return False
    return Int(cpython_table()[].PyType_IsSubtype(cls._obj_ptr, owner._obj_ptr)) != 0


comptime _AFTER_FIT_SLOT = "cluster_after_fit_free_v1"


def cluster_refuse_fit_callbacks(cls: PythonObject, message: String) raises:
    """Raise `message` when `cls` or an ancestor defines `_mojolearn_after_fit`.

    The ancestry walk (one class namespace read per ancestor) ran on every
    public call. A class that passed it is recorded per class state: any
    modification of the class or of an ancestor (attribute assignment or
    deletion, `__bases__`) invalidates the record and the walk runs again.
    Where type watchers are unavailable (Python 3.10, 3.11) the walk runs on
    every call, as before."""
    var mods = NativeModules()
    if class_record(mods, _AFTER_FIT_SLOT, cls) is not None:
        return
    var ancestry = _type_mro(cls)
    for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from a native call
        if py_contains(_class_fields(mods, ancestry[i]), "_mojolearn_after_fit"):
            raise Error(message)
    class_admit(mods, _AFTER_FIT_SLOT, cls, PythonObject(True))


def cluster_record_key(model: PythonObject, fields: PythonObject, required: PythonObject,
                       optional: PythonObject) raises -> PythonObject:
    """The identity key of a model record: the model's class, then for each
    name in `required` (all must be stored in `fields`) and `optional` (a
    presence word, then the value when stored) the name and the stored value,
    exact lists, tuples and dicts item by item to depth 4, any other
    non-scalar with its class. None when a required name is not stored in
    the instance dictionary (its value then comes from a class or a default
    the key does not cover): such a model is never recorded."""
    var mods = NativeModules()
    var key = Python.list(py_type(model))
    for i in range(len(required)):  # driver-loop: over connected constructor parameter names (identity key)
        var name = required[i]
        if not py_contains(fields, name):
            return PythonObject(None)
        records_key_append(key, name)
        trees_key_shallow(mods, key, fields[name], 0)
    for i in range(len(optional)):  # driver-loop: over fitted field names (identity key)
        var name = optional[i]
        records_key_append(key, name)
        if py_contains(fields, name):
            records_key_append(key, PythonObject(1))
            trees_key_shallow(mods, key, fields[name], 0)
            _cluster_array_words(key, fields[name])
        else:
            records_key_append(key, PythonObject(0))
    return key


def _cluster_array_words(key: PythonObject, value: PythonObject) raises:
    """p3-neighbors-cluster (hazard class C): for a fitted ndarray or
    mojolearn Array, its data address, rank and dimensions (exact ints), and
    for an ndarray its dtype object and C-contiguity flag. An in-place
    `arr.shape = ...`, `arr.dtype = ...`, `arr.resize(...)` or strides change
    keeps the object but changes these words, so the record misses and the
    full admission runs. Values are not part of the key: the records cache
    admission only, and the operands are read from the arrays every call."""
    var modules = native_loaded_module("sys").modules
    var cls = py_type(value)
    var arrays = py_get(modules, "mojolearn._array")
    var numpy = py_get(modules, "numpy")
    var is_ndarray = numpy is not None and cls is numpy.ndarray
    if not is_ndarray and (arrays is None or cls is not arrays.Array):
        return
    records_key_append(key, classical_address(value))
    var shape = value.shape
    records_key_append(key, PythonObject(len(shape)))
    for axis in range(len(shape)):  # small-loop(fitted array rank)
        records_key_append(key, PythonObject(Int(py=shape[axis])))
    if is_ndarray:
        records_key_append(key, value.dtype)
        records_key_append(key, value.flags.c_contiguous)


def cluster_record_get(slot: String, model: PythonObject, key: PythonObject) raises -> PythonObject:
    """The payload `slot` holds for `model` under an identical key, or None."""
    if key is None:
        return PythonObject(None)
    var held = records_get(RECORDS_CLUSTER, model, slot)
    if held is None or not trees_key_same(held[0], key):
        return PythonObject(None)
    return held[1]


def cluster_record_put(slot: String, model: PythonObject, key: PythonObject, payload: PythonObject) raises:
    """Keep `payload` (never None) for `model` under `key`, replacing any
    older record in `slot`. No-op without a key, and for a model that takes
    no weak reference (it then runs the full path on every call)."""
    if key is None:
        return
    _ = records_put(RECORDS_CLUSTER, model, slot, Python.tuple(key, payload))


def cluster_record_drop(slot: String, model: PythonObject) raises:
    """Forget `model`'s record in `slot` (a fit: the fitted arrays a record
    holds are released now, not at the next query)."""
    records_drop(RECORDS_CLUSTER, model, slot)
