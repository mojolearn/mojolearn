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
# The PACK half of the former `bindings/cluster_side_records.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/cluster_side_records_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.mode_metadata_native_door import _class_fields
from bindings.classical_attributes_door import _type_mro
from bindings.py_balanced import py_call, py_type, py_contains, py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_address
from bindings.trees_side_records_unpack import trees_key_same, trees_key_shallow
from bindings.native_model_records import records_get, records_put, records_drop, records_key_append, RECORDS_CLUSTER

# PyType_IsSubtype comes from the one CPython table (ClusterCPython,
# MojolearnClusterCPythonV1, folded into it by lane ne1-e-table, 2026-11-06).

from bindings.cluster_side_records_unpack import (
    _cluster_array_words,
)


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


def cluster_record_get(slot: String, model: PythonObject, key: PythonObject) raises -> PythonObject:
    """The payload `slot` holds for `model` under an identical key, or None."""
    if key is None:
        return PythonObject(None)
    var held = records_get(RECORDS_CLUSTER, model, slot)
    if held is None or not trees_key_same(held[0], key):
        return PythonObject(None)
    return held[1]
