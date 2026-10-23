# SPDX-License-Identifier: Apache-2.0
# The resident reference rows of DBSCAN.predict and
# AgglomerativeClustering.predict (lane mcb2-cfb): the fitted rows, their tie
# keys and their labels are uploaded once through the estimators binding's
# `labeled_reference_prepare` and kept under a handle in the model's
# `_resident_holder` dict as `(key, handle, binding)` (the shape the native
# finalizer of bindings/host_archive_lifetime.mojo releases); predict then
# uploads only its queries (`labeled_reference_predict_resident`). A binding
# without the resident entries (the CPU-only host binding) keeps the one-shot
# `labeled_reference_predict`. Same kernel, same operands: the bits do not
# move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call, family_entry, family_array
from bindings.cluster_native import _labeled_reference


def cfb_optional_entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The compiled entry `name` of `binding`, or None where the binding
    does not export it (an absent name is ImportError on a host binding
    proxy and AttributeError on a module; both mean "no residency")."""
    var b = Python.import_module("builtins")
    var entry = PythonObject(None)
    try:
        entry = py_call(b.getattr, binding, PythonObject(name))
    except:
        return PythonObject(None)
    if py_type(entry) is not context["family_builtin_type"]:
        return PythonObject(None)
    return entry


def cfb_release(model: PythonObject) raises:
    """Drop the model's resident reference set, if one is held, through the
    minting binding's `labeled_reference_release`. Quiet on a binding that
    cannot be reached any more and on a handle already released."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        return
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    try:
        _ = py_call(py_call(b.getattr, held[2], PythonObject("labeled_reference_release")), held[1])
    except:
        pass


def cfb_fresh_holder(model: PythonObject) raises:
    """After a fit: release the last fit's resident set and give the
    instance an empty holder (the holder's contents change at predict, never
    `__dict__`, which scikit-learn's check_dict_unchanged compares)."""
    cfb_release(model)
    py_setitem(py_fields(model), "_resident_holder", Python.dict())


def cfb_handle(model: PythonObject, binding: PythonObject, refs: PythonObject, keys: PythonObject,
               labels: PythonObject, n_refs: Int, n_features: Int, metric: Int,
               context: PythonObject) raises -> Int:
    """The resident handle of (refs, keys, labels, metric) on `binding`,
    prepared now when none is held for exactly these arrays; -1 where the
    binding has no resident entries."""
    var b = Python.import_module("builtins")
    var prepare = cfb_optional_entry(binding, "labeled_reference_prepare", context)
    if prepare is None or cfb_optional_entry(binding, "labeled_reference_predict_resident", context) is None:
        return -1
    var fields = py_fields(model)
    var holder = py_get(fields, PythonObject("_resident_holder"), PythonObject(None))
    if holder is None or py_type(holder) is not b.dict:
        holder = Python.dict()
        py_setitem(fields, "_resident_holder", holder)
    var held = py_get(holder, PythonObject("index"), PythonObject(None))
    if held is not None and py_type(held) is b.tuple and len(held) == 3 and held[2] is binding:
        var key = held[0]
        if (key[0] is refs and key[1] is keys and key[2] is labels and Int(py=key[3]) == n_refs
                and Int(py=key[4]) == n_features and Int(py=key[5]) == metric):
            return Int(py=held[1])
    cfb_release(model)
    var handle = family_call(prepare, Python.list(classical_address(refs), classical_address(keys), classical_address(labels)),
                             Python.list(PythonObject(n_refs), PythonObject(n_features), PythonObject(metric)))
    # The minting binding travels with the handle: the release (refit, the
    # native finalizer when the instance is dropped) is its own entry.
    var key = Python.tuple(refs, keys, labels, PythonObject(n_refs), PythonObject(n_features), PythonObject(metric))
    py_setitem(holder, "index", Python.tuple(key, handle, binding))
    return Int(py=handle)


def cfb_mint(model: PythonObject, binding: PythonObject, refs: PythonObject, keys: PythonObject,
             labels: PythonObject, n_refs: Int, n_features: Int, metric: Int, context: PythonObject) raises:
    """At fit: the reference set made resident for the predicts to come. A
    failure here (no device memory for the copy) leaves no handle; the
    first predict then prepares it and reports any refusal itself."""
    if n_refs < 1:
        return
    try:
        _ = cfb_handle(model, binding, refs, keys, labels, n_refs, n_features, metric, context)
    except:
        cfb_release(model)


def cfb_reference_labels(model: PythonObject, binding: PythonObject, refs: PythonObject, keys: PythonObject,
                         labels: PythonObject, q: PythonObject, n_refs: Int, n_features: Int, metric: Int,
                         eps: PythonObject, has_thresh: Int, context: PythonObject) raises -> PythonObject:
    """The label of each query's nearest reference row: on the resident set
    where the binding has one, else the one-shot `labeled_reference_predict`."""
    var handle = cfb_handle(model, binding, refs, keys, labels, n_refs, n_features, metric, context)
    if handle < 0:
        return _labeled_reference(binding, refs, keys, labels, q, n_refs, n_features, metric, eps, has_thresh, context)
    var nq = Int(py=q.shape[0])
    var out = family_array(Python.tuple(nq), "<i4", context)
    var entry = family_entry(binding, "labeled_reference_predict_resident", context)
    # ORDER MATCHES bindings/labeled_reference_resident_binding.mojo::
    # labeled_reference_predict_resident_binding.
    _ = family_call(entry, PythonObject(handle), Python.list(classical_address(q), classical_address(out)),
                    Python.list(PythonObject(nq), PythonObject(n_features), PythonObject(metric), eps,
                                PythonObject(has_thresh)))
    return out
