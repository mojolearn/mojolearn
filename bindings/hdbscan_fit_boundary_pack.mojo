# SPDX-License-Identifier: Apache-2.0
"""HDBSCAN fit PACK and the fit's one operator (UNPACK -> RUN -> PACK):
public output allocation, fitted-attribute publication, the prediction-data
mapping and the head copy of native scratch into a public array. UNPACK is
bindings/hdbscan_fit_boundary_unpack.mojo; RUN is the typed driver call of
core/hdbscan_fit_api.mojo, entered with the GIL released, and the number
refusals of providers/python/hdbscan_public_lifecycle_run.mojo."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.memory import unsafe_memcpy
from core.hdbscan_fit_api import HdbscanFitPlan, HdbscanPredictionRequest, HdbscanPredictionPlan
from core.kmeans_variants_api import VariantsSpan
from bindings.admitted_family_door import admit_family
from bindings.family_common_unpack import FAMILY_I32P, FAMILY_F32P
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address
from bindings.py_balanced import py_setattr, py_setitem
from bindings.hdbscan_fit_boundary_unpack import (
    hdbscan_fit_connection, hdbscan_buffer_span, hdbscan_fit_settings, hdbscan_fit_stage,
    hdbscan_fit_options, _hdbscan_mode,
)
from providers.python.hdbscan_public_lifecycle_run import hdbscan_fit_rows, hdbscan_fit_request


def hdbscan_pack_head(address: Int, count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    # Same allocation and 4-byte word copy as the replaced Array head helper.
    # The caller pins the native scratch owner until this synchronous copy ends.
    var result = family_array(Python.tuple(count), typestr, context)
    if count:
        unsafe_memcpy(dest=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(result))),
            src=FAMILY_I32P(unsafe_from_address=address), count=count)
        _ = result
    return result


def _set(model: PythonObject, var name: String, value: PythonObject) raises:
    """`setattr(model, name, value)`."""
    py_setattr(model, name^, value)


def hdbscan_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`HDBSCAN.fit(X)` on the given image; returns the new `labels_`.

    Same refusals in the same order, the same allocations and address reads
    at the same stages, the same driver call and the same published
    attributes as the body that lived in bindings/cluster_fit_a_native_door.mojo.
    The prediction data takes its counts and labels from this fit's typed
    result (it re-read three attributes it had just stored, by name)."""
    # UNPACK. The one admission path: the rank and emptiness refusals, then
    # the borrow or the cast.
    var admitted = admit_family(source, 2, "X", context)
    var x = admitted.array
    # `as_f32_c`'s `copied`: the cast made new storage.
    var copied = not admitted.borrowed
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    hdbscan_fit_rows(n, d)
    var settings = hdbscan_fit_settings(model, context)
    var want_pd = settings.prediction_data
    var labels = family_array(Python.tuple(n), "<i4", context)
    var core = family_array(Python.tuple(n), "<f4", context)
    var plan = HdbscanFitPlan(want_pd)
    # Preserve the original initial address admission before probabilities
    # allocation. Private info/tree addresses come from Mojo owners.
    var source_span = VariantsSpan(Int(py=classical_address(x)), n * d)
    var label_span = VariantsSpan(Int(py=classical_address(labels)), n)
    var core_span = VariantsSpan(Int(py=classical_address(core)), n)
    var probs = family_array(Python.tuple(n), "<f4", context)
    if want_pd:
        plan.storage[0].prepare_tree(n)
    var probability_span = VariantsSpan(Int(py=classical_address(probs)), n)
    _hdbscan_mode(model, binding, context)
    hdbscan_fit_stage(settings, model)
    var connection = hdbscan_fit_connection(binding)
    var options = hdbscan_fit_options(settings, n, d)
    var request = hdbscan_fit_request(source_span, label_span, core_span, probability_span, options)
    if connection.host:
        plan.capture_host_source(request)
    # RUN.
    var n_clusters = 0
    try:
        with GILReleased(Python()):
            n_clusters = connection.driver.fit(request, plan)
    finally:
        _ = connection.provider
        _ = x
        _ = labels
        _ = core
        _ = probs
        _ = plan
    plan.storage[0].host_source = List[Float32]()
    # PACK.
    var n_condensed = Int(plan.storage[0].info[3])
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "labels_", labels)
    _set(model, "core_distances_", core)
    _set(model, "probabilities_", probs)
    _set(model, "n_clusters_", PythonObject(n_clusters))
    _set(model, "n_outliers_", PythonObject(Int(plan.storage[0].info[1])))
    _set(model, "n_boruvka_rounds_", PythonObject(Int(plan.storage[0].info[2])))
    _set(model, "n_condensed_clusters_", PythonObject(n_condensed))
    _set(model, "_prediction_data", PythonObject(None))
    _set(model, "_raw_data", PythonObject(None))
    if want_pd:
        _set(model, "_prediction_data", _hdbscan_prediction_data(labels, binding, n,
            Int(plan.storage[0].info[4]), n_condensed, n_clusters, plan, context))
        # Keep the original borrowed-input copy and publication stage.
        var raw = x
        if not copied:
            raw = family_array(Python.tuple(n, d), "<f4", context)
            unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(raw))),
                src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=n * d)
            _ = raw
            _ = x
        _set(model, "_raw_data", raw)
    _ = x
    _ = plan
    return labels


def _hdbscan_prediction_data(labels: PythonObject, binding: PythonObject, n: Int, n_edges: Int,
    n_cond: Int, n_sel: Int, fit_plan: HdbscanFitPlan, context: PythonObject) raises -> PythonObject:
    """Same prediction-data algorithm and observable mapping; native scratch."""
    var pd = Python.dict()
    py_setitem(pd, PythonObject("n_edges"), PythonObject(n_edges))
    py_setitem(pd, PythonObject("n_condensed"), PythonObject(n_cond))
    var parents = hdbscan_pack_head(Int(fit_plan.storage[0].tree_parents.unsafe_ptr()), n_edges, "<i4", context)
    var children = hdbscan_pack_head(Int(fit_plan.storage[0].tree_children.unsafe_ptr()), n_edges, "<i4", context)
    var lambdas = hdbscan_pack_head(Int(fit_plan.storage[0].tree_lambdas.unsafe_ptr()), n_edges, "<f4", context)
    var sizes = hdbscan_pack_head(Int(fit_plan.storage[0].tree_sizes.unsafe_ptr()), n_edges, "<i4", context)
    var inverse = hdbscan_pack_head(Int(fit_plan.storage[0].inverse.unsafe_ptr()), max(n_sel, 1), "<i4", context)
    var deaths = family_array(Python.tuple(n_cond), "<f4", context)
    var selected = family_array(Python.tuple(max(n_sel, 1)), "<i4", context)
    var plan = HdbscanPredictionPlan(n)
    var offsets = family_array(Python.tuple(n_sel + 1), "<i4", context)
    var index_into = family_array(Python.tuple(n_edges + 1), "<i4", context)
    py_setitem(pd, PythonObject("parents"), parents)
    py_setitem(pd, PythonObject("children"), children)
    py_setitem(pd, PythonObject("lambdas"), lambdas)
    py_setitem(pd, PythonObject("sizes"), sizes)
    py_setitem(pd, PythonObject("inverse_label_map"), inverse)
    py_setitem(pd, PythonObject("deaths"), deaths)
    py_setitem(pd, PythonObject("selected_clusters"), selected)
    # The mapping is still private until return. Preserve its original key
    # insertion order without constructing the replaced full-capacity Array.
    py_setitem(pd, PythonObject("exemplar_idx"), PythonObject(None))
    py_setitem(pd, PythonObject("exemplar_label_offsets"), offsets)
    py_setitem(pd, PythonObject("index_into_children"), index_into)
    var connection = hdbscan_fit_connection(binding)
    var request = HdbscanPredictionRequest(hdbscan_buffer_span(labels, n), hdbscan_buffer_span(parents, n_edges),
        hdbscan_buffer_span(children, n_edges), hdbscan_buffer_span(lambdas, n_edges), hdbscan_buffer_span(sizes, n_edges), hdbscan_buffer_span(inverse, n_sel),
        hdbscan_buffer_span(deaths, n_cond), hdbscan_buffer_span(selected, n_sel), hdbscan_buffer_span(offsets, n_sel + 1), hdbscan_buffer_span(index_into, n_edges + 1),
        n, n_edges, n_cond, n_sel)
    plan.storage[0].capture(request)
    var n_exemplars = 0
    try:
        with GILReleased(Python()):
            n_exemplars = connection.driver.prediction(request, plan)
    finally:
        _ = connection.provider
        _ = labels
        _ = parents
        _ = children
        _ = lambdas
        _ = sizes
        _ = inverse
        _ = deaths
        _ = selected
        _ = offsets
        _ = index_into
        _ = plan
        _ = fit_plan
    plan.storage[0].release_inputs()
    py_setitem(pd, PythonObject("n_exemplars"), PythonObject(n_exemplars))
    py_setitem(pd, PythonObject("exemplar_idx"),
        hdbscan_pack_head(Int(plan.storage[0].exemplars.unsafe_ptr()), max(n_exemplars, 1), "<i4", context))
    _ = plan
    _ = labels
    return pd
