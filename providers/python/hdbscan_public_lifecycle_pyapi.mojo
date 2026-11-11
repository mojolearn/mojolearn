# SPDX-License-Identifier: Apache-2.0
# HDBSCAN's complete public boundary: metadata admission, current-provider
# selection, fit tracking, output publication and soft-membership batching.
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_loaded_module import native_loaded_module, native_namespace_string_keys
from bindings.classical_attributes import classical_connect_numpy
from bindings.cluster_side_records import (
    cluster_is_subclass, cluster_refuse_fit_callbacks, cluster_record_key, cluster_record_get,
    cluster_record_put, cluster_record_drop,
)
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_address, classical_dtype_str, classical_c_order
from bindings.classical_validate import classical_check_x, classical_view, classical_estimator_name
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.native_array_view import native_array_ndarray_view
from bindings.family_raw_attr import family_raw_get, family_raw_get_or
from bindings.family_common import family_array
from bindings.cluster_fit_a_native import cluster_fit_a_operation, _hdbscan_mode
from bindings.hdbscan_fit_boundary import hdbscan_buffer_span
from bindings.hdbscan_query_boundary import (
    HdbscanQueryConnection, HdbscanQueryRecord, hdbscan_query_connection, hdbscan_query_run,
)
from core.hdbscan_query_api import (
    HdbscanQueryDimensions, HdbscanQueryBuffers, HdbscanQueryRequest,
    HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_MEMBERSHIP, HDBSCAN_QUERY_ALL_POINTS,
)
from core.kmeans_variants_api import VariantsSpan
from bindings.metrics_native_support import metric_warn
from providers.python.estimator_binding_pyapi import estimator_binding
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal


def _schema(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var names = context["hdbscan_public_parameters"]
    if py_type(names) is not b.tuple:
        raise Error("TypeError: HDBSCAN constructor schema requires its native tuple")
    for i in range(len(names)):  # driver-loop: exact fixed parameter names before hashing or lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: HDBSCAN parameter names require exact strings")
    return names


def _initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var names = _schema(context)
    if len(names) != len(args):
        raise Error("TypeError: HDBSCAN constructor metadata differs")
    var replacements = List[PythonObject]()
    for i in range(len(names)):  # driver-loop: fixed constructor fields; admit every replaced owner first
        var value = args[i]
        if String(names[i]) == "numeric_mode" and value is None:
            value = family_raw_get_or(model, "numeric_mode", PythonObject(None))
        if py_contains(fields, names[i]) and fields[names[i]] is not value:
            native_release_admit(fields[names[i]], context)
        replacements.append(value)
    for i in range(len(names)):  # driver-loop: publish after complete lifetime admission
        py_setitem(fields, names[i], replacements[i])
    return PythonObject(None)


# Per-model records (bindings/cluster_side_records.mojo, lane perf2-cluster):
# the public admission passed for this key, and per query method the views of
# the fitted storage its kernel reads with their validated counts.
comptime _ADMIT_SLOT = "cluster_hdbscan_admitted_v1"


def _query_slot(method: String) -> String:
    return "cluster_hdbscan_" + method + "_v2"


def _scalars() raises -> PythonObject:
    return Python.tuple("n_features_in_", "n_clusters_", "n_outliers_", "n_boruvka_rounds_", "n_condensed_clusters_", "_mojolearn_fitted", "_mojolearn_fitting")


def _arrays() raises -> PythonObject:
    return Python.tuple("labels_", "probabilities_", "core_distances_", "_prediction_data", "_raw_data", "feature_names_in_")


def _admit(model: PythonObject, context: PythonObject, fitting: Bool, key: PythonObject) raises -> Bool:
    """The public admission. Perf (lane perf2-cluster): the ancestry walk runs
    once per class state, and a query whose `key` matches this model's
    admission record skips the per-value admissions of every setting and
    fitted field; the flag checks run on every call. Returns whether the
    record matched."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    cluster_refuse_fit_callbacks(py_type(model), "TypeError: HDBSCAN cannot invoke Python fit callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: HDBSCAN cannot invoke Python fit callbacks")
    var recorded = not fitting and cluster_record_get(_ADMIT_SLOT, model, key) is not None
    if not recorded:
        # (A recorded query's key was built from the schema this call.)
        var names = _schema(context)
        for i in range(len(names)):  # driver-loop: scalar settings before legacy numerical-owner casts
            var name = String(names[i])
            var value = family_raw_get_or(model, name, PythonObject(None)) if name == "numeric_mode" else family_raw_get(model, name)
            host_inference_scalar(value, context, True)
        var scalars = _scalars()
        var arrays = _arrays()
        for i in range(len(scalars)):  # driver-loop: fitted shape and count metadata
            if py_contains(fields, scalars[i]):
                host_inference_scalar(fields[scalars[i]], context)
                if fitting:
                    native_release_admit(fields[scalars[i]], context)
        for i in range(len(arrays)):  # driver-loop: fitted storage retained through native execution
            if py_contains(fields, arrays[i]):
                classical_admit_value(fields[arrays[i]], context)
                if fitting:
                    native_release_admit(fields[arrays[i]], context)
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.int and py_type(fitted) is not b.bool:
        raise Error("TypeError: native fitted flags require exact booleans or integers")
    return recorded


def _record_key(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The identity key of a query: every setting and fitted field the
    admission and the query read (`_prediction_data` item by item)."""
    var optional = Python.tuple("n_features_in_", "n_clusters_", "n_outliers_", "n_boruvka_rounds_",
                                "n_condensed_clusters_", "_mojolearn_fitted", "_mojolearn_fitting", "labels_",
                                "probabilities_", "core_distances_", "_prediction_data", "_raw_data",
                                "feature_names_in_")
    return cluster_record_key(model, py_fields(model), _schema(context), optional)


def _buffer(source: PythonObject, name: String, dtype: String, count: Int, context: PythonObject) raises -> PythonObject:
    var view = classical_view(source, context, name)
    if classical_dtype_str(view, context) != dtype or not classical_c_order(view, context):
        raise Error("TypeError: HDBSCAN " + name + " requires C-order " + dtype + " storage")
    if Int(py=view.size) < count:
        raise Error("ValueError: HDBSCAN " + name + " is smaller than its fitted metadata")
    return view


def _inference(model: PythonObject, method: String, args: PythonObject, fields: PythonObject,
               binding: PythonObject, context: PythonObject, key: PythonObject) raises -> PythonObject:
    """approximate_predict, membership_vector or all_points_membership_vectors.

    Perf (lane perf2-cluster): the prediction-data walk, the count checks,
    the nine views of the fitted storage (each a full view admission), the
    mode check and the entry lookup depend only on the model's fields and the
    binding. With a record `key`, their results are kept per model and query
    method while the key matches and the binding is the same module; the
    record is written only after the full path succeeded. The views share
    the fitted arrays' storage, so in-place value edits are still read."""
    var b = native_loaded_module("builtins")
    var approximate = method == "approximate_predict"
    var all_points = method == "all_points_membership_vectors"
    var slot = _query_slot(method)
    var held = cluster_record_get(slot, model, key)
    if held is not None:
        var cached = held.downcast_value_ptr[HdbscanQueryRecord]()
        if cached[].binding is not binding:
            held = PythonObject(None)
    var raw = PythonObject(None)
    var core = PythonObject(None)
    var labels = PythonObject(None)
    var parents = PythonObject(None)
    var lambdas = PythonObject(None)
    var deaths = PythonObject(None)
    var clusters = PythonObject(None)
    var children = PythonObject(None)
    var indices = PythonObject(None)
    var offsets = PythonObject(None)
    var connection = Optional[HdbscanQueryConnection]()
    var edges = 0
    var condensed = 0
    var exemplars = 0
    var selected = 0
    var columns = 0
    var rows = 0
    var neighbors = 0
    var pd = PythonObject(None)
    if held is not None:
        var cached = held.downcast_value_ptr[HdbscanQueryRecord]()
        connection = Optional(HdbscanQueryConnection(cached[].provider, cached[].driver, cached[].host))
        raw = cached[].raw
        core = cached[].core_distances
        labels = cached[].labels
        parents = cached[].parents
        lambdas = cached[].lambdas
        deaths = cached[].deaths
        clusters = cached[].clusters
        children = cached[].children
        indices = cached[].indices
        offsets = cached[].offsets
        var counts = cached[].dimensions
        rows = counts.rows
        columns = counts.features
        edges = counts.edges
        condensed = counts.condensed
        exemplars = counts.exemplars
        selected = counts.clusters
        neighbors = counts.neighbors
    else:
        if not py_contains(fields, "labels_"):
            raise Error("mojolearn " + method + ": this HDBSCAN instance is not fitted yet; call fit first")
        pd = py_get(fields, "_prediction_data")
        if pd is None:
            raise Error("mojolearn " + method + ": Prediction data not yet generated. Fit with HDBSCAN(prediction_data=True); cuML refuses the same call the same way (hdbscan.pyx:1104-1108)")
        if py_type(pd) is not b.dict:
            raise Error("TypeError: HDBSCAN prediction metadata requires an exact dictionary")
        if not native_namespace_string_keys(pd._obj_ptr, b.str):
            raise Error("TypeError: HDBSCAN prediction metadata requires exact string keys")
        for name in ["n_edges", "n_condensed", "n_exemplars"]:  # driver-loop: fixed tree counts
            if py_type(pd[name]) is not b.int or Int(py=pd[name]) < 0:
                raise Error("TypeError: HDBSCAN tree counts require non-negative native integers")
        edges = Int(py=pd["n_edges"])
        condensed = Int(py=pd["n_condensed"])
        exemplars = Int(py=pd["n_exemplars"])
        selected = Int(py=fields["n_clusters_"])
        columns = Int(py=fields["n_features_in_"])
        if selected < 0 or columns < 1:
            raise Error("ValueError: HDBSCAN fitted dimensions are invalid")
    var batch = 1
    if not approximate:
        var supplied = args[0] if all_points else args[1]
        if py_type(supplied) is not b.int:
            raise Error("TypeError: mojolearn " + method + ": batch_size must be an int")
        batch = Int(py=supplied)
        if batch <= 0:
            raise Error("mojolearn " + method + ": batch_size must be > 0, got " + String(batch))
    if approximate and selected == 0:
        metric_warn("Clusterer does not have any defined clusters, new data will be automatically predicted as outliers.", b.UserWarning, context)
    if held is None:
        raw = _buffer(fields["_raw_data"], "training data", "<f4", 0, context)
        if Int(py=raw.ndim) != 2 or Int(py=raw.shape[1]) != columns:
            raise Error("ValueError: HDBSCAN training storage differs from its fitted shape")
        rows = Int(py=raw.shape[0])
        if rows < 2:
            raise Error("ValueError: HDBSCAN training storage requires at least two rows")
    var queries = raw if all_points else admit_family(args[0], 2, "points_to_predict", context, finite=True).array
    var nq = Int(py=queries.shape[0])
    var d = Int(py=queries.shape[1])
    if d != columns:
        raise Error("mojolearn " + method + ": points_to_predict has " + String(d) + " features, the clusterer was fit on " + String(columns))
    var output = family_array(Python.tuple(nq), "<i4", context) if approximate else family_array(Python.tuple(nq, selected), "<f4", context)
    if not approximate and (selected == 0 or nq == 0):
        return output
    if held is None:
        var min_samples = family_raw_get(model, "min_samples")
        if min_samples is None or not Bool(py=min_samples):
            min_samples = family_raw_get(model, "min_cluster_size")
        neighbors = Int(py=min_samples)
        if neighbors < 1 or neighbors > rows:
            raise Error("ValueError: HDBSCAN min_samples must fit its training row count")
        core = _buffer(fields["core_distances_"], "core_distances_", "<f4", rows, context)
        labels = _buffer(fields["labels_"], "labels_", "<i4", rows, context)
        parents = _buffer(pd["parents"], "tree parents", "<i4", edges, context)
        lambdas = _buffer(pd["lambdas"], "tree lambdas", "<f4", edges, context)
        deaths = _buffer(pd["deaths"], "deaths", "<f4", condensed, context)
        clusters = _buffer(pd["selected_clusters"], "selected_clusters", "<i4", selected, context)
        children = _buffer(pd["index_into_children"], "index_into_children", "<i4", edges + 1, context)
        _hdbscan_mode(model, binding, context)
    if approximate:
        if held is None:
            connection = Optional(hdbscan_query_connection(binding))
        var probabilities = family_array(Python.tuple(nq), "<f4", context)
        var dims = HdbscanQueryDimensions(rows, columns, edges, condensed, exemplars, selected, neighbors)
        # Preserve address-read order from the old ten-address envelope.
        var training_span = hdbscan_buffer_span(raw, rows * columns)
        var core_span = hdbscan_buffer_span(core, rows)
        var label_span = hdbscan_buffer_span(labels, rows)
        var lambda_span = hdbscan_buffer_span(lambdas, edges)
        var death_span = hdbscan_buffer_span(deaths, condensed)
        var selected_span = hdbscan_buffer_span(clusters, selected)
        var child_span = hdbscan_buffer_span(children, edges + 1)
        var query_span = hdbscan_buffer_span(queries, nq * columns)
        var output_span = hdbscan_buffer_span(output, nq)
        var probability_span = hdbscan_buffer_span(probabilities, nq)
        var absent = VariantsSpan(0, 0)
        var buffers = HdbscanQueryBuffers(training_span, core_span, label_span, absent,
            lambda_span, death_span, selected_span, child_span, absent, absent,
            query_span, output_span, probability_span)
        hdbscan_query_run(connection.value(), HdbscanQueryRequest(dims, buffers, nq, 0, batch), HDBSCAN_QUERY_APPROXIMATE)
        if held is None:
            _record(model, slot, key, binding, connection.value(), raw, core, labels, parents, lambdas, deaths, clusters, children,
                    indices, offsets, rows, columns, edges, condensed, exemplars, selected, neighbors)
        _ = raw^
        _ = queries^
        _ = core^
        _ = labels^
        _ = parents^
        _ = lambdas^
        _ = deaths^
        _ = clusters^
        _ = children^
        return Python.tuple(output, probabilities)
    if held is None:
        indices = _buffer(pd["exemplar_idx"], "exemplar_idx", "<i4", exemplars, context)
        offsets = _buffer(pd["exemplar_label_offsets"], "exemplar_label_offsets", "<i4", selected + 1, context)
        connection = Optional(hdbscan_query_connection(binding))
    # Keep the old tree-first and query/output-before-training address reads.
    # These are boundary storage admissions, not numerical work.
    var parent_span = hdbscan_buffer_span(parents, edges)
    var lambda_span = hdbscan_buffer_span(lambdas, edges)
    var death_span = hdbscan_buffer_span(deaths, condensed)
    var selected_span = hdbscan_buffer_span(clusters, selected)
    var child_span = hdbscan_buffer_span(children, edges + 1)
    var exemplar_span = hdbscan_buffer_span(indices, exemplars)
    var offset_span = hdbscan_buffer_span(offsets, selected + 1)
    var query_span = VariantsSpan(0, 0) if all_points else hdbscan_buffer_span(queries, nq * columns)
    var output_span = hdbscan_buffer_span(output, nq * selected)
    var training_span = hdbscan_buffer_span(raw, rows * columns)
    var core_span = VariantsSpan(0, 0)
    var label_span = VariantsSpan(0, 0)
    if not all_points:
        core_span = hdbscan_buffer_span(core, rows)
        label_span = hdbscan_buffer_span(labels, rows)
    var dims = HdbscanQueryDimensions(rows, columns, edges, condensed, exemplars, selected, neighbors)
    var buffers = HdbscanQueryBuffers(training_span, core_span, label_span, parent_span,
        lambda_span, death_span, selected_span, child_span, exemplar_span, offset_span,
        query_span, output_span, VariantsSpan(0, 0))
    hdbscan_query_run(connection.value(), HdbscanQueryRequest(dims, buffers, nq, 0, batch),
        HDBSCAN_QUERY_ALL_POINTS if all_points else HDBSCAN_QUERY_MEMBERSHIP)
    if held is None:
        _record(model, slot, key, binding, connection.value(), raw, core, labels, parents, lambdas, deaths, clusters, children,
                indices, offsets, rows, columns, edges, condensed, exemplars, selected, neighbors)
    # Retain every input and output buffer owner through the final launch.
    _ = raw^
    _ = queries^
    _ = core^
    _ = labels^
    _ = parents^
    _ = lambdas^
    _ = deaths^
    _ = clusters^
    _ = children^
    _ = indices^
    _ = offsets^
    return output


def _record(model: PythonObject, slot: String, key: PythonObject, binding: PythonObject,
            connection: HdbscanQueryConnection, raw: PythonObject, core: PythonObject,
            labels: PythonObject, parents: PythonObject, lambdas: PythonObject,
            deaths: PythonObject, clusters: PythonObject, children: PythonObject,
            indices: PythonObject, offsets: PythonObject, rows: Int, columns: Int,
            edges: Int, condensed: Int, exemplars: Int, selected: Int, neighbors: Int) raises:
    """Same weak-key/provider cache with typed counts, driver and named pins."""
    var dims = HdbscanQueryDimensions(rows, columns, edges, condensed, exemplars, selected, neighbors)
    var record = HdbscanQueryRecord(binding, connection.provider, connection.driver, connection.host,
        dims, raw, core, labels, parents, lambdas, deaths, clusters, children, indices, offsets)
    cluster_record_put(slot, model, key, PythonObject(record^))


def hdbscan_public_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: HDBSCAN arguments require their native tuple")
    # Passive subclasses share the connected native owner: CPython's MRO
    # test (bindings/cluster_side_records.mojo), not a Python-level walk.
    if not cluster_is_subclass(py_type(model), context["hdbscan_public_class"]):
        raise Error("TypeError: HDBSCAN execution requires its connected class")
    var method = String(operation[byte=String("family_hdbscan_public_").byte_length():])
    if method == "initialize":
        return _initialize(model, args, context)
    var fitting = method == "fit" or method == "fit_predict"
    var query = method == "approximate_predict" or method == "membership_vector" or method == "all_points_membership_vectors"
    var key = _record_key(model, context) if query else PythonObject(None)
    var recorded = _admit(model, context, fitting, key)
    var fields = py_fields(model)
    if fitting:
        # A fit replaces the fitted state: release the old records now.
        cluster_record_drop(_ADMIT_SLOT, model)
        cluster_record_drop(_query_slot("approximate_predict"), model)
        cluster_record_drop(_query_slot("membership_vector"), model)
        cluster_record_drop(_query_slot("all_points_membership_vectors"), model)
    var binding = estimator_binding(model, "_mojolearn_hdbscan", context, False)
    if method == "binding":
        _hdbscan_mode(model, binding, context)
        return binding
    if not fitting:
        var result = _inference(model, method, args, fields, binding, context, key)
        if query and not recorded:
            cluster_record_put(_ADMIT_SLOT, model, key, PythonObject(True))
        return result
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    _target(args[1], rows, name, context)
    var depth = estimator_fit_begin(fields)
    var result = PythonObject(None)
    try:
        result = cluster_fit_a_operation(model, "family_cfa_hdbscan_" + method, Python.tuple(binding, x), context)
    except problem:
        var refused = _refusal(problem, True, depth > 0, rows, columns, name)
        raise refused^
    finally:
        estimator_fit_end(model, depth)
    estimator_fit_complete(model, args[0], False, context)
    if estimator_outside() and py_type(result) is context["array_class"] and py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result
