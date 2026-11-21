# SPDX-License-Identifier: Apache-2.0
# UNPACK of HDBSCAN's public lifecycle: constructor fields, the public
# admission and its per-model record, and a query's Python reads (the fitted
# storage views, the prediction-data mapping, batch_size, the query rows)
# into one `HdbscanQueryCall`: typed counts plus the lifetime pins of the
# buffers the provider reads. RUN (providers/python/hdbscan_public_lifecycle_run.mojo
# and core/hdbscan_query_api.mojo) sees the counts and the spans only; PACK
# and the operator are providers/python/hdbscan_public_lifecycle_pyapi_pack.mojo.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_type, py_get, py_contains, py_fields, py_setitem, py_item
from bindings.native_loaded_module_door import native_loaded_module, native_namespace_string_keys
from bindings.cluster_side_records_unpack import cluster_refuse_fit_callbacks
from bindings.cluster_side_records_pack import cluster_record_key, cluster_record_get
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_arrays_door import classical_dtype_str, classical_c_order
from bindings.classical_validate_unpack import classical_view
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.hdbscan_fit_boundary_unpack import hdbscan_buffer_span, _hdbscan_mode
from bindings.hdbscan_query_boundary_unpack import (
    HdbscanQueryConnection, HdbscanQueryRecord, hdbscan_query_connection,
)
from core.hdbscan_query_api import HdbscanQueryDimensions, HdbscanQueryBuffers
from core.kmeans_variants_api import VariantsSpan
from bindings.metrics_native_support_door import metric_warn
from providers.python.device_route_boundary_door import device_route_clear
from bindings.py_optional import py_present
from providers.python.hdbscan_public_lifecycle_run import (
    _query_slot, _query_method, hdbscan_fitted_dimensions, hdbscan_batch_size, hdbscan_training_shape,
    hdbscan_training_rows, hdbscan_query_width, hdbscan_query_neighbors,
)
from bindings.operation_names_hdbscan import (
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT,
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS)
from bindings.cluster_keys_unpack import cluster_keys


def _schema(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var nk = cluster_keys()
    var names = py_item(context, nk[].k_hdbscan_public_parameters)
    if py_type(names) is not b.tuple:
        raise Error("TypeError: HDBSCAN constructor schema requires its native tuple")
    for i in range(len(names)):  # driver-loop: exact fixed parameter names before hashing or lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: HDBSCAN parameter names require exact strings")
    return names


def hdbscan_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var names = _schema(context)
    # `device` is the constructor's last value, outside the names every
    # operation admits again: an instance without the field runs this
    # installation's device. Its check is the shared constructor door's (the
    # public constructor is the full-reset native constructor); a constructor
    # run again on a live estimator drops its binding route.
    if len(names) + 1 != len(args):
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
    py_setitem(fields, PythonObject("device"), args[len(names)])
    device_route_clear(model)
    return PythonObject(None)


# Per-model records (bindings/cluster_side_records.mojo, lane perf2-cluster):
# the public admission passed for this key, and per query method the views of
# the fitted storage its kernel reads with their validated counts.
comptime _ADMIT_SLOT = "cluster_hdbscan_admitted_v1"


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
    var nk = cluster_keys()
    if py_contains(fields, nk[].k__mojolearn_after_fit):
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
    var depth = py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
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


struct HdbscanQueryCall(Movable):
    """One query after UNPACK. The Python values are lifetime pins of the
    buffers whose spans RUN reads (and `pd`, the admitted prediction-data
    mapping, between the two UNPACK stages); everything RUN decides on is
    the typed counts. `held`: the per-model record supplied the views, the
    counts and the provider connection."""
    var held: Bool
    var raw: PythonObject
    var core: PythonObject
    var labels: PythonObject
    var parents: PythonObject
    var lambdas: PythonObject
    var deaths: PythonObject
    var clusters: PythonObject
    var children: PythonObject
    var indices: PythonObject
    var offsets: PythonObject
    var queries: PythonObject
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var pd: Optional[PythonObject]
    var connection: Optional[HdbscanQueryConnection]
    var rows: Int
    var columns: Int
    var edges: Int
    var condensed: Int
    var exemplars: Int
    var selected: Int
    var neighbors: Int
    var nq: Int
    var batch: Int

    def __init__(out self):
        self.held = False
        self.raw = PythonObject(None)
        self.core = PythonObject(None)
        self.labels = PythonObject(None)
        self.parents = PythonObject(None)
        self.lambdas = PythonObject(None)
        self.deaths = PythonObject(None)
        self.clusters = PythonObject(None)
        self.children = PythonObject(None)
        self.indices = PythonObject(None)
        self.offsets = PythonObject(None)
        self.queries = PythonObject(None)
        self.pd = Optional[PythonObject]()
        self.connection = Optional[HdbscanQueryConnection]()
        self.rows = 0
        self.columns = 0
        self.edges = 0
        self.condensed = 0
        self.exemplars = 0
        self.selected = 0
        self.neighbors = 0
        self.nq = 0
        self.batch = 1

    def dimensions(self) -> HdbscanQueryDimensions:
        return HdbscanQueryDimensions(self.rows, self.columns, self.edges, self.condensed, self.exemplars,
                                      self.selected, self.neighbors)


def hdbscan_query_unpack(mut call: HdbscanQueryCall, model: PythonObject, operation: Int, args: PythonObject,
                         fields: PythonObject, binding: PythonObject, context: PythonObject,
                         key: PythonObject) raises:
    """First UNPACK stage of approximate_predict, membership_vector or
    all_points_membership_vectors: everything read before the public output
    is allocated (the operator returns an empty membership result between
    the two stages, as it always did).

    Perf (lane perf2-cluster): the prediction-data walk, the count checks,
    the nine views of the fitted storage (each a full view admission), the
    mode check and the entry lookup depend only on the model's fields and the
    binding. With a record `key`, their results are kept per model and query
    method while the key matches and the binding is the same module; the
    record is written only after the full path succeeded. The views share
    the fitted arrays' storage, so in-place value edits are still read."""
    var b = native_loaded_module("builtins")
    # `operation` is a query's leaf code (lane bw2-cluster); `method` is its
    # static name for the refusal texts.
    var method = _query_method(operation)
    var approximate = operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT
    var all_points = operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS
    var held = cluster_record_get(_query_slot(operation), model, key)
    if held is not None:
        var cached = held.downcast_value_ptr[HdbscanQueryRecord]()
        if cached[].binding is not binding:
            held = PythonObject(None)
    var nk = cluster_keys()
    if held is not None:
        var cached = held.downcast_value_ptr[HdbscanQueryRecord]()
        call.held = True
        call.connection = Optional(HdbscanQueryConnection(cached[].provider, cached[].driver.copy(), cached[].host))
        call.raw = cached[].raw
        call.core = cached[].core_distances
        call.labels = cached[].labels
        call.parents = cached[].parents
        call.lambdas = cached[].lambdas
        call.deaths = cached[].deaths
        call.clusters = cached[].clusters
        call.children = cached[].children
        call.indices = cached[].indices
        call.offsets = cached[].offsets
        var counts = cached[].dimensions.copy()
        call.rows = counts.rows
        call.columns = counts.features
        call.edges = counts.edges
        call.condensed = counts.condensed
        call.exemplars = counts.exemplars
        call.selected = counts.clusters
        call.neighbors = counts.neighbors
    else:
        if not py_contains(fields, nk[].k_labels_):
            raise Error("mojolearn " + method + ": this HDBSCAN instance is not fitted yet; call fit first")
        var pd = py_present(py_get(fields, nk[].k__prediction_data))
        if not pd:
            raise Error("mojolearn " + method + ": Prediction data not yet generated. Fit with HDBSCAN(prediction_data=True); cuML refuses the same call the same way (hdbscan.pyx:1104-1108)")
        if py_type(pd.value()) is not b.dict:
            raise Error("TypeError: HDBSCAN prediction metadata requires an exact dictionary")
        if not native_namespace_string_keys(pd.value()._obj_ptr, b.str):
            raise Error("TypeError: HDBSCAN prediction metadata requires exact string keys")
        for name in ["n_edges", "n_condensed", "n_exemplars"]:  # driver-loop: fixed tree counts
            if py_type(pd.value()[name]) is not b.int or Int(py=pd.value()[name]) < 0:
                raise Error("TypeError: HDBSCAN tree counts require non-negative native integers")
        call.edges = Int(py=py_item(pd.value(), nk[].k_n_edges))
        call.condensed = Int(py=py_item(pd.value(), nk[].k_n_condensed))
        call.exemplars = Int(py=py_item(pd.value(), nk[].k_n_exemplars))
        call.selected = Int(py=py_item(fields, nk[].k_n_clusters_))
        call.columns = Int(py=py_item(fields, nk[].k_n_features_in_))
        hdbscan_fitted_dimensions(call.selected, call.columns)
        call.pd = Optional(pd.value())
    if not approximate:
        var supplied = args[0] if all_points else args[1]
        if py_type(supplied) is not b.int:
            raise Error("TypeError: mojolearn " + method + ": batch_size must be an int")
        call.batch = Int(py=supplied)
        hdbscan_batch_size(method, call.batch)
    # Issued here, where it always was: before the training storage and the
    # query rows are admitted (a warning raised as an error keeps its place).
    if approximate and call.selected == 0:
        metric_warn("Clusterer does not have any defined clusters, new data will be automatically predicted as outliers.", b.UserWarning, context)
    if not call.held:
        call.raw = _buffer(py_item(fields, nk[].k__raw_data), "training data", "<f4", 0, context)
        var ndim = Int(py=call.raw.ndim)
        var width = -1
        if ndim == 2:
            width = Int(py=call.raw.shape[1])
        hdbscan_training_shape(ndim, width, call.columns)
        call.rows = Int(py=call.raw.shape[0])
        hdbscan_training_rows(call.rows)
    call.queries = call.raw if all_points else admit_family(args[0], 2, "points_to_predict", context, finite=True).array
    call.nq = Int(py=call.queries.shape[0])
    hdbscan_query_width(method, Int(py=call.queries.shape[1]), call.columns)


def hdbscan_query_unpack_storage(mut call: HdbscanQueryCall, model: PythonObject, operation: Int,
                                 fields: PythonObject, binding: PythonObject, context: PythonObject) raises:
    """Second UNPACK stage: min_samples, the views of the fitted storage the
    query's kernel reads, the compiled-mode check and the provider
    connection, in the public order. A held record supplied all of it."""
    if call.held:
        return
    var approximate = operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT
    var nk = cluster_keys()
    var min_samples = family_raw_get(model, "min_samples")
    if min_samples is None or not Bool(py=min_samples):
        min_samples = family_raw_get(model, "min_cluster_size")
    call.neighbors = Int(py=min_samples)
    hdbscan_query_neighbors(call.neighbors, call.rows)
    call.core = _buffer(py_item(fields, nk[].k_core_distances_), "core_distances_", "<f4", call.rows, context)
    call.labels = _buffer(py_item(fields, nk[].k_labels_), "labels_", "<i4", call.rows, context)
    call.parents = _buffer(py_item(call.pd.value(), nk[].k_parents), "tree parents", "<i4", call.edges, context)
    call.lambdas = _buffer(py_item(call.pd.value(), nk[].k_lambdas), "tree lambdas", "<f4", call.edges, context)
    call.deaths = _buffer(py_item(call.pd.value(), nk[].k_deaths), "deaths", "<f4", call.condensed, context)
    call.clusters = _buffer(py_item(call.pd.value(), nk[].k_selected_clusters), "selected_clusters", "<i4", call.selected, context)
    call.children = _buffer(py_item(call.pd.value(), nk[].k_index_into_children), "index_into_children", "<i4", call.edges + 1, context)
    _hdbscan_mode(model, binding, context)
    if not approximate:
        call.indices = _buffer(py_item(call.pd.value(), nk[].k_exemplar_idx), "exemplar_idx", "<i4", call.exemplars, context)
        call.offsets = _buffer(py_item(call.pd.value(), nk[].k_exemplar_label_offsets), "exemplar_label_offsets", "<i4", call.selected + 1, context)
    call.connection = Optional(hdbscan_query_connection(binding))


def hdbscan_query_buffers(call: HdbscanQueryCall, operation: Int, output: PythonObject,
                          probabilities: PythonObject) raises -> HdbscanQueryBuffers:
    """The typed spans of one query: the address read of every pinned
    buffer, in the order the two envelopes always read them. These are
    boundary storage admissions, not numerical work."""
    var absent = VariantsSpan(0, 0)
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT:
        # Preserve address-read order from the old ten-address envelope.
        var training_span = hdbscan_buffer_span(call.raw, call.rows * call.columns)
        var core_span = hdbscan_buffer_span(call.core, call.rows)
        var label_span = hdbscan_buffer_span(call.labels, call.rows)
        var lambda_span = hdbscan_buffer_span(call.lambdas, call.edges)
        var death_span = hdbscan_buffer_span(call.deaths, call.condensed)
        var selected_span = hdbscan_buffer_span(call.clusters, call.selected)
        var child_span = hdbscan_buffer_span(call.children, call.edges + 1)
        var query_span = hdbscan_buffer_span(call.queries, call.nq * call.columns)
        var output_span = hdbscan_buffer_span(output, call.nq)
        var probability_span = hdbscan_buffer_span(probabilities, call.nq)
        return HdbscanQueryBuffers(training_span, core_span, label_span, absent,
            lambda_span, death_span, selected_span, child_span, absent, absent,
            query_span, output_span, probability_span)
    var all_points = operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS
    # Keep the old tree-first and query/output-before-training address reads.
    var parent_span = hdbscan_buffer_span(call.parents, call.edges)
    var lambda_span = hdbscan_buffer_span(call.lambdas, call.edges)
    var death_span = hdbscan_buffer_span(call.deaths, call.condensed)
    var selected_span = hdbscan_buffer_span(call.clusters, call.selected)
    var child_span = hdbscan_buffer_span(call.children, call.edges + 1)
    var exemplar_span = hdbscan_buffer_span(call.indices, call.exemplars)
    var offset_span = hdbscan_buffer_span(call.offsets, call.selected + 1)
    var query_span = VariantsSpan(0, 0) if all_points else hdbscan_buffer_span(call.queries, call.nq * call.columns)
    var output_span = hdbscan_buffer_span(output, call.nq * call.selected)
    var training_span = hdbscan_buffer_span(call.raw, call.rows * call.columns)
    var core_span = VariantsSpan(0, 0)
    var label_span = VariantsSpan(0, 0)
    if not all_points:
        core_span = hdbscan_buffer_span(call.core, call.rows)
        label_span = hdbscan_buffer_span(call.labels, call.rows)
    return HdbscanQueryBuffers(training_span, core_span, label_span, parent_span,
        lambda_span, death_span, selected_span, child_span, exemplar_span, offset_span,
        query_span, output_span, absent)
