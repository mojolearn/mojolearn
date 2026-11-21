# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024): LocalOutlierFactor, NearestCentroid and KNNImputer public methods
# (python/mojolearn/_expansion_neighbors.py).
# Each `family_xn_loc_*` operation validates, prepares, runs the family's compiled
# kernels and returns, with the former Python method's refusals, operations
# and arithmetic order.
#
# The shell hands over the binding `_bind()` resolved (the x_neighbors
# binding of the estimator's numeric mode) first in `args`. LOF and NC use
# native plans and fixed request envelopes. KNNImputer still uses the shared
# `_op` boundary helper below. The A/B arms the module
# read at import (MOJOLEARN_XN_UNFUSED_KNN, MOJOLEARN_XN_UNCOMPACT_IMPUTE,
# MOJOLEARN_XN_OLD_ITEMS) are the import-time values in the context
# (`family_xn_loc_flags`); NC_SPLIT_OPS is a build constant
# (`-D MOJOLEARN_NC_SPLIT_OPS`, neighbors/neighbors_local_run.mojo). The opt-in lazy output map (MOJOLEARN_XN_LAZY_OUT) is a
# memory arm only: outputs are zero-filled Arrays either way, same values.
# Scalars derived on the host (the contamination percentile, the prior
# checks) are the same binary64 operations in the same order; products that
# feed a sum are pinned (`pinned_mul_f64`), since Python never fuses.
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit_arguments, xn_execute, xn_module
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop, prepared_key_add, prepared_key_same
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas_door import classical_admit_value
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.label_admission_native_unpack import labels_primitive_kind
from bindings.host_sparse_native_door import host_sparse_format
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.family_common_unpack import (
    family_repr,
    family_isinstance,
)
from bindings.family_common_pack import (
    family_call,
    family_field,
)
from bindings.classical_labels_door import (
    labels_cast, classical_encode_labels,
    labels_decode_i32_codes, classical_accuracy,
)
from bindings.labels_descriptor_boundary_door import labels_class_ratio_f64
from bindings.nearest_centroid_provider_unpack import NcConnection
from bindings.lof_provider_unpack import LofConnection
from core.lof_plan import LofLabelsPlan, lof_f32_address, lof_offset
from neighbors.neighbors_local_run import (
    LOF_RESULT_SCORES, LOF_RESULT_DECISION, LOF_RESULT_LABELS, LOF_AUTO_OFFSET,
    local_check_n_neighbors, local_copy_f32, local_write_i32,
    lof_refuse_metric, lof_refuse_metric_params, lof_check_contamination, lof_check_fit_rows,
    lof_fit_neighbors, lof_plan, lof_refuse_fit_predict_novelty, lof_check_novelty,
    nc_refuse_metric, nc_check_label_rows, nc_check_classes, nc_check_priors_ndim, nc_check_priors_size,
    nc_check_variance, nc_refuse_decision_metric,
    imp_refuse_missing_values, imp_refuse_metric, imp_refuse_weights, imp_round_f32, imp_cells_length,
    imp_colmiss_length, imp_read_count, imp_fit_columns, imp_check_features, imp_cells_operation, imp_keep_plan,
)
from core.nearest_centroid_plan import NcLabelCodes, NcFitPlan
from neighbors.nearest_centroid_run import (
    NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN,
    NcFitRequest, NcFitOutcome, NcQueryRequest,
    nc_fit_count_run, nc_fit_priors_run, nc_fit_run, nc_labels_run, nc_decision_run, nc_proba_wide, nc_proba_run,
)
from bindings.admitted_input_unpack import admitted_array, admitted_vector, admitted_storage_native, admit_c_view, admitted_dtype
from bindings.admitted_input_unpack import admitted_finite_refuse



# ------------------------------------------------------------------ glue


comptime _LOCAL_STATE_SLOT = "neighbors_local_state_v1"


def _admit_local_state(model: PythonObject, context: PythonObject, fitting: Bool = False) raises -> PythonObject:
    # Mutable settings are re-admitted before comparison, coercion or truth
    # slots. Input metadata is captured by the consuming native buffer helper.
    # The admitted state's identity stamp (every slot object and, for exact
    # list/tuple/dict containers, their items) lives in the native prepared
    # registry, outside `__dict__`, weakly keyed by the model: while it is
    # unchanged a query skips the re-admission, as knn's resident stamp does.
    # `classes_` is re-admitted on every call (an object-dtype label array can
    # change in place without an identity change), like knn's class lists.
    # Perf (perf2-neighbors): returns that admitted label list (None when the
    # model has no `classes_` or at fit), so NearestCentroid predict/score
    # reuse it instead of building it a second time per call.
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546); PACK returns None when absent.
    var labels = Optional[PythonObject]()
    classical_check_layout(model, context)
    var fields = py_fields(model)
    var scalars = Python.tuple("metric", "p", "contamination", "n_neighbors", "novelty", "shrink_threshold",
        "missing_values", "weights", "add_indicator", "keep_empty_features", "n_features_in_",
        "n_neighbors_", "n_samples_fit_", "offset_")
    var values = Python.tuple("metric_params", "priors", "classes_", "_fit_X", "_fit_dist", "_lrd",
        "centroids_", "class_prior_", "within_class_std_dev_", "_valid", "_miss_cols",
        "negative_outlier_factor_")
    var mods = NativeModules()
    var stamp = Python.list()
    for i in range(len(scalars)):  # small-loop(fixed scalar parameter and fitted-state schema)
        prepared_key_add(mods, stamp, py_get(fields, scalars[i]))
    for i in range(len(values)):  # small-loop(fixed native container and buffer owners)
        if String(values[i]) != "classes_":
            prepared_key_add(mods, stamp, py_get(fields, values[i]))
    var record = prepared_get(mods, _LOCAL_STATE_SLOT, model)
    if record is not None and prepared_key_same(record, stamp):
        var classes = py_get(fields, "classes_")
        if classes is not None:
            var listed = classifier_label_list(classes, context)
            classical_admit_value(listed, context)
            labels = listed
    else:
        for i in range(len(scalars)):  # small-loop(fixed scalar parameter and fitted-state schema)
            var value = py_get(fields, scalars[i])
            if value is not None:
                host_inference_scalar(value, context, True)
        for i in range(len(values)):  # small-loop(fixed native container and buffer owners)
            var value = py_get(fields, values[i])
            if value is not None:
                if String(values[i]) == "classes_":
                    var listed = classifier_label_list(value, context)
                    classical_admit_value(listed, context)
                    labels = listed
                else:
                    classical_admit_value(value, context)
    if fitting:
        # Fit replaces the fitted slots: never retain the old arrays.
        prepared_drop(mods, _LOCAL_STATE_SLOT, model)
        return PythonObject(None)
    if record is None or not prepared_key_same(record, stamp):
        for i in range(len(stamp)):  # driver-loop: over the stamp objects, refusing a self-reference
            if stamp[i] is model:
                if labels:
                    return labels.value()
                return PythonObject(None)
        prepared_put(mods, _LOCAL_STATE_SLOT, model, stamp)
    if labels:
        return labels.value()
    return PythonObject(None)


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`, as an `if` or `and`/`or` took it."""
    # Perf (perf2-neighbors): an exact bool or None has no truth protocol to
    # run; read it by identity instead of a Python `bool()` call.
    if value is None:
        return False
    if py_type(value) is _b().bool:
        return value is PythonObject(True)
    return Bool(py=family_call(_b().bool, value))


def _int_word(value: PythonObject) raises -> Int:
    """`int(value)`; an exact int is read without the Python call
    (perf2-neighbors)."""
    if py_type(value) is _b().int:
        return Int(py=value)
    return Int(py=family_call(_b().int, value))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    # Perf (perf2-neighbors): two exact strs compare natively (str equality
    # runs no user protocol); anything else keeps `operator.eq`.
    var bi = _b()
    if py_type(a) is bi.str and py_type(b) is bi.str:
        return String(py=a) == String(py=b)
    return _truth(family_call(native_loaded_module("operator").eq, a, b))


def _ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    var bi = _b()
    if py_type(a) is bi.str and py_type(b) is bi.str:
        return String(py=a) != String(py=b)
    return _truth(family_call(native_loaded_module("operator").ne, a, b))


def _flag(context: PythonObject, name: String) raises -> Bool:
    """An A/B arm the module read at import."""
    return Bool(py=context["family_xn_loc_flags"][PythonObject(name)])


def _rows(a: PythonObject) raises -> Int:
    return Int(py=a.shape[0])


def _cols(a: PythonObject) raises -> Int:
    return Int(py=a.shape[1])


def _size(a: PythonObject) raises -> Int:
    return Int(py=a.size)


def _f32(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_f32(X)`: `as_f32_c(_dense(X), ndim=2, name="X")`."""
    return admit_family(source, 2, "X", context, dense=True).array


def _out(shape: PythonObject, typestr: String, context: PythonObject) raises -> PythonObject:
    """`empty(shape, typestr)` / `_empty_out(shape)`: a new zero-filled Array."""
    return admitted_array(shape, typestr, context)


def _op(binding: PythonObject, name: String, addrs: PythonObject, ints: PythonObject, floats: PythonObject,
        context: PythonObject) raises:
    """`_XNeighbors._op(name, bufs, ints, floats)`: the binding's `xn_<name>`
    with the address, int and float lists. The caller keeps every buffer
    alive across the call."""
    xn_execute(binding, name, addrs, ints, floats)


def _i(v: Int) -> PythonObject:
    return PythonObject(v)


def _f32_scalar(value: PythonObject) raises -> Float64:
    """`_f32_scalar(v)`: `float(v)` rounded once to float32."""
    return imp_round_f32(Float64(py=family_call(_b().float, value)))


def _copy_f32(x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`X.copy()` of a C-order float32 array: a new Array of its shape."""
    var out = _out(py_call(_b().tuple, x.shape), "<f4", context)
    local_copy_f32(Int(py=classical_address(out)), Int(py=classical_address(x)), _size(x))
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


def _i32_from_ints(values: List[Int], context: PythonObject) raises -> PythonObject:
    """`_i32(list)`: an int32 vector Array of a short int list (class
    offsets, kept columns, flags)."""
    var out = admitted_vector(len(values), "<i4", context)
    local_write_i32(values, Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the writes above
    return out


# ------------------------------------------------------------------ LOF


def _lof_error_scalar(value: PythonObject, context: PythonObject) raises -> String:
    # These owners have passed scalar admission. Array repr enters NumPy's
    # Python printing/formatter hooks even for an exact native zero-D buffer.
    if py_type(value) is py_get(context, "ndarray"):
        return "<native scalar array>"
    return family_repr(value)


def _lof_check(fields: PythonObject, context: PythonObject) raises:
    """`LocalOutlierFactor._check_params`."""
    var b = _b()
    var metric = family_field(fields, "metric")
    var p = family_field(fields, "p")
    if not (_eq(metric, PythonObject("euclidean")) or (_eq(metric, PythonObject("minkowski")) and _eq(p, PythonObject(2)))):
        lof_refuse_metric(_lof_error_scalar(metric, context), _lof_error_scalar(p, context))
    if _truth(family_field(fields, "metric_params")):
        lof_refuse_metric_params()
    var contamination = family_field(fields, "contamination")
    if _ne(contamination, PythonObject("auto")):
        lof_check_contamination(Float64(py=family_call(b.float, contamination)))
    local_check_n_neighbors(Int(py=family_call(b.int, family_field(fields, "n_neighbors"))))


def _lof_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> LofConnection:
    """Public admission/publication around the staged native fit plan."""
    var b = _b()
    var fields = py_fields(model)
    _lof_check(fields, context)
    var binding = args[0]
    var x = _f32(args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    lof_check_fit_rows(n, d)
    var k = lof_fit_neighbors(Int(py=family_call(b.int, family_field(fields, "n_neighbors"))), n)
    var unfused = _flag(context, "unfused_knn")
    var old_items = _flag(context, "old_items") if not unfused else False
    var plan = lof_plan(n, n, d, k, True, unfused, old_items)
    var connection = LofConnection(binding)
    var x_address = Int(py=classical_address(x))
    var dist = PythonObject(None)
    var lrd = PythonObject(None)
    var score = PythonObject(None)
    try:
        var status = 0
        with GILReleased(Python()):
            status = plan.select(connection.driver, x_address, x_address)
        dist = admitted_storage_native([n, k], "<f4", context, as_array=True)
        var dist_address = Int(py=classical_address(dist))
        with GILReleased(Python()):
            plan.roots(connection.driver, dist_address)
        admitted_finite_refuse(status, String(py_type(model).__name__))
        lrd = admitted_storage_native([n], "<f4", context, as_array=True)
        var lrd_address = Int(py=classical_address(lrd))
        with GILReleased(Python()):
            plan.density(connection.driver, dist_address, dist_address, lrd_address, n)
        score = admitted_storage_native([n], "<f4", context, as_array=True)
        var score_address = Int(py=classical_address(score))
        with GILReleased(Python()):
            plan.score(connection.driver, lrd_address, lrd_address, score_address, n)
        py_setitem(fields, "_fit_X", x)
        py_setitem(fields, "_fit_dist", dist)
        py_setitem(fields, "_lrd", lrd)
        py_setitem(fields, "n_neighbors_", _i(k))
        py_setitem(fields, "n_features_in_", _i(d))
        py_setitem(fields, "n_samples_fit_", _i(n))
        py_setitem(fields, "negative_outlier_factor_", score)
        var contamination = family_field(fields, "contamination")
        if _eq(contamination, PythonObject("auto")):
            py_setitem(fields, "offset_", PythonObject(LOF_AUTO_OFFSET))
        else:
            var c = Float64(py=family_call(b.float, contamination))
            var value = Float64(0)
            with GILReleased(Python()):
                value = lof_offset(connection.driver, score_address, n, c)
            py_setitem(fields, "offset_", PythonObject(value))
    finally:
        # Native integer addresses do not retain the owners or compiled image.
        # Pins cover success, native refusal, and exception unwinding.
        _ = plan
        _ = connection
        _ = x
        _ = dist
        _ = lrd
        _ = score
    return connection^


def _lof_fit_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if _truth(family_field(fields, "novelty")):
        lof_refuse_fit_predict_novelty()
    var connection = _lof_fit(model, args, context)
    var off = family_field(fields, "offset_")
    var score = family_field(fields, "negative_outlier_factor_")
    var size = _size(score)
    var labels = LofLabelsPlan(size)
    var result = PythonObject(None)
    try:
        if size != 0:
            var score_address = Int(py=classical_address(score))
            var threshold = Float64(py=family_call(_b().float, off))
            with GILReleased(Python()):
                labels.classify(connection.driver, score_address, 0, threshold)
        result = admitted_storage_native([size], "<i8", context, as_array=True)
        var result_address = Int(py=classical_address(result))
        with GILReleased(Python()):
            labels.write_i64(result_address)
        return result
    finally:
        _ = labels
        _ = connection
        _ = score
        _ = result


def _lof_novelty(fields: PythonObject, what: String) raises:
    lof_check_novelty(_truth(family_field(fields, "novelty")), what)


def _lof_query(binding: PythonObject, fields: PythonObject, source: PythonObject,
               context: PythonObject, name: String, result_kind: Int) raises -> PythonObject:
    """Unpack staged public state, run native query, pack its requested result.

    result_kind is LOF_RESULT_SCORES, LOF_RESULT_DECISION or LOF_RESULT_LABELS. Private scores and
    decisions never cross Python; fitted fields retain their old read stages.
    """
    var q = _f32(source, context)
    var k = _int_word(family_field(fields, "n_neighbors_"))
    var nf = _int_word(family_field(fields, "n_samples_fit_"))
    var fit_x = family_field(fields, "_fit_X")
    var nq = _rows(q)
    var features = _cols(q)
    var reference_rows = _rows(fit_x)
    var unfused = _flag(context, "unfused_knn")
    var old_items = _flag(context, "old_items") if not unfused else False
    var plan = lof_plan(nq, reference_rows, features, k, False, unfused, old_items)
    var connection = LofConnection(binding)
    var query_address = Int(py=classical_address(q))
    var fit_address = Int(py=classical_address(fit_x))
    var fit_dist = PythonObject(None)
    var fit_lrd = PythonObject(None)
    var result = PythonObject(None)
    try:
        var status = 0
        with GILReleased(Python()):
            status = plan.query_neighbors(connection.driver, query_address, fit_address)
        admitted_finite_refuse(status, name)
        fit_dist = family_field(fields, "_fit_dist")
        fit_lrd = family_field(fields, "_lrd")
        var fit_dist_address = Int(py=classical_address(fit_dist))
        with GILReleased(Python()):
            plan.query_density(connection.driver, fit_dist_address, nf)
        if result_kind == LOF_RESULT_SCORES:
            result = admitted_storage_native([nq], "<f4", context, as_array=True)
            var fit_lrd_address = Int(py=classical_address(fit_lrd))
            var result_address = Int(py=classical_address(result))
            with GILReleased(Python()):
                plan.score(connection.driver, fit_lrd_address, lof_f32_address(plan.densities), result_address, nf)
            return result
        var fit_lrd_address = Int(py=classical_address(fit_lrd))
        with GILReleased(Python()):
            plan.query_score(connection.driver, fit_lrd_address, nf)
        var offset = family_field(fields, "offset_")
        var shift: Float64
        if py_type(offset) is _b().float:
            shift = -Float64(py=offset)
        else:
            # Scalar unpack preserves native integer/NumPy negation semantics
            # (including unsigned wrap/refusal) before Float64 conversion.
            var op = native_loaded_module("operator")
            shift = Float64(py=family_call(_b().float, family_call(op.neg, offset)))
        if result_kind == LOF_RESULT_DECISION:
            result = admitted_storage_native([nq], "<f4", context, as_array=True)
            var result_address = Int(py=classical_address(result))
            with GILReleased(Python()):
                plan.decision(connection.driver, result_address, shift)
            return result
        with GILReleased(Python()):
            plan.query_decision(connection.driver, shift)
        var labels = LofLabelsPlan(nq)
        try:
            with GILReleased(Python()):
                labels.classify(connection.driver, lof_f32_address(plan.decisions), 1, 0.0)
            result = admitted_storage_native([nq], "<i8", context, as_array=True)
            var result_address = Int(py=classical_address(result))
            with GILReleased(Python()):
                labels.write_i64(result_address)
            return result
        finally:
            _ = labels
    finally:
        _ = connection
        _ = plan
        _ = q
        _ = fit_x
        _ = fit_dist
        _ = fit_lrd
        _ = result


def _lof_score_samples(binding: PythonObject, fields: PythonObject, source: PythonObject,
                       context: PythonObject, name: String) raises -> PythonObject:
    return _lof_query(binding, fields, source, context, name, LOF_RESULT_SCORES)


def _lof_decision(model: PythonObject, args: PythonObject, context: PythonObject,
                  predict: Bool = False) raises -> PythonObject:
    var fields = py_fields(model)
    _lof_novelty(fields, "decision_function")
    return _lof_query(args[0], fields, args[1], context, String(py_type(model).__name__),
                      LOF_RESULT_LABELS if predict else LOF_RESULT_DECISION)


def _lof_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    _lof_novelty(fields, "predict")
    return _lof_decision(model, args, context, True)


# ------------------------------------------------------------ NearestCentroid


def _class_array(classes: PythonObject, codes: List[Int32], context: PythonObject) raises -> PythonObject:
    """`_class_array(classes, codes)`: int or bool classes an int64 Array,
    real classes a float64 Array, other labels a list (`decode_labels`)."""
    # Perf (perf2-neighbors): an exact int/bool/float label answers the kind
    # tests by its type, and `int()`/`float()` of an exact int/float is the
    # object itself, so only other labels pay the Python calls.
    var b = _b()
    var k = len(classes)
    var all_int = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        var cls = py_type(classes[i])
        if cls is not b.int and cls is not b.bool and not family_isinstance(classes[i], Python.tuple(b.bool, b.int)):
            all_int = False
            break
    if all_int:
        var ints = Python.list()
        for i in range(k):  # driver-loop: over the k class values
            var label = classes[i]
            _ = py_call(ints.append, label if py_type(label) is b.int else family_call(b.int, label))
        return labels_decode_i32_codes(ints, codes, context)
    var all_real = True
    for i in range(k):  # driver-loop: over the k class labels (kind test)
        var cls = py_type(classes[i])
        if cls is not b.int and cls is not b.float and not family_isinstance(classes[i], Python.tuple(b.int, b.float)):
            all_real = False
            break
    if all_real:
        var reals = Python.list()
        for i in range(k):  # driver-loop: over the k class labels as floats
            var label = classes[i]
            _ = py_call(reals.append, label if py_type(label) is b.float else family_call(b.float, label))
        return labels_decode_i32_codes(reals, codes, context)
    return labels_decode_i32_codes(classes, codes, context)


# NearestCentroid is UNPACK / RUN / PACK. UNPACK (`_nc_fit_unpack`,
# `_nc_query_unpack`) reads every public value once into a typed request of
# neighbors/nearest_centroid_run.mojo and keeps the Python owners as pins.
# RUN is that module (no Python import), entered with the GIL released. PACK
# publishes the arrays UNPACK allocated and turns a status word into the
# public refusal.


struct _NcFitOperands(Movable):
    # Boundary owners only: lifetime pins of the storage the request addresses.
    var x: PythonObject
    var codes: PythonObject
    var classes: PythonObject
    var prior: PythonObject
    var stats: PythonObject
    var centroids: PythonObject
    var deviations: PythonObject
    var request: NcFitRequest

    def __init__(out self, x: PythonObject, codes: PythonObject, classes: PythonObject, prior: PythonObject,
                 stats: PythonObject, centroids: PythonObject, deviations: PythonObject, request: NcFitRequest):
        self.x = x
        self.codes = codes
        self.classes = classes
        self.prior = prior
        self.stats = stats
        self.centroids = centroids
        self.deviations = deviations
        self.request = request.copy()


def _nc_fit_unpack(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> _NcFitOperands:
    """UNPACK of NearestCentroid.fit: parameters, rows, label codes, the
    priors' kind (given priors cast to float64) and the four result arrays."""
    var b = _b()
    var metric = family_field(fields, "metric")
    if not py_contains(Python.tuple(PythonObject("euclidean"), PythonObject("manhattan")), metric):
        nc_refuse_metric()
    var x = _f32(args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    var encoded = classical_encode_labels(args[2], context)
    var classes = encoded.classes
    var lab = encoded.codes
    nc_check_label_rows(_size(lab), n)
    var C = len(classes)
    nc_check_classes(C)
    var priors = family_field(fields, "priors")
    var kind = NC_PRIORS_GIVEN
    if py_type(priors) is b.str:
        var word = String(priors)
        if word == "empirical":
            kind = NC_PRIORS_EMPIRICAL
        elif word == "uniform":
            kind = NC_PRIORS_UNIFORM
    var prior = admitted_storage_native([C], "<f8", context, as_array=True)
    var prior_address = Int(py=classical_address(prior))
    if kind == NC_PRIORS_GIVEN:
        var given = admit_c_view(priors, "priors", context)
        nc_check_priors_ndim(Int(py=given.ndim))
        nc_check_priors_size(Int(py=given.size), C)
        labels_cast(Int(py=classical_address(given)), admitted_dtype(given, context), prior_address, "<f8", C, context)
        _ = given
    var threshold = family_field(fields, "shrink_threshold")
    var shrink = Float64(py=family_call(b.float, threshold)) if _truth(threshold) else 0.0
    var stats = admitted_storage_native([d], "<f4", context, as_array=True)
    var new_cent = admitted_storage_native([C, d], "<f4", context, as_array=True)
    var devs = admitted_storage_native([C, d], "<f4", context, as_array=True)
    var request = NcFitRequest(Int(py=classical_address(x)), Int(py=classical_address(lab)), n, d, C,
                               _eq(metric, PythonObject("euclidean")), kind, shrink)
    request.prior = prior_address
    request.stats = Int(py=classical_address(stats))
    request.centroids = Int(py=classical_address(new_cent))
    request.deviations = Int(py=classical_address(devs))
    return _NcFitOperands(x, lab, classes, prior, stats, new_cent, devs, request)


def _nc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """NearestCentroid.fit: UNPACK, the typed RUN with the GIL released, PACK."""
    var fields = py_fields(model)
    var call = _nc_fit_unpack(fields, args, context)
    var connection = NcConnection(args[0])
    var plan = NcFitPlan(call.request.rows, call.request.features, call.request.classes)
    var empirical = call.request.priors == NC_PRIORS_EMPIRICAL
    with GILReleased(Python()):
        nc_fit_count_run(connection.driver, call.request, plan)
        if not empirical:
            nc_fit_priors_run(call.request)
    if empirical:
        # Empirical priors are the label family's ratio step (mode 1:
        # Float64(count) / Float64(n)), entered through its own boundary.
        _ = labels_class_ratio_f64(context, Int(plan.wide_counts.unsafe_ptr()), call.request.classes,
                                   call.request.rows, 1, call.request.prior)
    # The priors are published before the finite and variance refusals: a
    # refused fit leaves `class_prior_` set, as it always did.
    py_setitem(fields, "class_prior_", call.prior)
    var outcome = NcFitOutcome()
    with GILReleased(Python()):
        outcome = nc_fit_run(connection.driver, call.request, plan)
    if outcome.status != 0:
        admitted_finite_refuse(outcome.status, String(py_type(model).__name__))
    nc_check_variance(outcome.constant)
    _ = plan^
    _ = connection^
    py_setitem(fields, "centroids_", call.centroids)
    py_setitem(fields, "deviations_", call.deviations)
    py_setitem(fields, "within_class_std_dev_", call.stats)
    py_setitem(fields, "classes_", call.classes)
    py_setitem(fields, "_codes_classes", call.classes)
    py_setitem(fields, "n_features_in_", _i(call.request.features))
    _ = call^  # the rows and label codes outlive every kernel above


struct _NcQueryOperands(Movable):
    # Boundary owners only: lifetime pins of the storage the request addresses.
    var query: PythonObject
    var centroids: PythonObject
    var std: PythonObject
    var prior: PythonObject
    var request: NcQueryRequest

    def __init__(out self, query: PythonObject, centroids: PythonObject, std: PythonObject,
                 prior: PythonObject, request: NcQueryRequest):
        self.query = query
        self.centroids = centroids
        self.std = std
        self.prior = prior
        self.request = request.copy()


def _nc_query_unpack(fields: PythonObject, q: PythonObject, euclidean: Bool,
                     fused: Bool, wide: Bool) raises -> _NcQueryOperands:
    """UNPACK of a NearestCentroid query: the admitted float32 rows `q` and
    the fitted arrays as one typed request. The image's capabilities
    (`fused`, `wide`) come from its connection."""
    var cent = family_field(fields, "centroids_")
    var std = family_field(fields, "within_class_std_dev_")
    var prior = family_field(fields, "class_prior_")
    var request = NcQueryRequest(Int(py=classical_address(q)), Int(py=classical_address(cent)),
        Int(py=classical_address(std)), Int(py=classical_address(prior)), _rows(q), _cols(q), _size(prior),
        euclidean, fused, wide)
    return _NcQueryOperands(q, cent, std, prior, request)


def _nc_predict_values(model: PythonObject, binding: PythonObject, source: PythonObject,
                       context: PythonObject, labels: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var q = _f32(source, context)
    var classes = labels if labels is not None else classifier_label_list(family_field(fields, "classes_"), context)
    var connection = NcConnection(binding)
    var call = _nc_query_unpack(fields, q, _eq(family_field(fields, "metric"), PythonObject("euclidean")),
                                connection.fused, connection.wide)
    var codes = NcLabelCodes(call.request.rows)
    var status = 0
    with GILReleased(Python()):
        status = nc_labels_run(connection.driver, call.request, codes)
    if status != 0:
        admitted_finite_refuse(status, String(py_type(model).__name__))
    _ = call^
    _ = connection^
    var result = _class_array(classes, codes.values, context)
    _ = codes^  # code storage survives kernels, widening/gather and final label packing
    return result


def _nc_predict(model: PythonObject, args: PythonObject, context: PythonObject,
                labels: PythonObject) raises -> PythonObject:
    return _nc_predict_values(model, args[0], args[1], context, labels)


def _nc_scored_unpack(model: PythonObject, args: PythonObject, context: PythonObject,
                      connection: NcConnection) raises -> _NcQueryOperands:
    """UNPACK of decision_function / predict_proba: euclidean only, refused
    before the rows are admitted."""
    var fields = py_fields(model)
    if _ne(family_field(fields, "metric"), PythonObject("euclidean")):
        nc_refuse_decision_metric()
    return _nc_query_unpack(fields, _f32(args[1], context), True, connection.fused, connection.wide)


def _nc_decision(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var connection = NcConnection(args[0])
    var call = _nc_scored_unpack(model, args, context, connection)
    var n = call.request.rows
    var C = call.request.classes
    var output: PythonObject
    if C == 2:
        output = admitted_storage_native([n], "<f4", context, as_array=True)
    else:
        output = admitted_storage_native([n, C], "<f4", context, as_array=True)
    var dst = Int(py=classical_address(output))
    var status = 0
    with GILReleased(Python()):
        status = nc_decision_run(connection.driver, call.request, dst)
    if status != 0:
        admitted_finite_refuse(status, String(py_type(model).__name__))
    _ = call^
    _ = connection^
    return output


def _nc_probabilities(model: PythonObject, args: PythonObject, context: PythonObject,
                      logarithmic: Bool) raises -> PythonObject:
    var connection = NcConnection(args[0])
    var call = _nc_scored_unpack(model, args, context, connection)
    var output = admitted_storage_native([call.request.rows, call.request.classes],
        "<f8" if nc_proba_wide(call.request, logarithmic) else "<f4", context, as_array=True)
    var dst = Int(py=classical_address(output))
    var status = 0
    with GILReleased(Python()):
        status = nc_proba_run(connection.driver, call.request, dst, logarithmic)
    if status != 0:
        admitted_finite_refuse(status, String(py_type(model).__name__))
    _ = call^
    _ = connection^
    return output


def _nc_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _nc_probabilities(model, args, context, False)


def _nc_predict_log_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _nc_probabilities(model, args, context, True)


def _nc_score(model: PythonObject, args: PythonObject, context: PythonObject,
              labels: PythonObject) raises -> PythonObject:
    var prediction = _nc_predict_values(model, args[0], args[1], context, labels)
    return PythonObject(classical_accuracy(args[2], prediction, args[3], PythonObject(None), context))


# ------------------------------------------------------------- KNNImputer


def _imp_check(fields: PythonObject, context: PythonObject) raises:
    """`KNNImputer._check`."""
    var b = _b()
    var mv = family_field(fields, "missing_values")
    # The former numbers.Real test, by exact class identity: isinstance with
    # the numbers ABC runs ABCMeta.__instancecheck__ (a Python frame) on every
    # fit. Kinds 3 and 4 are the exact native integer and real scalar classes
    # (Python and connected NumPy); bool and NumPy bool_ (kind 2) are refused
    # as before.
    if (mv is None or family_isinstance(mv, Python.tuple(b.str, b.bool))
            or labels_primitive_kind(py_type(mv), context) < 3):
        imp_refuse_missing_values()
    if _ne(family_field(fields, "metric"), PythonObject("nan_euclidean")):
        imp_refuse_metric()
    if not py_contains(Python.tuple(PythonObject("uniform"), PythonObject("distance")), family_field(fields, "weights")):
        imp_refuse_weights()


def _imp_masked(binding: PythonObject, fields: PythonObject, source: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`KNNImputer._masked(X)`: a number's cells become NaN."""
    var x = _f32(source, context)
    var mv = family_field(fields, "missing_values")
    if _eq(mv, mv) and _size(x) != 0:
        var out = _out(py_call(_b().tuple, x.shape), "<f4", context)
        _op(binding, "p2m_mask_value", Python.list(classical_address(x), classical_address(out)),
            Python.list(_i(_size(x))), Python.list(PythonObject(_f32_scalar(mv))), context)
        _ = x  # keep the Python object alive through the native call above
        return out
    return x


def _imp_nan_cells(binding: PythonObject, x: PythonObject, colmiss_only: Int,
                   context: PythonObject) raises -> Tuple[PythonObject, PythonObject, Int]:
    """`KNNImputer._nan_cells(X, colmiss_only)`: (cells, colmiss, count),
    the per-column NaN counts as the int32 Array `xn_nan_cells` filled."""
    var b = _b()
    var n = _rows(x)
    var d = _cols(x)
    var lean = False
    if colmiss_only != 0:
        var probe = py_call(b.getattr, xn_module(binding), PythonObject("x_neighbors_nc_fit_lean"), PythonObject(None))
        if probe is not None:
            lean = Int(py=family_call(b.int, family_call(probe))) != 0
    var cells = _out(Python.tuple(imp_cells_length(n, d, lean)), "<i4", context)
    var colmiss = _out(Python.tuple(imp_colmiss_length(d)), "<i4", context)
    var info = _out(Python.tuple(1), "<i4", context)
    _op(binding, "nan_cells", Python.list(classical_address(x), classical_address(cells), classical_address(colmiss),
                                          classical_address(info)),
        Python.list(_i(n), _i(d), _i(colmiss_only)), Python.list(), context)
    _ = x  # keep the Python object alive through the native call above
    var count = imp_read_count(Int(py=classical_address(info)))
    _ = info  # keep the Python object alive through the read above
    return (cells, colmiss, count)


def _imp_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KNNImputer.fit(X)`."""
    var fields = py_fields(model)
    _imp_check(fields, context)
    var binding = args[0]
    var x = _imp_masked(binding, fields, args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    var found = _imp_nan_cells(binding, x, 1, context)
    var colmiss = found[1]
    var columns = imp_fit_columns(Int(py=classical_address(colmiss)), d, n)
    _ = colmiss  # keep the Python object alive through the native reads above
    var valid = Python.list()
    var miss = Python.list()
    for f in range(d):  # driver-loop: over the d feature columns (packing the fit's column masks)
        _ = py_call(valid.append, PythonObject(columns.valid[f]))
    for i in range(len(columns.missing)):  # driver-loop: over the columns missing at fit (packing)
        _ = py_call(miss.append, _i(columns.missing[i]))
    py_setitem(fields, "_valid", valid)
    py_setitem(fields, "_miss_cols", miss)
    py_setitem(fields, "_fit_X", x)
    py_setitem(fields, "n_features_in_", _i(d))


def _imp_transform(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`KNNImputer.transform(X)`."""
    var b = _b()
    var fields = py_fields(model)
    var binding = args[0]
    var x = _imp_masked(binding, fields, args[1], context)
    var n = _rows(x)
    var d = _cols(x)
    imp_check_features(d, Int(py=family_field(fields, "n_features_in_")))
    var fit_x = family_field(fields, "_fit_X")
    var m = _rows(fit_x)
    var out = _out(Python.tuple(n, d), "<f4", context)
    var k = _int_word(family_field(fields, "n_neighbors"))
    local_check_n_neighbors(k)
    var distance = 1 if _eq(family_field(fields, "weights"), PythonObject("distance")) else 0
    if _flag(context, "uncompact_impute"):
        _op(binding, "knn_impute", Python.list(classical_address(x), classical_address(fit_x), classical_address(out)),
            Python.list(_i(n), _i(m), _i(d), _i(k), _i(distance)), Python.list(), context)
    else:
        # one GPU thread per MISSING cell; a present cell keeps x
        var found = _imp_nan_cells(binding, x, 0, context)
        var cells = found[0]
        var nc = found[2]
        out = _copy_f32(x, context)
        if nc != 0:
            _op(binding, imp_cells_operation(_flag(context, "old_items")),
                Python.list(classical_address(cells), classical_address(x), classical_address(fit_x),
                            classical_address(out)),
                Python.list(_i(n), _i(m), _i(d), _i(k), _i(distance), _i(nc)), Python.list(), context)
        _ = cells  # keep the Python object alive through the native call above
    _ = fit_x  # keep the Python object alive through the native calls above
    var valid = family_field(fields, "_valid")
    var present = List[Bool]()
    var true_word = PythonObject(True)
    for f in range(d):  # driver-loop: over the d feature columns (the fit's column masks, read once)
        # Perf (perf2-neighbors): fit stores exact bools; read those by
        # identity instead of one Python `bool()` call per feature.
        var item = valid[f]
        present.append((item is true_word) if py_type(item) is b.bool else _truth(item))
    var columns = imp_keep_plan(present)
    if _truth(family_field(fields, "keep_empty_features")):
        if n != 0 and not columns.all_valid:
            var flag_array = _i32_from_ints(columns.flags, context)
            var z = _out(Python.tuple(n, d), "<f4", context)
            _op(binding, "p2m_zero_cols", Python.list(classical_address(out), classical_address(flag_array),
                                                      classical_address(z)),
                Python.list(_i(n), _i(d)), Python.list(), context)
            _ = out  # keep the Python object alive through the native call above
            _ = flag_array  # keep the Python object alive through the native call above
            out = z
    elif len(columns.keep) != d:
        # `_take_cols(out, keep)`
        var cols = _i32_from_ints(columns.keep, context)
        var rows = _rows(out)
        var taken = _out(Python.tuple(rows, len(columns.keep)), "<f4", context)
        _op(binding, "take_cols", Python.list(classical_address(out), classical_address(cols), classical_address(taken)),
            Python.list(_i(rows), _i(_cols(out)), _i(len(columns.keep))), Python.list(), context)
        _ = out  # keep the Python object alive through the native call above
        _ = cols  # keep the Python object alive through the native call above
        out = taken
    var miss = family_field(fields, "_miss_cols")
    if _truth(family_field(fields, "add_indicator")) and _truth(miss) and n != 0:
        var c = _cols(out)
        var q = len(miss)
        var miss_cols = List[Int]()
        for i in range(q):  # driver-loop: over the indicator columns (features missing at fit)
            miss_cols.append(Int(py=miss[i]))
        var miss_array = _i32_from_ints(miss_cols, context)
        var res = _out(Python.tuple(n, c + q), "<f4", context)
        var second = out if c != 0 else x
        _op(binding, "p2m_nan_indicator", Python.list(classical_address(x), classical_address(second),
                                                      classical_address(miss_array), classical_address(res)),
            Python.list(_i(n), _i(d), _i(c), _i(q)), Python.list(), context)
        _ = second  # keep the Python object alive through the native call above
        _ = miss_array  # keep the Python object alive through the native call above
        out = res
    _ = x  # keep the Python object alive through the native calls above
    return out


# ------------------------------------------------------------------ door


def _neighbors_local_operation(model: PythonObject, operation: String, var args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var labels = _admit_local_state(model, context, operation.endswith("_fit") or "_fit_" in operation)
    for i in range(1, len(args)):  # small-loop(public data arguments, before native preparation)
        if host_sparse_format(args[i]).byte_length() == 0:
            classical_admit_value(args[i], context)
    args = xn_admit_arguments(args, context)
    if operation == "family_xn_loc_lof_fit":
        _ = _lof_fit(model, args, context)
        return model
    if operation == "family_xn_loc_lof_fit_predict":
        return _lof_fit_predict(model, args, context)
    if operation == "family_xn_loc_lof_score_samples":
        var fields = py_fields(model)
        _lof_novelty(fields, "score_samples")
        return _lof_score_samples(args[0], fields, args[1], context, String(py_type(model).__name__))
    if operation == "family_xn_loc_lof_decision_function":
        return _lof_decision(model, args, context)
    if operation == "family_xn_loc_lof_predict":
        return _lof_predict(model, args, context)
    if operation == "family_xn_loc_nc_fit":
        _nc_fit(model, args, context)
        return model
    if operation == "family_xn_loc_nc_predict":
        return _nc_predict(model, args, context, labels)
    if operation == "family_xn_loc_nc_decision_function":
        return _nc_decision(model, args, context)
    if operation == "family_xn_loc_nc_predict_proba":
        return _nc_predict_proba(model, args, context)
    if operation == "family_xn_loc_nc_predict_log_proba":
        return _nc_predict_log_proba(model, args, context)
    if operation == "family_xn_loc_nc_score":
        return _nc_score(model, args, context, labels)
    if operation == "family_xn_loc_imp_fit":
        _imp_fit(model, args, context)
        return model
    if operation == "family_xn_loc_imp_transform":
        return _imp_transform(model, args, context)
    if operation == "family_xn_loc_imp_fit_transform":
        _imp_fit(model, args, context)
        return _imp_transform(model, args, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_local_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    var call = neighbors_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_local_operation(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return neighbors_public_finish(model, operation, result, call, context)
