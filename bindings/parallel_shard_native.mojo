# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The native owner of the Python-sharded multi-GPU drivers
(lane/native-parallel-driver-20261024,
docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md, "kind B"):
`parallel_classical.transform_rbf_sampler`, `parallel_preprocessing.fit_scaler`
and `transform_scaler`, `parallel_gaussian_process.fit_gaussian_process_classifier`
and `predict_gaussian_process_classifier`, and
`parallel_ensemble.fit_forest`, one call each into `parallel_shard_call`.

Each driver's partition, request order, driver negative control and merge
are its old Python driver's (bindings/parallel_shard_common.mojo for the
shared parts): the shards run their plain native owners in worker processes
on the native local process pool, one per device, and every shard's block is
placed at its own offset in shard order. The same compiled entries on the
same words: no bit moves.

The worker (`mojolearn._native_parallel_shard_worker`) imports the package,
whose driver modules register their classes, entries and contexts here at
import, and enters `shard_worker`. A worker rebuilds each shard's model from
its state record without a Python constructor. Its one Python call is the
binding getter (`_backend.binding`) the package registered, once per request,
to resolve the binding a plain owner takes as its first argument; the parent
never calls it (its shell resolves the bindings).
"""
from std.ffi import _Global
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.family_common import family_call, family_as_f32, family_view
from bindings.native_process_pool import (
    process_worker_connect, process_worker_next, process_send, process_pool_create, process_pool_map,
    process_pool_close,
)
from bindings.native_transport import transport_close
from bindings.native_codec import codec_initialize
from bindings.parallel_shard_common import (
    shard_b, shard_ranges, shard_devices, shard_read_shift, shard_run, shard_rows, shard_concat, shard_state,
    shard_model, shard_array, shard_join_rows, shard_row_vector, shard_open, shard_identity, shard_trace_take, shard_map, shard_faults_take,
)
from bindings.kernel_methods_native import _rbf_transform
from bindings.gp_classifier_native import (
    gpc_validate, _gpc_fit_binary, _gpc_binary, _gpc_set_fitted, _gpc_latent, _gpc_ovr_combine,
)
from bindings.gp_common_native import gp_kernel_arrays
from bindings.gp_labels_native import gp_encode_labels, gp_decode_labels
from core.parallel_forecast_impl import forecast_gather_cols, forecast_scatter_cols
from core.hotpath_impl import cast_elements, HP_I32, HP_I64, HP_U32

comptime _CHILD = "mojolearn._native_parallel_shard_worker"


struct ParallelShardBoundary(Defaultable, Movable):
    var tables: PythonObject

    def __init__(out self):
        self.tables = PythonObject(None)


comptime PARALLEL_SHARD_BOUNDARY = _Global[
    StorageType=ParallelShardBoundary, name="MojoParallelShardOwner",
    init_fn=ParallelShardBoundary.__init__,
]


def _t(name: String) raises -> PythonObject:
    """A registered table entry (a class, an entry, a context or a getter)."""
    var s = PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()
    if s[].tables is None:
        raise Error("RuntimeError: the parallel shard owner is not connected")
    var value = py_get(s[].tables, PythonObject(name), PythonObject(None))
    if value is None:
        raise Error("RuntimeError: the parallel shard owner has no " + name + "; import its driver module")
    return value


def _ctx() raises -> PythonObject:
    """The family owners' shared context (array types, codes, classes)."""
    return _t("context")


def _worker_binding(name: String) raises -> PythonObject:
    """The identical binding `name`, in a worker: the registered getter."""
    return py_call(_t("binding_getter"), PythonObject(name), PythonObject("identical"))


def _identical(fields: PythonObject) raises -> Bool:
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    return mode is None or (py_type(mode) is shard_b().str and String(mode) == "identical")


# ---------------------------------------------------------------- RBFSampler

def _transform_rbf_sampler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, rows_per_shard, binding): whole query rows
    in `rows_per_shard` blocks; each output row is a pure function of its
    input row, so blocks are joined by copying bytes in input order."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    if py_type(estimator) is not _t("rbf_class"):
        raise Error("TypeError: requires mojolearn.RBFSampler")
    var fields = py_fields(estimator)
    if not _identical(fields):
        raise Error("ValueError: parallel RBFSampler requires IDENTICAL numeric mode")
    if not py_contains(fields, PythonObject("random_weights_")):
        raise Error("ValueError: RBFSampler is not fitted")
    var rps = args[3]
    if py_type(rps) is not b.int or Int(py=rps) < 1:
        raise Error("ValueError: rows_per_shard must be a positive int")
    var x = family_as_f32(args[1], 2, "X", context)
    var d = Int(py=x.shape[1])
    if d != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: X has " + String(d) + " features, the fit had " + String(Int(py=fields["n_features_in_"])))
    var rows = Int(py=x.shape[0])
    if rows == 0:
        return _rbf_transform(estimator, Python.tuple(args[4], x), context)
    var n_devices = shard_devices(args[2])
    var ranges = shard_ranges(rows, rps, "rows_per_shard", "")
    var state = shard_state(fields, b.frozenset(Python.list("_resident_holder")), context)
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per row block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        _ = py_call(requests.append, Python.tuple("rbf_rows", state, Python.tuple(shard_rows(x, lo - shift, hi - shift, context))))
    var parts = shard_run(requests, args[2], _CHILD)
    var q = Int(py=fields["random_weights_"].shape[1])
    for r in range(len(parts)):  # driver-loop: checks one result block per row shard
        var part = parts[r]
        if Int(py=part.ndim) != 2 or Int(py=part.shape[1]) != q:
            raise Error("ValueError: RBFSampler worker returned an invalid result shape")
    return shard_join_rows(parts, rows, context)


# ---------------------------------------------------------------- scalers

def _scaler_kind(estimator: PythonObject) raises -> String:
    var cls = py_type(estimator)
    if cls is _t("standard_class"):
        return "standard"
    if cls is _t("minmax_class"):
        return "minmax"
    raise Error("TypeError: parallel preprocessing requires MinMaxScaler or StandardScaler")


def _scaler_stats(kind: String) raises -> PythonObject:
    if kind == "minmax":
        return Python.list("data_min_", "data_max_", "data_range_", "scale_", "min_")
    return Python.list("mean_", "var_", "scale_")


def _scaler_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ScalerProtocol._input(X)`'s refusals (float32, nonempty 2-D, the
    Int32 bound), then the C-contiguous float32 Array. The finiteness scan
    is the plain owners' (each shard's fit or transform)."""
    var view = family_view(source, -1, "input", context)
    var b = shard_b()
    var listed = py_type(source) is b.list or py_type(source) is b.tuple
    if not listed and String(py=view.dtype) != "<f4":
        raise Error("TypeError: Scaler input must have dtype float32")
    if Int(py=view.ndim) != 2 or Int(py=view.shape[0]) == 0 or Int(py=view.shape[1]) == 0:
        raise Error("ValueError: Scaler requires a nonempty two-dimensional input")
    if Int(py=view.shape[0]) * Int(py=view.shape[1]) > 2147483647:
        raise Error("ValueError: Scaler exceeds the native Int32 indexing bound")
    return family_as_f32(source, 2, "input", context)


def _columns(data: PythonObject, lo: Int, hi: Int, context: PythonObject) raises -> PythonObject:
    """`data[:, lo:hi]` of a C-ordered (n, d) float32 Array, as a C-ordered copy."""
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    var out = shard_array("<f4", Python.tuple(n, hi - lo), n * (hi - lo), context)
    if n > 0 and hi > lo:
        forecast_gather_cols(Int(py=classical_address(out)), Int(py=classical_address(data)), n, d, lo, hi - lo)
        _ = out  # keep the Python object alive through the native call above
        _ = data  # keep the Python object alive through the native call above
    return out


def _fit_scaler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, columns_per_shard, sample_weight): whole
    columns in `columns_per_shard` blocks; each statistic is per column, so
    every shard's statistics are placed at its column offset."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var kind = _scaler_kind(estimator)
    var names = _scaler_stats(kind)
    var fields = py_fields(estimator)
    if not _identical(fields):
        raise Error("ValueError: parallel scalers require IDENTICAL numeric mode")
    if args[4] is not None:
        raise Error("NotImplementedError: scalers do not implement sample_weight")
    var data = _scaler_input(args[1], context)
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    var ranges = shard_ranges(d, args[3], "columns_per_shard", "")
    # `estimator.get_params()` with numeric_mode='identical'
    var init = b.list()
    if kind == "minmax":
        init = Python.list(fields["feature_range"], fields["copy"], fields["clip"], "identical")
    else:
        init = Python.list(fields["copy"], fields["with_mean"], fields["with_std"], "identical")
    var n_devices = shard_devices(args[2])
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per column block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shift = shard_read_shift(r, lo, n_devices)
        _ = py_call(requests.append, Python.tuple("scaler_fit", Python.tuple(kind, py_call(b.tuple, init)),
                                                  Python.tuple(_columns(data, lo - shift, hi - shift, context))))
    var parts = shard_run(requests, args[2], _CHILD)
    var merged = b.dict()
    for k in range(len(names)):  # small-loop(the scaler's per-column statistics)
        var name = names[k]
        if parts[0][name] is None:
            py_setitem(merged, name, PythonObject(None))
            continue
        var out = shard_array("<f4", Python.tuple(d), d, context)
        for r in range(len(ranges) // 2):  # driver-loop: places one shard's statistic block
            var lo = ranges[2 * r]
            var w = ranges[2 * r + 1] - lo
            var value = parts[r][name]
            memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))) + lo,
                   src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(value))),
                   count=w)
            _ = out  # keep the Python object alive through the native call above
            _ = value  # keep the Python object alive through the native call above
        py_setitem(merged, name, out)
    # `result = type(estimator)(**params); result.__dict__.update(parts[0].__dict__)`
    _ = py_call(fields.clear)
    _ = py_call(fields.update, parts[0])
    _ = py_call(fields.update, merged)
    py_setitem(fields, "n_features_in_", d)
    py_setitem(fields, "n_samples_seen_", n)
    return estimator


def _transform_scaler(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, columns_per_shard, inverse): every column
    block transformed with its columns' statistics, the blocks joined at
    their column offsets."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var kind = _scaler_kind(estimator)
    var names = _scaler_stats(kind)
    var entry = _t(kind + "_call")
    var owner_context = _t(kind + "_context")
    if not Bool(py=py_call(b.bool, family_call(entry, estimator, PythonObject("is_fitted"), Python.tuple(), owner_context))):
        raise Error("ValueError: scaler is not fitted")
    var fields = py_fields(estimator)
    var fitted_mode = py_get(fields, PythonObject("numeric_mode_"), PythonObject(None))
    if fitted_mode is None or py_type(fitted_mode) is not b.str or String(fitted_mode) != "identical":
        raise Error("ValueError: parallel scaler transforms require an IDENTICAL fit")
    var inverse = args[4]
    if py_type(inverse) is not b.bool:
        raise Error("ValueError: inverse must be a bool")
    var data = _scaler_input(args[1], context)
    var n = Int(py=data.shape[0])
    var d = Int(py=data.shape[1])
    if d != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: input feature count differs from fit")
    var ranges = shard_ranges(d, args[3], "columns_per_shard", "")
    var base = shard_state(fields, b.frozenset(), context)
    var n_devices = shard_devices(args[2])
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per column block
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var part = py_call(b.dict, base)
        py_setitem(part, "n_features_in_", hi - lo)
        for k in range(len(names)):  # small-loop(the scaler's per-column statistics)
            var value = fields[names[k]]
            py_setitem(part, names[k], PythonObject(None) if value is None else shard_rows(value, lo, hi, context))
        var shift = shard_read_shift(r, lo, n_devices)
        _ = py_call(requests.append, Python.tuple("scaler_transform", Python.tuple(kind, part),
                                                  Python.tuple(_columns(data, lo - shift, hi - shift, context), inverse)))
    var parts = shard_run(requests, args[2], _CHILD)
    var out = shard_array("<f4", Python.tuple(n, d), n * d, context)
    for r in range(len(ranges) // 2):  # driver-loop: places one column block
        var lo = ranges[2 * r]
        var w = ranges[2 * r + 1] - lo
        var part = parts[r]
        if Int(py=part.shape[0]) != n or Int(py=part.shape[1]) != w:
            raise Error("ValueError: scaler worker returned an invalid block shape")
        forecast_scatter_cols(Int(py=classical_address(out)), Int(py=classical_address(part)), n, d, lo, w)
        _ = out  # keep the Python object alive through the native call above
        _ = part  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- GaussianProcessClassifier

def _gpc_admit(estimator: PythonObject, context: PythonObject) raises -> PythonObject:
    """`parallel_gaussian_process._validate`: the class, IDENTICAL, then the
    native parameter validation on a copy (`copy.copy(estimator)._validate_params()`),
    which is returned: it carries the validated `_kernel` and
    `_max_iter_predict` while the estimator itself is untouched until a fit
    publishes."""
    if py_type(estimator) is not _t("gpc_class"):
        raise Error("TypeError: requires mojolearn.GaussianProcessClassifier")
    var fields = py_fields(estimator)
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var b = shard_b()
    if mode is None or not Bool(py=py_call(b.bool, mode)):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str or String(mode) != "identical":
        raise Error("ValueError: parallel GPC requires IDENTICAL numeric mode")
    var work = shard_model(py_type(estimator), py_call(b.dict, fields))
    gpc_validate(work, context)
    return work


def _fit_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, y, devices, gp binding, base binding): one binary
    Laplace fit per one-vs-rest class column (class 1 alone for two
    classes), each on a worker, stored as the plain fit stores them."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var work = py_fields(_gpc_admit(estimator, context))
    var fields = py_fields(estimator)
    var source = family_as_f32(args[1], 2, "X", context)
    var copied = not (source is args[1])
    var x = source
    var encoded = gp_encode_labels(args[2], args[5], context)
    var classes = encoded[0]
    var codes = encoded[1]
    if Int(py=codes.size) != Int(py=x.shape[0]):
        raise Error("ValueError: y length differs from X rows")
    if len(classes) < 2:
        raise Error("ValueError: GaussianProcessClassifier requires at least two classes")
    var columns = b.list()
    if len(classes) == 2:
        _ = py_call(columns.append, PythonObject(1))
    else:
        for k in range(len(classes)):  # driver-loop: over the one-vs-rest classes
            _ = py_call(columns.append, PythonObject(k))
    var arrays = gp_kernel_arrays(work["_kernel"], context)
    var carried = b.list()
    for i in range(len(arrays)):  # small-loop(the kernel's five packed arrays and words)
        _ = py_call(carried.append, arrays[i])
    var max_iter = work["_max_iter_predict"]
    _ = shard_devices(args[3])
    var requests = b.list()
    for i in range(len(columns)):  # driver-loop: one worker request per class column
        _ = py_call(requests.append, Python.tuple("gpc_class_fit", Python.tuple(carried, max_iter),
                                                  Python.tuple(x, codes, columns[i])))
    var parts = shard_run(requests, args[3], _CHILD)
    var fits = b.list()
    for i in range(len(parts)):  # driver-loop: one binary fit per class column
        var f = parts[i]
        _ = py_call(fits.append, _gpc_binary(context, f["y_train_"], f["L_"], f["pi_"], f["W_sr_"],
                                             Float64(py=f["log_marginal_likelihood_value_"]), Int(py=f["n_iter_"]),
                                             Int(py=f["nb_"])))
    # published only now that every class fitted: `_fresh(estimator)` carried
    # the validated kernel, max_iter_predict and IDENTICAL
    py_setitem(work, "numeric_mode", "identical")
    py_setitem(work, "input_copied_", copied)
    _gpc_set_fitted(work, x, classes, fits, context)
    _ = py_call(fields.update, work)
    return estimator


def _predict_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, method, gp binding, base binding): one
    latent pass per fitted class on a worker; two classes take the binding's
    decoded codes or [1 - p, p] rows from the one fit, more take the
    one-vs-rest combination of the class probability columns."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var work = py_fields(_gpc_admit(estimator, context))
    var method = args[3]
    if py_type(method) is not b.str or (String(method) != "predict" and String(method) != "predict_proba"):
        raise Error("ValueError: method must be predict or predict_proba")
    var fields = py_fields(estimator)
    if not py_contains(fields, PythonObject("estimators_")):
        raise Error("ValueError: GaussianProcessClassifier: call fit() or load() first")
    var q = family_as_f32(args[1], 2, "X", context)
    if Int(py=q.shape[1]) != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: GaussianProcessClassifier: X has " + String(Int(py=q.shape[1])) + " features, fit saw "
                    + String(Int(py=fields["n_features_in_"])))
    var proba_method = String(method) == "predict_proba"
    var n_classes = Int(py=fields["n_classes_"])
    var want_proba = proba_method or n_classes > 2
    var out_kind = 0
    if n_classes == 2:
        out_kind = 2 if proba_method else 1
    var arrays = gp_kernel_arrays(work["_kernel"], context)
    var carried = b.list()
    for i in range(len(arrays)):  # small-loop(the kernel's five packed arrays and words)
        _ = py_call(carried.append, arrays[i])
    var fits = fields["estimators_"]
    _ = shard_devices(args[2])
    var requests = b.list()
    for i in range(len(fits)):  # driver-loop: one worker request per fitted class
        var fit_state = shard_state(py_fields(fits[i]), b.frozenset(), context)
        _ = py_call(requests.append, Python.tuple("gpc_class_predict", Python.tuple(carried, fields["X_train_"], fit_state),
                                                  Python.tuple(q, want_proba, out_kind)))
    var columns = shard_run(requests, args[2], _CHILD)
    var n_star = Int(py=q.shape[0])
    for i in range(len(columns)):  # driver-loop: checks one result column per class
        if Int(py=columns[i].shape[0]) != n_star:
            raise Error("ValueError: GPC worker returned an invalid row count")
    if n_classes == 2:
        if proba_method:
            return columns[0]
        return gp_decode_labels(fields["classes_"], columns[0], args[5], context)
    var combined = _gpc_ovr_combine(args[4], columns, n_star, context)
    if proba_method:
        return combined[0]
    # `codes32.astype("<i8")`: the exact widening
    var codes32 = combined[1]
    var codes = shard_array("<i8", Python.tuple(n_star), n_star, context)
    _ = cast_elements(Int(py=classical_address(codes32)), HP_I32, Int(py=classical_address(codes)), HP_I64, n_star)
    _ = codes32  # keep the Python object alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    return gp_decode_labels(fields["classes_"], codes, args[5], context)


# ---------------------------------------------------------------- forests

def _forest_y(y: PythonObject, regressor: Bool, context: PythonObject) raises -> PythonObject:
    """The targets a forest shard receives: float32 for a regressor (the old
    driver's `as_f32_c(y, ndim=1)`); a classifier's labels as given (a native
    Array or a list), or their `tolist()` (the old driver's flat label list)
    when the codec cannot carry the object itself."""
    if regressor:
        return family_as_f32(y, 1, "y", context)
    var b = shard_b()
    var cls = py_type(y)
    if cls is context["array_class"] or cls is b.list or cls is b.tuple:
        return y
    return py_call(py_attr(y, "tolist"))


def _fit_forest(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, y, devices, trees_per_shard): whole global tree-ID
    ranges, each a fresh forest of the estimator's parameters fitted from
    its first global tree ID; the trees joined in ID order, their node
    offsets rebased natively."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    var cls = py_type(estimator)
    var classes = _t("forest_classes")
    var regressors = _t("forest_regressors")
    if not Bool(py=py_call(py_attr(classes, "__contains__"), cls)):
        raise Error("TypeError: fit_forest requires a mojolearn RandomForest or ExtraTrees estimator")
    var tps = args[4]
    if py_type(tps) is not b.int or Int(py=tps) < 1:
        raise Error("ValueError: trees_per_shard must be a positive integer")
    var entry = _t("trees_entry")
    var trees_context = _t("trees_context")
    var params = family_call(entry, estimator, PythonObject("trees_forest_get_params"), Python.tuple(False), trees_context)
    var mode = py_get(params, PythonObject("numeric_mode"), PythonObject(None))
    if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
        raise Error("ValueError: parallel forests require IDENTICAL numeric mode")
    py_setitem(params, "numeric_mode", "identical")
    var x = family_as_f32(args[1], 2, "X", context)
    var regressor = Bool(py=py_call(py_attr(regressors, "__contains__"), cls))
    var y = _forest_y(args[2], regressor, context)
    var count = Int(py=params["n_estimators"])
    if count < 1:
        raise Error("ValueError: n_estimators must be positive")
    var ranges = shard_ranges(count, tps, "trees_per_shard", "")
    var n_devices = shard_devices(args[3])
    var requests = b.list()
    for r in range(len(ranges) // 2):  # driver-loop: one worker request per tree range
        var lo = ranges[2 * r]
        var hi = ranges[2 * r + 1]
        var shard_params = py_call(b.dict, params)
        py_setitem(shard_params, "n_estimators", hi - lo)
        var shift = shard_read_shift(r, lo, n_devices)
        _ = py_call(requests.append, Python.tuple("forest_fit", Python.tuple(cls.__name__, shard_params),
                                                  Python.tuple(x, y, lo - shift)))
    var parts = shard_run(requests, args[3], _CHILD)
    var merged = b.dict()
    # node offsets: one rebased table over every shard's trees
    var offsets = b.list()
    for r in range(len(parts)):  # driver-loop: one offset table per tree range
        _ = py_call(offsets.append, parts[r]["_offsets"])
    py_setitem(merged, "_offsets", _rebase_offsets(offsets, context))
    var node_names = Python.list("_colid", "_quesval", "_left_child", "_leaves")
    for k in range(len(node_names)):  # small-loop(the four flat node arrays)
        var name = String(node_names[k])
        var total = 0
        for r in range(len(parts)):  # driver-loop: one node array per tree range
            total += Int(py=parts[r][name].shape[0])
        py_setitem(merged, name, shard_concat(parts, name, total, context))
    # `result = type(estimator)(**params); result.__dict__.update(parts[0].__dict__)`
    var fields = py_fields(estimator)
    _ = py_call(fields.clear)
    _ = py_call(fields.update, params)
    py_setitem(fields, "_has_constructor_parameters", True)
    _ = py_call(fields.update, parts[0])
    _ = py_call(fields.update, merged)
    py_setitem(fields, "_n_trees", count)
    py_setitem(fields, "n_estimators", count)
    var cfg = py_get(fields, PythonObject("_cfg"), PythonObject(None))
    if cfg is not None:
        cfg = py_call(b.dict, cfg)
        py_setitem(cfg, "n_estimators", count)
        py_setitem(fields, "_cfg", cfg)
    if py_contains(fields, PythonObject("depth_cap_bound_")):
        var bound = False
        for r in range(len(parts)):  # driver-loop: one flag per tree range
            if Bool(py=py_call(b.bool, py_get(parts[r], PythonObject("depth_cap_bound_"), PythonObject(False)))):
                bound = True
        py_setitem(fields, "depth_cap_bound_", bound)
    # `result._resident_forest = None`: a fresh holder with no device snapshot
    var holder = b.dict()
    py_setitem(holder, "forest", PythonObject(None))
    py_setitem(fields, "_resident_holder", holder)
    return estimator


def _rebase_offsets(arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    """`parallel_ensemble._merge_offsets`: the shards' int32 node-offset
    tables joined, then rebased by the native `msel_rebase_offsets_i32` into
    one table of 1 + sum(sizes) - len(shards) entries."""
    var b = shard_b()
    var parts = b.list()
    var n_raw = 0
    for p in range(len(arrays)):  # driver-loop: one offset table per tree range
        var a = arrays[p]
        if String(py=a.dtype) != "<i4":
            raise Error("ValueError: forest worker returned non-int32 node offsets")
        _ = py_call(parts.append, a)
        n_raw += Int(py=a.shape[0])
    var raw = shard_join_rows(parts, n_raw, context)
    var lens = shard_array("<i8", Python.tuple(len(parts)), len(parts), context)
    var lp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(lens)))
    for p in range(len(parts)):  # driver-loop: one size word per tree range
        lp[p] = Int64(Int(py=parts[p].shape[0]))
    _ = lens  # keep the Python object alive through the native call above
    var n_out = 1 + n_raw - len(parts)
    var out = shard_array("<i4", Python.tuple(n_out), n_out, context)
    _ = family_call(_t("rebase_offsets_entry"), classical_address(raw), PythonObject(n_raw), classical_address(lens),
                    PythonObject(len(parts)), PythonObject(n_out), classical_address(out))
    _ = raw  # keep the Python object alive through the native call above
    _ = lens  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- persistent pools
# ReferenceShardedNeighbors, ParallelQueries and DistributedIVFIndex keep one
# native local pool for their lifetime (the old `DevicePool` they held): the
# workers start at the first request and keep their snapshots until close.
# Their per-shard work is the plain estimators' own query methods, which are
# not one native call each yet (python/mojolearn/_parallel_shard_ops.py);
# the partition, scheduling, merge and join here are native.

def _pool_open(devices: PythonObject) raises -> PythonObject:
    _ = shard_devices(devices)
    return shard_open(devices, _CHILD)


def _map(pool: PythonObject, requests: PythonObject) raises -> PythonObject:
    var result = shard_map(pool, requests)
    if len(result) != len(requests):
        raise Error("ValueError: shard workers returned an incomplete batch")
    return result


def _py_request(name: String, args: PythonObject) raises -> PythonObject:
    return Python.tuple("py", name, args)


def _join(parts: PythonObject, ragged: Bool, context: PythonObject) raises -> PythonObject:
    """`parallel_neighbors._join`: tuples joined component by component,
    lists either chained (ragged) or joined position by position, Arrays
    joined along their first axis in shard order."""
    var b = shard_b()
    var first = parts[0]
    if py_type(first) is b.tuple:
        var out = b.list()
        for i in range(len(first)):  # small-loop(the result tuple's components)
            var column = b.list()
            for p in range(len(parts)):  # driver-loop: one component per query shard
                _ = py_call(column.append, parts[p][i])
            _ = py_call(out.append, _join(column, ragged, context))
        return py_call(b.tuple, out)
    if py_type(first) is b.list:
        var out = b.list()
        if ragged:
            for p in range(len(parts)):  # driver-loop: one ragged row list per query shard
                _ = py_call(out.extend, parts[p])
            return out
        for i in range(len(first)):  # driver-loop: over the per-output result columns
            var column = b.list()
            for p in range(len(parts)):  # driver-loop: one column per query shard
                _ = py_call(column.append, parts[p][i])
            _ = py_call(out.append, _join(column, False, context))
        return out
    var total = 0
    for p in range(len(parts)):  # driver-loop: one block per query shard
        total += Int(py=parts[p].shape[0])
    return shard_join_rows(parts, total, context)


def _reference_query(args: PythonObject) raises -> PythonObject:
    """args (sharded, X, method, n_neighbors, return_distance, merge entry):
    `ReferenceShardedNeighbors._query`. Reference rows in blocks of
    `reference_rows_per_shard` (or an even split over the devices), queried
    in waves of len(devices) per block of `query_rows_per_shard` query rows;
    every block's candidates merged per query row by the native
    `shard_topk_merge_f32`; predictions voted over the merged neighbors on
    the first worker."""
    var b = shard_b()
    var context = _ctx()
    var holder = py_fields(args[0])
    var model = holder["estimator"]
    var pool = holder["_pool"]
    var method = String(args[2])
    var fields = py_fields(model)
    var cls = py_type(model)
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var algorithm = py_get(fields, PythonObject("algorithm"), PythonObject(None))
    if (not _identical(fields) or py_type(algorithm) is not b.str
            or (String(algorithm) != "auto" and String(algorithm) != "brute")):
        raise Error("ValueError: reference shards require IDENTICAL brute-force KNN")
    _ = mode
    var index = py_get(fields, PythonObject("_index"), PythonObject(None))
    if index is None:
        raise Error("ValueError: fit the estimator before querying")
    if method != "kneighbors":
        if cls is not _t("knc_class") and cls is not _t("knr_class"):
            raise Error("TypeError: prediction requires a classifier or regressor")
        if method == "predict_proba" and cls is not _t("knc_class"):
            raise Error("TypeError: predict_proba requires a classifier")
        if py_get(fields, PythonObject("_y_cols"), PythonObject(None)) is None:
            raise Error("ValueError: fit the estimator before prediction")
    var data = family_as_f32(args[1], 2, "X", context)
    var d = Int(py=fields["n_features_in_"])
    if Int(py=data.shape[1]) != d:
        raise Error("ValueError: input feature count differs from fit")
    var n = Int(py=fields["n_samples_fit_"])
    var k_obj = fields["n_neighbors"] if args[3] is None else args[3]
    if py_type(k_obj) is not b.int or Int(py=k_obj) < 1 or Int(py=k_obj) > n:
        raise Error("ValueError: n_neighbors must be an integer in [1, reference rows]")
    var k = Int(py=k_obj)
    if n > 2147483647:
        raise Error("ValueError: reference row count exceeds the native signed-int32 shape contract")
    var cell_limit = 2147483647 // d
    if cell_limit < 1:
        raise Error("ValueError: feature count exceeds the native signed-int32 shape contract")
    var devices = pool["devices"]
    var n_devices = len(devices)
    var width_obj = holder["reference_rows_per_shard"]
    var width = min(cell_limit, (n + n_devices - 1) // n_devices) if width_obj is None else Int(py=width_obj)
    if min(width, n) > cell_limit:
        raise Error("ValueError: reference shard exceeds the native signed-int32 cell-count contract")
    var ranges = shard_ranges(n, PythonObject(width), "reference_rows_per_shard", "")
    var params = b.dict()
    py_setitem(params, "n_neighbors", k)
    py_setitem(params, "query_tile", fields["query_tile"])
    py_setitem(params, "metric", fields["metric"])
    py_setitem(params, "algorithm", "brute")
    py_setitem(params, "p", fields["p"])
    py_setitem(params, "numeric_mode", "identical")
    var vote_state = PythonObject(None)
    if method != "kneighbors":
        vote_state = shard_state(fields, b.frozenset(Python.list("_index", "_resident", "_resident_holder")), context)
        py_setitem(vote_state, "_index", PythonObject(None))
        py_setitem(vote_state, "numeric_mode", "identical")
    var rows = Int(py=data.shape[0])
    var step = Int(py=holder["query_rows_per_shard"])
    var n_ranges = len(ranges) // 2
    var outputs = b.list()
    var diagnostics = b.list()
    var first = 0
    while first < rows:  # driver-loop: over the query row blocks
        var end = min(rows, first + step)
        var query = shard_rows(data, first, end, context)
        var parts = b.list()
        var wave = 0
        while wave < n_ranges:  # driver-loop: over waves of len(devices) reference blocks
            var requests = b.list()
            for r in range(wave, min(wave + n_devices, n_ranges)):  # driver-loop: one request per reference block
                var start = ranges[2 * r]
                var stop = ranges[2 * r + 1]
                var shift = shard_read_shift(r, start, n_devices)
                _ = py_call(requests.append, _py_request("reference_shard", Python.tuple(
                    params, shard_rows(index, start - shift, stop - shift, context), query, min(k, stop - start))))
            _ = py_call(parts.extend, _map(pool, requests))
            wave += n_devices
        var merged = _merge_candidates(parts, ranges, end - first, k, args[5], context)
        var output = PythonObject(None)
        if method == "kneighbors":
            var wide = shard_array("<i8", Python.tuple(end - first, k), (end - first) * k, context)
            _ = cast_elements(Int(py=classical_address(merged[1])), HP_U32, Int(py=classical_address(wide)), HP_I64,
                              (end - first) * k)
            _ = wide  # keep the Python object alive through the native call above
            output = Python.tuple(merged[0], wide) if Bool(py=py_call(b.bool, args[4])) else wide
        else:
            var vote = Python.list(_py_request("reference_vote", Python.tuple(cls.__name__, vote_state, merged[0],
                                                                               merged[1], method)))
            output = _map(pool, vote)[0]
        _ = py_call(outputs.append, output)
        for r in range(n_ranges):  # driver-loop: one diagnostic record per reference block
            var record = b.dict()
            py_setitem(record, "query_start", first)
            py_setitem(record, "query_end", end)
            py_setitem(record, "reference_start", ranges[2 * r])
            py_setitem(record, "reference_end", ranges[2 * r + 1])
            py_setitem(record, "reference_bytes", (ranges[2 * r + 1] - ranges[2 * r]) * d * 4)
            py_setitem(record, "device", devices[r % n_devices])
            _ = py_call(diagnostics.append, record)
        first = end
    var result = _join(outputs, False, context)
    py_setitem(holder, "last_shards_", diagnostics)
    return result


def _merge_candidates(parts: PythonObject, ranges: List[Int], n_queries: Int, k: Int, entry: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`parallel_neighbors_reference._merge`: (float32 distances, uint32
    global indices), each (n_queries, k), from the reference blocks'
    candidates by the native `shard_topk_merge_f32`."""
    var b = shard_b()
    var distances = shard_array("<f4", Python.tuple(n_queries, k), n_queries * k, context)
    var indices = shard_array("<u4", Python.tuple(n_queries, k), n_queries * k, context)
    if n_queries == 0:
        return Python.tuple(distances, indices)
    var count = len(parts)
    var table = shard_array("<i8", Python.tuple(5 * count), 5 * count, context)
    var tp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(table)))
    var keep = b.list()
    for p in range(count):  # driver-loop: one candidate table row per reference block
        var dist = family_as_f32(parts[p][0], 2, "distance shard", context)
        var ids = parts[p][1]
        if String(py=ids.dtype) != "<i8":
            var wide = shard_array("<i8", py_call(b.tuple, ids.shape), Int(py=ids.size), context)
            var code = HP_I32 if String(py=ids.dtype) == "<i4" else HP_U32
            _ = cast_elements(Int(py=classical_address(ids)), code, Int(py=classical_address(wide)), HP_I64,
                              Int(py=ids.size))
            _ = wide  # keep the Python object alive through the native call above
            ids = wide
        _ = py_call(keep.append, Python.tuple(dist, ids))
        tp[5 * p] = Int64(Int(py=classical_address(dist)))
        tp[5 * p + 1] = Int64(Int(py=classical_address(ids)))
        tp[5 * p + 2] = Int64(ranges[2 * p])
        tp[5 * p + 3] = Int64(ranges[2 * p + 1])
        tp[5 * p + 4] = Int64(Int(py=dist.shape[1]))
        _ = table  # keep the Python object alive through the native call above
    var rc = Int(py=family_call(entry, classical_address(table), PythonObject(count), PythonObject(n_queries),
                                PythonObject(k), classical_address(distances), classical_address(indices)))
    _ = keep  # keep the Python objects alive through the native call above
    _ = table  # keep the Python object alive through the native call above
    if rc == 1:
        raise Error("RuntimeError: reference worker returned an invalid neighbor index")
    if rc == 2:
        raise Error("RuntimeError: reference shards returned too few neighbors")
    return Python.tuple(distances, indices)


def _parallel_queries(args: PythonObject) raises -> PythonObject:
    """args (queries object, X, method, kwargs): `ParallelQueries.query`.
    Every worker stores one snapshot of the estimator under a fresh token;
    whole query rows in `rows_per_shard` blocks are answered by the plain
    query method (block i on worker i % len(devices)); the snapshots are
    released; the blocks joined in input order."""
    var b = shard_b()
    var context = _ctx()
    var holder = py_fields(args[0])
    var model = holder["estimator"]
    var pool = holder["_pool"]
    var fields = py_fields(model)
    var methods = py_get(_t("query_methods"), py_type(model), Python.tuple())
    var method = methods[0] if args[2] is None else args[2]
    if len(methods) == 0:
        raise Error("TypeError: parallel queries require an exact neighbors or KernelDensity estimator")
    if not Bool(py=py_call(py_attr(methods, "__contains__"), method)):
        raise Error("ValueError: unsupported parallel query method: " + String(py=py_call(b.str, method)))
    if not _identical(fields):
        raise Error("ValueError: parallel queries require IDENTICAL numeric mode")
    var index = py_get(fields, PythonObject("_index"), PythonObject(None))
    if index is None:
        index = py_get(fields, PythonObject("_x"), PythonObject(None))
    if index is None:
        raise Error("ValueError: fit the estimator before querying")
    var source = args[1]
    if source is None and String(method) == "radius_neighbors":
        source = index
    var data = family_as_f32(source, 2, "X", context)
    if Int(py=data.shape[1]) != Int(py=fields["n_features_in_"]):
        raise Error("ValueError: input feature count differs from fit")
    var n = Int(py=data.shape[0])
    var ranges = shard_ranges(n, holder["rows_per_shard"], "rows_per_shard", "")
    if len(ranges) == 0:
        ranges.append(0)
        ranges.append(0)
    var state = shard_state(fields, b.frozenset(Python.list("_resident", "_resident_holder")), context)
    py_setitem(state, "numeric_mode", "identical")
    var counter = Int(py=py_get(holder, PythonObject("_native_tokens"), PythonObject(0))) + 1
    py_setitem(holder, "_native_tokens", counter)
    var token = String("nq-") + String(Int(py=Python.import_module("os").getpid())) + "-" + String(Int(py=py_call(b.id, args[0]))) + "-" + String(counter)
    var devices = pool["devices"]
    var workers = len(devices)
    var name = py_type(model).__name__
    var stores = b.list()
    for i in range(workers):  # driver-loop: one snapshot request per worker
        _ = py_call(stores.append, _py_request("query_state", Python.tuple(token, name, state)))
    _ = _map(pool, stores)
    var results = PythonObject(None)
    var n_devices = workers
    try:
        var requests = b.list()
        for r in range(len(ranges) // 2):  # driver-loop: one worker request per query row block
            var lo = ranges[2 * r]
            var hi = ranges[2 * r + 1]
            var shift = shard_read_shift(r, lo, n_devices)
            _ = py_call(requests.append, _py_request("query_rows", Python.tuple(
                token, shard_rows(data, lo - shift, hi - shift, context), method, args[3])))
        results = _map(pool, requests)
    finally:
        if not Bool(py=pool["closed"]):
            var releases = b.list()
            for i in range(workers):  # driver-loop: one release request per worker
                _ = py_call(releases.append, _py_request("query_state", Python.tuple(token, name, PythonObject(None))))
            try:
                _ = _map(pool, releases)
            except:
                process_pool_close(pool)
    var outputs = b.list()
    var diagnostics = b.list()
    for r in range(len(results)):  # driver-loop: one result and diagnostic per query row block
        _ = py_call(outputs.append, results[r][0])
        var record = b.dict()
        py_setitem(record, "start", ranges[2 * r])
        py_setitem(record, "end", ranges[2 * r + 1])
        py_setitem(record, "device", devices[r % workers])
        _ = py_call(record.update, results[r][1])
        _ = py_call(diagnostics.append, record)
    var output = _join(outputs, String(method) == "radius_neighbors", context)
    py_setitem(holder, "last_shards_", diagnostics)
    return output


def _ivf_from_index(args: PythonObject) raises -> PythonObject:
    """args (class, index, devices, ivf binding, int32 list_indices_, int32
    list_offsets_): `DistributedIVFIndex.from_index`
    after its shell's checks. The native shard plan (`ivf_shard_plan`) gives
    every device its contiguous stored rows with list offsets and local IDs;
    each worker stores its shard under the object's token."""
    var b = shard_b()
    var context = _ctx()
    var index = args[1]
    var fields = py_fields(index)
    var native = args[3]
    var n = Int(py=fields["n_rows_"])
    var n_lists = Int(py=fields["n_lists_"])
    var ind = args[4]
    var offs = args[5]
    var all_devices = py_call(b.tuple, args[2])
    _ = shard_devices(all_devices)
    var parts_count = min(len(all_devices), n)
    var soff = shard_array("<i4", Python.tuple(parts_count, n_lists + 1), parts_count * (n_lists + 1), context)
    var local_all = shard_array("<i4", Python.tuple(n), n, context)
    var map_all = shard_array("<i4", Python.tuple(n), n, context)
    var status = Int(py=family_call(py_attr(native, "ivf_shard_plan"),
                                    Python.list(classical_address(ind), classical_address(offs),
                                                classical_address(soff), classical_address(local_all),
                                                classical_address(map_all)),
                                    Python.list(n, n_lists, parts_count)))
    _ = soff  # keep the Python object alive through the native call above
    _ = local_all  # keep the Python object alive through the native call above
    _ = map_all  # keep the Python object alive through the native call above
    if status == 1:
        raise Error("ValueError: IVF original IDs must be a permutation of stored row IDs")
    if status != 0:
        raise Error("ValueError: invalid global IVF list offsets")
    var devices = all_devices[py_call(b.slice, PythonObject(None), PythonObject(n))]
    var pool = shard_open(devices, _CHILD)
    var obj = py_call(b.object.__new__, args[0])
    var own = py_fields(obj)
    py_setitem(own, "_pool", pool)
    py_setitem(own, "_closed", False)
    py_setitem(own, "n_features_in_", fields["n_features_in_"])
    py_setitem(own, "n_neighbors", fields["n_neighbors"])
    py_setitem(own, "metric_code_", fields["metric_code_"])
    py_setitem(own, "devices", devices)
    py_setitem(own, "n_candidates_", PythonObject(None))
    py_setitem(own, "_native", native)
    var token = String("ivf-") + String(Int(py=Python.import_module("os").getpid())) + "-" + String(Int(py=py_call(b.id, obj)))
    py_setitem(own, "_token", token)
    var maps = b.list()
    var requests = b.list()
    var base = shard_state(fields, b.frozenset(Python.list("_resident", "_resident_holder", "list_data_",
                                                           "list_indices_", "list_offsets_", "n_rows_")), context)
    py_setitem(base, "numeric_mode", "identical")
    for part in range(parts_count):  # driver-loop: one stored shard per device
        var lo = part * n // parts_count
        var hi = (part + 1) * n // parts_count
        var state = py_call(b.dict, base)
        py_setitem(state, "n_rows_", hi - lo)
        py_setitem(state, "list_offsets_", shard_row_vector(soff, part, context))
        py_setitem(state, "list_indices_", shard_rows(local_all, lo, hi, context))
        py_setitem(state, "list_data_", shard_rows(fields["list_data_"], lo, hi, context))
        _ = py_call(maps.append, shard_rows(map_all, lo, hi, context))
        _ = py_call(requests.append, _py_request("ivf_store", Python.tuple(token, state)))
    py_setitem(own, "_id_maps", maps)
    try:
        var receipts = _map(pool, requests)
        for part in range(parts_count):  # driver-loop: one storage receipt per shard
            if Int(py=receipts[part]) != Int(py=maps[part].shape[0]):
                raise Error("ValueError: IVF storage receipts differ from shard sizes")
    except error:
        process_pool_close(pool)
        py_setitem(own, "_closed", True)
        raise error^
    return obj


def _ivf_search(args: PythonObject) raises -> PythonObject:
    """args (distributed index, queries): every worker's partial search of
    its stored shard, merged by the native `ivf_merge_shards` (cosine
    distances finalized natively)."""
    var b = shard_b()
    var context = _ctx()
    var own = py_fields(args[0])
    if Bool(py=own["_closed"]):
        raise Error("RuntimeError: distributed IVF index is closed")
    var q = family_as_f32(args[1], 2, "queries", context)
    if Int(py=q.shape[0]) < 1:
        raise Error("ValueError: at least one query is required")
    if Int(py=q.shape[1]) != Int(py=own["n_features_in_"]):
        raise Error("ValueError: query features differ from index")
    var pool = own["_pool"]
    var maps = own["_id_maps"]
    var native = own["_native"]
    try:
        var requests = b.list()
        for i in range(len(own["devices"])):  # driver-loop: one search request per stored shard
            _ = py_call(requests.append, _py_request("ivf_search", Python.tuple(own["_token"], q)))
        var parts = _map(pool, requests)
        if len(parts) != len(maps):
            raise Error("ValueError: incomplete IVF result shards")
        var m = Int(py=q.shape[0])
        var k = Int(py=own["n_neighbors"])
        for i in range(len(parts)):  # driver-loop: checks one result shard
            var p = parts[i]
            if (Int(py=p[0].shape[0]) != m or Int(py=p[0].shape[1]) != k or Int(py=p[1].shape[0]) != m
                    or Int(py=p[1].shape[1]) != k or Int(py=p[2].shape[0]) != m):
                raise Error("ValueError: invalid IVF shard output shapes")
        var result = shard_array("<f4", Python.tuple(m, k), m * k, context)
        var ids = shard_array("<i4", Python.tuple(m, k), m * k, context)
        var counts = shard_array("<i4", Python.tuple(m), m, context)
        var addrs = Python.list(classical_address(result), classical_address(ids), classical_address(counts))
        var sizes = Python.list(len(parts), m, k)
        for i in range(len(parts)):  # driver-loop: four buffer addresses per result shard
            var p = parts[i]
            var mapping = maps[i]
            _ = py_call(addrs.extend, Python.list(classical_address(p[0]), classical_address(p[1]),
                                                  classical_address(p[2]),
                                                  classical_address(mapping) if Int(py=mapping.shape[0]) > 0
                                                  else classical_address(p[2])))
            _ = py_call(sizes.append, Int(py=mapping.shape[0]))
        var merged = family_call(py_attr(native, "ivf_merge_shards"), addrs, sizes)
        _ = parts  # keep the Python objects alive through the native call above
        _ = maps  # keep the Python objects alive through the native call above
        _ = result  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        var status = Int(py=merged[0])
        if status != 0:
            if status == 1:
                raise Error("ValueError: invalid IVF candidate count")
            if status == 2:
                raise Error("ValueError: invalid IVF local row id")
            if status == 4:
                raise Error("ValueError: NaN distance in an IVF shard result")
            raise Error("ValueError: IVF shard merge failed")
        if Int(py=own["metric_code_"]) == 1:
            _ = family_call(py_attr(native, "ivf_finalize_distances"), classical_address(result),
                            PythonObject(m * k), own["metric_code_"])
            _ = result  # keep the Python object alive through the native call above
        py_setitem(own, "n_candidates_", counts)
        return Python.tuple(result, ids)
    except error:
        _ivf_close(args[0])
        raise error^


def _ivf_close(obj: PythonObject) raises:
    var own = py_fields(obj)
    py_setitem(own, "_closed", True)
    var pool = py_get(own, PythonObject("_pool"), PythonObject(None))
    if pool is not None and not Bool(py=pool["closed"]):
        process_pool_close(pool)


# ---------------------------------------------------------------- the worker

def shard_worker_dispatch(request: PythonObject) raises -> PythonObject:
    """One shard: its model rebuilt natively from the state record, then the
    plain native owner."""
    var b = shard_b()
    if len(request) == 1 and String(request[0]) == "identity":
        return shard_identity()
    var context = _ctx()
    if len(request) != 3:
        raise Error("parallel shards: invalid worker request envelope")
    var operation = String(request[0])
    var state = request[1]
    var args = request[2]
    if operation == "rbf_rows":
        var model = shard_model(_t("rbf_class"), state)
        return _rbf_transform(model, Python.tuple(_worker_binding("_mojolearn_kernel_methods"), args[0]), context)
    if operation == "scaler_fit":
        var kind = String(state[0])
        var model = py_call(b.object.__new__, _t(kind + "_class"))
        var entry = _t(kind + "_call")
        var owner_context = _t(kind + "_context")
        _ = family_call(entry, model, PythonObject("init"), state[1], owner_context)
        if kind == "standard":
            _ = family_call(_t("standard_fit"), model, args[0], PythonObject(None), PythonObject(None), owner_context)
        else:
            _ = family_call(entry, model, PythonObject("fit"), Python.tuple(args[0], PythonObject(None)), owner_context)
        return shard_state(py_fields(model), b.frozenset(), context)
    if operation == "scaler_transform":
        var kind = String(state[0])
        var model = shard_model(_t(kind + "_class"), state[1])
        var entry = _t(kind + "_call")
        var owner_context = _t(kind + "_context")
        if kind == "standard":
            return family_call(entry, model, PythonObject("transform"), Python.tuple(args[0], args[1], PythonObject(None)),
                               owner_context)
        return family_call(entry, model, PythonObject("transform"), Python.tuple(args[0], args[1]), owner_context)
    if operation == "gpc_class_fit":
        var ext = _worker_binding("_mojolearn_gp")
        var fit = _gpc_fit_binary(ext, args[0], args[1], state[0], state[1], args[2], context)
        return shard_state(py_fields(fit), b.frozenset(), context)
    if operation == "gpc_class_predict":
        var ext = _worker_binding("_mojolearn_gp")
        var f = state[2]
        var fit = _gpc_binary(context, f["y_train_"], f["L_"], f["pi_"], f["W_sr_"],
                              Float64(py=f["log_marginal_likelihood_value_"]), Int(py=f["n_iter_"]), Int(py=f["nb_"]))
        var holder = b.dict()
        py_setitem(holder, "X_train_", state[1])
        var out_kind = Int(py=args[2])
        var latent = _gpc_latent(ext, holder, fit, args[0], Bool(py=args[1]), out_kind, context, state[0])
        if out_kind != 0:
            return latent[3]
        return latent[2] if Bool(py=args[1]) else latent[0]
    if operation == "forest_fit":
        var model = shard_model(_t("forest_by_name")[state[0]], state[1])
        py_setitem(py_fields(model), "_has_constructor_parameters", True)
        _ = family_call(_t("trees_entry"), model, PythonObject("trees_public_forest_fit"),
                        Python.tuple(args[0], args[1], PythonObject(None), False, args[2]), _t("trees_context"))
        var fields = py_call(b.dict, py_fields(model))
        _ = py_call(fields.pop, "_resident_holder", PythonObject(None))
        return shard_state(fields, b.frozenset(), context)
    if operation == "py":
        # the neighbor and IVF drivers' per-shard plain query methods
        # (python/mojolearn/_parallel_shard_ops.py), by name
        var function = py_get(_t("py_ops"), state, PythonObject(None))
        if function is None:
            raise Error("parallel shards: unknown worker query operation")
        return py_call(function, args)
    raise Error("parallel shards: unsupported worker operation " + operation)


def shard_worker(argv: PythonObject) raises:
    var socket = process_worker_connect(argv)
    try:
        while True:  # driver-loop: over native local shard requests
            var request = process_worker_next(socket)
            if request is None:
                break
            var sequence = Int(py=py_get(request[0], "sequence", -1))
            try:
                process_send(socket, "result", sequence, shard_worker_dispatch(request[1]))
            except error:
                process_send(socket, "error", sequence, PythonObject(String(error)))
    except error:
        transport_close(socket)
        raise error^
    transport_close(socket)


def parallel_shard_call(operation: String, args: PythonObject) raises -> PythonObject:
    var s = PARALLEL_SHARD_BOUNDARY.get_or_create_ptr()
    if operation == "register":
        # A driver module's fixed classes, entries and contexts, at its import.
        if s[].tables is None:
            s[].tables = shard_b().dict()
        _ = py_call(s[].tables.update, args)
        var context = py_get(s[].tables, PythonObject("context"), PythonObject(None))
        if context is not None:
            codec_initialize(context["array_class"], PythonObject(None), PythonObject(None))
        return PythonObject(None)
    if operation == "transform_rbf_sampler":
        return _transform_rbf_sampler(args)
    if operation == "fit_scaler":
        return _fit_scaler(args)
    if operation == "transform_scaler":
        return _transform_scaler(args)
    if operation == "fit_gaussian_process_classifier":
        return _fit_gpc(args)
    if operation == "predict_gaussian_process_classifier":
        return _predict_gpc(args)
    if operation == "fit_forest":
        return _fit_forest(args)
    if operation == "trace_take":
        return shard_trace_take()
    if operation == "faults_take":
        return PythonObject(shard_faults_take())
    if operation == "pool_open":
        return _pool_open(args[0])
    if operation == "pool_close":
        if not Bool(py=args[0]["closed"]):
            process_pool_close(args[0])
        return PythonObject(None)
    if operation == "reference_query":
        return _reference_query(args)
    if operation == "parallel_queries":
        return _parallel_queries(args)
    if operation == "ivf_from_index":
        return _ivf_from_index(args)
    if operation == "ivf_search":
        return _ivf_search(args)
    if operation == "ivf_close":
        _ivf_close(args[0])
        return PythonObject(None)
    if operation == "worker":
        shard_worker(args)
        return PythonObject(None)
    raise Error("RuntimeError: parallel shards: unknown operation " + operation)


def parallel_shard_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = parallel_shard_call(String(args[0]), args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
