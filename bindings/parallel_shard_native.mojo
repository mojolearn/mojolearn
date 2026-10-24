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
from bindings.native_process_pool import process_worker_connect, process_worker_next, process_send
from bindings.native_transport import transport_close
from bindings.native_codec import codec_initialize
from bindings.parallel_shard_common import (
    shard_b, shard_ranges, shard_devices, shard_read_shift, shard_run, shard_rows, shard_concat, shard_state,
    shard_model, shard_array, shard_join_rows,
)
from bindings.kernel_methods_native import _rbf_transform
from bindings.gp_classifier_native import (
    gpc_validate, _gpc_fit_binary, _gpc_binary, _gpc_set_fitted, _gpc_latent, _gpc_ovr_combine,
)
from bindings.gp_common_native import gp_kernel_arrays
from bindings.gp_labels_native import gp_encode_labels, gp_decode_labels
from core.parallel_forecast_impl import forecast_gather_cols, forecast_scatter_cols
from core.hotpath_impl import cast_elements, HP_I32, HP_I64

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

def _gpc_admit(estimator: PythonObject, context: PythonObject) raises:
    """`parallel_gaussian_process._validate`: the class, IDENTICAL, then the
    native parameter validation (which sets `_kernel` and
    `_max_iter_predict`)."""
    if py_type(estimator) is not _t("gpc_class"):
        raise Error("TypeError: requires mojolearn.GaussianProcessClassifier")
    var fields = py_fields(estimator)
    var mode = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    var b = shard_b()
    if mode is None or not Bool(py=py_call(b.bool, mode)):
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str or String(mode) != "identical":
        raise Error("ValueError: parallel GPC requires IDENTICAL numeric mode")
    gpc_validate(estimator, context)


def _fit_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, y, devices, gp binding, base binding): one binary
    Laplace fit per one-vs-rest class column (class 1 alone for two
    classes), each on a worker, stored as the plain fit stores them."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    _gpc_admit(estimator, context)
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
    var arrays = gp_kernel_arrays(fields["_kernel"], context)
    var carried = b.list()
    for i in range(len(arrays)):  # small-loop(the kernel's five packed arrays and words)
        _ = py_call(carried.append, arrays[i])
    var max_iter = fields["_max_iter_predict"]
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
    # `_fresh(estimator)` carried the kernel, max_iter_predict and IDENTICAL
    py_setitem(fields, "numeric_mode", "identical")
    py_setitem(fields, "input_copied_", copied)
    _gpc_set_fitted(fields, x, classes, fits, context)
    return estimator


def _predict_gpc(args: PythonObject) raises -> PythonObject:
    """args (estimator, X, devices, method, gp binding, base binding): one
    latent pass per fitted class on a worker; two classes take the binding's
    decoded codes or [1 - p, p] rows from the one fit, more take the
    one-vs-rest combination of the class probability columns."""
    var b = shard_b()
    var context = _ctx()
    var estimator = args[0]
    _gpc_admit(estimator, context)
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
    var arrays = gp_kernel_arrays(fields["_kernel"], context)
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


# ---------------------------------------------------------------- the worker

def shard_worker_dispatch(request: PythonObject) raises -> PythonObject:
    """One shard: its model rebuilt natively from the state record, then the
    plain native owner."""
    var b = shard_b()
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
