# SPDX-License-Identifier: Apache-2.0
# One native call per public method (lane native-neighbors-20261024):
# KernelDensity `fit`, `score_samples` and `score` (python/mojolearn/density.py).
# Each `family_xn_kde_*` operation validates, prepares, runs the family's
# compiled kernels and returns, with the former Python method's refusals,
# operations and order. Numerical kernels and their defaults stay unchanged:
#
#   fit            `_release_resident_fit`, `as_f32_dense_c(X)` (and its
#                  `input_copied_` flag), the finiteness refusal, the
#                  binding's `kde_fast_immutable_fit` snapshot switch, then
#                  `check_sample_weight` (float32, scikit-learn's wording).
#   score_samples  `_scores(X, False)`: the fitted check, the query cast and
#   score          width refusal, the device-resident fit handle
#                  (`_resident_fit_handle`: the same key, the same
#                  `_validate_params` on a changed key, the same release and
#                  native preparation request), then direct native score /
#                  resident-score descriptors; `score` is
#                  the binding's fixed-order total (`want_total`).
#
# The shell hands over the binding `_bind("_mojolearn_estimators")` resolved
# for the model's numeric mode. ReferenceShardedNeighbors (also routed to
# this prefix by the lane's skeleton) is not served here: its dispatch is the
# Python `DevicePool` (python/mojolearn/_parallel_pool.py), whose workers run
# Python handlers.
from providers.python.estimator_binding_pyapi import estimator_arguments
from providers.python.neighbors_public_lifecycle_pyapi import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family, admit_family_weights
from bindings.native_loaded_module import native_loaded_module
from bindings.kde_admission_boundary import kde_admission_hit, kde_admission_remember, kde_admission_drop
from bindings.kde_native_boundary import KDEConnection, KDEResidentOwner, kde_owner_admit, kde_owner_prepare, kde_run_query, kde_run_resident, kde_array
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.decomp_scalar_compare import decomp_equal_admitted, decomp_name_admitted
from bindings.decomp_scalar_admission import decomp_scalar_repr
from bindings.decomp_truth_boundary import decomp_truth
from core.kde_native_api import KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_CAP_RESIDENT, KDE_CAP_IMMUTABLE
from core.kde_native_state import KDEFitKey
from bindings.mode_metadata_native import _fields
from bindings.py_balanced import py_call, py_get, py_contains, py_setitem, py_fields, py_type
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_call, family_field, family_is_str,
)
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense, _sparse_fields
# TOMBSTONE: _any_contiguous_address use (the labels admission copy's buffer address read) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from core.host_finite_scan import host_finite_status
from core.admitted_input import admitted_cast_f32
from core.cast_in import CAST_KIND_FLOAT
from bindings.admitted_input import admitted_finite_refuse
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.p3nc_fitted_freeze import fitted_freeze


comptime _KDE_STATE_SLOT = "neighbors_kde_state_v2"


def _kde_admit_state(model: PythonObject, context: PythonObject, fitting: Bool = False) raises:
    classical_check_layout(model, context)
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var names: List[String] = ["bandwidth", "algorithm", "kernel", "metric", "atol", "rtol", "breadth_first",
                               "leaf_size", "n_features_in_", "n_samples_fit_"]
    var arrays: List[String] = ["_x", "_w"]
    var hit = kde_admission_hit(model, PythonObject(None), False, _KDE_STATE_SLOT, context)
    if not hit:
        for i in range(len(names)):  # small-loop(fixed KDE settings and fitted metadata)
            var value = py_get(fields, names[i])
            if value is not None:
                host_inference_scalar(value, context, True)
        var params = py_get(fields, "metric_params")
        if params is not None:
            if py_type(params) is not native_loaded_module("builtins").dict:
                raise Error("TypeError: KernelDensity metric_params must be an exact native dictionary")
            classical_admit_value(params, context)
            var values = py_call(native_loaded_module("builtins").tuple, params.values())
            for i in range(len(values)):  # driver-loop: admitted metric-parameter scalar values
                host_inference_scalar(values[i], context)
        for i in range(len(arrays)):  # small-loop(fitted KDE buffer owners)
            var value = py_get(fields, arrays[i])
            if value is not None:
                classical_admit_value(value, context)
    if fitting:
        kde_admission_drop(model, _KDE_STATE_SLOT)
        return
    if not hit:
        kde_admission_remember(model, PythonObject(None), False, _KDE_STATE_SLOT, context)


def _kde_sparse_f32(source: PythonObject, format: String) raises -> Bool:
    """Whether `source.toarray()` is already float32 (the matrix's own
    dtype, read from its stored fields as `host_sparse_dense` reads it)."""
    var fields = _sparse_fields(source)
    var dtype = fields["dtype"] if (format == "dok" or format == "lil") else fields["data"].dtype
    return String(dtype.kind) == "f" and Int(py=dtype.itemsize) == 4


def _kde_borrowed(source: PythonObject, x: PythonObject, context: PythonObject) raises -> Bool:
    """`_materialize` borrowed `source` and the cast kept it: the converted
    array reads the source's own memory (an Array, or a contiguous buffer
    export); a list, a strided export or a cast is a copy."""
    var b = native_loaded_module("builtins")
    if py_type(source) is b.list or py_type(source) is b.tuple:
        return False
    var address = Int(py=classical_address(x))
    if py_type(source) is context["array_class"]:
        return address == Int(py=classical_address(source))
    try:
        # The exporter's own data address (classical_address: an ndarray's,
        # or a buffer export's first element).
        return Int(py=classical_address(source)) == address
    except:
        return False


@fieldwise_init
struct KDEDense(Movable):
    var array: PythonObject
    var copied: Bool


def _kde_dense(source: PythonObject, context: PythonObject) raises -> KDEDense:
    """`_buffer.as_f32_dense_c(source, ndim=2, name="X")`: typed array/copy metadata.
    A scipy.sparse matrix is densified exactly first (`toarray()`), and
    `copied` is the conversion's flag for the dense array it was handed, as
    `as_f32_c(obj.toarray())` reported it: a copy unless the matrix holds
    float32 (`toarray()` keeps its dtype)."""
    var sparse = host_sparse_format(source)
    if sparse.byte_length() > 0:
        if not py_contains(context, PythonObject("output_context")):
            py_setitem(context, "output_context", context)
        var dense = admit_family(host_sparse_dense(source, sparse, context), 2, "X", context, False, True,
                                 refusal=admitted_finite_wording("KernelDensity", "X"), defer_finite=True).array
        return KDEDense(dense, not _kde_sparse_f32(source, sparse))
    # Lane a7b-device-x-neighbors: a cast X is refused from the cast's own
    # verdict; a borrowed one by the fit's copy (`_kde_copy_finite`) or the
    # score entry's status word.
    var x = admit_family(source, 2, "X", context, False, True,
                         refusal=admitted_finite_wording("KernelDensity", "X"), defer_finite=True).array
    var copied = not _kde_borrowed(source, x, context)
    return KDEDense(x, copied)


def _kde_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value.copy()` of a C-order float32 array: a new mojolearn Array."""
    var b = native_loaded_module("builtins")
    var shape = List[Int]()
    for axis in range(len(value.shape)):  # driver-loop: admitted array rank
        shape.append(Int(py=value.shape[axis]))
    var result = kde_array(shape, context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))), count=Int(py=value.size))
    _ = result  # keep the Python object alive through the native call above
    _ = value  # keep the Python object alive through the native call above
    return result


def _kde_copy_finite(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kde_copy` of a borrowed fit set with the finite test in the same
    pass (lane a7b-device-x-neighbors: was a serial scan and then a memcpy):
    one parallel row-task copy (core/admitted_input.mojo admitted_cast_f32,
    float32 to float32: the same bits) whose verdict refuses a NaN or an
    infinity in `admitted_finite_refuse`'s wording."""
    var b = native_loaded_module("builtins")
    var shape = List[Int]()
    for axis in range(len(value.shape)):  # driver-loop: admitted array rank
        shape.append(Int(py=value.shape[axis]))
    var result = kde_array(shape, context)
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    # parallel: the copy and the finiteness verdict over row tasks across host cores
    var ok = admitted_cast_f32(Int(py=classical_address(value)), cols * 4, 4, rows, cols, 4, CAST_KIND_FLOAT,
                               FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))))
    _ = value  # keep the Python object alive through the native call above
    admitted_finite_refuse(0 if ok else 1, "KernelDensity")
    return result


def _kde_validate(fields: PythonObject, context: PythonObject) raises:
    """`KernelDensity._validate_params`: the constructor checks, in the
    Python method's order and wording."""
    var b = native_loaded_module("builtins")
    var bandwidth = family_field(fields, "bandwidth")
    var algorithm = family_field(fields, "algorithm")
    var kernel = family_field(fields, "kernel")
    var metric = family_field(fields, "metric")
    var atol = family_field(fields, "atol")
    var rtol = family_field(fields, "rtol")
    var breadth_first = family_field(fields, "breadth_first")
    var leaf_size = family_field(fields, "leaf_size")
    var metric_params = family_field(fields, "metric_params")
    if family_is_str(bandwidth):
        raise Error("NotImplementedError: mojolearn KernelDensity: bandwidth='" + family_str(bandwidth)
                    + "' is refused; compute "
                    + "the rule on the host and pass the float (sklearn's scott is "
                    + "n ** (-1/(d+4)), silverman is (n (d+2) / 4) ** (-1/(d+4)))")
    var width = family_call(b.float, bandwidth)
    var bw = Float64(py=width)
    if not (bw > 0.0) or bw != bw:
        raise Error("mojolearn KernelDensity: bandwidth must be positive, got " + family_repr(width))
    if algorithm is not None and not decomp_name_admitted(algorithm, "auto"):
        raise Error("mojolearn KernelDensity: algorithm=" + decomp_scalar_repr(algorithm, context) + " is refused; this is "
                    + "cuML's brute-force KDE, there is no tree to choose")
    var knobs = not decomp_equal_admitted(atol, PythonObject(0))
    if not knobs:
        knobs = not decomp_equal_admitted(rtol, PythonObject(0))
    if not knobs:
        knobs = not decomp_truth(breadth_first, context)
    if not knobs:
        knobs = not decomp_equal_admitted(leaf_size, PythonObject(40))
    if knobs:
        raise Error("mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
                    + "are tree-traversal knobs; this is brute force (refused)")
    if not family_is_str(kernel) or not family_is_str(metric):
        raise Error("mojolearn KernelDensity: kernel and metric are names")
    # cuML's rule, `kernel_density.py:302-307`: at most ONE entry, and its
    # VALUE is taken regardless of its key.
    if decomp_truth(metric_params, context):
        if len(metric_params) != 1:
            raise Error("mojolearn KernelDensity: only metrics with a single "
                        + "argument are supported (cuML raises the same, "
                        + "kernel_density.py:304-306)")
        var values = py_call(b.tuple, metric_params.values())
        var p = family_call(b.float, values[0])
        if Float64(py=p) != 2.0:
            raise Error("NotImplementedError: mojolearn KernelDensity: metric_params=" + _kde_params_repr(metric_params, context)
                        + " asks for "
                        + "Minkowski p=" + family_repr(p) + ". The KERNEL computes any finite "
                        + "positive normal p (kde/impl/distance/distance.mojo, "
                        + "DEVIATION 552) and kde/estimator.mojo takes it as "
                        + "`metric_arg`; what is missing is a SLOT IN THE "
                        + "BINDING -- kde_score_samples_binding length-checks "
                        + "its params list at exactly 5 entries "
                        + "(bindings/_mojolearn_estimators.mojo:647, and the same "
                        + "check in bindings/_mojolearn_estimators_host.mojo:213). "
                        + "Pass p=2, "
                        + "or apply the two-line binding change in "
                        + "kde/README.md's HAND-OFF.")


def _kde_params_repr(params: PythonObject, context: PythonObject) raises -> String:
    # Do not enter NumPy arrayprint through a dict's element repr. Exact
    # primitive scalar diagnostics retain their usual representation.
    var pairs = py_call(native_loaded_module("builtins").tuple, params.items())
    var result = String("{")
    for i in range(len(pairs)):  # driver-loop: diagnostic parameter metadata only
        if i:
            result += ", "
        result += decomp_scalar_repr(pairs[i][0], context) + ": " + decomp_scalar_repr(pairs[i][1], context)
    return result + "}"


def _kde_release(fields: PythonObject) raises:
    lifecycle_admit_fields(fields)
    var cached = py_get(fields, "_resident")
    kde_owner_admit(cached)
    if cached is not None:
        # Dropping the last pin invokes the minting driver's quiet release.
        # In-flight queries retain their own pin until native work completes.
        py_setitem(fields, "_resident", PythonObject(None))
    _ = cached


def _kde_text(value: PythonObject, context: PythonObject) raises -> String:
    if py_type(value) is py_get(context, "ndarray"):
        # Such a value is not a supported kernel/metric name. Avoid the
        # process-wide Python NumPy formatter on its diagnostic route.
        return decomp_scalar_repr(value, context)
    return family_str(value)


@fieldwise_init
struct KDEResidentLookup(Movable):
    var owner: PythonObject
    var released_gil: Bool


def _kde_handle(model: PythonObject, fields: PythonObject, connection: KDEConnection,
                context: PythonObject) raises -> KDEResidentLookup:
    if (connection.capabilities & KDE_CAP_RESIDENT) == 0:
        return KDEResidentLookup(PythonObject(None), False)
    var b = native_loaded_module("builtins")
    var x = family_field(fields, "_x")
    var w = family_field(fields, "_w")
    var w_address = 0 if w is None else Int(py=classical_address(w))
    var kernel = family_field(fields, "kernel")
    var metric = family_field(fields, "metric")
    var bandwidth = family_field(fields, "bandwidth")
    if family_is_str(bandwidth):
        _kde_validate(fields, context)
    var key = KDEFitKey(Int(py=classical_address(x)), Int(py=x.shape[0]), Int(py=x.shape[1]), w_address,
                        _kde_text(kernel, context), _kde_text(metric, context), Float64(py=family_call(b.float, bandwidth)))
    var cached = py_get(fields, "_resident")
    kde_owner_admit(cached)
    if cached is not None:
        if cached.downcast_value_ptr[KDEResidentOwner]()[].state.matches(Int(model._obj_ptr),
                Int(connection.provider._obj_ptr), key):
            return KDEResidentLookup(cached, False)
        _kde_validate(fields, context)
    _kde_release(fields)
    cached = PythonObject(None)
    lifecycle_admit_fields(fields)
    # The fit's native finite-copy/cast already tested the training set. Keep
    # finite_tested=1 and the existing lazy, once-per-key device upload.
    var request = KDEPrepareRequest(key.train, key.weights, key.rows, key.features, key.bandwidth,
                                    w is not None, True, key.kernel.copy(), key.metric.copy())
    var owner = kde_owner_prepare(model, connection, key, request, x, w)
    lifecycle_admit_fields(fields)
    py_setitem(fields, "_resident", owner)
    return KDEResidentLookup(owner, True)


def _kde_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`KernelDensity.fit(X, y=None, sample_weight=None)`."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var binding = args[0]
    # The constructor stores its settings verbatim (scikit-learn's deferred
    # validation); fit checks them first, so a refused setting raises the
    # documented error here instead of scoring with it.
    _kde_validate(fields, context)
    # A refit drops the device copy of the previous fit set HERE, as
    # NearestNeighbors.fit does (DEVIATION 2921's key rule).
    _kde_release(fields)
    lifecycle_admit_fields(fields)
    var dense = _kde_dense(args[1], context)
    var x = dense.array
    py_setitem(fields, "input_copied_", PythonObject(dense.copied))
    # Opt-in immutable snapshot: transport-only copy of caller bytes.
    # Residency never relies on external mutable pointer identity.
    var connection = KDEConnection(binding)
    var immutable = (connection.capabilities & KDE_CAP_IMMUTABLE) != 0
    if not dense.copied:
        # p3-neighbors-cluster (caller-aliased inputs): the conversion handed
        # back the caller's own float32 buffer, and the resident fit handle
        # is keyed by its address. Fit keeps an owned snapshot instead, so a
        # later in-place write into the caller's X never reaches a score.
        # `input_copied_` keeps reporting what the conversion did. Lane a7b:
        # this copy is also the fit's finite test (DEVIATION 604 at fit, as
        # scikit-learn refuses; a cast X was refused by its cast).
        x = _kde_copy_finite(x, context)
    elif immutable:
        x = _kde_copy(x, context)
    if immutable:
        py_setitem(fields, "input_copied_", PythonObject(True))
    # The fitted set is read-only (bindings/p3nc_fitted_freeze.mojo).
    _ = fitted_freeze(x, context)
    py_setitem(fields, "_x", x)
    py_setitem(fields, "n_features_in_", x.shape[1])
    py_setitem(fields, "n_samples_fit_", x.shape[0])
    var sample_weight = args[2]
    if sample_weight is not None:
        # scikit-learn's _check_sample_weight (shape, finite, >= 0, not all
        # zero, its wording) by the base binding's reductions
        var w = admit_family_weights(sample_weight, Int(py=x.shape[0]), False, context).keep
        # Always an owned snapshot (the resident key holds its address).
        w = _kde_copy(w, context)
        _ = fitted_freeze(w, context)
        py_setitem(fields, "_w", w)
    else:
        py_setitem(fields, "_w", PythonObject(None))


def _kde_scores(model: PythonObject, args: PythonObject, want_total: Bool,
                context: PythonObject) raises -> PythonObject:
    """`KernelDensity._scores(X, want_total)`: score_samples(X), or with
    `want_total` their sum, folded in the binding (on the device on a GPU
    install: `device_sum_f32_fixed`; the host binding's
    `host_sum_f32_fixed`, the same order)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var binding = args[0]
    if not py_contains(fields, PythonObject("_x")):
        raise Error("mojolearn KernelDensity: call fit() first")
    var q = args[1]
    if host_sparse_format(q).byte_length() > 0:
        var dense = _kde_dense(q, context)
        q = dense.array
    var features = family_field(fields, "n_features_in_")
    if not decomp_equal_admitted(q.shape[1], features):
        raise Error("mojolearn KernelDensity: X has " + family_str(q.shape[1]) + " features, "
                    + "fit saw " + _kde_text(features, context))
    var rows = Int(py=q.shape[0])
    var dst = kde_array(List[Int](length=1, fill=rows), context)
    var w = family_field(fields, "_w")
    var connection = KDEConnection(binding)
    var lookup = _kde_handle(model, fields, connection, context)
    var owner = lookup.owner
    if lookup.released_gil:
        # Only actual preparation/teardown releases the GIL here. Re-admit
        # changed public state and the retained original feature-count owner.
        lifecycle_admit_fields(fields)
        _kde_admit_state(model, context)
        host_inference_scalar(features, context, True)
    var total = Float64(0)
    if owner is not None:
        var status = List[Int32](length=1, fill=Int32(0))
        var query_address = Int(py=classical_address(q))
        var dst_address = Int(py=classical_address(dst))
        var d = Int(py=family_call(b.int, features))
        var width = Float64(py=family_call(b.float, family_field(fields, "bandwidth")))
        var request = KDEResidentQuery(owner.downcast_value_ptr[KDEResidentOwner]()[].state.handle,
            query_address, dst_address, Int(status.unsafe_ptr()), rows, d, width, want_total,
            _kde_text(family_field(fields, "kernel"), context), _kde_text(family_field(fields, "metric"), context))
        total = kde_run_resident(owner, request)
        _ = q
        _ = dst
        admitted_finite_refuse(Int(status[0]), "KernelDensity")
        _ = status^
    else:
        # Preserve the host route's parallel query finite scan before the
        # provider's own bound/cosine checks and its existing weights copy.
        admitted_finite_refuse(Int(host_finite_status(Int(py=classical_address(q)), Int(py=q.size))), "KernelDensity")
        var x = family_field(fields, "_x")
        var weights_address = 0 if w is None else Int(py=classical_address(w))
        var train_address = Int(py=classical_address(x))
        var query_address = Int(py=classical_address(q))
        var dst_address = Int(py=classical_address(dst))
        var train_rows = Int(py=x.shape[0])
        var d = Int(py=family_call(b.int, features))
        var width = Float64(py=family_call(b.float, family_field(fields, "bandwidth")))
        var request = KDEQueryRequest(train_address, query_address, weights_address, dst_address,
            train_rows, rows, d, width, w is not None, want_total,
            _kde_text(family_field(fields, "kernel"), context), _kde_text(family_field(fields, "metric"), context))
        total = kde_run_query(connection, request)
        _ = x
        _ = w
        _ = q
        _ = dst
    if want_total:
        return PythonObject(total)
    return dst


def _neighbors_kde_operation(model: PythonObject, operation: String, var args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _kde_admit_state(model, context, operation == "family_xn_kde_fit")
    for i in range(1, len(args)):  # small-loop(public input owners before sparse/native preparation)
        if host_sparse_format(args[i]).byte_length() == 0:
            classical_admit_value(args[i], context)
    if operation != "family_xn_kde_release":
        args = estimator_arguments(model, "_mojolearn_estimators", args, context)
    if operation == "family_xn_kde_fit":
        _kde_fit(model, args, context)
        return model
    if operation == "family_xn_kde_score_samples":
        return _kde_scores(model, args, False, context)
    if operation == "family_xn_kde_score":
        return _kde_scores(model, args, True, context)
    if operation == "family_xn_kde_release":
        _kde_release(py_fields(model))
        return PythonObject(None)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_kde_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    if operation == "family_xn_kde_getstate":
        var fields = _fields(model)
        var state = py_call(native_loaded_module("builtins").dict, fields)
        _ = py_call(state.pop, "_resident", PythonObject(None))
        return state
    if operation == "family_xn_kde_release":
        return _neighbors_kde_operation(model, operation, args, context)
    var call = neighbors_public_begin(model, operation, args, context)
    var result = PythonObject(None)
    try:
        result = _neighbors_kde_operation(model, operation, call.args, context)
    except error:
        var failure = neighbors_public_refusal(error, call, model)
        raise failure^
    finally:
        neighbors_public_end(model, call)
    return neighbors_public_finish(model, operation, result, call, context)
