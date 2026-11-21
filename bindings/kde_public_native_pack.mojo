# SPDX-License-Identifier: Apache-2.0
# KernelDensity's public operations: the operator of `fit`, `score_samples`
# and `score` (UNPACK -> RUN -> PACK) and the PACK half: the owned snapshot
# copies of the fitted set, the fitted-field publication, the resident fit
# handle and the returned scores. One native call per public method (lane
# native-neighbors-20261024), with the former Python method's refusals,
# operations and order; numerical kernels and their defaults unchanged:
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
# UNPACK is bindings/kde_public_native_unpack.mojo; RUN is
# neighbors/neighbors_kde_run.mojo (typed keys, requests and refusals) with
# the typed provider calls of core/kde_native_api.mojo. The image is the
# estimator's binding route (`estimator_arguments`). (This file and the
# UNPACK file were the single door bindings/neighbors_kde_native_door.mojo.)
from providers.python.estimator_binding_pyapi_door import estimator_arguments
from providers.python.neighbors_public_lifecycle_pyapi_door import neighbors_public_begin, neighbors_public_end, neighbors_public_finish, neighbors_public_refusal
from std.python import PythonObject
from bindings.admitted_family_door import admit_family, admit_family_weights
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.kde_admission_boundary_unpack import kde_admission_hit, kde_admission_remember, kde_admission_drop
from bindings.kde_native_boundary_unpack import KDEConnection, KDEResidentOwner, kde_owner_admit, kde_run_query, kde_run_resident
from bindings.kde_native_boundary_pack import kde_owner_prepare, kde_array
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.decomp_scalar_compare_unpack import decomp_equal_admitted, decomp_name_admitted
from bindings.decomp_scalar_admission_unpack import decomp_scalar_repr
from bindings.decomp_truth_boundary_unpack import decomp_truth
from neighbors.neighbors_kde_run import (
    KDE_DEFAULT_LEAF_SIZE, kde_refuse_bandwidth_name, kde_bandwidth_positive, kde_refuse_bandwidth,
    kde_refuse_algorithm, kde_refuse_tree_knobs, kde_refuse_names, kde_check_metric_params_count,
    kde_minkowski_supported, kde_refuse_minkowski, kde_refuse_unfitted, kde_refuse_features,
    kde_resident_capable, kde_immutable_capable, kde_copy_f32, kde_copy_finite_f32, kde_query_finite_status,
    kde_fit_key, kde_prepare_request, kde_query_request, kde_resident_query,
)
from bindings.py_balanced import py_call, py_get, py_contains, py_setitem, py_fields, py_type
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.family_common_unpack import (
    family_repr,
    family_str,
    family_is_str,
)
from bindings.family_common_pack import (
    family_call,
    family_field,
)
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense, _sparse_fields
# TOMBSTONE: _any_contiguous_address use (the labels admission copy's buffer address read) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.admitted_input_unpack import admitted_finite_refuse
from bindings.admitted_finite_wording import admitted_finite_wording
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze
from bindings.kde_public_native_unpack import (
    KDEDense, _kde_admit_state, _kde_dense, _kde_validate, _kde_text,
)


def _kde_copy(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value.copy()` of a C-order float32 array: a new mojolearn Array."""
    var shape = List[Int]()
    for axis in range(len(value.shape)):  # driver-loop: admitted array rank
        shape.append(Int(py=value.shape[axis]))
    var result = kde_array(shape, context)
    kde_copy_f32(Int(py=classical_address(result)), Int(py=classical_address(value)), Int(py=value.size))
    _ = result  # keep the Python object alive through the native call above
    _ = value  # keep the Python object alive through the native call above
    return result


def _kde_copy_finite(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kde_copy` of a borrowed fit set with the finite test in the same
    pass (lane a7b-device-x-neighbors: was a serial scan and then a memcpy):
    one parallel row-task copy (core/admitted_input.mojo admitted_cast_f32,
    float32 to float32: the same bits) whose verdict refuses a NaN or an
    infinity in `admitted_finite_refuse`'s wording."""
    var shape = List[Int]()
    for axis in range(len(value.shape)):  # driver-loop: admitted array rank
        shape.append(Int(py=value.shape[axis]))
    var result = kde_array(shape, context)
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var ok = kde_copy_finite_f32(Int(py=classical_address(value)), Int(py=classical_address(result)), rows, cols)
    _ = value  # keep the Python object alive through the native call above
    admitted_finite_refuse(0 if ok else 1, "KernelDensity")
    return result


def _kde_release(fields: PythonObject) raises:
    lifecycle_admit_fields(fields)
    var cached = py_get(fields, "_resident")
    kde_owner_admit(cached)
    if cached is not None:
        # Dropping the last pin invokes the minting driver's quiet release.
        # In-flight queries retain their own pin until native work completes.
        py_setitem(fields, "_resident", PythonObject(None))
    _ = cached


@fieldwise_init
struct KDEResidentLookup(Movable):
    var owner: PythonObject
    var released_gil: Bool


def _kde_handle(model: PythonObject, fields: PythonObject, connection: KDEConnection,
                context: PythonObject) raises -> KDEResidentLookup:
    # A changed string bandwidth ('scott') revalidates on every route; the
    # CPU (non-resident) route returned before this check (smoke 2026-11-15,
    # test_resident_key_change_revalidates_like_former_path).
    var bandwidth = family_field(fields, "bandwidth")
    if family_is_str(bandwidth):
        _kde_validate(fields, context)
    if not kde_resident_capable(connection.capabilities):
        return KDEResidentLookup(PythonObject(None), False)
    var b = native_loaded_module("builtins")
    var x = family_field(fields, "_x")
    var w = family_field(fields, "_w")
    var w_address = 0 if w is None else Int(py=classical_address(w))
    var kernel = family_field(fields, "kernel")
    var metric = family_field(fields, "metric")
    var key = kde_fit_key(Int(py=classical_address(x)), Int(py=x.shape[0]), Int(py=x.shape[1]), w_address,
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
    var request = kde_prepare_request(key, w is not None)
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
    var immutable = kde_immutable_capable(connection.capabilities)
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
        var w_pin = ArrayPin()
        _ = admit_family_weights(sample_weight, Int(py=x.shape[0]), False, context, w_pin)
        var w = w_pin.keep
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
        kde_refuse_unfitted()
    var q = args[1]
    if host_sparse_format(q).byte_length() > 0:
        var dense = _kde_dense(q, context)
        q = dense.array
    var features = family_field(fields, "n_features_in_")
    if not decomp_equal_admitted(q.shape[1], features):
        kde_refuse_features(family_str(q.shape[1]), _kde_text(features, context))
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
        var request = kde_resident_query(owner.downcast_value_ptr[KDEResidentOwner]()[].state.handle,
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
        admitted_finite_refuse(kde_query_finite_status(Int(py=classical_address(q)), Int(py=q.size)), "KernelDensity")
        var x = family_field(fields, "_x")
        var weights_address = 0 if w is None else Int(py=classical_address(w))
        var train_address = Int(py=classical_address(x))
        var query_address = Int(py=classical_address(q))
        var dst_address = Int(py=classical_address(dst))
        var train_rows = Int(py=x.shape[0])
        var d = Int(py=family_call(b.int, features))
        var width = Float64(py=family_call(b.float, family_field(fields, "bandwidth")))
        var request = kde_query_request(train_address, query_address, weights_address, dst_address,
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
    args = estimator_arguments(model, "_mojolearn_estimators", args, context)
    if operation == "family_xn_kde_fit":
        _kde_fit(model, args, context)
        return model
    if operation == "family_xn_kde_score_samples":
        return _kde_scores(model, args, False, context)
    if operation == "family_xn_kde_score":
        return _kde_scores(model, args, True, context)
    raise Error("RuntimeError: unknown neighbors operation " + operation)


def neighbors_kde_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
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
