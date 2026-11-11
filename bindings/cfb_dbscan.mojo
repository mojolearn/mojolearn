# SPDX-License-Identifier: Apache-2.0
# DBSCAN `fit`, `fit_predict` and `predict`, one native call each
# (python/mojolearn/density.py). The shell hands over the binding `_bind`
# resolved; this owner makes the Python method's refusals in its order and
# wording, then calls typed native fit/core-gather/reference descriptors.
# With `prediction_data=True` the core rows become resident on the device
# at fit, so `predict` uploads only its queries. Existing kernels/defaults
# and transfer counts are retained.
from bindings.admitted_family import admit_family, admit_family_weights
from bindings.p3nc_fitted_freeze import fitted_freeze
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_arrays import classical_address, classical_is_array
from core.host_finite_scan import host_all_finite_f32
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_call, family_array, family_min, family_field, family_is_str,
)
from bindings.cluster_native import _reference_queries
from bindings.dbscan_native_boundary import (
    DBSCANConnection, dbscan_array, dbscan_publish, dbscan_run_fit, dbscan_run_count, dbscan_run_gather,
    dbscan_fresh_holder, dbscan_mint, dbscan_reference_labels,
)
from core.dbscan_native_api import DBSCANFitRequest, DBSCANCoreRequest, DBSCANGatherRequest, DBSCANFitWorkspace, dbscan_empty_labels
from core.dbscan_native_state import dbscan_metric_code, dbscan_method_code, dbscan_method_accepts
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.host_inference_scalars import host_inference_scalar
from bindings.decomp_scalar_admission import decomp_scalar_repr
from bindings.native_loaded_module import native_loaded_module
from bindings.family_raw_attr import family_raw_get, family_raw_get_or, family_raw_has

comptime _METRIC_PRECOMPUTED = 3
comptime _EPS_NN_BRUTE_FORCE = 0


def cfb_borrow_copy(x: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`x if input_copied_ else x.copy()` as a mojolearn Array: an Array the
    cast made is kept, anything else (the caller's own buffer) is copied so
    a later write by the caller cannot move what predict reads."""
    if classical_is_array(x, context) and not (x is source):
        return x
    var b = native_loaded_module("builtins")
    var out = family_array(py_call(b.tuple, x.shape), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=Int(py=x.size))
    _ = out  # keep the Python object alive through the native call above
    return out


def _dbscan_text(value: PythonObject, context: PythonObject) raises -> String:
    host_inference_scalar(value, context, True)
    if py_type(value) is py_get(context, "ndarray"):
        return decomp_scalar_repr(value, context)
    return String(py=family_call(native_loaded_module("builtins").str, value))


def _dbscan_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`DBSCAN.fit(X, y=None, sample_weight=None)`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var binding = args[0]
    var source = args[1]
    var weights_source = args[2]
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn DBSCAN: prediction_data must be a bool, got "
                    + String(py_call(b.type.__dict__["__name__"].__get__, py_type(prediction_data))))
    var metric_attr = family_field(fields, "metric")
    var metric_key = _dbscan_text(metric_attr, context).lower()
    var metric = dbscan_metric_code(metric_key)
    if metric < 0:
        raise Error("mojolearn DBSCAN: metric=" + decomp_scalar_repr(metric_attr, context) + " is refused; it "
                    + "must be one of " + "['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'precomputed']" + ". 'euclidean'/'l2' is the "
                    + "implemented cuML arm; 'manhattan'/'l1'/'cityblock' is this "
                    + "library's own L1 arm (DEVIATION 27, dbscan/impl/neighbors/"
                    + "epsilon_neighborhood.mojo); 'cosine' and 'precomputed' "
                    + "are cuML's other two (dbscan.pyx:110-115, DEVIATIONS 5113 "
                    + "and 5114)")
    var algorithm = family_field(fields, "algorithm")
    host_inference_scalar(algorithm, context, True)
    # Preserve native unhashable-option refusals before the name refusal.
    _ = family_call(b.hash, algorithm)
    var method = dbscan_method_code(String(py=algorithm)) if family_is_str(algorithm) else -1
    if method < 0:
        raise Error("mojolearn DBSCAN: algorithm=" + decomp_scalar_repr(algorithm, context) + " is refused; "
                    + "it must be one of " + "['brute', 'rbc']" + ". Both name an "
                    + "eps-neighborhood SEARCH: 'rbc' is the random ball cover "
                    + "(the default, DEVIATION 35) and 'brute' is the fused "
                    + "all-pairs scan. scikit-learn's 'kd_tree' and 'ball_tree' "
                    + "are not offered because a recursive, data-dependent tree "
                    + "descent is the wrong shape for a GPU -- divergent branches "
                    + "and pointer chasing instead of coalesced reads -- and its "
                    + "pruning decays past about ten features anyway. The ball "
                    + "cover prunes arithmetically over two flat arrays and "
                    + "measured 2.7x-27x over brute force at 16k-200k rows on "
                    + "this hardware; see the class docstring")
    if not dbscan_method_accepts(method, metric):
        raise Error("mojolearn DBSCAN: metric=" + decomp_scalar_repr(metric_attr, context) + " with "
                    + "algorithm=" + decomp_scalar_repr(algorithm, context) + " is refused. The ball cover "
                    + "computes Euclidean distances for its landmark radii and "
                    + "its three pruning bounds (neighbors/impl/"
                    + "ball_cover/) against the radius itself, so it serves "
                    + "'euclidean' only ('cosine' compares unit rows against "
                    + "2 * eps, a threshold the index does not take). Pass "
                    + "algorithm='brute', which serves this metric. This is a "
                    + "scope boundary and not a property of the algorithm: the "
                    + "ball cover's pruning rests on the triangle inequality, "
                    + "which L1 satisfies, so an L1 index is reachable work in "
                    + "that lane")
    var predicting = Bool(py=prediction_data)
    if metric == _METRIC_PRECOMPUTED and predicting:
        raise Error("mojolearn DBSCAN: prediction_data=True with "
                    + "metric='precomputed' is refused by name: predict needs each "
                    + "query's distances to the core samples, which a distance "
                    + "matrix over the training rows does not give")
    var eps_value = family_field(fields, "eps")
    host_inference_scalar(eps_value, context, True)
    var eps = Float64(py=family_call(b.float, eps_value))
    if eps <= 0:
        raise Error("mojolearn DBSCAN eps must be positive")
    var min_value = family_field(fields, "min_samples")
    host_inference_scalar(min_value, context, True)
    var min_samples = Int(py=family_call(b.int, min_value))
    if min_samples < 1:
        raise Error("mojolearn DBSCAN min_samples must be at least 1")
    var cap_attr = family_field(fields, "max_iterations")
    host_inference_scalar(cap_attr, context, True)
    var cap = 0 if cap_attr is None else Int(py=family_call(b.int, cap_attr))
    if cap < 0:
        raise Error("mojolearn DBSCAN max_iterations must be None (the fixed "
                    + "point) or a positive cap")
    # args[3] (present and True) marks `source` as the X the public door
    # already admitted with classical_check_x (providers/python/
    # density_public_lifecycle_pyapi.mojo): 2-D, non-empty, C-order float32
    # and refused there when not finite. That X is used as is: no second
    # snapshot and no second finite pass over it (perf3-cluster-gp-gate;
    # perf2-cluster row 24: one extra full pass + one admission per fit).
    # Any other caller keeps the full admission below.
    if len(args) > 3 and py_type(args[3]) is not b.bool:
        raise Error("TypeError: DBSCAN admission marker requires an exact bool")
    var admitted = len(args) > 3 and Bool(py=args[3])
    var x = source if admitted else admit_family(source, 2, "X", context).array
    py_setitem(fields, "input_copied_", PythonObject(not (x is source)))
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if not admitted and not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), Int(py=x.size)):
        # Before this check a NaN row reached the kernel and came back
        # labelled noise (-1), silently; scikit-learn refuses it.
        raise Error("mojolearn DBSCAN: X contains a NaN or an infinity; a "
                    + "non-finite row has no distance to any other, so it is "
                    + "refused by name")
    _ = x  # keep the Python object alive through the native calls above
    if metric == _METRIC_PRECOMPUTED:
        if n != d:
            raise Error("mojolearn DBSCAN: metric='precomputed' needs a square "
                        + "distance matrix, got shape " + family_repr(py_call(b.tuple, x.shape)))
        if Float64(family_min(x)) < 0:
            raise Error("mojolearn DBSCAN: metric='precomputed' refuses a "
                        + "negative distance, as scikit-learn does")
    var labels = dbscan_array(List[Int](length=1, fill=n), "<i4", context)
    var budget_attr = family_field(fields, "max_mbytes_per_batch")
    host_inference_scalar(budget_attr, context, True)
    var budget = 0 if budget_attr is None else Int(py=family_call(b.int, budget_attr))
    if budget < 0:
        raise Error("mojolearn DBSCAN max_mbytes_per_batch cannot be negative")
    # sample_weight: `0` is the binding's `sample_weight == nullptr`. The
    # weights are held until after the call (their address is in flight).
    var w = PythonObject(None)
    var weight_addr = 0
    if weights_source is not None:
        w = admit_family_weights(weights_source, n, True, context).keep
        weight_addr = Int(py=classical_address(w))
        _ = w  # keep the Python object alive through the native call above
    # ORDER MATCHES bindings/_mojolearn_estimators.mojo::dbscan_fit_binding:
    # n_rows, n_features, eps, min_samples, budget_mb, max_iter,
    # eps_nn_method, metric. precomputed takes the brute arm's adjacency path
    # whatever `algorithm` names (DEVIATION 5114).
    if metric == _METRIC_PRECOMPUTED:
        method = _EPS_NN_BRUTE_FORCE
    var workspace = DBSCANFitWorkspace(n)
    var connection = DBSCANConnection(binding)
    var x_address = Int(py=classical_address(x))
    var labels_address = Int(py=classical_address(labels))
    var request = DBSCANFitRequest(x_address, labels_address, weight_addr, workspace.address(),
        n, d, min_samples, budget, cap, method, metric, eps)
    var n_iter = dbscan_run_fit(connection, request)
    _ = labels
    _ = x
    _ = workspace
    _ = w^
    dbscan_publish(fields, "n_iter_", PythonObject(n_iter), context)
    dbscan_publish(fields, "labels_", labels, context)
    dbscan_publish(fields, "n_features_in_", PythonObject(d), context)
    dbscan_publish(fields, "n_samples_fit_", PythonObject(n), context)
    # Keep the original count and gather passes, publication order and exact
    # destination sizes. The private mask and inter-call dimensions are Mojo.
    var count_request = DBSCANCoreRequest(x_address, labels_address, workspace.address(), n, d)
    var m = dbscan_run_count(connection, count_request)
    var indices = dbscan_array(List[Int](length=1, fill=m), "<i4", context)
    var shape: List[Int] = [m, d]
    var components = dbscan_array(shape, "<f4", context)
    var core_labels = dbscan_array(List[Int](length=1, fill=m), "<i4", context)
    var gather_request = DBSCANGatherRequest(x_address, labels_address, workspace.address(),
        Int(py=classical_address(indices)), Int(py=classical_address(components)), Int(py=classical_address(core_labels)), n, d, m)
    dbscan_run_gather(connection, gather_request)
    _ = workspace^
    _ = x^
    _ = labels
    # p3-neighbors-cluster: the core rows behind predict's resident device
    # copy (native identity key, bindings/dbscan_native_boundary.mojo) are read-only.
    _ = fitted_freeze(indices, context)
    _ = fitted_freeze(components, context)
    _ = fitted_freeze(core_labels, context)
    dbscan_publish(fields, "core_sample_indices_", indices, context)
    dbscan_publish(fields, "components_", components, context)
    dbscan_publish(fields, "_core_labels", core_labels, context)
    dbscan_fresh_holder(model)
    if predicting:
        dbscan_mint(model, connection, components, indices, core_labels, m, d, metric, context)


def _dbscan_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject, resident: Bool = True) raises -> PythonObject:
    """`DBSCAN.predict`: the nearest core sample's label within eps, over
    the resident core rows."""
    var b = native_loaded_module("builtins")
    if not family_raw_has(model, "labels_"):
        raise Error("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
    var components = family_raw_get_or(model, "components_", PythonObject(None))
    var predicting = family_raw_get(model, "prediction_data")
    host_inference_scalar(predicting, context, True)
    if not Bool(py=predicting) or components is None:
        raise Error("mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                    + "DBSCAN(prediction_data=True), which keeps the core samples this "
                    + "rule needs (DEVIATION 2740)")
    var features = family_raw_get(model, "n_features_in_")
    host_inference_scalar(features, context)
    var n_features = Int(py=features)
    var q = _reference_queries(source, n_features, "DBSCAN.predict", context)
    var nq = Int(py=q.shape[0])
    var n_core = Int(py=components.shape[0])
    if n_core == 0:
        var labels = dbscan_array(List[Int](length=1, fill=nq), "<i4", context)
        dbscan_empty_labels(Int(py=classical_address(labels)), nq)
        _ = labels  # keep the Python object alive through the native call above
        return labels
    var metric_name = _dbscan_text(family_raw_get(model, "metric"), context).lower()
    var metric = dbscan_metric_code(metric_name)
    if metric < 0:
        raise Error("KeyError: " + family_repr(PythonObject(metric_name)))
    var eps_value = family_raw_get(model, "eps")
    host_inference_scalar(eps_value, context, True)
    var eps = Float64(py=family_call(b.float, eps_value))
    var result = dbscan_reference_labels(model, binding, components,
                                      family_raw_get(model, "core_sample_indices_"),
                                      family_raw_get(model, "_core_labels"), q, n_core,
                                      n_features, metric, eps, context, resident)
    _ = q^
    return result


def cfb_dbscan_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """args: (binding, X[, sample_weight[, admitted]]), unpacked here once."""
    lifecycle_admit_fields(py_fields(model))
    if operation == "family_cfb_dbscan_fit":
        _dbscan_fit(model, args, context)
        return model
    if operation == "family_cfb_dbscan_fit_predict":
        _dbscan_fit(model, args, context)
        return py_fields(model)["labels_"]
    if operation == "family_cfb_dbscan_predict":
        return _dbscan_predict(model, args[0], args[1], context)
    if operation == "family_xc_dbscan_predict":
        # The historical internal entry always used direct reference uploads.
        return _dbscan_predict(model, args[0], args[1], context, False)
    raise Error("RuntimeError: unknown DBSCAN operation " + operation)
