# SPDX-License-Identifier: Apache-2.0
# UNPACK of DBSCAN's public operations (`fit`, `fit_predict`, `predict`):
# the estimator's settings read once into the typed `DBSCANFitSettings`
# (the Python method's refusals in its order and wording; the refusals that
# compare a number are the typed RUN of cluster/dbscan_door_run.mojo), the
# admission marker, the batch budget, and predict's fitted-state reads.
# RUN is cluster/dbscan_door_run.mojo with the typed provider calls of
# core/dbscan_native_api.mojo; PACK and the operators are
# bindings/dbscan_public_native_pack.mojo. (This file and that one were the
# single door bindings/cfb_dbscan_door.mojo.)
from bindings.admitted_family_door import admit_family, admit_family_weights
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_arrays_door import classical_address, classical_is_array
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_is_str,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
    family_field,
)
from cluster.dbscan_door_run import (
    dbscan_check_method, dbscan_check_prediction, dbscan_check_eps, dbscan_check_min_samples,
    dbscan_check_max_iterations, dbscan_check_budget, dbscan_check_x, dbscan_fit_request, dbscan_core_request,
)
from core.dbscan_native_state import dbscan_method_accepts
from bindings.cluster_native_door import _reference_queries
from bindings.dbscan_native_boundary_unpack import DBSCANConnection, dbscan_run_fit, dbscan_run_count, dbscan_run_gather
from bindings.dbscan_native_boundary_pack import (
    dbscan_array, dbscan_publish, dbscan_fresh_holder, dbscan_mint, dbscan_reference_labels,
)
from core.dbscan_native_api import DBSCANGatherRequest, DBSCANFitWorkspace, dbscan_empty_labels
from core.dbscan_native_state import dbscan_metric_code, dbscan_method_code
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.decomp_scalar_admission_unpack import decomp_scalar_repr
from bindings.operation_names_cfb import (
    CFB_OPN_FAMILY_CFB_DBSCAN_FIT, CFB_OPN_FAMILY_CFB_DBSCAN_FIT_PREDICT, CFB_OPN_FAMILY_CFB_DBSCAN_PREDICT,
    CFB_OPN_FAMILY_XC_DBSCAN_PREDICT)
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_unpack import family_raw_has
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.cluster_keys_unpack import cluster_keys
from bindings.py_balanced import py_item


def _dbscan_text(value: PythonObject, context: PythonObject) raises -> String:
    host_inference_scalar(value, context, True)
    var nk = cluster_keys()
    if py_type(value) is py_get(context, nk[].k_ndarray):
        return decomp_scalar_repr(value, context)
    return String(py=family_call(native_loaded_module("builtins").str, value))


@fieldwise_init
struct DBSCANFitSettings(Copyable, Movable):
    """The fit's settings as RUN takes them: Mojo values only."""
    var predicting: Bool
    var metric: Int
    var method: Int
    var eps: Float64
    var min_samples: Int
    var cap: Int


def dbscan_fit_settings(fields: PythonObject, context: PythonObject) raises -> DBSCANFitSettings:
    """prediction_data, metric, algorithm, eps, min_samples and
    max_iterations, each refused in the public order and wording."""
    var b = native_loaded_module("builtins")
    var nk = cluster_keys()
    var prediction_data = family_field(fields, "prediction_data")
    if py_type(prediction_data) is not b.bool:
        raise Error("TypeError: mojolearn DBSCAN: prediction_data must be a bool, got "
                    + String(py_call(py_item(b.type.__dict__, nk[].k___name__).__get__, py_type(prediction_data))))
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
    # The two reprs are built only on refusal, as before the door split.
    if not dbscan_method_accepts(method, metric):
        dbscan_check_method(method, metric, decomp_scalar_repr(metric_attr, context),
                            decomp_scalar_repr(algorithm, context))
    var predicting = Bool(py=prediction_data)
    dbscan_check_prediction(metric, predicting)
    var eps_value = family_field(fields, "eps")
    host_inference_scalar(eps_value, context, True)
    var eps = Float64(py=family_call(b.float, eps_value))
    dbscan_check_eps(eps)
    var min_value = family_field(fields, "min_samples")
    host_inference_scalar(min_value, context, True)
    var min_samples = Int(py=family_call(b.int, min_value))
    dbscan_check_min_samples(min_samples)
    var cap_attr = family_field(fields, "max_iterations")
    host_inference_scalar(cap_attr, context, True)
    var cap = 0 if cap_attr is None else Int(py=family_call(b.int, cap_attr))
    dbscan_check_max_iterations(cap)
    return DBSCANFitSettings(predicting, metric, method, eps, min_samples, cap)


def dbscan_fit_marker(args: PythonObject) raises -> Bool:
    """args[3] (present and True) marks X as the one the public door
    already admitted with classical_check_x."""
    if len(args) > 3 and py_type(args[3]) is not native_loaded_module("builtins").bool:
        raise Error("TypeError: DBSCAN admission marker requires an exact bool")
    return len(args) > 3 and Bool(py=args[3])


def dbscan_fit_budget(fields: PythonObject, context: PythonObject) raises -> Int:
    """max_mbytes_per_batch (0: the provider's own budget)."""
    var b = native_loaded_module("builtins")
    var budget_attr = family_field(fields, "max_mbytes_per_batch")
    host_inference_scalar(budget_attr, context, True)
    var budget = 0 if budget_attr is None else Int(py=family_call(b.int, budget_attr))
    dbscan_check_budget(budget)
    return budget


def dbscan_predict_components(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The fitted and prediction-data refusals of predict; the core rows."""
    if not family_raw_has(model, "labels_"):
        raise Error("mojolearn DBSCAN.predict: this DBSCAN instance is not fitted yet; call fit first")
    var components = family_raw_get_or(model, "components_", PythonObject(None))
    var predicting = family_raw_get(model, "prediction_data")
    host_inference_scalar(predicting, context, True)
    if not Bool(py=predicting) or components is None:
        raise Error("mojolearn DBSCAN.predict: prediction data was not stored. Fit with "
                    + "DBSCAN(prediction_data=True), which keeps the core samples this "
                    + "rule needs (DEVIATION 2740)")
    return components


def dbscan_predict_features(model: PythonObject, context: PythonObject) raises -> Int:
    var features = family_raw_get(model, "n_features_in_")
    host_inference_scalar(features, context)
    return Int(py=features)


def dbscan_predict_metric(model: PythonObject, context: PythonObject) raises -> Int:
    var metric_name = _dbscan_text(family_raw_get(model, "metric"), context).lower()
    var metric = dbscan_metric_code(metric_name)
    if metric < 0:
        raise Error("KeyError: " + family_repr(PythonObject(metric_name)))
    return metric


def dbscan_predict_eps(model: PythonObject, context: PythonObject) raises -> Float64:
    var eps_value = family_raw_get(model, "eps")
    host_inference_scalar(eps_value, context, True)
    return Float64(py=family_call(native_loaded_module("builtins").float, eps_value))
