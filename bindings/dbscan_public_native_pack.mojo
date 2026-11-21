# SPDX-License-Identifier: Apache-2.0
# DBSCAN `fit`, `fit_predict` and `predict`, one native call each
# (python/mojolearn/density.py): the operators (UNPACK -> RUN -> PACK) and
# the PACK half: the public output arrays, the fitted-field publication and
# the resident core rows. The image is the estimator's binding route, handed
# over by the density lifecycle. With `prediction_data=True` the core rows
# become resident on the device at fit, so `predict` uploads only its
# queries. Existing kernels/defaults and transfer counts are retained.
# UNPACK is bindings/dbscan_public_native_unpack.mojo; RUN is
# cluster/dbscan_door_run.mojo with the typed provider calls of
# core/dbscan_native_api.mojo. (This file and the UNPACK file were the
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
from bindings.dbscan_public_native_unpack import (
    DBSCANFitSettings, dbscan_fit_settings, dbscan_fit_marker, dbscan_fit_budget,
    dbscan_predict_components, dbscan_predict_features, dbscan_predict_metric, dbscan_predict_eps,
)


def _dbscan_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`DBSCAN.fit(X, y=None, sample_weight=None)`."""
    var fields = py_fields(model)
    lifecycle_admit_fields(fields)
    var binding = args[0]
    var source = args[1]
    var weights_source = args[2]
    # UNPACK: the settings, typed. The builtins module is resolved once
    # per fit, as it was, and handed to the three UNPACK steps.
    var b = native_loaded_module("builtins")
    var settings = dbscan_fit_settings(fields, b, context)
    var metric = settings.metric
    var nk = cluster_keys()
    # args[3] (present and True) marks `source` as the X the public door
    # already admitted with classical_check_x (providers/python/
    # density_public_lifecycle_pyapi.mojo): 2-D, non-empty, C-order float32
    # and refused there when not finite. That X is used as is: no second
    # snapshot and no second finite pass over it (perf3-cluster-gp-gate;
    # perf2-cluster row 24: one extra full pass + one admission per fit).
    # Any other caller keeps the full admission below.
    var admitted = dbscan_fit_marker(args, b)
    var x = source if admitted else admit_family(source, 2, "X", context).array
    py_setitem(fields, nk[].k_input_copied_, PythonObject(not (x is source)))
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var x_address = Int(py=classical_address(x))
    dbscan_check_x(x_address, n, d, admitted, metric)
    _ = x  # keep the Python object alive through the native checks above
    var labels = dbscan_array(List[Int](length=1, fill=n), "<i4", context)
    var budget = dbscan_fit_budget(fields, b, context)
    # sample_weight: `0` is the binding's `sample_weight == nullptr`. The
    # weights are held until after the call (their address is in flight).
    var w = PythonObject(None)
    var weight_addr = 0
    if weights_source is not None:
        var w_pin = ArrayPin()
        _ = admit_family_weights(weights_source, n, True, context, w_pin)
        w = w_pin.keep
        weight_addr = Int(py=classical_address(w))
        _ = w  # keep the Python object alive through the native call above
    var workspace = DBSCANFitWorkspace(n)
    var connection = DBSCANConnection(binding)
    var labels_address = Int(py=classical_address(labels))
    var request = dbscan_fit_request(x_address, labels_address, weight_addr, workspace.address(),
        n, d, settings.min_samples, budget, settings.cap, settings.method, metric, settings.eps)
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
    var count_request = dbscan_core_request(x_address, labels_address, workspace.address(), n, d)
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
    # copy (native identity key, bindings/dbscan_native_boundary_unpack.mojo) are read-only.
    _ = fitted_freeze(indices, context)
    _ = fitted_freeze(components, context)
    _ = fitted_freeze(core_labels, context)
    dbscan_publish(fields, "core_sample_indices_", indices, context)
    dbscan_publish(fields, "components_", components, context)
    dbscan_publish(fields, "_core_labels", core_labels, context)
    dbscan_fresh_holder(model)
    if settings.predicting:
        dbscan_mint(model, connection, components, indices, core_labels, m, d, metric, context)


def _dbscan_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject, resident: Bool = True) raises -> PythonObject:
    """`DBSCAN.predict`: the nearest core sample's label within eps, over
    the resident core rows. Same reads in the same order as the single
    body this replaced."""
    var components = dbscan_predict_components(model, context)
    var n_features = dbscan_predict_features(model, context)
    var q = _reference_queries(source, n_features, "DBSCAN.predict", context)
    var nq = Int(py=q.shape[0])
    var n_core = Int(py=components.shape[0])
    if n_core == 0:
        var labels = dbscan_array(List[Int](length=1, fill=nq), "<i4", context)
        dbscan_empty_labels(Int(py=classical_address(labels)), nq)
        _ = labels  # keep the Python object alive through the native call above
        return labels
    var metric = dbscan_predict_metric(model, context)
    var eps = dbscan_predict_eps(model, context)
    var result = dbscan_reference_labels(model, binding, components,
                                      family_raw_get(model, "core_sample_indices_"),
                                      family_raw_get(model, "_core_labels"), q, n_core,
                                      n_features, metric, eps, context, resident)
    _ = q^
    return result


def cfb_dbscan_operation(model: PythonObject, operation: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """args: (binding, X[, sample_weight[, admitted]]), unpacked here once.
    `operation` is the leaf code of bindings/operation_names_cfb.mojo
    (DBSCAN block; lane bw2-cluster: it was the name, four String compares)."""
    lifecycle_admit_fields(py_fields(model))
    if operation == CFB_OPN_FAMILY_CFB_DBSCAN_FIT:
        _dbscan_fit(model, args, context)
        return model
    if operation == CFB_OPN_FAMILY_CFB_DBSCAN_FIT_PREDICT:
        var nk = cluster_keys()
        _dbscan_fit(model, args, context)
        return py_item(py_fields(model), nk[].k_labels_)
    if operation == CFB_OPN_FAMILY_CFB_DBSCAN_PREDICT:
        return _dbscan_predict(model, args[0], args[1], context)
    if operation == CFB_OPN_FAMILY_XC_DBSCAN_PREDICT:
        # The historical internal entry always used direct reference uploads.
        return _dbscan_predict(model, args[0], args[1], context, False)
    raise Error("RuntimeError: unknown DBSCAN operation code " + String(operation))
