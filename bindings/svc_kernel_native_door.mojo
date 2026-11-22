# SPDX-License-Identifier: Apache-2.0
# SVC `fit`, `decision_function`, `predict`, `predict_proba`,
# `predict_log_proba`, `score`, `coef_` and SVR `fit`, `predict`, `score`,
# `coef_` (python/mojolearn/_svm_impl.py), one native call each, in three
# parts. This file only sequences them.
#   UNPACK  svc_kernel_params_unpack.mojo (`_validate_params` with its
#           refusals in the same wording and order, the settings record,
#           the bounds' operands), svc_kernel_rows_unpack.mojo (the fit and
#           query X, the status-word refusals), svc_kernel_provider_unpack.mojo
#           (the binding's entry tables), svc_kernel_fit_unpack.mojo,
#           svc_kernel_query_unpack.mojo, svc_kernel_svr_unpack.mojo: every
#           Python read of the operation, the owners, the typed plan;
#   RUN     svm/svc_kernel_fit_run.mojo, svm/svc_kernel_native_run.mojo and
#           core/svr_native_api.mojo over core/svc_native_api.mojo's typed
#           drivers, the GIL released, no Python object: the exact gamma
#           (the binding's `scale_gamma_limbs`, then
#           core/scale_gamma_exact.mojo's once-rounded rational), the
#           per-row bounds, the binary machine, the one-vs-one machines over
#           the class pairs, libsvm's Platt cross-validation per pair (five
#           folds), then the decision, vote and probability epilogues;
#   PACK    svc_kernel_fit_pack.mojo, svc_kernel_query_pack.mojo,
#           svc_kernel_svr_pack.mojo: the public arrays and the fitted
#           fields, published together after RUN.
# Every number comes from the same owners called with the same words on the
# same operands in the same order as the by-name entries the door used to
# call through CPython (`svc_fit`, `svc_predict`, `svc_predict_resident`,
# `svc_pair_epilogue`, `svc_splitmix_perm`, `svc_platt_train`,
# `svc_ovo_layout`, `svc_dual_gemv`, `scale_gamma_limbs`), so the bits do
# not move.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_type, py_get, py_fields, py_item
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.gp_svm_records_door import gpsvm_no_fit_callbacks
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.native_array_view_pack import native_array_ndarray_view
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from std.python._cpython import GILReleased
from core.svc_native_api import SvcDrivers
from svm.svc_kernel_native_run import svc_query_run, svc_coef_run, svc_gather_columns_run
from svm.svc_kernel_fit_run import SvcFitResult, svc_fit_run, svc_fit_finish_run, svc_gamma_run, svc_bounds_run
from bindings.svc_kernel_fit_unpack import svc_fit_unpack, svc_fit_refused
from bindings.svc_kernel_fit_pack import svc_fit_vectors, svc_fit_pack
from bindings.svc_kernel_provider_unpack import SvcConnection, svc_connection, svc_connection_publish
from bindings.svc_kernel_query_unpack import svc_query_unpack, svc_query_refused, svc_coef_unpack, SVC_QUERY_DECISION, SVC_QUERY_CODES, SVC_QUERY_PROBA, SVC_QUERY_LOG_PROBA
from bindings.svc_kernel_query_pack import svc_query_result, svc_query_pack, svc_coef_output
from bindings.svc_kernel_svr_unpack import svr_fit_unpack, svr_query_unpack, svr_query_refused, svr_score_unpack
from bindings.svc_kernel_svr_pack import svr_fit_pack, svr_query_output, svr_score_pack
from bindings.svc_common_unpack import svc_field
from bindings.svc_common_pack import svc_decode
from bindings.svc_common_door import svc_accuracy_codes
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.operation_names_svc import SVC_OPN_FAMILY_SVC_SVC_COEF, SVC_OPN_FAMILY_SVC_SVC_DECISION, SVC_OPN_FAMILY_SVC_SVC_FIT, SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA, SVC_OPN_FAMILY_SVC_SVC_PREDICT, SVC_OPN_FAMILY_SVC_SVC_PROBA, SVC_OPN_FAMILY_SVC_SVC_SCORE, SVC_OPN_FAMILY_SVC_SVC_VALIDATE, SVC_OPN_FAMILY_SVC_SVR_FIT, SVC_OPN_FAMILY_SVC_SVR_PREDICT, SVC_OPN_FAMILY_SVC_SVR_SCORE, SVC_OPN_FAMILY_SVC_SVR_VALIDATE
from bindings.svm_keys_unpack import svm_keys
from bindings.svc_kernel_rows_unpack import _addr
from bindings.svc_kernel_params_unpack import _svm_validate

# ---------------------------------------------------------------- SVC fit


def _svc_fit(model: PythonObject, args: PythonObject, context: PythonObject, drivers: SvcDrivers) raises:
    """`SVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight):
    UNPACK; RUN (gamma, the bounds, the binary machine or the one-vs-one
    machines and their layout) in one release of the GIL; the refusals of
    what the solves reported; then, when there is more to run (the
    one-vs-one support vectors gathered into their public array, the
    probability fit), a second release; PACK."""
    var unpacked = svc_fit_unpack(model, args, context, drivers)
    var result = SvcFitResult()
    with GILReleased(Python()):
        svc_fit_run(unpacked.drivers, unpacked.plan, result)
    svc_fit_refused(unpacked, result)
    var vectors = svc_fit_vectors(unpacked, result, context)
    if vectors is not None or unpacked.plan.probability:
        var vectors_address = _addr(vectors) if vectors is not None else 0
        with GILReleased(Python()):
            svc_fit_finish_run(unpacked.drivers, unpacked.plan, vectors_address, result)
    svc_fit_pack(unpacked, result, vectors, context)
    # The owners of X, the labels, the codes and the weights outlive RUN and PACK.
    _ = unpacked


# ---------------------------------------------------------------- SVC inference


def _svc_query(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject, kind: Int,
               context: PythonObject, drivers: SvcDrivers) raises -> PythonObject:
    """One inference operation (`kind`: SVC_QUERY_*): UNPACK, the public
    array, every machine and the epilogue in ONE release of the GIL, the
    refusal of a query the device entry flagged, PACK."""
    var unpacked = svc_query_unpack(model, fields, binding, source, kind, context, drivers)
    var result = svc_query_result(unpacked, context)
    var complete = False
    with GILReleased(Python()):
        complete = svc_query_run(unpacked.drivers, unpacked.plan)
    if not complete:
        svc_query_refused(unpacked)
        raise Error("RuntimeError: mojolearn SVC: the device entry flagged the query")
    var packed = svc_query_pack(unpacked, result, context)
    # The owners of X, the fitted arrays, the scratch and the lease outlive RUN and PACK.
    _ = unpacked
    return packed


# ---------------------------------------------------------------- SVR


def _svr_fit(model: PythonObject, args: PythonObject, context: PythonObject, connection: SvcConnection) raises:
    """`SVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight):
    UNPACK; in ONE release of the GIL the gamma of host words, the bounds
    C * sample_weight and the typed solve; PACK."""
    var unpacked = svr_fit_unpack(model, args, context, connection.drivers, connection.candidate)
    var count = 0
    with GILReleased(Python()):
        if unpacked.host_gamma:
            unpacked.request.gamma = svc_gamma_run(connection.drivers, unpacked.gamma_kind, unpacked.request.gamma,
                                                   unpacked.request.x, unpacked.cells, unpacked.request.features)
        if unpacked.request.bounds != 0:
            svc_bounds_run(connection.drivers, unpacked.request.rows, unpacked.sample_weights, 0, 0, 0,
                           unpacked.request.c, unpacked.request.bounds)
        count = unpacked.connection.driver.fit(unpacked.request)
    unpacked.plan.count = count
    unpacked.gamma = unpacked.request.gamma
    svr_fit_pack(unpacked, context)
    # The owners of X, the targets, the weights and the provider outlive RUN and PACK.
    _ = unpacked


def _svr_predict(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject, connection: SvcConnection) raises -> PythonObject:
    """`SVR.predict`: UNPACK, the public vector, then in one release of
    the GIL the cross-kernel's support columns (kernel='precomputed') and
    one typed query; then the refusal of a query the device entry flagged."""
    var unpacked = svr_query_unpack(model, fields, binding, source, context, connection.candidate)
    var result = svr_query_output(unpacked.request.rows, context)
    unpacked.request.dst = _addr(result)
    with GILReleased(Python()):
        if unpacked.gather_columns != 0:
            svc_gather_columns_run(connection.drivers, unpacked.q.address, unpacked.q.rows, unpacked.q.cols,
                                   unpacked.gather_columns, unpacked.gather_count, unpacked.request.x)
        _ = unpacked.connection.driver.query(unpacked.request, unpacked.resident)
    svr_query_refused(unpacked)
    # The owners of X, the fitted arrays, the lease and the provider outlive RUN.
    _ = unpacked
    return result


def _svr_score(model: PythonObject, fields: PythonObject, binding: PythonObject, source: PythonObject, y: PythonObject,
               context: PythonObject, connection: SvcConnection) raises -> PythonObject:
    """`SVR.score`: R^2 of the prediction against y."""
    var pred = _svr_predict(model, fields, binding, source, context, connection)
    svr_score_unpack(Int(py=pred.shape[0]), y, context)
    # The R^2 sums run on the estimator's device's metrics image (labels_scope).
    return svr_score_pack(pred, y, labels_scope(model, context))


def _linear_coef(fields: PythonObject, classifier: Bool, context: PythonObject, drivers: SvcDrivers) raises -> PythonObject:
    """Public `coef_` and composed RFE: UNPACK (every fitted buffer captured
    once), one GEMV per fitted machine with the GIL released, the matrix."""
    var unpacked = svc_coef_unpack(fields, classifier, context)
    var result = svc_coef_output(len(unpacked.rows), unpacked.features, context)
    var address = _addr(result)
    with GILReleased(Python()):
        svc_coef_run(drivers, unpacked.rows, unpacked.features, unpacked.sign, address)
    _ = unpacked
    return result


# ---------------------------------------------------------------- dispatch


def _svm_lifecycle_admit(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """Fit bookkeeping read by the public lifecycle: exact native values,
    and no Python completion callback a native fit could be asked to run."""
    var b = native_loaded_module("builtins")
    var nk = svm_keys()
    if py_type(py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))) is not b.int:
        raise Error("TypeError: SVM fitting depth requires an exact integer")
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: SVM fitted state requires a native boolean or integer")
    # The ancestry scan runs once per class state (bindings/gp_svm_records.mojo).
    gpsvm_no_fit_callbacks(model, fields, "TypeError: SVM fitting cannot invoke Python completion callbacks")


def _svc_method(operation: Int) -> String:
    """The public method a kernel-machine operation serves, for the
    not-fitted text (a static literal; the old door built it per call from
    five suffix tests)."""
    if operation == SVC_OPN_FAMILY_SVC_SVC_DECISION:
        return "decision_function"
    if operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA:
        return "predict_log_proba"
    if operation == SVC_OPN_FAMILY_SVC_SVC_PROBA:
        return "predict_proba"
    if operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT or operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT:
        return "predict"
    if operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE:
        return "score"
    return "fit"


def svc_kernel_operation(model: PythonObject, operation: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The kernel machines' whole public operation: argument admission, the
    fitted and feature-name checks, fit bookkeeping (depth, fitted flag,
    feature names, public class labels), the algorithm through one admitted
    provider, and NumPy results for callers outside the package.
    `operation` is the leaf code of bindings/operation_names_svc.mojo."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: SVM public arguments require an exact tuple")
    var fields = py_fields(model)
    var classifier = operation >= SVC_OPN_FAMILY_SVC_SVC_COEF and operation <= SVC_OPN_FAMILY_SVC_SVC_VALIDATE
    if operation == SVC_OPN_FAMILY_SVC_SVC_VALIDATE or operation == SVC_OPN_FAMILY_SVC_SVR_VALIDATE:
        _svm_validate(model, fields, classifier, context)
        return PythonObject(None)
    var fitting = operation == SVC_OPN_FAMILY_SVC_SVC_FIT or operation == SVC_OPN_FAMILY_SVC_SVR_FIT
    var public = (operation == SVC_OPN_FAMILY_SVC_SVC_DECISION or operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT
                  or operation == SVC_OPN_FAMILY_SVC_SVC_PROBA or operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA
                  or operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT
                  or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE)
    if not fitting and not public:
        # coef_ (the native property schema) and composed callers: algorithm only.
        var connected = svc_connection(args[0])
        var plain = _linear_coef(fields, operation == SVC_OPN_FAMILY_SVC_SVC_COEF, context, connected.drivers)
        svc_connection_publish(fields, connected, context)
        return plain
    _svm_lifecycle_admit(model, fields, context)
    var nk = svm_keys()
    var depth = Int(py=py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0)))
    if public:
        classical_require_fitted(model, False, _svc_method(operation), context)
        if depth == 0:
            estimator_feature_check(model, fields, args[1], context)
    # One provider admission per public operation; RUN takes its typed drivers.
    var connection = svc_connection(args[0])
    if fitting:
        var initial_depth = estimator_fit_begin(fields)
        try:
            if operation == SVC_OPN_FAMILY_SVC_SVC_FIT:
                _svc_fit(model, args, context, connection.drivers)
            else:
                _svr_fit(model, args, context, connection)
        finally:
            estimator_fit_end(model, initial_depth)
        # A nested set_params may have replaced the dictionary's contents.
        fields = py_fields(model)
        svc_connection_publish(fields, connection, context)
        estimator_fit_complete(model, args[1], classifier, context)
        return model
    var result = _svc_kernel_operation(model, operation, args, context, fields, connection)
    # A nested set_params may have replaced the dictionary's contents.
    fields = py_fields(model)
    svc_connection_publish(fields, connection, context)
    # predict has one result for every caller (Andrew, 2026-11-22, lane
    # predict-dtype): the kernel's float32 regression values (or the decoded
    # labels) as a NumPy view, with no caller test and no float64 widening
    # pass into a second buffer. The other public results keep the caller test.
    var predicting = operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT or operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT
    if (operation == SVC_OPN_FAMILY_SVC_SVC_SCORE or operation == SVC_OPN_FAMILY_SVC_SVR_SCORE
            or py_get(context, nk[].k_ndarray) is None
            or py_type(result) is not py_item(context, nk[].k_array_class)
            or (not predicting and not estimator_outside())):
        return result
    return native_array_ndarray_view(result, context)


def _svc_kernel_operation(model: PythonObject, operation: Int, args: PythonObject,
                          context: PythonObject, fields: PythonObject,
                          connection: SvcConnection) raises -> PythonObject:
    if operation == SVC_OPN_FAMILY_SVC_SVC_DECISION:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_DECISION, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_PREDICT:
        # the predicted codes decoded to the fitted classes
        var nk = svm_keys()
        return svc_decode(model, svc_field(fields, nk[].k_classes_, "classes_"),
                          _svc_query(model, fields, args[0], args[1], SVC_QUERY_CODES, context, connection.drivers),
                          labels_scope(model, context))
    if operation == SVC_OPN_FAMILY_SVC_SVC_PROBA:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_PROBA, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_LOG_PROBA:
        return _svc_query(model, fields, args[0], args[1], SVC_QUERY_LOG_PROBA, context, connection.drivers)
    if operation == SVC_OPN_FAMILY_SVC_SVC_SCORE:
        # N382: accuracy on codes. y is encoded once against classes_ and
        # compared with the predicted codes in the x_metrics COUNT program
        # (classical_score_from_codes, bindings/classical_labels_door.mojo); the
        # prediction is never decoded to label objects and re-encoded. The
        # same matched count over the same row count: no bit change.
        var codes = _svc_query(model, fields, args[0], args[1], SVC_QUERY_CODES, context, connection.drivers)
        var nk = svm_keys()
        return PythonObject(svc_accuracy_codes(args[2], svc_field(fields, nk[].k_classes_, "classes_"), codes,
                                               Int(py=codes.shape[0]), labels_scope(model, context)))
    if operation == SVC_OPN_FAMILY_SVC_SVR_PREDICT:
        return _svr_predict(model, fields, args[0], args[1], context, connection)
    if operation == SVC_OPN_FAMILY_SVC_SVR_SCORE:
        return _svr_score(model, fields, args[0], args[1], args[2], context, connection)
    raise Error("RuntimeError: unknown kernel SVM operation " + String(operation))
