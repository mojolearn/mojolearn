# SPDX-License-Identifier: Apache-2.0
# LinearSVC `fit`, `predict`, `score` and LinearSVR `fit`, `score`
# (python/mojolearn/svm.py), one native call each, in three parts:
#   UNPACK  bindings/svc_linear_boundary_unpack.mojo: the parameter checks of
#           `_LinearSVMBase._check_common`, the label encoding, the per-row
#           weights of `linear_model._qn_row_weights`, the typed requests and
#           the pins of every addressed buffer;
#   RUN     the typed driver calls of bindings/linear_svc_boundary_unpack.mojo
#           (`linear_svc_run_fit`, `linear_svr_run_fit`, `linear_svc_run_query`:
#           core/linear_svc_native_api.mojo requests, the GIL released), the
#           decision scores of bindings/svm_mixture_native_door.mojo and the
#           native threshold / argmax over them;
#   PACK    bindings/svc_linear_boundary_pack.mojo: the fitted public fields,
#           the decoded labels, the score.
# This file only sequences them. The one `qn_fit_ovr` (or the per-target
# `qn_fit`) call with the same parameter lists, the prediction entries and the
# score metrics are the Python methods' own, in the same order with the same
# arguments: the bits do not move. The driver is the estimator's device route,
# resolved once (first operation, a changed mode, a new __init__ / set_params)
# and held on its DeviceRouteOwner (`linear_svc_route`); the class-weight
# programs and the metrics image come from its binding route: no call here asks
# which device the process runs on.
from std.python import PythonObject
from bindings.py_balanced import py_fields
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.operation_names_svc import (
    SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT, SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE, SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT,
    SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT, SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE,
)
from bindings.svm_mixture_native_door import _qn_scores, _qn_scores_native
from bindings.linear_svc_boundary_unpack import (
    LinearSvcConnection, linear_svc_route, linear_svc_run_fit, linear_svc_run_query, linear_svr_run_fit, LinearSvcPrediction,
    LINEAR_SVC_ROUTE_ROWS, LINEAR_SVC_ROUTE_MULTICLASS,
)
from bindings.svc_linear_boundary_unpack import linear_svr_fit_unpack, linear_svc_fit_unpack, linear_svc_codes_unpack
from bindings.svc_linear_boundary_pack import (
    linear_svr_fit_pack, linear_svc_fit_pack, linear_svc_predict_pack, linear_svc_score_pack, linear_svr_score_pack,
)
from core.linear_svc_native_api import LinearSvcFitPlan
from core.qn_single_fit import QnSingleFitPlan
from core.logistic_native_api import QN_NATIVE_BINARY, QN_NATIVE_MULTICLASS
from core.labels_native_api import LabelsThresholdRequest
from bindings.labels_descriptor_boundary_door import labels_threshold, labels_argmax_rows, labels_scope
from bindings.svc_common_unpack import svc_field, svc_has
from bindings.svm_keys_unpack import svm_keys

# ---------------------------------------------------------------- LinearSVR


def _linear_svr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVR.fit(X, y, sample_weight)`; args = (None, X, y, sample_weight)."""
    var unpacked = linear_svr_fit_unpack(model, args, context)
    var plan = QnSingleFitPlan()
    linear_svr_run_fit(unpacked.connection, unpacked.request.copy(), plan)
    linear_svr_fit_pack(unpacked, plan, context)
    # The pins of X, y, the weights and the coefficient store outlive RUN and PACK.
    _ = plan
    _ = unpacked


def _linear_svr_predict(model: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var nk = svm_keys()
    if not svc_has(py_fields(model), nk[].k__w):
        raise Error("mojolearn LinearSVR: call fit before predict")
    return _qn_scores(model, source, 1, context)


# ---------------------------------------------------------------- LinearSVC


def _linear_svc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVC.fit(X, y, sample_weight)`; args = (None, X, y, sample_weight)."""
    var unpacked = linear_svc_fit_unpack(model, args, context)
    var plan = LinearSvcFitPlan(unpacked.n_fits, unpacked.n_coefs)
    plan.storage[0].prepare_info(unpacked.n_fits)
    linear_svc_run_fit(unpacked.connection, unpacked.request, plan)
    linear_svc_fit_pack(unpacked, plan, context)
    # The pins of X, the codes and the row weights outlive RUN and PACK.
    _ = plan
    _ = unpacked


def _linear_svc_prediction(model: PythonObject, source: PythonObject,
                        context: PythonObject, label_context: PythonObject) raises -> LinearSvcPrediction:
    var fields = py_fields(model)
    var nk = svm_keys()
    if svc_has(fields, nk[].k_classes_):
        var classes = svc_field(fields, nk[].k_classes_, "classes_")
        var k = len(classes)
        # One route read per call (the former door connected the provider
        # again for the decision-score fallback below).
        var connection = linear_svc_route(model, fields, context)
        if k == 2 and (connection.flags & LINEAR_SVC_ROUTE_ROWS) != 0:
            var query = linear_svc_codes_unpack(model, fields, source, 2, context)
            if query.rows > 0:
                linear_svc_run_query(connection, QN_NATIVE_BINARY, query.request)
            var codes = List[Int64]()
            swap(codes, query.codes)
            _ = query
            return LinearSvcPrediction(classes, codes^)
        if k > 2 and (connection.flags & LINEAR_SVC_ROUTE_MULTICLASS) != 0:
            var query = linear_svc_codes_unpack(model, fields, source, k, context)
            if query.rows > 0:
                linear_svc_run_query(connection, QN_NATIVE_MULTICLASS, query.request)
            var codes = List[Int64]()
            swap(codes, query.codes)
            _ = query
            return LinearSvcPrediction(classes, codes^)
        if not svc_has(fields, nk[].k__w):
            raise Error("mojolearn LinearSVC: call fit first")
        return _linear_svc_decided(model, connection, classes, k, source, context, label_context)
    if not svc_has(fields, nk[].k__w):
        raise Error("mojolearn LinearSVC: call fit first")
    var classes = svc_field(fields, nk[].k_classes_, "classes_")
    var k = len(classes)
    var connection = linear_svc_route(model, fields, context)
    return _linear_svc_decided(model, connection, classes, k, source, context, label_context)


def _linear_svc_decided(model: PythonObject, connection: LinearSvcConnection, classes: PythonObject, k: Int,
                        source: PythonObject, context: PythonObject,
                        label_context: PythonObject) raises -> LinearSvcPrediction:
    """The decision-score fallback: scores, then the native threshold (two
    classes) or argmax over them."""
    var scores = _qn_scores_native(model, connection, source, 1 if k == 2 else k, context)
    if k != 2 and k < 3:
        # A rebound class table of length 0/1 produced a 1-D decision Array;
        # the old argmax boundary then refused its missing second dimension.
        raise Error("IndexError: tuple index out of range")
    var codes = List[Int64](length=scores.rows, fill=0)
    # The label steps run on the estimator's device (`label_context`: the
    # operation's one labels_scope read).
    if k == 2:
        if scores.rows > 0:
            _ = labels_threshold(label_context, LabelsThresholdRequest(Int(scores.values.unsafe_ptr()),
                Int(codes.unsafe_ptr()), 0, scores.rows, True, 0, 1, 0))
    elif scores.rows > 0:
        labels_argmax_rows(label_context, False, Int(scores.values.unsafe_ptr()), scores.rows, k, Int(codes.unsafe_ptr()))
    _ = scores
    return LinearSvcPrediction(classes, codes^)



def svc_linear_operation(model: PythonObject, operation: Int, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The linear machines; args[0] is None (the driver is the estimator's
    held route, bindings/linear_svc_boundary_unpack.mojo `linear_svc_route`);
    `operation` is the leaf code of bindings/operation_names_svc.mojo."""
    classical_connect_numpy(context)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVR_FIT:
        _linear_svr_fit(model, args, context)
        return model
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVR_SCORE:
        var pred = _linear_svr_predict(model, args[1], context)
        return linear_svr_score_pack(pred, args[2], labels_scope(model, context))
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_FIT:
        _linear_svc_fit(model, args, context)
        return model
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_PREDICT:
        # One label scope read per operation (the fallback and the decode).
        var scope = labels_scope(model, context)
        var prediction = _linear_svc_prediction(model, args[1], context, scope)
        return linear_svc_predict_pack(model, prediction, scope)
    if operation == SVC_OPN_FAMILY_SVC_LINEAR_SVC_SCORE:
        var scope = labels_scope(model, context)
        var prediction = _linear_svc_prediction(model, args[1], context, scope)
        return linear_svc_score_pack(model, prediction, args[2], context, scope)
    raise Error("RuntimeError: unknown linear SVM operation " + String(operation))
