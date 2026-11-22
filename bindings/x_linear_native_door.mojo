# SPDX-License-Identifier: Apache-2.0
# The native owner of the linear expansion estimators' public operations
# (python/mojolearn/_expansion_linear.py and the coordinate-descent pair in
# python/mojolearn/_solver_impl.py). Each public method is one call into
# x_linear_c, which validates the arguments with the Python wording, encodes
# labels (bindings/classical_labels_door.mojo), assembles the same int and float
# parameter lists, makes the same fit / decision / class-prep calls through
# the native x_linear operations table (core/linear_entry_table_ext.mojo, no
# CPython between the two modules) and the `cd_fit` / `cd_predict` calls
# through the native solver operations table (core/solver_entry_table.mojo),
# with the same operands in the same order, and
# writes the fitted attributes the Python wrote (same names, types, dtypes
# and shapes). No kernel changes and no reduction order changes, so the bits
# are the ones the Python path produced.
#
# What the single call removes: the Python between the calls. HuberRegressor
# predicts its training X from the float32 matrix the fit already converted
# (no second conversion); IsotonicRegression.fit_transform transforms the
# column the fit converted; a classifier's score scores the prediction the
# same call decoded; Lasso / ElasticNet score the prediction buffer as it is.
#
# Every operation here is three calls: UNPACK and PACK in the group's
# bindings/x_linear_<group>_unpack.mojo / _pack.mojo (cd, infer, isotonic,
# regressors, sgd, cv), RUN in x_linear/door_run.mojo with the GIL released
# and no Python object. The scores are a split predict and the metric door's
# own operation. RidgeCV's split path is the one loop driven from here (a
# RUN per cv split): bindings/x_linear_cv_unpack.mojo says why.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.x_linear_route_unpack import XLinearOps, xl_route, xl_route_host, xl_device_admit, xl_route_drop, xl_route_vendor_used
from bindings.x_linear_common_unpack import _b, _truth, _name, _xl_scope, xl_intercept_words
from x_linear.door_run import xl_cd_fit_run, xl_cd_predict_run
from bindings.x_linear_cd_unpack import xl_cd_fit_unpack, xl_cd_predict_unpack, xl_cd_host_prediction_unpack
from std.python import PythonObject
from bindings.py_balanced import py_setitem
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_common_unpack import _xl_float
from bindings.x_linear_cd_unpack import XlCdFit, XlCdPredict
from bindings.x_linear_common_unpack import xl_decision_released
from core.linear_entry_table_ext import XL_OP_DECISION, XL_OP_DECISION_CODES, XL_OP_DECISION_ROWS
from x_linear.door_run import LINK_IDENTITY, LINK_SIGMOID, ROWS_SGD_PROBA, ROWS_LRCV_PROBA, xl_run_word
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    XlDecision, xl_decision_build, _attr, _eq, _int, _name, _repr, _xl_bayes_refuse, _xl_check_fitted, _xl_matrix,
    _xl_width, _xl_words,
)
from bindings.py_balanced import py_fields, py_get, py_item
from bindings.admitted_input_unpack import AdmittedX
from bindings.x_linear_keys_unpack import (
    x_linear_keys, X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION, X_LINEAR_OPN_ONE_CLASS_PREDICT,
)
from bindings.classical_labels_door import labels_full_f32
from bindings.py_balanced import py_fields, py_item
from bindings.admitted_input_unpack import admitted_vector
from bindings.x_linear_common_unpack import XlDecision, xl_decision_refuse
from bindings.classical_labels_door import labels_address, labels_cast, classical_decode_labels
from x_linear.door_run import xl_isotonic_fit_run
from bindings.x_linear_common_unpack import xl_fit_released
from bindings.x_linear_isotonic_unpack import (
    xl_isotonic_fit_unpack, xl_isotonic_transform_unpack, xl_isotonic_refit_transform_unpack,
)
from bindings.x_linear_isotonic_pack import xl_isotonic_fit_pack, xl_isotonic_transform_pack
from x_linear.door_run import xl_huber_fit_run
from bindings.x_linear_regressors_unpack import (
    xl_fit_glm_unpack, xl_fit_huber_unpack, xl_fit_bayes_unpack, xl_fit_ard_unpack, xl_fit_lars_unpack,
    xl_fit_quantile_unpack,
)
from bindings.x_linear_regressors_pack import (
    xl_fit_glm_pack, xl_fit_huber_pack, xl_fit_bayes_pack, xl_fit_ard_pack, xl_fit_lars_pack, xl_fit_quantile_pack,
)
from bindings.x_linear_sgd_unpack import (
    xl_fit_sgd_regressor_unpack, xl_fit_one_class_unpack, xl_fit_sgd_classifier_unpack,
    xl_fit_ridge_classifier_unpack,
)
from bindings.x_linear_sgd_pack import (
    xl_fit_sgd_regressor_pack, xl_fit_one_class_pack, xl_fit_sgd_classifier_pack, xl_fit_ridge_classifier_pack,
)
from x_linear.door_run import (
    xl_logcv_fit_run, xl_ridge_cv_plain_run, xl_ridge_cv_kfold_run, xl_ridge_cv_fold_run, xl_ridge_cv_choose_run,
)
from bindings.x_linear_sgd_unpack import xl_class_fit_released
from bindings.x_linear_cv_unpack import (
    XL_RIDGE_CV_PLAIN, XL_RIDGE_CV_KFOLD, xl_fit_ridge_cv_unpack, xl_ridge_cv_fold_unpack, xl_fit_lasso_cv_unpack,
    xl_fit_elastic_net_cv_unpack, xl_fit_logistic_cv_unpack,
)
from bindings.x_linear_cv_pack import (
    xl_ridge_singular_refuse, xl_fit_ridge_cv_pack, xl_fit_ridge_cv_kfold_pack, xl_fit_ridge_cv_splits_pack,
    xl_fit_lasso_cv_pack, xl_fit_elastic_net_cv_pack, xl_fit_logistic_cv_pack,
)
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_attributes_door import _type_mro
from bindings.mode_metadata_native_door import _class_fields
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.classical_arrays_door import classical_address
from bindings.native_array_view_pack import native_array_ndarray_view
from providers.python.array_metadata_pyapi_door import array_slot_once
from providers.python.estimator_lifecycle_pyapi_unpack import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)
from providers.python.linear_score_boundary_pack import classical_r2_score
from bindings.admitted_input_unpack import admit_f32, admitted_storage
from bindings.admitted_input_run import admitted_finite_entry_refuse
from bindings.x_linear_keys_unpack import (
    x_linear_keys, X_LINEAR_OPN_CD_FIT, X_LINEAR_OPN_CD_PREDICT, X_LINEAR_OPN_CD_SCORE,
    X_LINEAR_OPN_CLASSIFIER_PREDICT, X_LINEAR_OPN_CLASSIFIER_SCORE, X_LINEAR_OPN_DECISION_FUNCTION,
    X_LINEAR_OPN_FIT_ARD, X_LINEAR_OPN_FIT_BAYES, X_LINEAR_OPN_FIT_ELASTIC_NET_CV, X_LINEAR_OPN_FIT_GLM,
    X_LINEAR_OPN_FIT_HUBER, X_LINEAR_OPN_FIT_LARS, X_LINEAR_OPN_FIT_LASSO_CV, X_LINEAR_OPN_FIT_LASSO_LARS,
    X_LINEAR_OPN_FIT_LOGISTIC_CV, X_LINEAR_OPN_FIT_ONE_CLASS, X_LINEAR_OPN_FIT_PA_CLASSIFIER,
    X_LINEAR_OPN_FIT_PA_REGRESSOR, X_LINEAR_OPN_FIT_PERCEPTRON, X_LINEAR_OPN_FIT_QUANTILE,
    X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER, X_LINEAR_OPN_FIT_RIDGE_CV, X_LINEAR_OPN_FIT_SGD_CLASSIFIER,
    X_LINEAR_OPN_FIT_SGD_REGRESSOR, X_LINEAR_OPN_ISOTONIC_FIT, X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM,
    X_LINEAR_OPN_ISOTONIC_PREDICT, X_LINEAR_OPN_ISOTONIC_SCORE, X_LINEAR_OPN_ISOTONIC_TRANSFORM,
    X_LINEAR_OPN_LRCV_PREDICT_PROBA, X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION, X_LINEAR_OPN_ONE_CLASS_PREDICT,
    X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES, X_LINEAR_OPN_REGRESSOR_PREDICT, X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE,
    X_LINEAR_OPN_REGRESSOR_SCORE, X_LINEAR_OPN_DEVICE_ADMIT, X_LINEAR_OPN_SGD_PREDICT_PROBA,
    X_LINEAR_OPN_CD_VENDOR_USED,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item
from bindings.classical_labels_door import labels_cast, classical_accuracy, labels_r2


# ---- from bindings/x_linear_infer_unpack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# UNPACK of the linear expansion estimators' inference operations
# (python/mojolearn/_expansion_linear.py: the regressors' predict, the
# classifiers' decision_function / predict / predict_proba,
# LogisticRegressionCV.predict_proba and SGDOneClassSVM's three scores):
# the fitted and feature-count refusals in the order and words the Python
# made them, X through the one admission, and the typed request
# (x_linear/door_run.mojo `XlDecisionRequest`), one function per public
# operation. RUN (`xl_decision_run`) holds no Python object; PACK is
# bindings/x_linear_infer_pack.mojo.


def _xl_infer_x(model: PythonObject, fields: PythonObject, source: PythonObject, entries: XLinearOps,
                context: PythonObject) raises -> AdmittedX:
    """What every inference operation checks before its decision: the
    model is fitted, its route holds an entry, X is admitted, X has the
    fitted feature count."""
    _xl_check_fitted(fields, model)
    xl_ops_require(entries)
    var x = _xl_matrix(source, _name(model), context)
    _xl_width(x.view(), fields, model)
    return x^


def xl_regressor_predict_unpack(model: PythonObject, args: PythonObject, entries: XLinearOps,
                                context: PythonObject) raises -> XlDecision:
    """`_LinearRegressorMixin.predict(X)`: link(X @ coef^T + intercept),
    (n,) for one target. BayesianRidge and ARDRegression pass return_std
    and refuse first."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    if len(args) > 1:
        _xl_bayes_refuse(model, fields, args[1])
    var x = _xl_infer_x(model, fields, args[0], entries, context)
    var stored = py_get(fields, nk[].k__link)
    if stored is None:
        stored = _attr(model, "_link", PythonObject(LINK_IDENTITY))
    return xl_decision_build(XL_OP_DECISION, x.view(), x.array, py_item(fields, nk[].k_coef_),
                             py_item(fields, nk[].k_intercept_), _int(stored), True, False, 0, 0, 0, entries, context)


def xl_decision_function_unpack(model: PythonObject, source: PythonObject, entries: XLinearOps,
                                context: PythonObject) raises -> XlDecision:
    """`_LinearClassifierMixin.decision_function(X)`."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var x = _xl_infer_x(model, fields, source, entries, context)
    return xl_decision_build(XL_OP_DECISION, x.view(), x.array, py_item(fields, nk[].k_coef_),
                             py_item(fields, nk[].k_intercept_), LINK_IDENTITY, True, False, 0, 0, 0, entries,
                             context)


def xl_classifier_predict_unpack(model: PythonObject, source: PythonObject, entries: XLinearOps,
                                 context: PythonObject) raises -> XlDecision:
    """`_LinearClassifierMixin.predict(X)`: each row's int32 class code."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var x = _xl_infer_x(model, fields, source, entries, context)
    return xl_decision_build(XL_OP_DECISION_CODES, x.view(), x.array, py_item(fields, nk[].k_coef_),
                             py_item(fields, nk[].k_intercept_), LINK_IDENTITY, False, True, 0, 1, 0, entries,
                             context)


def xl_predict_proba_unpack(model: PythonObject, source: PythonObject, lrcv: Bool, entries: XLinearOps,
                            context: PythonObject) raises -> XlDecision:
    """SGDClassifier.predict_proba (the sigmoid scores; only for
    loss='log_loss') and, with `lrcv`, LogisticRegressionCV.predict_proba
    (the decision scores): the decision and the row link in one call."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    if not lrcv and not _eq(py_item(fields, nk[].k_loss), PythonObject("log_loss")):
        raise Error("AttributeError: probability estimates are not available for loss=" + _repr(py_item(fields, nk[].k_loss)))
    var x = _xl_infer_x(model, fields, source, entries, context)
    # ne2-g-linear (N362): decision and row link in one call (XL_OP_DECISION_ROWS,
    # the same two kernels' words). Both x_linear bindings publish it, so the
    # former two-call path (decision, then `py2mojo_rows`) is deleted.
    return xl_decision_build(XL_OP_DECISION_ROWS, x.view(), x.array, py_item(fields, nk[].k_coef_),
                             py_item(fields, nk[].k_intercept_), LINK_IDENTITY if lrcv else LINK_SIGMOID, False,
                             False, 0, 0, ROWS_LRCV_PROBA if lrcv else ROWS_SGD_PROBA, entries, context)


def _xl_offset(fields: PythonObject, context: PythonObject) raises -> Float64:
    """`float(self.offset_[0])`."""
    var nk = x_linear_keys()
    var off = _xl_words(py_item(fields, nk[].k_offset_), "offset_", context)
    var value = Float64(xl_run_word(off.address, 0))
    _ = off^
    return value


def xl_one_class_unpack(model: PythonObject, operation: Int, source: PythonObject, entries: XLinearOps,
                        context: PythonObject) raises -> XlDecision:
    """SGDOneClassSVM.decision_function (X @ coef - offset), score_samples
    (X @ coef) and predict (the int32 codes -1 / +1 of the decision)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var x = _xl_infer_x(model, fields, source, entries, context)
    var offset = _xl_offset(fields, context)
    if operation == X_LINEAR_OPN_ONE_CLASS_PREDICT:
        # `Array.from_list([-offset], '<f4')`
        var icpt = labels_full_f32(1, -offset, context)
        return xl_decision_build(XL_OP_DECISION_CODES, x.view(), x.array, py_item(fields, nk[].k_coef_), icpt,
                                 LINK_IDENTITY, False, False, -1, 1, 0, entries, context)
    var intercept = PythonObject(-offset) if operation == X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION else PythonObject(0.0)
    return xl_decision_build(XL_OP_DECISION, x.view(), x.array, py_item(fields, nk[].k_coef_), intercept,
                             LINK_IDENTITY, True, False, 0, 0, 0, entries, context)

# ---- from bindings/x_linear_infer_pack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# PACK of the linear expansion estimators' inference operations: after
# x_linear/door_run.mojo `xl_decision_run` wrote the result words, refuse a
# non-finite X the entry found (scikit-learn's wording) and return the
# object the public method returns. UNPACK is
# bindings/x_linear_infer_unpack.mojo.


def xl_scores_pack(model: PythonObject, call: XlDecision) raises -> PythonObject:
    """The float32 scores as they are: the regressors' predict, the
    classifiers' decision_function and predict_proba,
    LogisticRegressionCV.predict_proba, SGDOneClassSVM.decision_function
    and score_samples."""
    xl_decision_refuse(call, model)
    return call.result


def xl_classifier_predict_pack(model: PythonObject, call: XlDecision, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.predict`: `decode_labels(classes_, codes)`."""
    xl_decision_refuse(call, model)
    var nk = x_linear_keys()
    return classical_decode_labels(py_item(py_fields(model), nk[].k_classes_), call.result, context)


def xl_one_class_predict_pack(model: PythonObject, call: XlDecision, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.predict: the codes -1 / +1 as int64."""
    xl_decision_refuse(call, model)
    var rows = call.request.rows
    var wide = admitted_vector(rows, "<i8", context)
    labels_cast(labels_address(call.result), "<i4", labels_address(wide), "<i8", rows, context)
    return wide

# ---- from bindings/x_linear_cd_pack.mojo (inlined: its one importer is this file) ----
# SPDX-License-Identifier: Apache-2.0
# PACK of the coordinate-descent pair's operations (Lasso, ElasticNet):
# the typed result of x_linear/door_run.mojo `xl_cd_fit_run` /
# `xl_cd_predict_run` becomes the fitted attributes and the returned
# objects, one function per public operation. UNPACK is
# bindings/x_linear_cd_unpack.mojo.


def xl_cd_fit_pack(model: PythonObject, fields: PythonObject, call: XlCdFit, n_iter: Int) raises -> PythonObject:
    """ElasticNet.fit / Lasso.fit: publish `input_copied_`,
    `fortran_copied_`, `coef_`, `intercept_`, `n_iter_` and
    `n_features_in_` (the names, types and order the fit always wrote) and
    return the model. Nothing is published before the solver returned: a
    fit that refuses or fails leaves the estimator's fields as they were."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k_input_copied_, PythonObject(call.copied))
    py_setitem(fields, nk[].k_fortran_copied_, PythonObject(call.copied))
    py_setitem(fields, nk[].k_coef_, call.coef)
    py_setitem(fields, nk[].k_intercept_, _xl_float(call.info, 0))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(call.request.n_cols))
    return model


def xl_cd_predict_pack(call: XlCdPredict) raises -> PythonObject:
    """ElasticNet.predict / Lasso.predict: the (n,) float32 prediction."""
    return call.prediction

# x_linear/dispatch.mojo algorithm codes (`_expansion_linear.ALGO_*`).
# The public method an operation serves (`_xl_method`).
comptime _XL_M_NONE = 0
comptime _XL_M_FIT = 1
comptime _XL_M_FIT_TRANSFORM = 2
comptime _XL_M_PREDICT = 3
comptime _XL_M_DECISION_FUNCTION = 4
comptime _XL_M_SCORE = 5
comptime _XL_M_PREDICT_PROBA = 6
comptime _XL_M_SCORE_SAMPLES = 7
comptime _XL_M_TRANSFORM = 8
# `_lbfgs_work`'s memory.


# ------------------------------------------------------------ bindings


# TOMBSTONE (lane dw-xlinear, 2026-11-20): `_xl_mode`, `_xl_entries` and the
# door's `XLinearOps` (the per-call resolution: a mode String with strip and
# lower, `answer_mode_word`, the `xl_bindings` read by a new str, the
# connection owner read and the table slot read) are replaced by the
# estimator's route (bindings/x_linear_route_unpack.mojo `xl_route`), resolved
# once per estimator. Code recoverable at 0121caad3d.


# ------------------------------------------------------------------- SGD
# Each fit is UNPACK (bindings/x_linear_sgd_unpack.mojo), RUN
# (x_linear/door_run.mojo `xl_fit_run`, GIL released, no Python object) and
# PACK (bindings/x_linear_sgd_pack.mojo).


def xl_fit_sgd_regressor(model: PythonObject, args: PythonObject, passive: Bool, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDRegressor.fit(X, y, sample_weight) and (`passive`)
    PassiveAggressiveRegressor.fit(X, y)."""
    var fields = py_fields(model)
    var call = xl_fit_sgd_regressor_unpack(model, fields, args, passive, entries, context)
    xl_fit_released(call, model)
    return xl_fit_sgd_regressor_pack(model, fields, call, context)


def xl_fit_one_class(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.fit(X, y=None, sample_weight=None)."""
    var fields = py_fields(model)
    var call = xl_fit_one_class_unpack(model, fields, args, entries, context)
    xl_fit_released(call, model)
    return xl_fit_one_class_pack(model, fields, call, context)



def xl_one_class(model: PythonObject, operation: Int, source: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.decision_function / score_samples / predict: UNPACK,
    RUN, PACK (bindings/x_linear_infer_unpack.mojo, _pack.mojo)."""
    var call = xl_one_class_unpack(model, operation, source, entries, context)
    xl_decision_released(call)
    if operation == X_LINEAR_OPN_ONE_CLASS_PREDICT:
        return xl_one_class_predict_pack(model, call, context)
    return xl_scores_pack(model, call)


# -------------------------------------------------------------- predict
# Each inference operation is UNPACK (bindings/x_linear_infer_unpack.mojo:
# the fitted and width checks, X through the one admission, the typed
# XlDecisionRequest), RUN (x_linear/door_run.mojo `xl_decision_run`, GIL
# released, no Python object) and PACK (bindings/x_linear_infer_pack.mojo).


def xl_regressor_predict(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_LinearRegressorMixin.predict(X)` (BayesianRidge and ARDRegression
    pass return_std and refuse first)."""
    var call = xl_regressor_predict_unpack(model, args, entries, context)
    xl_decision_released(call)
    return xl_scores_pack(model, call)


def xl_classifier_decision(model: PythonObject, source: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.decision_function(X)`."""
    var call = xl_decision_function_unpack(model, source, entries, context)
    xl_decision_released(call)
    return xl_scores_pack(model, call)


def xl_classifier_predict(model: PythonObject, source: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.predict(X)`: the codes on the binding, the
    labels decoded natively (`decode_labels(classes_, codes)`)."""
    var call = xl_classifier_predict_unpack(model, source, entries, context)
    xl_decision_released(call)
    return xl_classifier_predict_pack(model, call, _xl_scope(entries, context))


def xl_predict_proba(model: PythonObject, source: PythonObject, lrcv: Bool, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.predict_proba (the sigmoid scores) and
    LogisticRegressionCV.predict_proba (the decision scores), finished per
    row on the binding in the one call (XL_OP_DECISION_ROWS)."""
    var call = xl_predict_proba_unpack(model, source, lrcv, entries, context)
    xl_decision_released(call)
    return xl_scores_pack(model, call)


def xl_classifier_score(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.score(X, y)`: predict, then
    `accuracy_fraction(y, prediction)` in the same call (the labels door's
    own operation, in the scope of the route's device)."""
    var prediction = xl_classifier_predict(model, args[0], entries, context)
    return PythonObject(classical_accuracy(args[1], prediction, PythonObject(None), PythonObject(None),
                                           _xl_scope(entries, context)))



# ------------------------------------------------------------ regressors
# Each fit is UNPACK (bindings/x_linear_regressors_unpack.mojo), RUN
# (x_linear/door_run.mojo `xl_fit_run`, GIL released, no Python object) and
# PACK (bindings/x_linear_regressors_pack.mojo).


def xl_fit_glm(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_GLMBase.fit(X, y, sample_weight)`: Poisson, Gamma and Tweedie."""
    var fields = py_fields(model)
    var call = xl_fit_glm_unpack(model, fields, args, entries, context)
    xl_fit_released(call, model)
    return xl_fit_glm_pack(model, fields, call, context)


def xl_fit_huber(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """HuberRegressor.fit(X, y, sample_weight): the fit, then the outlier
    mask of the training rows from the X the fit already holds (one RUN,
    `xl_huber_fit_run`)."""
    var fields = py_fields(model)
    var call = xl_fit_huber_unpack(model, fields, args, entries, context)
    var py = Python()
    with GILReleased(py):
        xl_huber_fit_run(call.fit.request, call.targets, call.flags_address, call.link, call.epsilon)
    return xl_fit_huber_pack(model, fields, call, context)


def xl_regressor_score(model: PythonObject, args: PythonObject, entries: XLinearOps,
                       context: PythonObject) raises -> PythonObject:
    """`_LinearRegressorMixin.score(X, y)`: R^2 of the linear prediction
    against y, in one call. The prediction is `xl_regressor_predict`'s
    (link(X @ coef^T + intercept) on the estimator's x_linear entry, the
    kernel `predict` runs) and the R^2 is `metrics.r2_score`'s float32
    kernel on the metrics image of the estimator's device (`_xl_scope`),
    with numeric_mode None as before. It replaces the separate score door
    (bindings/linear_score_native_unpack.mojo), which selected the process
    images by name on every call and so could not follow `device`."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var name = _name(model)
    # BayesianRidge / ARDRegression refuse what their predict refuses.
    var compute_score = py_get(fields, nk[].k_compute_score)
    if compute_score is not None and _truth(compute_score):
        raise Error("mojolearn " + name + ": compute_score is not implemented")
    # A multi-output model is refused BEFORE the prediction, as the old score
    # did (lane fx-audit, audit-classical minor 3): the prediction is one
    # column exactly when the fitted block has one class row (k = the
    # intercept's words, `xl_block_dims`; the predict squeezes k == 1).
    var intercept = py_get(fields, nk[].k_intercept_)
    if intercept is not None and xl_intercept_words(intercept) != 1:
        raise Error("mojolearn " + name + ": a regressor's score needs one output column")
    var prediction = xl_regressor_predict(model, Python.tuple(args[0]), entries, context)
    # The targets as one float32 vector (float64 targets are cast, as the
    # score always cast them); R^2 refuses a NaN or infinity in either.
    var target = admit_f32(args[1], 1, "y", context, finite=False)
    var result = labels_r2(target.array, prediction, PythonObject(None), PythonObject(None),
                           _xl_scope(entries, context))
    _ = target^
    return PythonObject(result)


def xl_fit_bayes(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """BayesianRidge.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    var call = xl_fit_bayes_unpack(model, fields, args, entries, context)
    xl_fit_released(call, model)
    return xl_fit_bayes_pack(model, fields, call, context)


def xl_fit_ard(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ARDRegression.fit(X, y)."""
    var fields = py_fields(model)
    var call = xl_fit_ard_unpack(model, fields, args, entries, context)
    xl_fit_released(call, model)
    return xl_fit_ard_pack(model, fields, call, context)


def xl_fit_lars(model: PythonObject, args: PythonObject, lasso: Bool, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """Lars.fit(X, y) and (`lasso`) LassoLars.fit(X, y): `_lars_fit`."""
    var fields = py_fields(model)
    var call = xl_fit_lars_unpack(model, fields, args, lasso, entries, context)
    xl_fit_released(call, model)
    return xl_fit_lars_pack(model, fields, call, context)


def xl_fit_quantile(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """QuantileRegressor.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    var call = xl_fit_quantile_unpack(model, fields, args, entries, context)
    xl_fit_released(call, model)
    return xl_fit_quantile_pack(model, fields, call, context)



# ----------------------------------------------------------- classifiers


def xl_fit_sgd_classifier(model: PythonObject, args: PythonObject, operation: Int, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.fit(X, y, sample_weight), Perceptron.fit(X, y,
    sample_weight) and PassiveAggressiveClassifier.fit(X, y): one SGD fit
    over the class codes."""
    var fields = py_fields(model)
    var call = xl_fit_sgd_classifier_unpack(model, fields, args, operation, entries, context)
    xl_class_fit_released(call, model)
    return xl_fit_sgd_classifier_pack(model, fields, call, context)


def xl_fit_ridge_classifier(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """RidgeClassifier.fit(X, y, sample_weight): `_ridge_run` over the int32
    class codes (the +-1 targets are built in the binding); a CV fold
    stays on the device."""
    var fields = py_fields(model)
    var call = xl_fit_ridge_classifier_unpack(model, fields, args, entries, context)
    xl_class_fit_released(call, model)
    return xl_fit_ridge_classifier_pack(model, fields, call, context)



# -------------------------------------------------------------- isotonic


def xl_isotonic(model: PythonObject, operation: Int, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """IsotonicRegression.fit / fit_transform / transform / predict: UNPACK
    (bindings/x_linear_isotonic_unpack.mojo), RUN (x_linear/door_run.mojo,
    GIL released), PACK (bindings/x_linear_isotonic_pack.mojo).
    fit_transform is the fit's three steps, then the transform's over the
    column the fit already converted."""
    if operation == X_LINEAR_OPN_ISOTONIC_FIT or operation == X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM:
        var fields = py_fields(model)
        var fitting = xl_isotonic_fit_unpack(model, fields, args, entries, context)
        var rising = False
        var py = Python()
        with GILReleased(py):
            rising = xl_isotonic_fit_run(fitting.fit.request, fitting.auto, fitting.targets)
        xl_isotonic_fit_pack(fields, fitting, rising, context)
        if operation == X_LINEAR_OPN_ISOTONIC_FIT:
            return model
        var again = xl_isotonic_refit_transform_unpack(model, fields, fitting, entries, context)
        xl_fit_released(again.fit, model)
        return xl_isotonic_transform_pack(again, context)
    var call = xl_isotonic_transform_unpack(model, args[0], entries, context)
    xl_fit_released(call.fit, model)
    return xl_isotonic_transform_pack(call, context)



# ------------------------------------------------- Lasso / ElasticNet (cd)


def _xl_host_only(model: PythonObject) raises -> Bool:
    """An explicit host inference class (`_HOST_INFERENCE_ONLY` in its class
    dictionaries): it predicts a saved model on the CPU and never fits.
    Depends only on the class rows: resolved once per class state
    (native_class_admission re-checks after any class modification)."""
    var nk = x_linear_keys()
    var mods = NativeModules()
    var cls = py_type(model)
    var record = class_record(mods, _XL_HOST_ONLY_SLOT, cls)
    if record is not None:
        return Bool(py=record)
    var b = _b()
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, cls)
    var found = False
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var class_fields = py_call(py_item(b.type.__dict__, nk[].k___dict__).__get__, mro[k])
        if py_get(class_fields, nk[].k__HOST_INFERENCE_ONLY) is not None:
            found = True
            break
    class_admit(mods, _XL_HOST_ONLY_SLOT, cls, PythonObject(found))
    return found


comptime _XL_HOST_ONLY_SLOT = "x_linear_host_inference_only_v1"
comptime _XL_AFTER_FIT_SLOT = "x_linear_no_after_fit_v1"


# TOMBSTONE (lane dw-xlinear, 2026-11-20): `_xl_cd_entry` (per fit and per
# predict the `cd_solver` context read, the mode word, two integer conversions
# and the solver table slot read) is the same route's solver entry
# (`xl_route(..., cd=True)`). Code recoverable at 0121caad3d.


def xl_cd_fit(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ElasticNet.fit / Lasso.fit(X, y, sample_weight): UNPACK
    (bindings/x_linear_cd_unpack.mojo), RUN (`cd_fit` on the route's solver
    entry, GIL released), PACK (bindings/x_linear_cd_pack.mojo)."""
    var fields = py_fields(model)
    var call = xl_cd_fit_unpack(model, fields, args, entries, context)
    var n_iter = 0
    var py = Python()
    with GILReleased(py):
        n_iter = xl_cd_fit_run(call.request)
    return xl_cd_fit_pack(model, fields, call, n_iter)


def xl_cd_predict(model: PythonObject, source: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ElasticNet.predict / Lasso.predict(X): UNPACK, RUN (`cd_predict`,
    GIL released), PACK."""
    var call = xl_cd_predict_unpack(model, source, entries, context)
    var py = Python()
    with GILReleased(py):
        xl_cd_predict_run(call.request)
    return xl_cd_predict_pack(call)


def xl_cd_score(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ElasticNet.score / Lasso.score(X, y): `_r2_host(self.predict(X), y)`.
    The prediction is the R^2 program's input as it is (the metric door's
    own operation, in the scope of the route's device). An explicit host
    class predicts through its own native host entry."""
    if entries.host_only:
        return classical_r2_score(xl_cd_host_prediction_unpack(model, args[0], context), args[1],
                                  _xl_scope(entries, context))
    return classical_r2_score(xl_cd_predict(model, args[0], entries, context), args[1], _xl_scope(entries, context))



# ------------------------------------------------------- cross-validated
# RidgeCV, LassoCV, ElasticNetCV and LogisticRegressionCV: each fit is UNPACK
# (bindings/x_linear_cv_unpack.mojo: the refusals, the admission, the folds,
# the typed requests), RUN (x_linear/door_run.mojo, GIL released, no Python
# object) and PACK (bindings/x_linear_cv_pack.mojo).


def xl_fit_ridge_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit(X, y, sample_weight). cv=None and the k-fold path are one
    RUN each. The split path runs one RUN per cv split (that split's alpha
    grid: fit, prediction and R^2 per alpha, all typed calls) and one for
    the choice and the refit; the splits are driven here because a split's
    rows are gathered by the base binding's Python-callable entry
    (`xl_ridge_cv_fold_unpack`)."""
    var fields = py_fields(model)
    var call = xl_fit_ridge_cv_unpack(model, fields, args, entries, context)
    var py = Python()
    var best = 0
    if call.path == XL_RIDGE_CV_PLAIN:
        with GILReleased(py):
            xl_ridge_cv_plain_run(call.request)
        return xl_fit_ridge_cv_pack(model, fields, call, context)
    if call.path == XL_RIDGE_CV_KFOLD:
        with GILReleased(py):
            best = xl_ridge_cv_kfold_run(call.request)
        return xl_fit_ridge_cv_kfold_pack(model, fields, call, best, context)
    var name = _name(model)
    var undefined = False
    for s in range(call.request.k):  # driver-loop: over the cv splits (rows gathered at the boundary, split by split)
        var fold = xl_ridge_cv_fold_unpack(call, s, context)
        if not fold.scored:
            undefined = True
            continue
        var singular = -1
        with GILReleased(py):
            singular = xl_ridge_cv_fold_run(call.request, fold.request, name)
        _ = fold.pins  # keep the gathered rows alive through the native calls above
        if singular >= 0:
            xl_ridge_singular_refuse(model, call.fit_words, call.request.d)
    with GILReleased(py):
        best = xl_ridge_cv_choose_run(call.request, undefined)
    return xl_fit_ridge_cv_splits_pack(model, fields, call, best, undefined, context)


def xl_fit_lasso_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """LassoCV.fit(X, y, sample_weight): the l1_ratio grid [1.0]."""
    var fields = py_fields(model)
    var call = xl_fit_lasso_cv_unpack(model, fields, args, entries, context)
    xl_fit_released(call.fit, model)
    return xl_fit_lasso_cv_pack(model, fields, call, context)


def xl_fit_elastic_net_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ElasticNetCV.fit(X, y, sample_weight): l1_ratio_ is the grid entry
    nearest the best one."""
    var fields = py_fields(model)
    var call = xl_fit_elastic_net_cv_unpack(model, fields, args, entries, context)
    xl_fit_released(call.fit, model)
    return xl_fit_elastic_net_cv_pack(model, fields, call, context)


def xl_fit_logistic_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """LogisticRegressionCV.fit(X, y, sample_weight): the class-weight step,
    the stratified fold check and the fit are one RUN (`xl_logcv_fit_run`)."""
    var fields = py_fields(model)
    var call = xl_fit_logistic_cv_unpack(model, fields, args, entries, context)
    var py = Python()
    with GILReleased(py):
        xl_logcv_fit_run(call.fit.request, call.prep, call.stratified, call.folds)
    return xl_fit_logistic_cv_pack(model, fields, call, context)


# ------------------------------------------------------------------ door


def x_linear_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var b = native_loaded_module("builtins")
        if len(call) != 4:
            raise Error("TypeError: x_linear_native_call requires model, operation, arguments and connected context")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: x_linear_native_call requires an operation name, an argument tuple and its context")
        var model = call[0]
        var nk = x_linear_keys()
        # One lookup with the caller's own str (bindings/py_keys.mojo): the
        # former door copied the name into a String, classified it by up to
        # 17 compares (`_xl_method`) and walked a chain of up to 40 more.
        var operation = py_op_code(nk[].ops, call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = _xl_public_operation(model, operation, call[1], args, context)
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:"):
            return classical_python_error(error^, args_ptr)
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def _xl_method(operation: Int) -> Int:
    """The public method an x_linear operation serves (`_XL_M_NONE` for
    none, and for an unknown operation, which `_xl_dispatch` refuses). The
    coordinate-descent door (ElasticNet/Lasso `cd_*`) runs the same native
    lifecycle as the other linear estimators, so those classes carry
    `_mojolearn_native_public` and get no `_mode` class-creation wrappers.
    Integer compares on the operation code (lane fx-boundary-core,
    2026-11-19; was x_linear/door_run.mojo `_xl_method`, a prefix test and
    up to 16 String compares per call that returned the method as a String)."""
    if (operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER or operation == X_LINEAR_OPN_FIT_PERCEPTRON or operation == X_LINEAR_OPN_FIT_PA_CLASSIFIER
            or operation == X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER or operation == X_LINEAR_OPN_FIT_SGD_REGRESSOR
            or operation == X_LINEAR_OPN_FIT_PA_REGRESSOR or operation == X_LINEAR_OPN_FIT_ONE_CLASS or operation == X_LINEAR_OPN_FIT_GLM
            or operation == X_LINEAR_OPN_FIT_HUBER or operation == X_LINEAR_OPN_FIT_BAYES or operation == X_LINEAR_OPN_FIT_ARD
            or operation == X_LINEAR_OPN_FIT_LARS or operation == X_LINEAR_OPN_FIT_LASSO_LARS or operation == X_LINEAR_OPN_FIT_QUANTILE
            or operation == X_LINEAR_OPN_FIT_RIDGE_CV or operation == X_LINEAR_OPN_FIT_LASSO_CV or operation == X_LINEAR_OPN_FIT_ELASTIC_NET_CV
            or operation == X_LINEAR_OPN_FIT_LOGISTIC_CV or operation == X_LINEAR_OPN_ISOTONIC_FIT or operation == X_LINEAR_OPN_CD_FIT):
        return _XL_M_FIT
    if operation == X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM:
        return _XL_M_FIT_TRANSFORM
    # "regressor_predict_score" (lane fx-rows-rest, 2026-11-18): the regressors'
    # predict for a package-internal caller that reads float32 words (a meta
    # door, an explainer, a native scorer); the same lifecycle as predict.
    if (operation == X_LINEAR_OPN_REGRESSOR_PREDICT or operation == X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE
            or operation == X_LINEAR_OPN_CLASSIFIER_PREDICT or operation == X_LINEAR_OPN_ONE_CLASS_PREDICT
            or operation == X_LINEAR_OPN_ISOTONIC_PREDICT or operation == X_LINEAR_OPN_CD_PREDICT):
        return _XL_M_PREDICT
    if operation == X_LINEAR_OPN_DECISION_FUNCTION or operation == X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION:
        return _XL_M_DECISION_FUNCTION
    if (operation == X_LINEAR_OPN_CLASSIFIER_SCORE or operation == X_LINEAR_OPN_ISOTONIC_SCORE or operation == X_LINEAR_OPN_CD_SCORE
            or operation == X_LINEAR_OPN_REGRESSOR_SCORE):
        return _XL_M_SCORE
    if operation == X_LINEAR_OPN_SGD_PREDICT_PROBA or operation == X_LINEAR_OPN_LRCV_PREDICT_PROBA:
        return _XL_M_PREDICT_PROBA
    if operation == X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES:
        return _XL_M_SCORE_SAMPLES
    if operation == X_LINEAR_OPN_ISOTONIC_TRANSFORM:
        return _XL_M_TRANSFORM
    return _XL_M_NONE


def _xl_method_name(method: Int) -> String:
    """The method's public name, for the not-fitted refusal's wording."""
    if method == _XL_M_PREDICT:
        return "predict"
    if method == _XL_M_DECISION_FUNCTION:
        return "decision_function"
    if method == _XL_M_SCORE:
        return "score"
    if method == _XL_M_PREDICT_PROBA:
        return "predict_proba"
    if method == _XL_M_SCORE_SAMPLES:
        return "score_samples"
    if method == _XL_M_TRANSFORM:
        return "transform"
    if method == _XL_M_FIT_TRANSFORM:
        return "fit_transform"
    return "fit"


def _xl_dispatch(model: PythonObject, operation: Int, name: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
        var nk = x_linear_keys()
        var result = PythonObject(None)
        if operation == X_LINEAR_OPN_REGRESSOR_PREDICT or operation == X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE:
            result = xl_regressor_predict(model, args, entries, context)
        elif operation == X_LINEAR_OPN_REGRESSOR_SCORE:
            result = xl_regressor_score(model, args, entries, context)
        elif operation == X_LINEAR_OPN_DECISION_FUNCTION:
            result = xl_classifier_decision(model, args[0], entries, context)
        elif operation == X_LINEAR_OPN_CLASSIFIER_PREDICT:
            result = xl_classifier_predict(model, args[0], entries, context)
        elif operation == X_LINEAR_OPN_CLASSIFIER_SCORE:
            result = xl_classifier_score(model, args, entries, context)
        elif operation == X_LINEAR_OPN_SGD_PREDICT_PROBA:
            result = xl_predict_proba(model, args[0], False, entries, context)
        elif operation == X_LINEAR_OPN_LRCV_PREDICT_PROBA:
            result = xl_predict_proba(model, args[0], True, entries, context)
        elif operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER:
            result = xl_fit_sgd_classifier(model, args, operation, entries, context)
        elif operation == X_LINEAR_OPN_FIT_PERCEPTRON:
            result = xl_fit_sgd_classifier(model, args, operation, entries, context)
        elif operation == X_LINEAR_OPN_FIT_PA_CLASSIFIER:
            result = xl_fit_sgd_classifier(model, args, operation, entries, context)
        elif operation == X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER:
            result = xl_fit_ridge_classifier(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_SGD_REGRESSOR:
            result = xl_fit_sgd_regressor(model, args, False, entries, context)
        elif operation == X_LINEAR_OPN_FIT_PA_REGRESSOR:
            result = xl_fit_sgd_regressor(model, args, True, entries, context)
        elif operation == X_LINEAR_OPN_FIT_ONE_CLASS:
            result = xl_fit_one_class(model, args, entries, context)
        elif operation == X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION:
            result = xl_one_class(model, operation, args[0], entries, context)
        elif operation == X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES:
            result = xl_one_class(model, operation, args[0], entries, context)
        elif operation == X_LINEAR_OPN_ONE_CLASS_PREDICT:
            result = xl_one_class(model, operation, args[0], entries, context)
        elif operation == X_LINEAR_OPN_FIT_GLM:
            result = xl_fit_glm(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_HUBER:
            result = xl_fit_huber(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_BAYES:
            result = xl_fit_bayes(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_ARD:
            result = xl_fit_ard(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_LARS:
            result = xl_fit_lars(model, args, False, entries, context)
        elif operation == X_LINEAR_OPN_FIT_LASSO_LARS:
            result = xl_fit_lars(model, args, True, entries, context)
        elif operation == X_LINEAR_OPN_FIT_QUANTILE:
            result = xl_fit_quantile(model, args, entries, context)
        elif operation == X_LINEAR_OPN_ISOTONIC_FIT:
            result = xl_isotonic(model, operation, args, entries, context)
        elif operation == X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM:
            result = xl_isotonic(model, operation, args, entries, context)
        elif operation == X_LINEAR_OPN_ISOTONIC_TRANSFORM or operation == X_LINEAR_OPN_ISOTONIC_PREDICT:
            result = xl_isotonic(model, X_LINEAR_OPN_ISOTONIC_TRANSFORM, args, entries, context)
        elif operation == X_LINEAR_OPN_ISOTONIC_SCORE:
            # RegressorScoreMixin.score: metrics.r2_score(y, predict(X),
            # sample_weight, numeric_mode=self.numeric_mode) on the float32
            # prediction, which is handed straight to the R^2 kernel.
            var prediction = xl_isotonic(model, X_LINEAR_OPN_ISOTONIC_TRANSFORM, Python.tuple(args[0]), entries, context)
            result = PythonObject(labels_r2(args[1], prediction, args[2],
                                            py_get(py_fields(model), nk[].k_numeric_mode),
                                            _xl_scope(entries, context)))
        elif operation == X_LINEAR_OPN_FIT_RIDGE_CV:
            result = xl_fit_ridge_cv(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_LASSO_CV:
            result = xl_fit_lasso_cv(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_ELASTIC_NET_CV:
            result = xl_fit_elastic_net_cv(model, args, entries, context)
        elif operation == X_LINEAR_OPN_FIT_LOGISTIC_CV:
            result = xl_fit_logistic_cv(model, args, entries, context)
        elif operation == X_LINEAR_OPN_CD_FIT:
            result = xl_cd_fit(model, args, entries, context)
        elif operation == X_LINEAR_OPN_CD_PREDICT:
            result = xl_cd_predict(model, args[0], entries, context)
        elif operation == X_LINEAR_OPN_CD_SCORE:
            result = xl_cd_score(model, args, entries, context)
        else:
            raise Error("unknown native x_linear operation " + String(py=name))
        return result


def _xl_lifecycle_admit(model: PythonObject, fields: PythonObject) raises:
    var nk = x_linear_keys()
    var b = native_loaded_module("builtins")
    if py_type(py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))) is not b.int:
        raise Error("TypeError: linear fitting depth requires an exact integer")
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: linear fitted state requires a native boolean or integer")
    # The class rows' verdict (no `_mojolearn_after_fit` anywhere in the MRO)
    # is kept per class state (native_class_admission: any class or ancestor
    # modification invalidates it); only a passing class is recorded.
    var mods = NativeModules()
    var cls = py_type(model)
    if class_record(mods, _XL_AFTER_FIT_SLOT, cls) is None:
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from native fit
            if py_contains(_class_fields(ancestry[i]), nk[].k__mojolearn_after_fit):
                raise Error("TypeError: linear fitting cannot invoke Python completion callbacks")
        class_admit(mods, _XL_AFTER_FIT_SLOT, cls, PythonObject(True))
    if py_contains(fields, nk[].k__mojolearn_after_fit):
        raise Error("TypeError: linear fitting cannot invoke Python completion callbacks")


def _xl_widen(result: PythonObject, context: PythonObject) raises -> PythonObject:
    """A float32 Array widened exactly to float64 (scikit-learn's dtype for
    regression predictions and outlier scores); other Arrays as views."""
    if String(py=array_slot_once(result, "dtype", context)) != "<f4":
        return native_array_ndarray_view(result, context)
    var shape = array_slot_once(result, "shape", context)
    var count = 1
    for i in range(len(shape)):  # driver-loop: over admitted result rank metadata
        count *= Int(py=shape[i])
    # fx-rows (2026-11-18): the cast writes every element, so the float64
    # result is uninitialised storage (`admitted_storage`: an ndarray with
    # NumPy, else an Array, the types `classical_typed` returned); it was a
    # zero-filled 0.8 MB bytearray per isotonic/regressor predict, then
    # overwritten. The dtype and the values are unchanged.
    var wide = admitted_storage(shape, "<f8", context)
    labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8", count, context)
    _ = result
    return wide


def _xl_public_operation(model: PythonObject, operation: Int, name: PythonObject, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The linear expansion estimators' whole public operation: fitted and
    feature-name checks, fit bookkeeping (depth, fitted flag, feature names,
    public class labels), the algorithm, and NumPy results for callers
    outside the package (the former Python `_mode` wrappers)."""
    var nk = x_linear_keys()
    if operation == X_LINEAR_OPN_DEVICE_ADMIT:
        # A constructor given a device, a constructor run again on a live
        # estimator, or `set_params(device=...)`, before any field changes:
        # the value is checked in the shared words and the route is dropped.
        if len(args) != 1:
            raise Error("TypeError: device admission expects the device")
        xl_device_admit(model, args[0], context)
        xl_route_drop(model)
        return PythonObject(None)
    if operation == X_LINEAR_OPN_CD_VENDOR_USED:
        # Lasso / ElasticNet `vendor_used()`: metadata, no input and no
        # lifecycle step. An explicit host class runs the CPU image.
        if _xl_host_only(model):
            return PythonObject("cpu")
        return xl_route_vendor_used(model, py_fields(model), context, True)
    var method = _xl_method(operation)
    if method == _XL_M_NONE:
        raise Error("unknown native x_linear operation " + String(py=name))
    var fields = py_fields(model)
    _xl_lifecycle_admit(model, fields)
    var fitting = method == _XL_M_FIT or method == _XL_M_FIT_TRANSFORM
    var depth = Int(py=py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0)))
    if not fitting:
        classical_require_fitted(model, False, _xl_method_name(method), context)
        if depth == 0:
            estimator_feature_check(model, fields, args[0], context)
    var initial_depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        # The estimator's route: a hit is one model-record read and three
        # pointer compares; it holds the entry, the pins of its image and the
        # context of its device for this whole operation. An explicit host
        # class (the coordinate-descent pair's fit and score ask, as they
        # did) resolves no entry.
        var cd = (operation == X_LINEAR_OPN_CD_FIT or operation == X_LINEAR_OPN_CD_PREDICT
                  or operation == X_LINEAR_OPN_CD_SCORE)
        if cd and operation != X_LINEAR_OPN_CD_PREDICT and _xl_host_only(model):
            var hosted = xl_route_host(context)
            result = _xl_dispatch(model, operation, name, args, hosted, context)
        else:
            var entries = xl_route(model, fields, context, cd)
            result = _xl_dispatch(model, operation, name, args, entries, context)
    except error:
        if fitting:
            estimator_fit_end(model, initial_depth)
        # The fit entries test the X and y they upload (defer_finite in
        # _xl_matrix / _xl_vector) and refuse in their own words: the one
        # scikit-learn wording replaces them (lane a7c).
        admitted_finite_entry_refuse(error, _name(model))
        raise error^
    if fitting:
        estimator_fit_end(model, initial_depth)
    if fitting:
        var classifier = (operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER or operation == X_LINEAR_OPN_FIT_PERCEPTRON
                          or operation == X_LINEAR_OPN_FIT_PA_CLASSIFIER or operation == X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER
                          or operation == X_LINEAR_OPN_FIT_LOGISTIC_CV)
        estimator_fit_complete(model, args[0], classifier, context)
        if method == _XL_M_FIT:
            return result
    # Lane fx-rows-rest (2026-11-18): "regressor_predict_score" is the plan's
    # internal predict (`scorer_predict_operation`, python/mojolearn/
    # _native_msel_prep.py; Ridge and LinearRegression have had `predict_score`).
    # Its caller is a compiled door reading float32 words (KernelExplainer /
    # PermutationExplainer, the stacking / voting / multi-output doors, a native
    # scorer), so it gets the kernel's fresh float32 Array, as a caller inside
    # the package always did. A door has no Python frame, so those callers were
    # judged "outside": every model call widened the n predictions to a float64
    # ndarray (`_xl_widen`) and the door then narrowed them back to float32
    # through a zero-filled snapshot (kernel-shap: 20 M predictions per explain,
    # +114 ms on a 334 ms row against 0.8.55, whose callers read the float32
    # output). The widening is exact, so the float32 words are the same words.
    # The public predict (operation "regressor_predict") is unchanged: float64.
    if (method == _XL_M_SCORE or operation == X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE or not estimator_outside()
            or py_get(context, nk[].k_ndarray) is None):
        return result
    var b = native_loaded_module("builtins")
    if py_type(result) is b.tuple:
        # predict(X, return_std=True): each Array member as a NumPy view.
        var members = Python.list()
        for i in range(len(result)):  # driver-loop: over the members of a result tuple
            var member = result[i]
            _ = py_call(members.append, native_array_ndarray_view(member, context)
                        if py_type(member) is py_item(context, nk[].k_array_class) else member)
        return py_call(b.tuple, members)
    if py_type(result) is not py_item(context, nk[].k_array_class):
        return result
    if (method == _XL_M_PREDICT and (operation == X_LINEAR_OPN_REGRESSOR_PREDICT or operation == X_LINEAR_OPN_ISOTONIC_PREDICT
                                     or operation == X_LINEAR_OPN_CD_PREDICT)) or (
            operation == X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION or operation == X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES):
        return _xl_widen(result, context)
    return native_array_ndarray_view(result, context)
