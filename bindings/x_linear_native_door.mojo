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
from std.ffi import c_int, c_long
from std.math import isinf
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from core.linear_entry_table_ext import XL_OP_DECISION_ROWS, XL_OP_HUBER_OUT, XL_OP_BEST_ALPHA
from x_linear.door_run import (
    ALGO_SGD, ALGO_GLM, ALGO_HUBER, ALGO_BAYES, ALGO_ARD, ALGO_LARS, ALGO_QUANTILE, ALGO_RIDGE, ALGO_ENETCV,
    ALGO_LOGCV, ALGO_RIDGE_KFOLD, LINK_IDENTITY, LINK_EXP, xl_run_parts, xl_parts_total, xl_seed_halves,
    xl_lbfgs_work, xl_sgd_class_floats, XlSplitWords, xl_cv_fold_ids_run, xl_cv_roles_limit, xl_cv_roles_run,
)
from bindings.x_linear_route_unpack import (
    XLinearOps, xl_route, xl_route_host, xl_ops_require, xl_device_admit, xl_route_drop,
)
from bindings.x_linear_common_unpack import (
    _b, _op, _truth, _float, _int, _repr, _str, _gt, _ge, _lt, _le, _eq, _isinstance, _is_int, _name, _class_attr,
    _attr, _sorted_repr, _xl_scope, _xl_call, _xl_word, _xl_float, _xl_count, _xl_slice, _xl_matrix, _xl_vector,
    _xl_block, _xl_decision, _xl_run, _xl_fold_matrix, _xl_run_x, _xl_with_weights, _xl_owned, _xl_reals,
    _xl_class_prep,
)
from x_linear.door_run import xl_cd_fit_run, xl_cd_predict_run
from bindings.x_linear_cd_unpack import xl_cd_fit_unpack, xl_cd_predict_unpack, xl_cd_host_prediction_unpack
from bindings.x_linear_cd_pack import xl_cd_fit_pack, xl_cd_predict_pack
from bindings.x_linear_common_unpack import xl_decision_released, _xl_bayes_refuse
from bindings.x_linear_infer_unpack import (
    xl_regressor_predict_unpack, xl_decision_function_unpack, xl_classifier_predict_unpack,
    xl_predict_proba_unpack, xl_one_class_unpack,
)
from bindings.x_linear_infer_pack import xl_scores_pack, xl_classifier_predict_pack, xl_one_class_predict_pack
from x_linear.door_run import xl_isotonic_fit_run
from bindings.x_linear_common_unpack import xl_fit_released, _xl_or
from bindings.x_linear_isotonic_unpack import (
    xl_isotonic_fit_unpack, xl_isotonic_transform_unpack, xl_isotonic_refit_transform_unpack,
)
from bindings.x_linear_isotonic_pack import xl_isotonic_fit_pack, xl_isotonic_transform_pack
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_row_means_f64, base_reduce_stat_int
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_attributes_door import _type_mro
from bindings.mode_metadata_native_door import _class_fields
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_lists import py_list_int
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.classical_arrays_door import classical_address
from bindings.native_array_view_pack import native_array_ndarray_view
from providers.python.array_metadata_pyapi_door import array_slot_once
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from bindings.classical_arrays_door import classical_wrap_array
from providers.python.linear_score_boundary_pack import classical_r2_score
from bindings.classical_warnings_unpack import classical_warning
from bindings.prep_native_estimator_door import prep_native_context, prep_native_split
from bindings.msel_inputs_unpack import msel_admit
from bindings.admitted_input_unpack import (
    AdmittedX, admit_f32, admit_weight_words, admitted_words, admitted_array, admitted_vector, admitted_storage,
    admit_ids,
)
from bindings.admitted_input_run import admitted_finite_entry_refuse
from bindings.admitted_input_run import AdmittedXView
from bindings.py_optional import py_or_none, py_present
from bindings.x_linear_keys_unpack import (
    x_linear_keys,
    X_LINEAR_OPN_CD_FIT,
    X_LINEAR_OPN_CD_PREDICT,
    X_LINEAR_OPN_CD_SCORE,
    X_LINEAR_OPN_CLASSIFIER_PREDICT,
    X_LINEAR_OPN_CLASSIFIER_SCORE,
    X_LINEAR_OPN_DECISION_FUNCTION,
    X_LINEAR_OPN_FIT_ARD,
    X_LINEAR_OPN_FIT_BAYES,
    X_LINEAR_OPN_FIT_ELASTIC_NET_CV,
    X_LINEAR_OPN_FIT_GLM,
    X_LINEAR_OPN_FIT_HUBER,
    X_LINEAR_OPN_FIT_LARS,
    X_LINEAR_OPN_FIT_LASSO_CV,
    X_LINEAR_OPN_FIT_LASSO_LARS,
    X_LINEAR_OPN_FIT_LOGISTIC_CV,
    X_LINEAR_OPN_FIT_ONE_CLASS,
    X_LINEAR_OPN_FIT_PA_CLASSIFIER,
    X_LINEAR_OPN_FIT_PA_REGRESSOR,
    X_LINEAR_OPN_FIT_PERCEPTRON,
    X_LINEAR_OPN_FIT_QUANTILE,
    X_LINEAR_OPN_FIT_RIDGE_CLASSIFIER,
    X_LINEAR_OPN_FIT_RIDGE_CV,
    X_LINEAR_OPN_FIT_SGD_CLASSIFIER,
    X_LINEAR_OPN_FIT_SGD_REGRESSOR,
    X_LINEAR_OPN_ISOTONIC_FIT,
    X_LINEAR_OPN_ISOTONIC_FIT_TRANSFORM,
    X_LINEAR_OPN_ISOTONIC_PREDICT,
    X_LINEAR_OPN_ISOTONIC_SCORE,
    X_LINEAR_OPN_ISOTONIC_TRANSFORM,
    X_LINEAR_OPN_LRCV_PREDICT_PROBA,
    X_LINEAR_OPN_ONE_CLASS_DECISION_FUNCTION,
    X_LINEAR_OPN_ONE_CLASS_PREDICT,
    X_LINEAR_OPN_ONE_CLASS_SCORE_SAMPLES,
    X_LINEAR_OPN_REGRESSOR_PREDICT,
    X_LINEAR_OPN_REGRESSOR_PREDICT_SCORE,
    X_LINEAR_OPN_REGRESSOR_SCORE,
    X_LINEAR_OPN_DEVICE_ADMIT,
    X_LINEAR_OPN_SGD_PREDICT_PROBA,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item
from bindings.classical_labels_door import (
    ClassicalLabels, labels_full_f32, labels_address, labels_cast, labels_reshape, classical_encode_labels,
    classical_accuracy, labels_r2, labels_real_type, labels_integral_type,
)

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


def _xl_sgd_refuse(model: PythonObject, early_stopping: PythonObject, average: PythonObject,
                   warm_start: PythonObject) raises:
    """`_sgd_refuse`."""
    var name = _name(model)
    if _truth(early_stopping):
        raise Error("mojolearn " + name + ": early_stopping is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if _truth(average):
        raise Error("mojolearn " + name + ": average is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if _truth(warm_start):
        raise Error("mojolearn " + name + ": warm_start is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")


def _xl_seed(random_state: PythonObject) raises -> Tuple[Int, Int]:
    """`_seed`: the seed's two Int32 halves (None -> 0)."""
    if random_state is None:
        return (0, 0)
    if not _is_int(random_state) or _lt(random_state, PythonObject(0)):
        raise Error("mojolearn: random_state must be None or a non-negative int")
    var op = _op()
    var mask = py_call(op.sub, py_call(op.lshift, PythonObject(1), PythonObject(64)), PythonObject(1))
    var s = py_call(op.and_, random_state, mask)
    var lo = Int(py=py_call(op.and_, s, PythonObject(4294967295)))
    var hi = Int(py=py_call(op.rshift, s, PythonObject(32)))
    return xl_seed_halves(lo, hi)


def _xl_sgd_auto_sum(batch_size: PythonObject, loss: PythonObject, learning_rate: PythonObject,
                     context: PythonObject) raises -> Bool:
    """`_sgd_auto_sum`."""
    var nk = x_linear_keys()
    return (_isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
            and py_contains(py_item(context, nk[].k_xl_sgd_sum_losses), loss)
            and not py_contains(Python.tuple("pa1", "pa2"), learning_rate))


def _xl_sgd(model: PythonObject, fields: PythonObject, entries: XLinearOps, x: AdmittedXView, y_keep: PythonObject,
            y_addr: Int,
            n_classes: Int, loss_code: Int, penalty: PythonObject, lr: PythonObject, alpha: PythonObject,
            l1_ratio: PythonObject, eta0: PythonObject, power_t: PythonObject, epsilon: PythonObject,
            fit_intercept: PythonObject, max_iter: PythonObject, tol: PythonObject,
            n_iter_no_change: PythonObject, shuffle: PythonObject, random_state: PythonObject,
            sample_weight: PythonObject, class_weight: PythonObject, classes: PythonObject, codes: PythonObject,
            batch_size: PythonObject, var batch_sum: Bool, auto_cap: Int,
            context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_sgd_fit`: the SGD fit on the binding; (coef (problems, d),
    intercept (problems,)). The n float32 targets or class codes are at
    `y_addr` (`y_keep` owns them); 0 is no targets (SGDOneClassSVM: zeros)."""
    var nk = x_linear_keys()
    var name = _name(model)
    var n = x.rows
    var d = x.cols
    var penalties = py_item(context, nk[].k_xl_sgd_penalty)
    var rates = py_item(context, nk[].k_xl_sgd_lr)
    if not py_contains(penalties, penalty):
        raise Error("mojolearn " + name + ": penalty must be 'l2', 'l1', 'elasticnet' or None")
    if not py_contains(rates, lr):
        raise Error("mojolearn " + name + ": unknown learning_rate " + _repr(lr))
    if py_contains(Python.tuple("constant", "invscaling", "adaptive"), lr) and not _gt(eta0, PythonObject(0)):
        raise Error("mojolearn " + name + ": eta0 must be > 0")
    if _eq(lr, PythonObject("optimal")) and not _gt(alpha, PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be > 0 for learning_rate='optimal'")
    var seed = _xl_seed(random_state)
    var problems = n_classes if n_classes > 2 else 1
    var ip: List[Int] = [n_classes, loss_code, _int(penalties[penalty]), _int(rates[lr]),
                         1 if _truth(fit_intercept) else 0, _int(max_iter), _int(n_iter_no_change),
                         1 if _truth(shuffle) else 0, seed[0], seed[1]]
    var fp: List[Float64] = [_float(alpha), _float(l1_ratio), _float(eta0), _float(power_t), _float(epsilon),
                             -3.0e38 if tol is None else _float(tol)]
    var targets = y_keep
    var t_addr = y_addr
    if y_addr == 0:
        targets = admitted_vector(n, "<f4", context)
        t_addr = labels_address(targets)
        _ = targets  # keep the Python object alive through the native call above
    var has_sw = 0
    var t = admitted_words(targets, t_addr, n, 1)
    var yy = _xl_with_weights(t, sample_weight, n, has_sw, context)
    var has_cw = 0
    if class_weight is not None and n_classes >= 2:
        # k scalars: the fit's parameters
        var prep = _xl_class_prep(model, entries, class_weight, classes, codes, 0, False, False, context)
        xl_sgd_class_floats(fp, labels_address(prep[0]), n_classes)
        # The class-weight words are prep's Array: retained through the read.
        _ = prep^
        has_cw = 1
    var auto = _isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
    var batch = -1
    if auto:
        batch = -auto_cap
    elif _truth(batch_size) and (not _is_int(batch_size) or _lt(batch_size, PythonObject(1))):
        raise Error("mojolearn " + name + ": batch_size must be 'auto', 0 or a positive int")
    elif batch_size is not None:
        batch = _int(batch_size)
    # TOMBSTONE: MOJOLEARN_SGD_BATCH_SUM (env override, the B arm of MOJOLEARN_SGD_AUTO_STEPS_256_OFF, whose old path was
    # deleted 2026-11-15 by lane/expdel2-21-linear) deleted 2026-11-18 by lane/fx-env-dead; code recoverable at 0c65d61fed.
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_BATCH_SUM.patch
    ip.append(has_sw)
    ip.append(has_cw)
    ip.append(batch)
    ip.append(1 if batch_sum else 0)
    var out = _xl_run(entries, ALGO_SGD, x.address, n * d, n, d, yy.address, yy.rows, ip, fp,
                      problems * d + problems + 2, problems * (n + d + 1), problems * n, context)
    _ = yy  # keep the Python object alive through the native call above
    var n_out = problems * d + problems + 2
    if _xl_word(out, n_out - 1) != 0:
        raise Error("Floating-point under-/overflow occurred. Scaling input data with "
                    + "StandardScaler or MinMaxScaler might help.")
    _ = yy  # keep the Python object alive through the native calls above
    var coef = _xl_slice(out, 0, problems * d, Python.tuple(problems, d), context)
    _ = yy  # keep the Python object alive through the native call above
    var intercept = _xl_slice(out, problems * d, problems, Python.tuple(problems), context)
    _ = yy  # keep the Python object alive through the native call above
    var n_iter = _xl_count(out, n_out - 2)
    _ = yy  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_t_, PythonObject(Float64(n_iter * n + 1)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = yy^
    _ = targets^
    return (coef, intercept)


def xl_fit_sgd_regressor(model: PythonObject, args: PythonObject, passive: Bool, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDRegressor.fit(X, y, sample_weight) and (`passive`)
    PassiveAggressiveRegressor.fit(X, y)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    var losses = py_item(context, nk[].k_xl_sgd_reg_loss)
    var loss = py_item(fields, nk[].k_loss)
    if passive:
        if not py_contains(Python.tuple("epsilon_insensitive", "squared_epsilon_insensitive"), loss):
            raise Error("mojolearn PassiveAggressiveRegressor: loss must be 'epsilon_insensitive' "
                        + "or 'squared_epsilon_insensitive'")
        if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveRegressor: C must be > 0")
    elif not py_contains(losses, loss):
        raise Error("mojolearn SGDRegressor: loss must be one of " + _sorted_repr(losses))
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var y = _xl_vector(args[1], x.rows, context)
    var fit: Tuple[PythonObject, PythonObject]
    if passive:
        var lr = PythonObject("pa1") if _eq(loss, PythonObject("epsilon_insensitive")) else PythonObject("pa2")
        fit = _xl_sgd(model, fields, entries, x.view(), y.array, y.address, 0, _int(py_item(losses, nk[].k_epsilon_insensitive)),
                      PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5),
                      py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), PythonObject(None),
                      PythonObject(None), PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 256,
                      context)
        _ = y  # keep the Python object alive through the native call above
    else:
        fit = _xl_sgd(model, fields, entries, x.view(), y.array, y.address, 0, _int(losses[loss]), py_item(fields, nk[].k_penalty),
                      py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0), py_item(fields, nk[].k_power_t),
                      py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                      PythonObject(None), PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 4096,
                      context)
        _ = y  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_coef_, labels_reshape(fit[0], Python.tuple(x.cols), context))
    py_setitem(fields, nk[].k_intercept_, fit[1])
    _ = x^
    _ = y^
    return model


def xl_fit_one_class(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.fit(X, y=None, sample_weight=None)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    _xl_sgd_refuse(model, PythonObject(False), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    var nu = py_item(fields, nk[].k_nu)
    if not (_lt(PythonObject(0), nu) and _le(nu, PythonObject(1))):
        raise Error("mojolearn SGDOneClassSVM: nu must be in (0, 1]")
    if py_contains(Python.tuple("pa1", "pa2"), py_item(fields, nk[].k_learning_rate)):
        raise Error("mojolearn SGDOneClassSVM: learning_rate must be constant, optimal, invscaling or adaptive")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var fit = _xl_sgd(model, fields, entries, x.view(), PythonObject(None), 0, 1, _int(py_item(py_item(context, nk[].k_xl_sgd_clf_loss), nk[].k_hinge)),
                      PythonObject("l2"), py_item(fields, nk[].k_learning_rate), nu, PythonObject(0.0), py_item(fields, nk[].k_eta0),
                      py_item(fields, nk[].k_power_t), PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      PythonObject(5), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2], PythonObject(None),
                      PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 4096, context)
    py_setitem(fields, nk[].k_coef_, labels_reshape(fit[0], Python.tuple(x.cols), context))
    # the binding returns offset_ itself for one class
    py_setitem(fields, nk[].k_offset_, _xl_slice(fit[1], 0, 1, Python.tuple(1), context))
    _ = x^
    return model


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
    return xl_classifier_predict_pack(model, call, context)


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


def _xl_set_linear(fields: PythonObject, out_buf: PythonObject, d: Int, context: PythonObject) raises:
    """coef_ = vals[:d] (float32 Array), intercept_ = float(vals[d])."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k_coef_, _xl_slice(out_buf, 0, d, Python.tuple(d), context))
    py_setitem(fields, nk[].k_intercept_, _xl_float(out_buf, d))


def _xl_tweedie_link(fields: PythonObject) raises -> Int:
    """TweedieRegressor._link_code."""
    var nk = x_linear_keys()
    var p = _float(py_item(fields, nk[].k_power))
    if p < 0 or (p > 0 and p < 1):
        raise Error("mojolearn TweedieRegressor: power=" + _str(PythonObject(p)) + " is not implemented (power 0 or >= 1)")
    var link = py_item(fields, nk[].k_link)
    if _eq(link, PythonObject("auto")):
        return 0 if p == 0 else 1
    if _eq(link, PythonObject("identity")):
        if p != 0:
            raise Error("mojolearn TweedieRegressor: the identity link is implemented for power=0 only")
        return 0
    if _eq(link, PythonObject("log")):
        return 1
    raise Error("mojolearn TweedieRegressor: link must be 'auto', 'identity' or 'log'")


def xl_fit_glm(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_GLMBase.fit(X, y, sample_weight)`: Poisson, Gamma and Tweedie."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    # TweedieRegressor reads its own power and link; Poisson and Gamma fix
    # them (`_power`, log link)
    var tweedie = py_contains(fields, nk[].k_power) and py_contains(fields, nk[].k_link)
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(py_item(fields, nk[].k_power)) if tweedie else _float(_class_attr(model, "_power"))
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, link, has_sw]
    var fp: List[Float64] = [power, _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]
    var out = _xl_run(entries, ALGO_GLM, x.address, n * d, n, d, yy.address, yy.rows, ip, fp,
                      d + 3, 3 * n + m * m + 3 * m, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    if _xl_word(out, d + 2) < 0:
        var loss_name = _class_attr(model, "_loss_name")
        raise Error("Some value(s) of y are out of the valid range of the loss '"
                    + (_str(loss_name) if _truth(loss_name) else String("HalfTweedieLoss")) + "'.")
    _ = yy  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    py_setitem(fields, nk[].k__link, PythonObject(LINK_EXP if link == 1 else LINK_IDENTITY))
    _ = x^
    _ = y^
    _ = yy^
    return model


def _xl_huber_mask(flags: PythonObject, address: Int, count: Int) raises -> PythonObject:
    # The private <u1 output already has one completed native flag per row.
    # Keep its owner live while publishing directly into one final bool list.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    # Own the NULL-filled list immediately: partial failures release every
    # initialized slot, while CPython skips the uninitialized NULL slots.
    var out = PythonObject(from_owned=raw)
    var box = cpy.lib.get_function[PyObjectPtr]("PyBool_FromLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: publish completed native Huber result bytes as Python bool slots
        var item = box(c_long(data[i] != 0))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on both success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = flags
    return out


def xl_fit_huber(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """HuberRegressor.fit(X, y, sample_weight): the fit, then the outlier
    mask of the training rows from the X the fit already holds."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var epsilon = py_item(fields, nk[].k_epsilon)
    if not _ge(epsilon, PythonObject(1.0)):
        raise Error("mojolearn HuberRegressor: epsilon must be >= 1.0")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn HuberRegressor: warm_start is not implemented")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var intercept = _truth(py_item(fields, nk[].k_fit_intercept))
    var p = d + 2 if intercept else d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if intercept else 0, has_sw]
    var fp: List[Float64] = [_float(epsilon), _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]
    var out = _xl_run(entries, ALGO_HUBER, x.address, n * d, n, d, yy.address, yy.rows, ip, fp,
                      d + 4 + p, xl_lbfgs_work(p) + n, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var scale = _xl_float(out, d + 1)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_scale_, scale)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 2)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # `self.predict(a)`: the fitted model on the float32 X this call holds
    var link = _int(_attr(model, "_link", PythonObject(LINK_IDENTITY)))
    # ne2-g-linear (N302): `float(scale_ * epsilon)` as one binary64 product
    # in Mojo (Python's float multiply on the same two doubles).
    var thr = _float(scale) * _float(epsilon)
    var flags = admitted_vector(n, "<u1", context)
    var flag_address = labels_address(flags)
    # The prediction and |y - pred| > thr in one device session
    # (XL_OP_HUBER_OUT). Both x_linear bindings publish it, so the former
    # two-call path (decision, then `py2mojo_rows` ROWS_HUBER_OUT) is deleted.
    var block = _xl_block(py_item(fields, nk[].k_coef_), py_item(fields, nk[].k_intercept_), context)
    var wb = block[0]
    _ = _xl_call(entries, XL_OP_HUBER_OUT, [x.address, labels_address(wb), y.address, flag_address],
                 [n, d, link], [thr])
    _ = wb  # keep the Python object alive through the native call above
    _ = y  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_outliers_, _xl_huber_mask(flags, flag_address, n))
    _ = wb^
    _ = flags^
    _ = x^
    _ = y^
    _ = yy^
    return model


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
    var prediction = xl_regressor_predict(model, Python.tuple(args[0]), entries, context)
    if len(prediction.shape) != 1:
        raise Error("mojolearn " + name + ": a regressor's score needs one output column")
    # The targets as one float32 vector (float64 targets are cast, as the
    # score always cast them); R^2 refuses a NaN or infinity in either.
    var target = admit_f32(args[1], 1, "y", context, finite=False)
    var result = labels_r2(target.array, prediction, PythonObject(None), PythonObject(None),
                           _xl_scope(entries, context))
    _ = target^
    return PythonObject(result)


def xl_fit_bayes(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """BayesianRidge.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    _xl_bayes_refuse(model, fields, PythonObject(False))
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, has_sw]
    var fp: List[Float64] = [_float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)), _float(py_item(fields, nk[].k_alpha_2)),
                             _float(py_item(fields, nk[].k_lambda_1)), _float(py_item(fields, nk[].k_lambda_2)),
                             _xl_or(py_item(fields, nk[].k_alpha_init), -1.0), _xl_or(py_item(fields, nk[].k_lambda_init), -1.0)]
    var out = _xl_run(entries, ALGO_BAYES, x.address, n * d, n, d, yy.address, yy.rows, ip, fp,
                      d + 4, 3 * d * d + 5 * d + n, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_alpha_, _xl_float(out, d + 1))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_lambda_, _xl_float(out, d + 2))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 3)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = x^
    _ = y^
    _ = yy^
    return model


def xl_fit_ard(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ARDRegression.fit(X, y)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    _xl_bayes_refuse(model, fields, PythonObject(False))
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    if n < 2:
        raise Error("mojolearn ARDRegression: at least 2 samples are required")
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0]
    var fp: List[Float64] = [_float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)), _float(py_item(fields, nk[].k_alpha_2)),
                             _float(py_item(fields, nk[].k_lambda_1)), _float(py_item(fields, nk[].k_lambda_2)), _float(py_item(fields, nk[].k_threshold_lambda))]
    var out = _xl_run(entries, ALGO_ARD, x.address, n * d, n, d, y.address, y.rows, ip, fp,
                      2 * d + 4, 3 * d * d + 4 * d + n, 2 * d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_alpha_, _xl_float(out, d + 1))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_lambda_, _xl_slice(out, d + 2, d, Python.tuple(d), context))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, 2 * d + 2)))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = x^
    _ = y^
    return model


def xl_fit_lars(model: PythonObject, args: PythonObject, lasso: Bool, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """Lars.fit(X, y) and (`lasso`) LassoLars.fit(X, y): `_lars_fit`."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var name = _name(model)
    # LassoLars has max_iter and alpha; Lars has n_nonzero_coefs
    var max_iter: PythonObject
    var alpha_min = Float64(0.0)
    if lasso:
        if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
            raise Error("mojolearn LassoLars: alpha must be >= 0")
        max_iter = py_item(fields, nk[].k_max_iter)
        alpha_min = _float(py_item(fields, nk[].k_alpha))
    else:
        max_iter = py_item(fields, nk[].k_n_nonzero_coefs)
    if _attr(model, "jitter", PythonObject(None)) is not None:
        raise Error("mojolearn " + name + ": jitter is not implemented")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(max_iter), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, 1 if lasso else 0,
                         1 if _truth(_attr(model, "positive", PythonObject(False))) else 0]
    var fp: List[Float64] = [alpha_min]
    var out = _xl_run(entries, ALGO_LARS, x.address, n * d, n, d, y.address, y.rows, ip, fp,
                      2 * d + 4, 2 * d * d + 8 * d, 2 * d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 1)))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_alpha_, _xl_float(out, d + 2))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var k = _xl_count(out, d + 3)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # the active feature indices read natively, published as one list
    # (py_list_int), not a bound `append` call per feature
    var active_words = List[Int](capacity=max(k, 0))
    for i in range(k):  # driver-loop: over the active features (model structure)
        active_words.append(_xl_count(out, d + 4 + i))
    _ = out  # keep the result words alive through the reads above
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_active_, py_list_int(active_words))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = x^
    _ = y^
    return model


def xl_fit_quantile(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """QuantileRegressor.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var q = py_item(fields, nk[].k_quantile)
    if not (_lt(PythonObject(0), q) and _lt(q, PythonObject(1))):
        raise Error("mojolearn QuantileRegressor: quantile must be strictly between 0 and 1")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn QuantileRegressor: alpha must be >= 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var tol = py_item(fields, nk[].k_tol)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, has_sw]
    var fp: List[Float64] = [_float(q), _float(py_item(fields, nk[].k_alpha)), _float(py_call(_op().truediv, tol, PythonObject(100.0))),
                             _float(tol), 1.0]  # fp[4]: unread since MOJOLEARN_XQ_ABS_BALANCE went (TOMBSTONE in x_linear/quantile.mojo)
    var out = _xl_run(entries, ALGO_QUANTILE, x.address, n * d, n, d, yy.address, yy.rows, ip, fp,
                      d + 3, m * m + 3 * m + 4 * n + 3 * d, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = x^
    _ = y^
    _ = yy^
    return model


# ----------------------------------------------------------- classifiers


def _xl_class_codes(model: PythonObject, source: PythonObject, n: Int, context: PythonObject) raises -> ClassicalLabels:
    """`_class_codes(est, y, n)`: the labels encoded once at entry."""
    var labels = classical_encode_labels(source, context)
    if labels.n != n:
        raise Error("mojolearn " + _name(model) + ": X and y lengths differ")
    if len(labels.classes) < 2:
        raise Error("mojolearn " + _name(model) + ": y has one class")
    return labels^


def xl_fit_sgd_classifier(model: PythonObject, args: PythonObject, operation: Int, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.fit(X, y, sample_weight) ("sgd"), Perceptron.fit(X, y,
    sample_weight) ("perceptron") and PassiveAggressiveClassifier.fit(X, y)
    ("pa"): one SGD fit over the class codes."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var losses = py_item(context, nk[].k_xl_sgd_clf_loss)
    if operation == X_LINEAR_OPN_FIT_PERCEPTRON:
        _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), PythonObject(False), py_item(fields, nk[].k_warm_start))
    else:
        _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    if operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER and not py_contains(losses, py_item(fields, nk[].k_loss)):
        raise Error("mojolearn SGDClassifier: loss must be one of " + _sorted_repr(losses))
    if operation == X_LINEAR_OPN_FIT_PA_CLASSIFIER:
        if not py_contains(Python.tuple("hinge", "squared_hinge"), py_item(fields, nk[].k_loss)):
            raise Error("mojolearn PassiveAggressiveClassifier: loss must be 'hinge' or 'squared_hinge'")
        if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveClassifier: C must be > 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var labels = _xl_class_codes(model, args[1], x.rows, context)
    var codes = admitted_vector(x.rows, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", x.rows, context)
    _ = codes  # keep the Python object alive through the native call above
    var k = len(labels.classes)
    py_setitem(fields, nk[].k_classes_, labels.classes)
    var fit: Tuple[PythonObject, PythonObject]
    if operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER:
        var batch_sum = _xl_sgd_auto_sum(py_item(fields, nk[].k_batch_size), py_item(fields, nk[].k_loss), py_item(fields, nk[].k_learning_rate), context)
        fit = _xl_sgd(model, fields, entries, x.view(), codes, labels_address(codes), k, _int(losses[py_item(fields, nk[].k_loss)]),
                      py_item(fields, nk[].k_penalty), py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0),
                      py_item(fields, nk[].k_power_t), py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                      py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), batch_sum, 4096,
                      context)
        _ = codes  # keep the Python object alive through the native call above
    elif operation == X_LINEAR_OPN_FIT_PERCEPTRON:
        fit = _xl_sgd(model, fields, entries, x.view(), codes, labels_address(codes), k, _int(py_item(losses, nk[].k_perceptron)),
                      py_item(fields, nk[].k_penalty), PythonObject("constant"), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0),
                      PythonObject(0.5), PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                      py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), True, 256, context)
        _ = codes  # keep the Python object alive through the native call above
    else:
        var lr = PythonObject("pa1") if _eq(py_item(fields, nk[].k_loss), PythonObject("hinge")) else PythonObject("pa2")
        # scikit-learn's PassiveAggressive*.fit takes no sample_weight
        fit = _xl_sgd(model, fields, entries, x.view(), codes, labels_address(codes), k, _int(py_item(losses, nk[].k_hinge)),
                      PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5),
                      PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                      py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), PythonObject(None),
                      py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), False, 256, context)
        _ = codes  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_coef_, fit[0])
    py_setitem(fields, nk[].k_intercept_, fit[1])
    _ = codes^
    _ = x^
    return model


def xl_fit_ridge_classifier(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """RidgeClassifier.fit(X, y, sample_weight): `_ridge_run` over the int32
    class codes (the +-1 targets are built in the binding)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var name = _name(model)
    if _truth(_attr(model, "positive", PythonObject(False))):
        raise Error("mojolearn " + name + ": positive=True is not implemented")
    if not py_contains(Python.tuple("auto", "cholesky"), _attr(model, "solver", PythonObject("auto"))):
        raise Error("mojolearn " + name + ": solver must be 'auto' or 'cholesky'")
    var alpha = py_item(fields, nk[].k_alpha)
    if not _ge(alpha, PythonObject(0)):
        raise Error("mojolearn RidgeClassifier: alpha must be >= 0")
    xl_ops_require(entries)
    # lane a8c: a CV fold stays on the device (x_linear_fit_fold)
    var x = _xl_fold_matrix(args[0], _name(model), entries, context)
    var n = x.rows
    var d = x.cols
    var labels = classical_encode_labels(args[1], context)
    if labels.n != n:
        raise Error("mojolearn RidgeClassifier: X and y lengths differ")
    var k = len(labels.classes)
    if k < 2:
        raise Error("mojolearn RidgeClassifier: y has one class")
    var T = 1 if k == 2 else k
    var sample_weight = args[2]
    if py_item(fields, nk[].k_class_weight) is not None:
        # theirs: sample_weight * compute_sample_weight(class_weight, y),
        # 'balanced' from the unweighted class counts
        var sw_keep = PythonObject(None)
        var sw_addr = 0
        if sample_weight is not None:
            var w = admit_weight_words(sample_weight, n, context)
            sw_keep = w.array
            sw_addr = w.address
            _ = w  # keep the Python object alive through the native call above
        var prep = _xl_class_prep(model, entries, py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, sw_addr,
                                  False, True, context)
        sample_weight = prep[1]
        _ = sw_keep^
    # `_ridge_run(..., codes_mode=True)`: the codes' 32-bit words, then the
    # checked weights
    var has_sw = 0
    var y = admitted_words(labels.codes, labels_address(labels.codes), n, 1)
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    var ip: List[Int] = [T, 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, 1, has_sw, 1]
    var fp: List[Float64] = [_float(alpha)]
    var out = _xl_run_x(model, entries, ALGO_RIDGE, x.view(), yy.address, yy.rows, ip, fp,
                        T * d + T + 2 + 1 + 1, 3 * d * d + 3 * d + T + d * T + 2 * n, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # status 2: X'X + alpha I does not factor even in float-float
    if _xl_word(out, T * d + T + 2 + 1) == 2.0:
        raise Error("mojolearn " + name + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(out, T * d + T), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    py_setitem(fields, nk[].k_classes_, labels.classes)
    py_setitem(fields, nk[].k_coef_, _xl_slice(out, 0, T * d, Python.tuple(T, d), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_intercept_, _xl_slice(out, T * d, T, Python.tuple(T), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = x^
    _ = yy^
    return model


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



# ------------------------------------------------------------ cv helpers
# RidgeCV, LassoCV, ElasticNetCV and LogisticRegressionCV. An int (or None)
# cv takes the same integer fold ids the Python assigned (the base binding's
# `fold_ids` / `select_fold_i64`, or the binding's own StratifiedKFold ids);
# a user splitter object is the user-API boundary: its `split(X, y)` is
# called once, as the Python did, and its (train, test) index arrays are
# read natively.



def _xl_helper(context: PythonObject, name: String) raises -> PythonObject:
    """A base binding helper (`_buffer._native(name)`) connected at import,
    or the refusal recorded then."""
    var nk = x_linear_keys()
    var entry = py_get(py_item(context, nk[].k_xl_helpers), PythonObject(name))
    if entry is None:
        raise Error("ImportError: mojolearn: the base binding has no `" + name + "`")
    if py_type(entry) is _b().str:
        raise Error(String(py=entry))
    return entry


def _xl_is_int_cv(cv: PythonObject, context: PythonObject) raises -> Bool:
    """`_is_int_cv`: an exact native integer that is not a bool (identity,
    never the numbers.Integral ABC hook)."""
    return labels_integral_type(py_type(cv), context)


def _xl_index_i64(values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_index_i64(values, name=name)`: row indices as a 1-D int64 Array
    (a list, tuple or range through `array('q', ...)` in C; a buffer through
    the native cast)."""
    var nk = x_linear_keys()
    var b = _b()
    var cls = py_type(values)
    if cls is b.list or cls is b.tuple or cls is b.range:
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("q"), values)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)
    # The one integer admission (bindings/admitted_input_unpack.mojo admit_ids):
    # one parallel read of the exporter's strides into fresh int64 words.
    return admit_ids(values, name, context)


def _xl_cv_splits(cv: PythonObject, x_source: PythonObject, y_source: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`_cv_splits(cv, X, y)`: a list of (train, test) int64 Arrays from a
    splitter's `split(X, y)` (called once) or an iterable of pairs."""
    var nk = x_linear_keys()
    var b = _b()
    var pairs = PythonObject(None)
    if py_type(py_type(cv)) is b.type and py_contains(py_item(prep_native_context(context), nk[].k_family_msel_splitter_layouts), py_type(cv)):
        pairs = prep_native_split(cv, x_source, y_source, context)
    elif py_type(cv) is b.list or py_type(cv) is b.tuple:
        msel_admit(cv, prep_native_context(context))
        pairs = cv
    else:
        raise Error("TypeError: native linear CV requires a native splitter or list of index pairs; Python callbacks cannot execute")
    var listed = py_call(b.list, pairs)
    var splits = Python.list()
    for k in range(len(listed)):  # driver-loop: over the cv splits
        var pair = py_call(b.tuple, listed[k])
        if len(pair) != 2:
            if len(pair) > 2:
                raise Error("too many values to unpack (expected 2, got " + String(len(pair)) + ")")
            raise Error("not enough values to unpack (expected 2, got " + String(len(pair)) + ")")
        _ = py_call(splits.append, Python.tuple(_xl_index_i64(pair[0], "train", context),
                                                _xl_index_i64(pair[1], "test", context)))
    if len(splits) == 0:
        raise Error("mojolearn: cv produced no (train, test) split")
    return splits


def _xl_default_folds(n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    """`KFold(n_splits=k).split(...)` unshuffled: contiguous folds by the
    core helpers (`fold_ids` with no codes, `select_fold_i64`)."""
    var fold_store = admitted_vector(n, "<i4", context)
    var fold_counts = admitted_vector(k, "<i8", context)
    var class_counts = admitted_vector(1, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0), PythonObject(k),
                      PythonObject(labels_address(class_counts)), PythonObject(labels_address(fold_store)),
                      PythonObject(labels_address(fold_counts)))
    _ = class_counts  # keep the Python object alive through the native call above
    _ = fold_store  # keep the Python object alive through the native call above
    _ = fold_counts  # keep the Python object alive through the native call above
    var select = _xl_helper(context, "select_fold_i64")
    var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=labels_address(fold_counts))
    _ = fold_counts  # keep the Python object alive through the native call above
    var out = Python.list()
    for fold in range(k):  # driver-loop: over the cv folds
        var size = Int(counts[fold])
        _ = fold_counts  # keep the Python object alive through the native call above
        var test = admitted_vector(size, "<i8", context)
        var train = admitted_vector(n - size, "<i8", context)
        var scratch = admitted_vector(n, "<i8", context)
        var got = _int(py_call_typed(select, PythonObject(labels_address(fold_store)), PythonObject(n), PythonObject(fold),
                                     PythonObject(labels_address(test) if size > 0 else labels_address(scratch)),
                                     PythonObject(labels_address(train) if n - size > 0 else labels_address(scratch))))
        _ = fold_store  # keep the Python object alive through the native call above
        _ = test  # keep the Python object alive through the native call above
        _ = scratch  # keep the Python object alive through the native call above
        _ = train  # keep the Python object alive through the native call above
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with fold_ids")
        _ = scratch  # keep the Python object alive through the native calls above
        _ = train  # keep the Python object alive through the native calls above
        _ = test  # keep the Python object alive through the native calls above
        _ = fold_store  # keep the Python object alive through the native calls above
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def _xl_kfold_ids_f32(n: Int, k: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kfold_ids_f32(n, k)`: KFold's contiguous fold ids as float32."""
    if not _is_int(k) or _lt(k, PythonObject(2)):
        raise Error("mojolearn: cv must be None or an int in [2, n_samples]")
    var folds = _int(k)
    if folds > n:
        raise Error("Cannot have number of splits n_splits=" + String(folds) + " greater than the "
                    + "number of samples: n_samples=" + String(n) + ".")
    var ids = admitted_vector(n, "<i4", context)
    var counts = admitted_vector(folds, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0),
                      PythonObject(folds), PythonObject(0), PythonObject(labels_address(ids)),
                      PythonObject(labels_address(counts)))
    _ = ids  # keep the Python object alive through the native call above
    _ = counts  # keep the Python object alive through the native call above
    var out = admitted_vector(n, "<f4", context)
    labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
    _ = out  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    return out


def _xl_split_words(splits: PythonObject) raises -> XlSplitWords:
    """UNPACK of the cv splits: each side's int64 index address (0 when
    empty) and count. The caller keeps `splits` alive through RUN."""
    var words = XlSplitWords(List[Int](), List[Int](), List[Int](), List[Int]())
    for k in range(len(splits)):  # driver-loop: over the cv splits
        var tr = splits[k][0]
        var te = splits[k][1]
        var tr_size = Int(py=tr.size)
        var te_size = Int(py=te.size)
        words.train.append(labels_address(tr) if tr_size > 0 else 0)
        words.train_sizes.append(tr_size)
        words.test.append(labels_address(te) if te_size > 0 else 0)
        words.test_sizes.append(te_size)
    return words^


def _xl_full_i32(n: Int, value: Int, context: PythonObject) raises -> PythonObject:
    """`full((n,), value, '<i4')`: the store repeated in C."""
    var nk = x_linear_keys()
    var one = py_call(py_item(context, nk[].k_array_type), PythonObject("i"), Python.list(PythonObject(value)))
    var store = py_call(py_attr(one, "__mul__"), PythonObject(n))
    return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context)


def _xl_cv_fold_ids_f32(splits: PythonObject, n: Int, name: String,
                        context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_cv_fold_ids_f32(splits, n, name)`: (fold ids, None) for a
    partition, else (the -(n + 1) words, the per-split roles).

    `_cv_fold_ids(splits, n)`: the int32 fold of every row when the test
    sides partition the rows and every train side is the complement of its
    test side (RUN: x_linear/door_run.mojo, on the base helper table).
    The split words and the helper key are read once for both RUNs, and
    both RUNs run with the GIL released: the former `assign_fold_i64`,
    `check_indices_i64`, `count_fold_hits_i64` and `mark_split_roles_f32`
    exports released it around their work (lane perf-audit-2-a)."""
    var nk = x_linear_keys()
    var key = base_helpers_key(py_item(context, nk[].k_xl_helpers))
    var words = _xl_split_words(splits)
    var ids = _xl_full_i32(n, -1, context)
    var ids_addr = labels_address(ids)
    var partition = False
    var py = Python()
    with GILReleased(py):
        partition = xl_cv_fold_ids_run(key, words, n, ids_addr)
    _ = ids  # keep the Python object alive through the native calls above
    _ = splits  # keep the index Arrays alive through the native calls above
    if partition:
        var out = admitted_vector(n, "<f4", context)
        labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        return (out, PythonObject(None))
    # `_cv_roles_f32`: per-split roles (x_linear/cv_roles.mojo)
    xl_cv_roles_limit(n, name)
    var roles = admitted_vector(len(splits) * n, "<f4", context)
    var roles_addr = labels_address(roles)
    with GILReleased(py):
        xl_cv_roles_run(key, words, n, roles_addr, name)
    _ = words^
    _ = splits  # keep the index Arrays alive through the native calls above
    return (labels_full_f32(n, -Float64(n + 1), context), roles)


def _xl_parts(addrs: List[Int], counts: List[Int], context: PythonObject) raises -> PythonObject:
    """`_concat_f32_parts(parts)`: the parts' 32-bit words in order, one
    float32 vector (one byte copy per part)."""
    var out = admitted_vector(xl_parts_total(counts), "<f4", context)
    xl_run_parts(labels_address(out), addrs, counts)
    return out


# ------------------------------------------------------------------ RidgeCV


def _xl_ridge(model: PythonObject, entries: XLinearOps, x_addr: Int, n: Int, d: Int, y: AdmittedX,
              alphas: List[Float64], sample_weight: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ridge_run(est, a, n, d, Y, 1, alphas, sample_weight)`: one target,
    the alphas' Gram solves; refused when X'X + alpha I does not factor even
    in float-float."""
    var nk = x_linear_keys()
    var A = len(alphas)
    var has_sw = 0
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    var ip: List[Int] = [1, 1 if _truth(py_item(py_fields(model), nk[].k_fit_intercept)) else 0, A, has_sw]
    var out = _xl_run(entries, ALGO_RIDGE, x_addr, n * d, n, d, yy.address, yy.rows, ip, alphas,
                      d + 1 + 2 + A + 1, 3 * d * d + 3 * d + 1 + d + 2 * n, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    if _xl_word(out, d + 1 + 2 + A) == 2.0:
        raise Error("mojolearn " + _name(model) + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(out, d + 1), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
        _ = yy  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native calls above
    _ = yy^
    return out


def _xl_gather(src: Int, rows: PythonObject, n: Int, width: Int, context: PythonObject) raises -> PythonObject:
    """`take(src, idx, width)`: rows idx of src (width float32 words a row)."""
    var count = Int(py=rows.size)
    var shape = Python.tuple(count, width) if width > 1 else Python.tuple(count)
    var out = admitted_array(shape, "<f4", context)
    _ = py_call_typed(_xl_helper(context, "gather_rows_bytes"), PythonObject(src), PythonObject(labels_address(out)),
                      PythonObject(labels_address(rows)), PythonObject(n), PythonObject(count), PythonObject(4 * width))
    _ = out  # keep the Python object alive through the native call above
    return out


def _xl_undefined_warning(message: String, context: PythonObject) raises:
    """`_expansion_metrics._undefined_warning(message)`: scikit-learn's
    UndefinedMetricWarning when sklearn.exceptions is loaded, else
    mojolearn's, at the public method's frame."""
    var nk = x_linear_keys()
    var category = py_item(context, nk[].k_xl_undefined_warning)
    var sk = py_get(py_item(context, nk[].k_modules), "sklearn.exceptions")
    if sk is not None:
        category = py_call(_b().getattr, sk, PythonObject("UndefinedMetricWarning"), category)
    var frame = py_call(native_loaded_module("sys")._getframe, PythonObject(0))
    var module = py_get(frame.f_globals, nk[].k___name__, "")
    classical_warning(message, category, context, Python.tuple(frame.f_code.co_filename, frame.f_lineno, module),
                      "UndefinedMetricWarning")


def xl_fit_ridge_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var b = _b()
    if py_item(fields, nk[].k_scoring) is not None or _truth(py_item(fields, nk[].k_alpha_per_target)):
        raise Error("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
    var given = py_item(fields, nk[].k_alphas)
    var listed = given if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, listed))
    var alphas = List[Float64]()
    for i in range(len(floats)):  # driver-loop: over the alpha grid
        alphas.append(Float64(py=floats[i]))
    var refuse = len(alphas) == 0
    for i in range(len(alphas)):  # driver-loop: over the alpha grid
        refuse = refuse or not (alphas[i] > 0)
    if refuse:
        raise Error("mojolearn RidgeCV: alphas must be positive")
    xl_ops_require(entries)
    var score_scope = _xl_scope(entries, context)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var sample_weight = args[2]
    var cv = py_item(fields, nk[].k_cv)
    var A = len(alphas)
    if cv is None:
        var out = _xl_ridge(model, entries, x.address, n, d, y, alphas, sample_weight, context)
        _ = x  # keep the Python object alive through the native call above
        _xl_set_linear(fields, out, d, context)
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_alpha_, _xl_float(out, d + 1))
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_best_score_, _xl_float(out, d + 2))
        _ = x  # keep the Python object alive through the native call above
        if _truth(py_item(fields, nk[].k_store_cv_results)):
            py_setitem(fields, nk[].k_cv_results_, _xl_slice(out, d + 3, A, Python.tuple(A), context))
            _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        _ = x^
        _ = y^
        return model
    var int_cv = _xl_is_int_cv(cv, context)
    if int_cv and sample_weight is None and _int(cv) >= 2 and n // _int(cv) >= 2:
        # `_fit_kfold`: the k-fold scores on the binding, the first best
        # finite score chosen in Mojo
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        if _truth(py_item(fields, nk[].k_store_cv_results)):
            raise Error("cv!=None and store_cv_results=True are incompatible")
        var n_fw = 2 * d * d + 3 * d + 1 + A * (d + 2) + A * (n // folds + 1)
        var ip: List[Int] = [folds, 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, A]
        var scores = _xl_run(entries, ALGO_RIDGE_KFOLD, x.address, n * d, n, d, y.address, y.rows, ip,
                             alphas, A, n_fw, 1, context)
        var best = _xl_call(entries, XL_OP_BEST_ALPHA, [labels_address(scores)], [A], List[Float64]())
        _ = scores  # keep the Python object alive through the native call above
        if best < 0:
            raise Error("mojolearn RidgeCV: X'X + alpha I is singular in float32 for every alpha "
                        + "(rescale X, or use larger alphas)")
        _ = scores  # keep the Python object alive through the native calls above
        var one: List[Float64] = [alphas[best]]
        _ = scores  # keep the Python object alive through the native call above
        var out = _xl_ridge(model, entries, x.address, n, d, y, one, PythonObject(None), context)
        _xl_set_linear(fields, out, d, context)
        py_setitem(fields, nk[].k_alpha_, PythonObject(alphas[best]))
        _ = scores  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_best_score_, _xl_float(scores, best))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        _ = x^
        _ = y^
        return model
    # `_fit_splits`: a native fit, predict and r2_score per split and alpha
    if _truth(py_item(fields, nk[].k_store_cv_results)):
        raise Error("cv!=None and store_cv_results=True are incompatible")
    var splits = PythonObject(None)
    var yv = _xl_owned(y, context)
    if int_cv:
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        splits = _xl_default_folds(n, folds, context)
    else:
        splits = _xl_cv_splits(cv, args[0], args[1], context)
    var w_keep = PythonObject(None)
    var w_addr = 0
    if sample_weight is not None:
        var w = admit_weight_words(sample_weight, n, context)
        w_keep = _xl_owned(w, context)
        w_addr = labels_address(w_keep)
        _ = w_keep  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    var rows = Python.list()
    for i in range(A):  # driver-loop: over the alpha grid
        _ = py_call(rows.append, Python.list())
    var undefined = False
    for s in range(len(splits)):  # driver-loop: over the cv splits
        var tr = splits[s][0]
        var te = splits[s][1]
        var n_tr = Int(py=tr.size)
        var n_te = Int(py=te.size)
        if n_tr == 0 or n_te == 0:
            raise Error("mojolearn RidgeCV: every cv split needs training and test rows")
        if n_te < 2:
            # scikit-learn's r2_score: NaN, with its warning, for fewer than
            # two test rows; every alpha's mean is then NaN (the first wins)
            _xl_undefined_warning("R^2 score is not well-defined with less than two samples.", context)
            undefined = True
            continue
        var xtr = _xl_gather(x.address, tr, n, d, context)
        var ytr = _xl_gather(labels_address(yv), tr, n, 1, context)
        _ = yv  # keep the Python object alive through the native call above
        var xte = _xl_gather(x.address, te, n, d, context)
        var yte = _xl_gather(labels_address(yv), te, n, 1, context)
        _ = yv  # keep the Python object alive through the native call above
        var wtr = PythonObject(None)
        var wte = PythonObject(None)
        if w_addr != 0:
            wtr = _xl_gather(w_addr, tr, n, 1, context)
            wte = _xl_gather(w_addr, te, n, 1, context)
        var ytr_m = admitted_words(ytr, labels_address(ytr), n_tr, 1)
        _ = ytr  # keep the Python object alive through the native call above
        _ = yv  # keep the Python object alive through the native call above
        var xte_m = admitted_words(xte, labels_address(xte), n_te, d)
        _ = xte  # keep the Python object alive through the native call above
        for i in range(A):  # driver-loop: over the alpha grid
            var one: List[Float64] = [alphas[i]]
            var out = _xl_ridge(model, entries, labels_address(xtr), n_tr, d, ytr_m, one, wtr, context)
            _ = ytr  # keep the Python object alive through the native call above
            _ = xtr  # keep the Python object alive through the native call above
            _ = yv  # keep the Python object alive through the native call above
            var coef = _xl_slice(out, 0, d, Python.tuple(d), context)
            _ = xtr  # keep the Python object alive through the native call above
            var pred = _xl_decision(model, xte_m.view(), coef, _xl_float(out, d), LINK_IDENTITY, True, entries, context)
            _ = xte  # keep the Python object alive through the native call above
            _ = xtr  # keep the Python object alive through the native call above
            var score = labels_r2(yte, pred, wte, PythonObject(None), score_scope)
            _ = yv  # keep the Python object alive through the native call above
            _ = py_call(rows[i].append, PythonObject(score))
        _ = xtr^
        _ = xte^
    var best = 0
    var best_score = py_call(b.float, PythonObject("nan"))
    if not undefined:
        # `_row_means(rows)`: fsum / len per alpha (`row_means_f64`), the
        # first best by `reduce_stat`
        var k = len(rows[0])
        # UNPACK: the alpha grid's Python score rows into one native float64
        # (A, k) table (it was an array('d') extended per row and read back
        # through buffer_info).
        var flat = admitted_vector(max(A * k, 1), "<f8", context)
        var flat_words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=labels_address(flat))
        for i in range(A):  # driver-loop: over the alpha grid
            var row = rows[i]
            if len(row) != k:
                raise Error("ValueError: every alpha needs one score per split")
            for j in range(k):  # driver-loop: over the splits of one alpha (UNPACK of its Python floats)
                flat_words[i * k + j] = Float64(py=row[j])
        var means = admitted_vector(max(A, 1), "<f8", context)
        var bkey = base_helpers_key(py_item(context, nk[].k_xl_helpers))
        if A > 0 and k > 0:
            if base_row_means_f64(bkey, labels_address(flat), A, k, labels_address(means)) != 0:
                raise Error("-inf + inf in fsum")
        best = base_reduce_stat_int(bkey, labels_address(means), 1, A, 3)
        _ = flat  # keep the score table alive through the native calls above
        _ = means  # keep the Python object alive through the native call above
        best_score = PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=labels_address(means))[best])
        _ = means  # keep the Python object alive through the native call above
        _ = flat^
        _ = means^
    var one: List[Float64] = [alphas[best]]
    var w_final = PythonObject(None) if w_addr == 0 else w_keep
    var out = _xl_ridge(model, entries, x.address, n, d, y, one, w_final, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, nk[].k_alpha_, PythonObject(alphas[best]))
    py_setitem(fields, nk[].k_best_score_, best_score)
    _ = w_keep^
    _ = yv^
    _ = x^
    _ = y^
    return model


# ------------------------------------------------------ LassoCV / ElasticNetCV


def _xl_enetcv(model: PythonObject, fields: PythonObject, args: PythonObject, l1_ratios: PythonObject,
               entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """`_enetcv_fit(est, X, y, l1_ratios, sample_weight)`: the alpha grid,
    the folds and the path on the binding; returns the l1_ratios entry
    nearest the best one."""
    var nk = x_linear_keys()
    var name = _name(model)
    var b = _b()
    if not _eq(py_item(fields, nk[].k_selection), PythonObject("cyclic")):
        raise Error("mojolearn " + name + ": selection='random' is not implemented")
    var L = len(l1_ratios)
    for i in range(L):  # driver-loop: over the l1_ratio grid
        var r = Float64(py=l1_ratios[i])
        if not (r > 0 and r <= 1):
            raise Error("mojolearn " + name + ": l1_ratio must be in (0, 1]")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var cv = py_item(fields, nk[].k_cv)
    var ids = PythonObject(None)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var roles = Optional[PythonObject]()
    var folds = 0
    if cv is None or _xl_is_int_cv(cv, context):
        var k = PythonObject(5) if cv is None else py_call(b.int, cv)
        ids = _xl_kfold_ids_f32(n, k, context)
        folds = _int(k)
    else:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, name, context)
        ids = got[0]
        roles = py_present(got[1])
    var alphas = py_item(fields, nk[].k_alphas)
    var n_alphas = _attr(model, "n_alphas", PythonObject(None))
    var explicit = False
    var grid = 0
    var values = Python.list()
    if _is_int(alphas):
        grid = _int(alphas)
    elif alphas is None or py_contains(Python.tuple("warn", "deprecated"), alphas):
        grid = _int(n_alphas) if _is_int_plain(n_alphas) else 100
    else:
        explicit = True
        values = py_call(b.sorted, py_call(b.map, b.float, alphas), reverse=PythonObject(True))
        grid = len(values)
    if grid < 1:
        raise Error("mojolearn " + name + ": at least one alpha is required")
    var sample_weight = args[2]
    # A plain number is no weight vector (the former numbers.Number test,
    # by exact native type identity).
    if labels_real_type(py_type(sample_weight), context) or py_type(sample_weight) is _b().complex:
        sample_weight = PythonObject(None)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(y.address)
    _ = y  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(ids))
    _ = ids  # keep the Python object alive through the native call above
    counts.append(n)
    var weighted = 0
    var w_keep = PythonObject(None)
    if Bool(roles):
        # per-split roles ride the weighted arm, after the weights
        if sample_weight is None:
            w_keep = labels_full_f32(n, 1.0, context)
        else:
            w_keep = _xl_owned(admit_weight_words(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        addrs.append(labels_address(roles.value()))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.value().size))
        weighted = 1
    elif sample_weight is not None:
        w_keep = _xl_owned(admit_weight_words(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        weighted = 1
    var yy = _xl_parts(addrs, counts, context)
    var fp = List[Float64]()
    fp.append(_float(py_item(fields, nk[].k_eps)))
    fp.append(_float(py_item(fields, nk[].k_tol)))
    for i in range(L):  # driver-loop: over the l1_ratio grid
        fp.append(Float64(py=l1_ratios[i]))
    if explicit:
        for i in range(grid):  # driver-loop: over the user alpha grid
            fp.append(Float64(py=values[i]))
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, grid, folds, L,
                         1 if explicit else 0, 1 if _truth(py_item(fields, nk[].k_positive)) else 0, weighted]
    var out = _xl_run(entries, ALGO_ENETCV, x.address, n * d, n, d, labels_address(yy), Int(py=yy.size), ip,
                      fp, d + 4 + L * grid + L * grid * folds, d * d + 4 * d + 3 + grid * (d + 2), 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_alpha_, _xl_float(out, d + 1))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var l1_best = Float64(_xl_word(out, d + 2))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, d + 3)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var off = d + 4
    if L == 1:
        py_setitem(fields, nk[].k_alphas_, _xl_slice(out, off, grid, Python.tuple(grid), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_mse_path_, _xl_slice(out, off + grid, grid * folds, Python.tuple(grid, folds), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, nk[].k_alphas_, _xl_slice(out, off, L * grid, Python.tuple(L, grid), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_mse_path_, _xl_slice(out, off + L * grid, L * grid * folds, Python.tuple(L, grid, folds), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # the user's own value (the kernel carried it as float32): the first
    # entry nearest the returned one
    var pick = 0
    var gap = abs(Float64(py=l1_ratios[0]) - l1_best)
    for i in range(1, L):  # driver-loop: over the l1_ratio grid
        var g = abs(Float64(py=l1_ratios[i]) - l1_best)
        if g < gap:
            gap = g
            pick = i
    _ = yy^
    _ = w_keep^
    _ = ids^
    _ = roles^
    _ = x^
    _ = y^
    return l1_ratios[pick]


def _is_int_plain(value: PythonObject) raises -> Bool:
    """`isinstance(value, int)` (bools included, as the Python test was)."""
    return _isinstance(value, _b().int)


def xl_fit_lasso_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """LassoCV.fit(X, y, sample_weight): the l1_ratio grid [1.0]."""
    var fields = py_fields(model)
    _ = _xl_enetcv(model, fields, args, Python.list(PythonObject(1.0)), entries, context)
    return model


def xl_fit_elastic_net_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """ElasticNetCV.fit(X, y, sample_weight): l1_ratio_ is the grid entry
    nearest the best one."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var b = _b()
    var given = py_item(fields, nk[].k_l1_ratio)
    var ratios = py_call(b.list, given) if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, ratios))
    py_setitem(fields, nk[].k_l1_ratio_, _xl_enetcv(model, fields, args, floats, entries, context))
    return model


# ---------------------------------------------------- LogisticRegressionCV


def _xl_cs_grid(cs: PythonObject, context: PythonObject) raises -> List[Float64]:
    """The Cs grid: `np.logspace(-4, 4, m)` through `_pm.powr` (10 ** y by
    the decimal module at 50 digits, one correctly rounded conversion), or
    the user's values."""
    var nk = x_linear_keys()
    var grid = List[Float64]()
    if _is_int(cs):
        var m = _int(cs)
        if m <= 1:
            grid.append(1e-4)
            return grid^
        var ten = py_call(py_item(context, nk[].k_xl_decimal), PythonObject(10.0))
        var ctx = py_item(context, nk[].k_xl_pow_context)
        for i in range(m):  # driver-loop: over the Cs grid
            var exponent = -4.0 + Float64(8 * i) / Float64(m - 1)
            var value = Float64(py=py_call(_b().float, py_call(ctx.power, ten, py_call(py_item(context, nk[].k_xl_decimal), PythonObject(exponent)))))
            if isinf(value):
                raise Error("OverflowError: math range error")
            grid.append(value)
        return grid^
    var floats = py_call(_b().list, py_call(_b().map, _b().float, cs))
    for i in range(len(floats)):  # driver-loop: over the user Cs grid
        grid.append(Float64(py=floats[i]))
    return grid^


def xl_fit_logistic_cv(model: PythonObject, args: PythonObject, entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """LogisticRegressionCV.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    var b = _b()
    if not _eq(py_item(fields, nk[].k_penalty), PythonObject("l2")) or _truth(py_item(fields, nk[].k_dual)) or py_item(fields, nk[].k_l1_ratios) is not None:
        raise Error("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
    if py_item(fields, nk[].k_scoring) is not None or not _truth(py_item(fields, nk[].k_refit)):
        raise Error("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
    if not py_contains(Python.tuple("lbfgs", "newton-cg", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var labels = _xl_class_codes(model, args[1], n, context)
    var K = len(labels.classes)
    var kp = 1 if K == 2 else K
    var Cs = _xl_cs_grid(py_item(fields, nk[].k_Cs), context)
    var cv = py_item(fields, nk[].k_cv)
    var explicit = not (cv is None or _xl_is_int_cv(cv, context))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var roles = Optional[PythonObject]()
    var fold_ids = PythonObject(None)
    var folds = 0
    if explicit:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, "LogisticRegressionCV", context)
        fold_ids = got[0]
        roles = py_present(got[1])
    else:
        folds = 5 if cv is None else _int(cv)
    var sample_weight = args[2]
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var raw = Optional[PythonObject]()
    if sample_weight is not None:
        raw = _xl_owned(admit_weight_words(sample_weight, n, context), context)
    var class_weight = py_item(fields, nk[].k_class_weight)
    var weigh = sample_weight is not None or class_weight is not None or Bool(roles)
    var prep = _xl_class_prep(model, entries, class_weight, labels.classes, labels.codes,
                              0 if not raw else labels_address(raw.value()), True, class_weight is not None, context)
    _ = raw  # keep the Python object alive through the native call above
    var fitw = prep[1]
    _ = raw  # keep the Python object alive through the native call above
    var largest = prep[2]
    _ = raw  # keep the Python object alive through the native call above
    if not explicit:
        # `_check_stratified_folds(largest, folds)`
        if folds < 2 or folds > largest:
            raise Error("mojolearn: cv must be None or an int in [2, the largest class size]")
    var codes = admitted_vector(n, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", n, context)
    _ = codes  # keep the Python object alive through the native call above
    var zeros = admitted_vector(n, "<f4", context)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(labels_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(fold_ids) if explicit else labels_address(zeros))
    _ = zeros  # keep the Python object alive through the native call above
    _ = fold_ids  # keep the Python object alive through the native call above
    counts.append(n)
    var has_sw = 0
    if weigh:
        if not raw:
            raw = labels_full_f32(n, 1.0, context)
        addrs.append(labels_address(raw.value()) if fitw is None else labels_address(fitw))
        _ = fitw  # keep the Python object alive through the native call above
        _ = raw  # keep the Python object alive through the native call above
        counts.append(n)
        addrs.append(labels_address(py_or_none(raw)))
        _ = raw  # keep the Python object alive through the native call above
        counts.append(n)
        has_sw = 1
    if Bool(roles):
        addrs.append(labels_address(roles.value()))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.value().size))
    var yy = _xl_parts(addrs, counts, context)
    var p = kp * (d + 1)
    var nc = len(Cs)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, kp, nc, folds, has_sw]
    if explicit:
        ip.append(1)
    var fp = List[Float64]()
    fp.append(_float(py_item(fields, nk[].k_tol)))
    for i in range(nc):  # driver-loop: over the Cs grid
        fp.append(Cs[i])
    var out = _xl_run(entries, ALGO_LOGCV, x.address, n * d, n, d, labels_address(yy), Int(py=yy.size), ip,
                      fp, kp * d + kp + 2 + folds * nc, p + 1 + n * (kp + 1) + xl_lbfgs_work(p), 5, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_classes_, labels.classes)
    py_setitem(fields, nk[].k_coef_, _xl_slice(out, 0, kp * d, Python.tuple(kp, d), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_intercept_, _xl_slice(out, kp * d, kp, Python.tuple(kp), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # the grid entry nearest the returned best C (the first on a tie)
    var best_word = Float64(_xl_word(out, kp * d + kp))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var pick = 0
    var gap = abs(Cs[0] - best_word)
    for i in range(1, nc):  # driver-loop: over the Cs grid
        var g = abs(Cs[i] - best_word)
        if g < gap:
            gap = g
            pick = i
    var cs_list = _xl_reals(Cs)
    py_setitem(fields, nk[].k_Cs_, cs_list)
    var chosen = Python.list()
    for c in range(kp):  # driver-loop: over the fitted problems (model structure)
        _ = py_call(chosen.append, PythonObject(Cs[pick]))
    py_setitem(fields, nk[].k_C_, chosen)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(out, kp * d + kp + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var off = kp * d + kp + 2
    var names = py_call(b.list, labels.classes)
    var scores = Python.dict()
    var first = 1 if kp == 1 else 0
    for c in range(first, len(names)):  # driver-loop: over the class labels (one score table each)
        py_setitem(scores, names[c], _xl_slice(out, off, folds * nc, Python.tuple(folds, nc), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_scores_, scores)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _ = yy^
    _ = codes^
    _ = zeros^
    _ = raw^
    _ = fitw^
    _ = fold_ids^
    _ = roles^
    _ = x^
    return model


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
