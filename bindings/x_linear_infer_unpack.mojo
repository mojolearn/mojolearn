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
from std.python import PythonObject
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
