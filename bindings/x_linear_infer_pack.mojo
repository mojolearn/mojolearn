# SPDX-License-Identifier: Apache-2.0
# PACK of the linear expansion estimators' inference operations: after
# x_linear/door_run.mojo `xl_decision_run` wrote the result words, refuse a
# non-finite X the entry found (scikit-learn's wording) and return the
# object the public method returns. UNPACK is
# bindings/x_linear_infer_unpack.mojo.
from std.python import PythonObject
from bindings.py_balanced import py_fields, py_item
from bindings.admitted_input_unpack import admitted_vector
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_common_unpack import XlDecision, xl_decision_refuse
from bindings.classical_labels_door import labels_address, labels_cast, classical_decode_labels


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
