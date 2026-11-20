# SPDX-License-Identifier: Apache-2.0
"""Typed operation-name tests of the neighbors public lifecycle (RUN)."""


def _method(operation: String) raises -> String:
    var names = ["fit_predict", "fit_transform", "predict_log_proba", "predict_proba", "score_samples",
                 "decision_function", "transform", "predict", "score", "fit"]
    for suffix in names:  # small-loop(public neighbor operation names)
        if operation.endswith(suffix):
            return suffix
    raise Error("RuntimeError: unknown native neighbor lifecycle operation " + operation)


def _knn_fold_operation(operation: String) -> Bool:
    """The k-NN query operations model selection calls on a fold (the
    classifier's and regressor's predict / predict_proba / score)."""
    return (operation == "family_xn_knn_clf_predict" or operation == "family_xn_knn_clf_predict_proba"
            or operation == "family_xn_knn_clf_score" or operation == "family_xn_knn_reg_predict"
            or operation == "family_xn_knn_reg_score")
