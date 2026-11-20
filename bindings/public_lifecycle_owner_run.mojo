# SPDX-License-Identifier: Apache-2.0
"""RUN of the native estimator lifecycle (bindings/public_lifecycle_owner.mojo):
which public methods fit and which need a fitted instance. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word."""


def lifecycle_fitting(method: String) -> Bool:
    """The methods that fit (`_mode._FITS`)."""
    return method == "fit" or method == "partial_fit" or method == "fit_transform" or method == "fit_predict"


def lifecycle_needs_fit(method: String) -> Bool:
    """The methods that refuse an unfitted instance (`_mode._NEEDS_FIT`)."""
    return (method == "predict" or method == "predict_proba" or method == "predict_log_proba"
            or method == "decision_function" or method == "transform" or method == "inverse_transform"
            or method == "score" or method == "score_samples" or method == "predict_joint_log_proba")
