# SPDX-License-Identifier: Apache-2.0
"""The GaussianMixture scoring operation word of each former entry name (RUN)."""
from core.mixture_native_api import (
    MIXTURE_NATIVE_SCORE_SAMPLES, MIXTURE_NATIVE_PREDICT_PROBA, MIXTURE_NATIVE_PREDICT, MIXTURE_NATIVE_SCORE_BIC_AIC,
)


def _gmm_operation(name: String) raises -> Int:
    """The scoring operation of the former Python entry `name`."""
    if name == "gmm_score_samples":
        return MIXTURE_NATIVE_SCORE_SAMPLES
    if name == "gmm_predict_proba":
        return MIXTURE_NATIVE_PREDICT_PROBA
    if name == "gmm_predict":
        return MIXTURE_NATIVE_PREDICT
    if name == "gmm_score_bic_aic":
        return MIXTURE_NATIVE_SCORE_BIC_AIC
    raise Error("RuntimeError: unknown GaussianMixture scoring entry " + name)
