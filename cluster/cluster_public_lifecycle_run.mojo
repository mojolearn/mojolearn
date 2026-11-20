# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `providers/python/cluster_public_lifecycle_pyapi.mojo` (wave 8
run split): the public method of a cluster operation. Lane bw2-cluster: the
method is chosen by the operation's owner and leaf code (it was a String
built per call from up to eight suffix tests of the name, then compared up
to six times); its text is a static literal, for the not-fitted message."""
from bindings.native_operation_codes import FR_CFA, FR_XC
from bindings.operation_names_xc import XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM, XC_OPN_FAMILY_XC_VARIANTS_SCORE
from bindings.operation_names_cfa import (
    CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT, CFA_OPN_FAMILY_CFA_FIT_PREDICT, CFA_OPN_FAMILY_CFA_FIT_TRANSFORM,
    CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT, CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT,
    CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT, CFA_OPN_FAMILY_CFA_HDBSCAN_FIT_PREDICT)
from bindings.operation_names_mix import (
    MIX_OPN_FAMILY_MIX_BGMM_FIT, MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT, MIX_OPN_FAMILY_MIX_BGMM_PREDICT,
    MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA, MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES)

# The fits are first, so `method <= CLUSTER_METHOD_FIT_TRANSFORM` is a fit.
comptime CLUSTER_METHOD_FIT = 0
comptime CLUSTER_METHOD_PARTIAL_FIT = 1
comptime CLUSTER_METHOD_FIT_PREDICT = 2
comptime CLUSTER_METHOD_FIT_TRANSFORM = 3
comptime CLUSTER_METHOD_PREDICT = 4
comptime CLUSTER_METHOD_PREDICT_PROBA = 5
comptime CLUSTER_METHOD_SCORE_SAMPLES = 6
comptime CLUSTER_METHOD_TRANSFORM = 7
comptime CLUSTER_METHOD_SCORE = 8


def _method(owner: Int, operation: Int) -> Int:
    """The public method of a flagged cluster operation: FR_CFA and FR_XC
    leaves, any other owner a `family_mix_bgmm_*` leaf (the three owners the
    family door sends here). Each name maps as its old suffix test did."""
    if owner == FR_CFA:
        if operation == CFA_OPN_FAMILY_CFA_MBK_PARTIAL_FIT:
            return CLUSTER_METHOD_PARTIAL_FIT
        if (operation == CFA_OPN_FAMILY_CFA_FIT_PREDICT or operation == CFA_OPN_FAMILY_CFA_MEANSHIFT_FIT_PREDICT
                or operation == CFA_OPN_FAMILY_CFA_OPTICS_FIT_PREDICT or operation == CFA_OPN_FAMILY_CFA_AP_FIT_PREDICT
                or operation == CFA_OPN_FAMILY_CFA_HDBSCAN_FIT_PREDICT):
            return CLUSTER_METHOD_FIT_PREDICT
        if operation == CFA_OPN_FAMILY_CFA_FIT_TRANSFORM:
            return CLUSTER_METHOD_FIT_TRANSFORM
        return CLUSTER_METHOD_FIT
    if owner == FR_XC:
        if operation == XC_OPN_FAMILY_XC_VARIANTS_TRANSFORM:
            return CLUSTER_METHOD_TRANSFORM
        if operation == XC_OPN_FAMILY_XC_VARIANTS_SCORE:
            return CLUSTER_METHOD_SCORE
        return CLUSTER_METHOD_PREDICT
    if operation == MIX_OPN_FAMILY_MIX_BGMM_FIT:
        return CLUSTER_METHOD_FIT
    if operation == MIX_OPN_FAMILY_MIX_BGMM_FIT_PREDICT:
        return CLUSTER_METHOD_FIT_PREDICT
    if operation == MIX_OPN_FAMILY_MIX_BGMM_PREDICT:
        return CLUSTER_METHOD_PREDICT
    if operation == MIX_OPN_FAMILY_MIX_BGMM_PREDICT_PROBA:
        return CLUSTER_METHOD_PREDICT_PROBA
    if operation == MIX_OPN_FAMILY_MIX_BGMM_SCORE_SAMPLES:
        return CLUSTER_METHOD_SCORE_SAMPLES
    return CLUSTER_METHOD_SCORE


def _method_text(method: Int) -> String:
    """The method's public name (a static literal)."""
    if method == CLUSTER_METHOD_FIT:
        return "fit"
    if method == CLUSTER_METHOD_PARTIAL_FIT:
        return "partial_fit"
    if method == CLUSTER_METHOD_FIT_PREDICT:
        return "fit_predict"
    if method == CLUSTER_METHOD_FIT_TRANSFORM:
        return "fit_transform"
    if method == CLUSTER_METHOD_PREDICT:
        return "predict"
    if method == CLUSTER_METHOD_PREDICT_PROBA:
        return "predict_proba"
    if method == CLUSTER_METHOD_SCORE_SAMPLES:
        return "score_samples"
    if method == CLUSTER_METHOD_TRANSFORM:
        return "transform"
    return "score"
