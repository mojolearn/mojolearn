# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN body of `providers/python/cluster_public_lifecycle_pyapi.mojo` (wave 8
run split): the public method name of a cluster operation word, moved
unchanged."""


def _method(operation: String) -> String:
    if operation.endswith("partial_fit"):
        return String("partial_fit")
    if operation.endswith("fit_predict"):
        return String("fit_predict")
    if operation.endswith("fit_transform"):
        return String("fit_transform")
    if operation.endswith("predict_proba"):
        return String("predict_proba")
    if operation.endswith("score_samples"):
        return String("score_samples")
    if operation.endswith("transform"):
        return String("transform")
    if operation.endswith("predict"):
        return String("predict")
    return String("fit") if operation.endswith("fit") else String("score")
