# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Feature selection and univariate feature scores, under scikit-learn's module name."""

from ._expansion_prep import RFE, SelectKBest, VarianceThreshold, chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression, r_regression

__all__ = ["RFE", "SelectKBest", "VarianceThreshold", "chi2", "f_classif", "f_regression", "mutual_info_classif", "mutual_info_regression", "r_regression"]
