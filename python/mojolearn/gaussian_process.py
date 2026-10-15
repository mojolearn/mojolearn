# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gaussian process regression and classification and their kernels, under scikit-learn's module name."""

from ._gp_impl import GaussianProcessRegressor, ConstantKernel, Matern, RBF, WhiteKernel
from ._gpc_impl import GaussianProcessClassifier

__all__ = ["GaussianProcessRegressor", "ConstantKernel", "Matern", "RBF", "WhiteKernel", "GaussianProcessClassifier"]
