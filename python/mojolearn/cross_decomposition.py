# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Partial least squares and canonical correlation analysis, under scikit-learn's module name."""

from ._expansion_decomp import CCA, PLSCanonical, PLSRegression

__all__ = ["CCA", "PLSCanonical", "PLSRegression"]
