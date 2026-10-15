# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Linear and quadratic discriminant analysis, under scikit-learn's module name."""

from ._expansion_prep import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]
