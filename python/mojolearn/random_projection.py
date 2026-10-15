# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Random projections, under scikit-learn's module name."""

from ._expansion_decomp import GaussianRandomProjection, SparseRandomProjection

__all__ = ["GaussianRandomProjection", "SparseRandomProjection"]
