# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Robust covariance estimation and outlier detection, under scikit-learn's module name."""

from ._expansion_decomp import EllipticEnvelope, MinCovDet

__all__ = ["EllipticEnvelope", "MinCovDet"]
