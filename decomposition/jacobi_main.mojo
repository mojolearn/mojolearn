# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the device eigensolver checks.

Separate from `pca_main.mojo` because these run the solver DIRECTLY, across
sizes that cross where the old `JACOBI_MAX_N = 32` cap sat, without a PCA
around it to blur what failed.
"""

from decomposition.checks.jacobi_check import (
    check_jacobi_device_sizes,
    check_jacobi_reaches_past_32,
    check_jacobi_scale_invariance,
)


def main() raises:
    check_jacobi_device_sizes()
    check_jacobi_reaches_past_32()
    check_jacobi_scale_invariance()
