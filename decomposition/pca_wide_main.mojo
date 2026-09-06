# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the WIDE PCA checks, whose output feeds the sklearn oracle.

Separate from `pca_main.mojo` because those all run at four features, which
is why a 32-feature cap in the eigensolver shipped unnoticed.
"""

from decomposition.checks.pca_check import check_pca_truncation, check_pca_wide


def main() raises:
    check_pca_wide()
    check_pca_truncation()
