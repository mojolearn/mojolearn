# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the PCA checks.

The Gram dispatch checks run here too: the covariance product is the one
consumer every fit in this section shares, and `PORTING_RULES.md 8` wants
both sides of `gemm_tn`'s split-K/vendor switch exercised by name.
"""

from decomposition.original.pca_check import (
    check_covariance_fused_and_fallback_restore,
    check_input_restored,
    check_covariance_is_symmetric,
    check_pca_fit,
    check_pca_invariants,
    check_tsvd_against_pca,
)
from original.gram_splitk_check import (
    check_gram_centered_fused,
    check_gram_dispatch,
    check_gram_splitk_oracle,
    check_gram_vendor_arm,
)
from original.hardware_matrix_check import check_hardware_matrix


def main() raises:
    check_hardware_matrix()
    check_gram_splitk_oracle()
    check_gram_vendor_arm()
    check_gram_dispatch()
    check_gram_centered_fused()
    check_covariance_is_symmetric()
    check_covariance_fused_and_fallback_restore()
    check_pca_fit()
    check_pca_invariants()
    check_input_restored()
    check_tsvd_against_pca()
