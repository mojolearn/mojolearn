# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the `svd_solver='full'` checks.

Separate from `pca_main.mojo` because these run the QR and the one-sided
Jacobi DIRECTLY, with sabotage arms, and because a failure here should name
the R-SVD rather than arrive wrapped in the covariance arm's gates.

Order matters. The fold widths first, because every number below is folded
by them; then the QR, because the SVD's input is its output; then the SVD;
then the arm against the one already shipped, which is the only part that
compares two estimators rather than one estimator against a property.
"""

from decomposition.checks.svd_full_check import (
    check_full_beats_covariance_on_ill_conditioning,
    check_full_components_are_orthonormal,
    check_full_matches_covariance_on_well_conditioned,
    check_full_mean_matches_covariance_arm,
    check_full_refuses_a_wide_matrix,
    check_full_refuses_unconverged,
    check_full_reflector_sign_earns_its_place,
    check_full_scale_invariance,
    check_full_spectrum_matches_float64,
    check_full_survives_a_constant_column,
    check_full_truncated_reconstruction_is_optimal,
    check_qr_fold_width_is_pinned,
    check_qr_gram_matches_float64,
    check_tsqr_agrees_with_one_block,
)


def main() raises:
    check_qr_fold_width_is_pinned()
    check_qr_gram_matches_float64()
    check_tsqr_agrees_with_one_block()
    check_full_reflector_sign_earns_its_place()
    check_full_spectrum_matches_float64()
    check_full_components_are_orthonormal()
    check_full_truncated_reconstruction_is_optimal()
    check_full_scale_invariance()
    check_full_refuses_unconverged()
    check_full_survives_a_constant_column()
    check_full_refuses_a_wide_matrix()
    check_full_mean_matches_covariance_arm()
    check_full_matches_covariance_on_well_conditioned()
    check_full_beats_covariance_on_ill_conditioning()
