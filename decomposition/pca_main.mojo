"""Entry point for the PCA checks."""

from decomposition.mojo_only.pca_check import (
    check_input_restored,
    check_pca_fit,
    check_pca_invariants,
    check_tsvd_against_pca,
)


def main() raises:
    check_pca_fit()
    check_pca_invariants()
    check_input_restored()
    check_tsvd_against_pca()
