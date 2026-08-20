"""Entry point for the PCA checks.

The Gram dispatch checks run here too: the covariance product is the one
consumer every fit in this section shares, and `PORTING_RULES.md 8` wants
both sides of `gemm_tn`'s split-K/vendor switch exercised by name.
"""

from decomposition.mojo_only.pca_check import (
    check_input_restored,
    check_covariance_is_symmetric,
    check_pca_fit,
    check_pca_invariants,
    check_tsvd_against_pca,
)
from mojo_only.gram_splitk_check import (
    check_gram_dispatch,
    check_gram_splitk_oracle,
    check_gram_vendor_arm,
)
from mojo_only.hardware_matrix_check import check_hardware_matrix


def main() raises:
    check_hardware_matrix()
    check_gram_splitk_oracle()
    check_gram_vendor_arm()
    check_gram_dispatch()
    check_covariance_is_symmetric()
    check_pca_fit()
    check_pca_invariants()
    check_input_restored()
    check_tsvd_against_pca()
