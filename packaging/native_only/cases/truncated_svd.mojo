# SPDX-License-Identifier: Apache-2.0
# Native truncated SVD fit compared with PCA on centered data.
from decomposition.checks.pca_check import check_tsvd_against_pca


def main() raises:
    check_tsvd_against_pca()
    print("NATIVE_ONLY_PASS truncated_svd")
