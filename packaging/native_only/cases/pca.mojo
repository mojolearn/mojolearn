# SPDX-License-Identifier: Apache-2.0
# Native PCA fit and consumed eigenvalue/component orthogonality and residual assertions.
from decomposition.checks.pca_check import check_pca_fit


def main() raises:
    check_pca_fit()
    print("NATIVE_ONLY_PASS pca")
