# SPDX-License-Identifier: Apache-2.0
# Native whole spectral clustering fit:144 rows,3 planted blobs; consumed labels must reproduce the planted partition bijectively.
from spectral.checks.spectral_check import check_spectral_clustering_labels as check_0


def main() raises:
    check_0()
    print("NATIVE_ONLY_PASS spectral_clustering")
