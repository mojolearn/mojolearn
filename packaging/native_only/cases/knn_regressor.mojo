# SPDX-License-Identifier: Apache-2.0
# Native regression aggregation of existing neighbor ids; does not exercise end-to-end search plus regression.
from neighbors.checks.knn_regress_check import check_knn_regress_matches_host_transcription


def main() raises:
    check_knn_regress_matches_host_transcription()
    print("NATIVE_ONLY_PASS knn_regressor")
