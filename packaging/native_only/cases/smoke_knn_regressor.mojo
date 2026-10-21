# SPDX-License-Identifier: Apache-2.0
# Native search plus regression. The fold oracle reuses native search ids.
from neighbors.checks.knn_regress_check import check_knn_regress_matches_host_transcription


def main() raises:
    check_knn_regress_matches_host_transcription()
    print("NATIVE_ONLY_PASS knn_regressor")
