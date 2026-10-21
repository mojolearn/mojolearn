# SPDX-License-Identifier: Apache-2.0
# Native voting/probability aggregation of existing neighbor ids; does not exercise end-to-end search plus classification.
from neighbors.checks.knn_classify_check import check_knn_classify_matches_host_transcription


def main() raises:
    check_knn_classify_matches_host_transcription()
    print("NATIVE_ONLY_PASS knn_classifier")
