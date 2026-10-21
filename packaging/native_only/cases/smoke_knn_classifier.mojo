# SPDX-License-Identifier: Apache-2.0
# Native search plus classification. The vote oracle reuses native search ids.
from neighbors.checks.knn_classify_check import check_knn_classify_matches_host_transcription


def main() raises:
    check_knn_classify_matches_host_transcription()
    print("NATIVE_ONLY_PASS knn_classifier")
