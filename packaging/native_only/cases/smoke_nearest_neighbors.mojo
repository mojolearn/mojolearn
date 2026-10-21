# SPDX-License-Identifier: Apache-2.0
# Native pointer-facing nearest-neighbor search including input upload and consumed indices against brute force.
from neighbors.checks.estimator_check import check_knn_search_matches_host


def main() raises:
    check_knn_search_matches_host()
    print("NATIVE_ONLY_PASS nearest_neighbors")
