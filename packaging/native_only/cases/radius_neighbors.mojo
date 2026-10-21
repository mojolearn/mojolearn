# SPDX-License-Identifier: Apache-2.0
# Native pointer-facing radius count/fill with consumed membership/distances against brute force.
from neighbors.checks.radius_check import check_radius_neighbors_matches_host


def main() raises:
    check_radius_neighbors_matches_host()
    print("NATIVE_ONLY_PASS radius_neighbors")
