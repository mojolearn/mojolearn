# SPDX-License-Identifier: Apache-2.0
# Native caller-facing fit with input/output host buffers, Lloyd loop, final label assignment, consumed planted memberships and centers.
from cluster.checks.estimator_check import check_kmeans_fit_recovers_planted


def main() raises:
    check_kmeans_fit_recovers_planted()
    print("NATIVE_ONLY_PASS kmeans")
