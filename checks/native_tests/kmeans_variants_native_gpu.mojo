# SPDX-License-Identifier: Apache-2.0
"""Standalone entrypoint for the reusable KMeans variant device fixtures."""
from checks.native_tests.kmeans_variants_native_gpu_suite import run_checks


def main() raises:
    run_checks()
