# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the resident non-symmetric GBDT predict suite."""
from checks.native_tests.gbdt_resident_ns_predict_gpu_suite import run_checks


def main() raises:
    run_checks()
