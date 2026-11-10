# SPDX-License-Identifier: Apache-2.0
"""Standalone entry point for the reusable native StandardScaler suite."""
from checks.native_tests.archive_standard_suite import run_checks


def main() raises:
    run_checks()
