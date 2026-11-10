# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the importable native linear archive checks."""
from checks.native_tests.archive_linear_suite import run_checks


def main() raises:
    run_checks()
