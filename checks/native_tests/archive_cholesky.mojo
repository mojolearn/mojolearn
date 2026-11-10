# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the reusable native Cholesky archive suite."""
from checks.native_tests.archive_cholesky_suite import run_checks


def main() raises:
    run_checks()
