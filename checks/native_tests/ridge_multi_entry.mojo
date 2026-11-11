# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the ridge multi entry suite."""
from checks.native_tests.ridge_multi_entry_suite import run_checks


def main() raises:
    run_checks()
