# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the host forest entry suite."""
from checks.native_tests.host_entry_forest_suite import run_checks


def main() raises:
    run_checks()
