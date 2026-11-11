# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the base helper entry suite."""
from checks.native_tests.base_helpers_entry_suite import run_checks


def main() raises:
    run_checks()
