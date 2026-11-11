# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process NET-P01 native production composition proof."""
from checks.native_tests.linear_entry_production_suite import run_checks


def main() raises:
    run_checks()
