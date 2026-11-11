# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the gbdt entry suite."""
from checks.native_tests.trees_entry_gbdt_suite import run_checks


def main() raises:
    run_checks()
