# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the trees DART entry suite."""
from checks.native_tests.trees_entry_dart_suite import run_checks


def main() raises:
    run_checks()
