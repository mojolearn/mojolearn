# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the sequence optimizer entry suite."""
from checks.native_tests.seq_opt_entry_suite import run_checks


def main() raises:
    run_checks()
