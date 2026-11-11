# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the neural-blocks entry table suite."""
from checks.native_tests.neural_entry_suite import run_checks


def main() raises:
    run_checks()
