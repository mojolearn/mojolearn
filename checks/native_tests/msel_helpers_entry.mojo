# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the model-selection helper entry suite."""
from checks.native_tests.msel_helpers_entry_suite import run_checks


def main() raises:
    run_checks()
