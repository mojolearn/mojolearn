# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the byte LM trainer table suite."""
from checks.native_tests.byte_lm_entry_suite import run_checks


def main() raises:
    run_checks()
