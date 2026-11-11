# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the Samba entry table suite."""
from checks.native_tests.samba_entry_table_suite import run_checks


def main() raises:
    run_checks()
