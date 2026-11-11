# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the parallel and archive entry table suite."""
from checks.native_tests.parallel_archive_entry_suite import run_checks


def main() raises:
    run_checks()
