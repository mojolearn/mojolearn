# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the sequence glue entry suite."""
from checks.native_tests.xseq_entry_glue_suite import run_checks


def main() raises:
    run_checks()
