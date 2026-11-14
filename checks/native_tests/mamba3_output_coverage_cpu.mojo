# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the Mamba-3 output-coverage suite (CPU column)."""
from checks.native_tests.mamba3_output_coverage_cpu_suite import run_checks


def main() raises:
    run_checks()
