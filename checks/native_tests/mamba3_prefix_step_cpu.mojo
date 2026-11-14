# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the Mamba-3 prefix + step suite (CPU column)."""
from checks.native_tests.mamba3_prefix_step_cpu_suite import run_checks


def main() raises:
    run_checks()
