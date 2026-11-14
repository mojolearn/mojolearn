# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the CNN blocks suite (host column)."""
from checks.native_tests.cnn_blocks_cpu_suite import run_checks


def main() raises:
    run_checks()
