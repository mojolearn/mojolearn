# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the CNN batch-normalization suite (host column)."""
from checks.native_tests.cnn_batchnorm_cpu_suite import run_checks


def main() raises:
    run_checks()
