# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the xseq neural RUN suite."""
from checks.native_tests.xseq_neural_run_suite import run_checks


def main() raises:
    run_checks()
