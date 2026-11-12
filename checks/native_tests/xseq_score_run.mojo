# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the xseq score RUN suite."""
from checks.native_tests.xseq_score_run_suite import run_checks


def main() raises:
    run_checks()
