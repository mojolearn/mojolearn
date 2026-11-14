# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the CNN trainer Linear suite (host column)."""
from checks.native_tests.cnn_trainer_cpu_suite import run_checks


def main() raises:
    run_checks()
