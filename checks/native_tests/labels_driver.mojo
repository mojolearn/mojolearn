# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the importable native label driver suite."""
from checks.native_tests.labels_driver_suite import run_checks


def main() raises:
    run_checks()
