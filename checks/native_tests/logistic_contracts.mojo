# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the reusable native Logistic ABI suite."""
from checks.native_tests.logistic_contracts_suite import run_checks


def main() raises:
    run_checks()
