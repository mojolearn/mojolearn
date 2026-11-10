# SPDX-License-Identifier: Apache-2.0
"""Standalone entry point for the reusable native test suite."""
from checks.native_tests.theta_state_gpu_suite import run_checks


def main() raises:
    run_checks()
