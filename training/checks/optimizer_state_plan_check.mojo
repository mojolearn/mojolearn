# SPDX-License-Identifier: Apache-2.0
"""Standalone entry point for the reusable native test suite."""
from training.checks.optimizer_state_plan_check_suite import run_checks


def main() raises:
    run_checks()
