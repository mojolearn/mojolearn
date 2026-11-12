# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the trees door RUN suite."""
from checks.native_tests.trees_door_run_suite import run_checks


def main() raises:
    run_checks()
