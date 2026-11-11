# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the pinned standard-normal helper suite."""
from checks.native_tests.portable_math_suite import run_checks


def main() raises:
    run_checks()
