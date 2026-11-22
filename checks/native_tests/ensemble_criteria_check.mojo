# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
from std.testing import TestSuite
from ensemble.checks.criteria_check import run_checks


def test_criteria_check() raises:
    """`ensemble/checks/criteria_check.mojo`, whole."""
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
