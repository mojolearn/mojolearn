# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`holtwinters/checks/hw_predict_ptr_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from holtwinters.checks.hw_predict_ptr_check import run_checks


def test_hw_predict_ptr() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
