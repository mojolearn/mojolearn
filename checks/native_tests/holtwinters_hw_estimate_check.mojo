# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`holtwinters/checks/hw_estimate_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from holtwinters.checks.hw_estimate_check import run_checks


def test_hw_estimate() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
