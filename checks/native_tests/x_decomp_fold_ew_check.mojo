# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`x_decomp/checks/fold_ew_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from x_decomp.checks.fold_ew_check import run_checks


def test_fold_ew() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
