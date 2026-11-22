# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
from std.testing import TestSuite
from ensemble.checks.sibling_column_overlap_check import run_checks


def test_sibling_column_overlap_check() raises:
    """`ensemble/checks/sibling_column_overlap_check.mojo`, whole."""
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
