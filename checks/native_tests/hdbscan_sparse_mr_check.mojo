# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
from std.testing import TestSuite
from hdbscan.checks.sparse_mr_check import run_checks


def test_sparse_mr_check() raises:
    """`hdbscan/checks/sparse_mr_check.mojo`, whole."""
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
