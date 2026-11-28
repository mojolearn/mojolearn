# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`x_ann/checks/knn_graph_check.mojo`'s CPU-column checks as one TestSuite test (lane sc-ann-1)."""
from std.testing import TestSuite
from x_ann.checks.knn_graph_check import run_host_checks


def test_knn_graph_host() raises:
    run_host_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
