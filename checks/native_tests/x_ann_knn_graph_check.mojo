# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`x_ann/checks/knn_graph_check.mojo`: the device graph equals the CPU column word for word (lane sc-ann-1)."""
from std.testing import TestSuite
from x_ann.checks.knn_graph_check import run_checks


def test_knn_graph_device() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
