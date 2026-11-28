# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`umap/checks/neighbors_graph_check.mojo`: the device graphs equal the CPU column word for word (lane sc-leftovers-1)."""
from std.testing import TestSuite
from umap.checks.neighbors_graph_check import run_checks


def test_neighbors_graph_device() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
