# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`umap/checks/neighbors_graph_check.mojo`'s CPU-column checks as one TestSuite test (lane sc-leftovers-1)."""
from std.testing import TestSuite
from umap.checks.neighbors_graph_check import run_host_checks


def test_neighbors_graph_host() raises:
    run_host_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
