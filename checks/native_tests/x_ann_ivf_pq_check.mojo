# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`x_ann/checks/ivf_pq_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from x_ann.checks.ivf_pq_check import run_checks


def test_ivf_pq() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
