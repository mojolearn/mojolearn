# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""`x_cluster/checks/spectral_assign_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from x_cluster.checks.spectral_assign_check import run_checks


def test_spectral_assign() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
