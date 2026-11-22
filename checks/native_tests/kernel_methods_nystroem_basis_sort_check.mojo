# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`kernel_methods/checks/nystroem_basis_sort_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from kernel_methods.checks.nystroem_basis_sort_check import run_checks


def test_nystroem_basis_sort() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
