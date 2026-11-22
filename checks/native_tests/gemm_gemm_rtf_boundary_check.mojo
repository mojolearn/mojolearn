# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`gemm/checks/gemm_rtf_boundary_check.mojo` as one TestSuite test: its `run_checks()` runs every fixture of that check."""
from std.testing import TestSuite
from gemm.checks.gemm_rtf_boundary_check import run_checks


def test_gemm_rtf_boundary() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
