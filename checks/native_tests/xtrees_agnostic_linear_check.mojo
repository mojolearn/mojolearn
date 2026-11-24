# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""One TestSuite check over `xtrees.checks.agnostic_linear_check.run_checks()` (its body lives there, outside
checks/native_tests): KernelExplainer's resident linear device route equals the device member route and the host
column, bit for bit (lane vdf-explain, 2026-11-24)."""
from std.testing import TestSuite
from xtrees.checks.agnostic_linear_check import run_checks


def test_kshap_solve_linear() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
