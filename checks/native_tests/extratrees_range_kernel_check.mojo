# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""One TestSuite check over `extratrees.checks.range_kernel_check.run_checks()` (its body lives there, outside checks/native_tests)."""
from std.testing import TestSuite
from extratrees.checks.range_kernel_check import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `extratrees/checks/range_kernel_check.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
# (its TestSuite file).
def test_range_kernel_check() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
