# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The device-column entry of `hierarchy/checks/linkage_host_check.mojo`: its `run_checks()` as one test (the suite needs a
GPU, so a host-column build never instantiates it)."""
from std.sys.compile import is_defined
from std.testing import TestSuite
from hierarchy.checks.linkage_host_check import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `hierarchy/checks/linkage_host_check.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
# (the TestSuite runner).
def test_linkage_host() raises:
    comptime if not is_defined["MOJOLEARN_COLUMN_CPU"]():
        run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
