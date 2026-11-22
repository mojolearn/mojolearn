# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The one native check of `gbdt/checks/host_binarize_identical.mojo`: its `run_checks()` as one test.

Device column only (`# native-check: gpu`): the suite needs a GPU, so the CPU column never builds this file.
"""
from std.testing import TestSuite
from gbdt.checks.host_binarize_identical import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `gbdt/checks/host_binarize_identical.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
def test_run_checks() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
