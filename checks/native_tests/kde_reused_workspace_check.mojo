# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
from std.testing import TestSuite
from kde.checks.reused_workspace_check import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `kde/checks/reused_workspace_check.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this file is its one
# TestSuite entry.
# Device column only: the suite needs a GPU, so the file is built on the device column alone.
def test_reused_workspace_check() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
