# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
from std.testing import TestSuite
from autograd.checks.pass1_cpu import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `autograd/checks/pass1_cpu.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this file is its one
# TestSuite entry.
# Host column: the suite is host work and does not branch on the target.
def test_pass1_cpu() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
