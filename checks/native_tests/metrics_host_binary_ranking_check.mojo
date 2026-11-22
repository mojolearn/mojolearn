# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The one native check of `metrics/checks/host_binary_ranking_check.mojo`: its `run_checks()` as one test.

Host column (`# native-check: cpu`): the suite is host work and does not branch on the target.
"""
from std.testing import TestSuite
from metrics.checks.host_binary_ranking_check import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `metrics/checks/host_binary_ranking_check.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
def test_run_checks() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
