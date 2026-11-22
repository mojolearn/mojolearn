# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Registered by lane fx-checks-orphans (2026-11-18): `arima/checks/fit_check.mojo` was a standalone main() that no
pixi task, workflow, development group or tools script ran. Its main() is now `run_checks()` there; this TestSuite
file is its one entry.
"""
from std.testing import TestSuite
from arima.checks.fit_check import run_checks


def test_fit_check() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
