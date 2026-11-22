# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Registered by lane fx-checks-orphans (2026-11-18): `cluster/checks/host_gemm_check.mojo` was a standalone main()
that no pixi task, workflow, development group or tools script ran. Its main() is now `run_checks()` there; this
TestSuite file is its one entry.
"""
from std.testing import TestSuite
from cluster.checks.host_gemm_check import run_checks


def test_host_gemm() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
