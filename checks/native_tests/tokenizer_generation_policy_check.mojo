# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Registered by lane fx-checks-orphans (2026-11-18): `tokenizer/checks/generation_policy_check.mojo` was a standalone
main() that no pixi task, workflow, development group or tools script ran. Its main() is now `run_checks()` there;
this TestSuite file is its one entry.
"""
from std.testing import TestSuite
from tokenizer.checks.generation_policy_check import run_checks


def test_generation_policy() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
