# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The one native check of `transformer/checks/attention_zdot_routing_check.mojo`: its `run_checks()` as one test.

Device column only (`# native-check: gpu`): the suite needs a GPU, so the CPU column never builds this file.
"""
from std.testing import TestSuite
from transformer.checks.attention_zdot_routing_check import run_checks


# Registered by lane fx-checks-orphans (2026-11-18): `transformer/checks/attention_zdot_routing_check.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
def test_run_checks() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
