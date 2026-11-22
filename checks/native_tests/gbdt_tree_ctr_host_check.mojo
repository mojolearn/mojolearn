# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The one native check of `gbdt/checks/tree_ctr_host_check.mojo`: its `run_checks()` as one test.

Device column only (`# native-check: gpu`): the suite needs a GPU, so the CPU column never builds this file.
"""
from std.testing import TestSuite
from gbdt.checks.tree_ctr_host_check import run_checks


# lane cat-ctr (2026-11-21): the one entry of `gbdt/checks/tree_ctr_host_check.mojo`
# against the device fit, model text byte for byte.
def test_run_checks() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
