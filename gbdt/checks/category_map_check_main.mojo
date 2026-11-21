# SPDX-License-Identifier: Apache-2.0
"""Standalone wrapper of `gbdt/checks/category_map_check.mojo`."""
from gbdt.checks.category_map_check import run_checks


def main() raises:
    run_checks()
    print("PASS category map: keys, distinct set, dense codes, unseen code, stored map")
