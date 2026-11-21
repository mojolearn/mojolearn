# SPDX-License-Identifier: Apache-2.0
"""Standalone wrapper of `gbdt/checks/combination_map_check.mojo`."""
from gbdt.checks.combination_map_check import run_checks


def main() raises:
    run_checks()
    print("PASS combination map: derived codes, unseen tuples, tree-CTR model columns, text")
