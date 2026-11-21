# SPDX-License-Identifier: Apache-2.0
"""Standalone wrapper of `gbdt/checks/tree_ctr_check.mojo`."""
from gbdt.checks.tree_ctr_check import run_checks


def main() raises:
    run_checks()
    print("PASS tree CTR host statement: tensor, candidates, groups, ordered Borders, FeatureFreq, grid")
