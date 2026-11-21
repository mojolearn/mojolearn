# SPDX-License-Identifier: Apache-2.0
"""Standalone wrapper of `gbdt/checks/learn_order_check.mojo`."""
from gbdt.checks.learn_order_check import run_checks


def main() raises:
    run_checks()
    print("PASS learn orders: bijections, inverse walk, whole blocks, seeded, Ordered plan")
