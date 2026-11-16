# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Entry point for the BLOCK-LEVEL batch-invariance gate.

A driver of its own, at the repository root beside `checks/probe_main.mojo` and
`checks/oracle_main.mojo`, for the same reason `neighbors/warpsort_probe_main.mojo`
is separate from `neighbors/knn_main.mojo`: the claim spans TWO lanes
(`mamba/` and `transformer/`) and one layer below them (`gemm/`), so it
belongs to none of their gate files and putting it in either would make a
cross-lane property read as that lane's clause.

The check itself is `checks/batch_invariance_check.mojo`. Read its docstring
first: it states which TIER the claim is in and why, what "the same tokens
at a different batch size" has to mean for a recurrence, and what the
sabotage and the three vacuity controls are.

    tools/with_identical_mode.sh pixi run check-native-gpu batch_invariance_main

Under FAST this driver still runs and still prints every
number; it REPORTS instead of asserting, and says so on its banner
([[fast-is-not-identical]]).
"""

from checks.batch_invariance_check import check_block_batch_invariance


def run_checks() raises:
    check_block_batch_invariance()


# Registered by lane checks-toplevel (2026-11-16): a standalone main() under
# checks/ that no CI job ran. One process per check name in CI keeps any
# fresh-process precondition. Device column only: the suite needs a GPU, so the CPU build never instantiates it.
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("batch_invariance_main", cpu=False, gpu=True, serial="", run=run_cases)
