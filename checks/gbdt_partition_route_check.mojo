# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Host-only routing gate; this does not validate NVIDIA kernel execution.

Run default, IDENTICAL opt-in, and opt-in plus kill-switch builds. The
large-input device permutation test is still required before enabling the
IDENTICAL candidate by default.
"""
from std.sys.compile import is_defined
from checks.kernel_matrix import (
    COLUMN_APPLE, COLUMN_NVIDIA, COLUMN_AMD, COLUMN_AMD_RDNA,
    reorder_single_pass_for,
)


def run_checks() raises:
    comptime disabled = is_defined["MOJOLEARN_2042_FAST_NO_LOOKBACK"]()
    comptime opted_in = is_defined["MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION"]()
    if reorder_single_pass_for[COLUMN_NVIDIA, False]() != (not disabled):
        raise Error("NVIDIA FAST route")
    if reorder_single_pass_for[COLUMN_NVIDIA, True]() != (opted_in and not disabled):
        raise Error("NVIDIA IDENTICAL opt-in / kill-switch precedence")
    if (reorder_single_pass_for[COLUMN_APPLE, True]()
        or reorder_single_pass_for[COLUMN_APPLE, False]()
        or reorder_single_pass_for[COLUMN_AMD, True]()
        or reorder_single_pass_for[COLUMN_AMD, False]()
        or reorder_single_pass_for[COLUMN_AMD_RDNA, True]()
        or reorder_single_pass_for[COLUMN_AMD_RDNA, False]()):
        raise Error("unsupported vendor routed to lookback")
    print("PARTITION ROUTES GREEN; opt_in", opted_in, "disabled", disabled)


# Registered by lane checks-toplevel (2026-11-16): a standalone main() under
# checks/ that no CI job ran. One process per check name in CI keeps any
# fresh-process precondition. Host column: the suite is host work and does not branch on the target.
from checks.native_tests.runner import Check, Column, Report


def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_partition_route_check", cpu=True, gpu=False, serial="", run=run_cases)
