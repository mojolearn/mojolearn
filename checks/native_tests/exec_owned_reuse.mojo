# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.exec_owned_reuse_suite import run_checks
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


# Registered by lane checks-register (2026-11-16): it was a standalone main()
# that no CI job ran. One process per check name in CI keeps any fresh-process
# precondition.
# Device column only: the suite asserts a non-CPU target, so the
# CPU build never instantiates it.
def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("exec_owned_reuse", cpu=False, gpu=True, serial="", run=run_cases)
