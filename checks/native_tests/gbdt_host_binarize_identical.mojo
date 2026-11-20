# SPDX-License-Identifier: Apache-2.0
from gbdt.checks.host_binarize_identical import run_checks
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


# Registered by lane fx-checks-orphans (2026-11-18): `gbdt/checks/host_binarize_identical.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
# (`pixi run check-native-gpu gbdt_host_binarize_identical`).
# Device column only: the suite needs a GPU, so the CPU build never instantiates it.
def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_host_binarize_identical", cpu=False, gpu=True, serial="", run=run_cases)
