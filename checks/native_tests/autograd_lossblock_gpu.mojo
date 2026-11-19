# SPDX-License-Identifier: Apache-2.0
from autograd.checks.lossblock_gpu import run_checks
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


# Registered by lane fx-checks-orphans (2026-11-18): `autograd/checks/lossblock_gpu.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
# (`pixi run check-native-gpu autograd_lossblock_gpu`).
# Device column only: the suite needs a GPU, so the CPU build never instantiates it.
def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("autograd_lossblock_gpu", cpu=False, gpu=True, serial="", run=run_cases)
