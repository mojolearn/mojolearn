# SPDX-License-Identifier: Apache-2.0
from autograd.checks.fold_cpu import run_checks
from checks.native_tests.runner import Check, Column, Report


# Registered by lane fx-checks-orphans (2026-11-18): `autograd/checks/fold_cpu.mojo` was a
# standalone main() that no pixi task, workflow, development group or tools
# script ran. Its main() is now `run_checks()` there; this is its one entry
# (`pixi run check-native-cpu autograd_fold_cpu`).
# Host column: the suite is host work and does not branch on the target.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("autograd_fold_cpu", cpu=True, gpu=False, serial="", run=run_cases)
