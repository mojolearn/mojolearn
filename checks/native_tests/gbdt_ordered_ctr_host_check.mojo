# SPDX-License-Identifier: Apache-2.0
from gbdt.checks.ordered_ctr_host_check import run_checks
from checks.native_tests.runner import Check, Column, Report, CPU_COLUMN_BUILD


# lane cat-ctr (2026-11-21): the one entry of `gbdt/checks/ordered_ctr_host_check.mojo`
# (`pixi run check-native-gpu gbdt_ordered_ctr_host_check`): the host Ordered fit with
# CTR categorical columns against the device fit, model text byte for byte.
# Device column only: the suite needs a GPU for the device arm, so the CPU build never instantiates it.
def run_cases(column: Column, mut report: Report):
    try:
        comptime if not CPU_COLUMN_BUILD:
            run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("gbdt_ordered_ctr_host_check", cpu=False, gpu=True, serial="", run=run_cases)
