# SPDX-License-Identifier: Apache-2.0
"""The registered check of the splitters' typed RUN (bindings/msel_split_plan_run.mojo)."""
from checks.native_tests.msel_split_plan_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite is host work (its cases run across host cores inside
# run_checks; the check counts as one case). It publishes the host helper
# table itself, so it runs in its own process like every registered check.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("msel_split_plan", cpu=True, gpu=False, serial="", run=run_cases)
