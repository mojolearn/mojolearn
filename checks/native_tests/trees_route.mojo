# SPDX-License-Identifier: Apache-2.0
"""The tree estimators' device route plan (core/trees_route.mojo), no Python."""
from checks.native_tests.trees_route_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite is a decision table over words (one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("trees_route", cpu=True, gpu=False, serial="", run=run_cases)
