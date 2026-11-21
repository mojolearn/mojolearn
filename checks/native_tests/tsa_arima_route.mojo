# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process proof of the ARIMA operations table and device route."""
from checks.native_tests.tsa_arima_route_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# One process per check name in CI keeps the fresh-process precondition
# (the missing-table assertion runs before the publication).
# Host column: the suite is host work (its fixtures run across host
# cores inside run_checks; the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("tsa_arima_route", cpu=True, gpu=False, serial="", run=run_cases)
