# SPDX-License-Identifier: Apache-2.0
"""The registered check of the search score tables' typed RUN (bindings/msel_search_stats_run.mojo)."""
from checks.native_tests.msel_search_stats_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite is host work (its cases run across host cores inside
# run_checks; the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("msel_search_stats", cpu=True, gpu=False, serial="", run=run_cases)
