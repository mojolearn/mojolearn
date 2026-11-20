# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.kde_contract_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Registered by lane checks-register (2026-11-16): it was a standalone main()
# that no CI job ran. One process per check name in CI keeps any fresh-process
# precondition.
# Host column: the suite is host work (its fixtures run across host
# cores inside run_checks; the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("kde_contract", cpu=True, gpu=False, serial="", run=run_cases)
