# SPDX-License-Identifier: Apache-2.0
"""The registered check (one fresh process per check) of the prep entry table suite."""
from checks.native_tests.prep_entry_table_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Registered by lane checks-register (2026-11-16): it was a standalone main()
# that no CI job ran. One process per check name in CI keeps any fresh-process
# precondition.
# Both columns: the suite branches on TARGET_COLUMN (host cases in
# parallel, device cases on one context).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("prep_entry_table", cpu=True, gpu=True, serial="", run=run_cases)
