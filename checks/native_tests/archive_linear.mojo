# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the importable native linear archive checks."""
from checks.native_tests.archive_linear_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# The one runner (runner.mojo): the fixtures run across host cores inside
# run_checks, which raises after printing each failed fixture; the check counts
# as one case. Host column only: archive reading and writing is host work.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("archive_linear", cpu=True, gpu=False, serial="", run=run_cases)
