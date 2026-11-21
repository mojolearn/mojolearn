# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.parallel_byte_session_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the typed session's state and lifetime contracts over a
# stand-in schedule table (its fixtures run across host cores inside
# run_checks; the check counts as one case). No device is touched.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("parallel_byte_session", cpu=True, gpu=False, serial="", run=run_cases)
