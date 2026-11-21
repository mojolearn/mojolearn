# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.cross_vendor_run_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the typed cross-vendor run's state contracts (its fixtures run
# across host cores inside run_checks; the check counts as one case). No
# socket, session or device is touched.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("cross_vendor_run", cpu=True, gpu=False, serial="", run=run_cases)
