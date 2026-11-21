# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.cross_vendor_loopback_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: one gathered and one chained step of the typed cross-vendor
# coordinator and workers over local socket pairs, and the refusal words of
# malformed frames (the fixtures run across host cores inside run_checks; the
# check counts as one case). No device is touched.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("cross_vendor_loopback", cpu=True, gpu=False, serial="", run=run_cases)
