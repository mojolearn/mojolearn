# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.transport_frame_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the typed transport frame and the typed byte-LM archive write
# are host work (the four fixtures run across host cores inside run_checks;
# the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("transport_frame", cpu=True, gpu=False, serial="", run=run_cases)
