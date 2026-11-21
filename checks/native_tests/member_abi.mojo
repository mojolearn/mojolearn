# SPDX-License-Identifier: Apache-2.0
from checks.native_tests.member_abi_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the member ABI's call envelope and the linear member state
# store's lifetime are host metadata work (five fixtures across host cores
# inside run_checks; the check counts as one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("member_abi", cpu=True, gpu=False, serial="", run=run_cases)
