# SPDX-License-Identifier: Apache-2.0
"""The registered check of the orchestration families' member device rule suite."""
from checks.native_tests.member_device_rule_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite is word compares with no device work (it counts as
# one case; see the suite's docstring for why it is not split).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("member_device_rule", cpu=True, gpu=False, serial="", run=run_cases)
