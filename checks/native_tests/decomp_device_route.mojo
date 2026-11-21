# SPDX-License-Identifier: Apache-2.0
"""The decomposition estimators' device route rules (core/device_route_dw_decomp.mojo)."""
from checks.native_tests.decomp_device_route_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: scalar rule checks with no numerical work (one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("decomp_device_route", cpu=True, gpu=False, serial="", run=run_cases)
