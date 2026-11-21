# SPDX-License-Identifier: Apache-2.0
"""The forest door's typed configuration, request and predict state
(bindings/forest_config_run.mojo, bindings/forest_predict_run.mojo), no Python."""
from checks.native_tests.forest_door_values_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite is a decision table over words (one case).
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("forest_door_values", cpu=True, gpu=False, serial="", run=run_cases)
