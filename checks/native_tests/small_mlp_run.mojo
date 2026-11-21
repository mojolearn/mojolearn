# SPDX-License-Identifier: Apache-2.0
"""Standalone proof of SmallMLPTrainer's RUN file on the host program entry."""
from checks.native_tests.small_mlp_run_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column: the suite drives the host program image through the typed
# RUN functions (serial by nature: see the suite's docstring); the check
# counts as one case.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("small_mlp_run", cpu=True, gpu=False, serial="", run=run_cases)
