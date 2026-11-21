# SPDX-License-Identifier: Apache-2.0
"""Standalone entry for the importable native fitted-digest checks."""
from checks.native_tests.fitted_digest_suite import run_checks
from checks.native_tests.runner import Check, Column, Report


# Host column only: the digest is host work over the described fitted fields.
def run_cases(column: Column, mut report: Report):
    try:
        run_checks()
    except error:
        report.abort(error)


def register() -> Check:
    return Check("fitted_digest", cpu=True, gpu=False, serial="each fixture splits its fields and chunks across the host pool itself",
                 run=run_cases)
