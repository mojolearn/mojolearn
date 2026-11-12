# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the forecaster RUN suite."""
from checks.native_tests.xseq_forecast_run_suite import run_checks


def main() raises:
    run_checks()
