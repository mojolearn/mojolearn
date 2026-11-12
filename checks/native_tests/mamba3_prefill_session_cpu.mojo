# SPDX-License-Identifier: Apache-2.0
"""Standalone fresh-process run of the Mamba-3 prefill session suite (CPU column)."""
from checks.native_tests.mamba3_prefill_session_cpu_suite import run_checks


def main() raises:
    run_checks()
