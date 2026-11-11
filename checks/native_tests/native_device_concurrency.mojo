# SPDX-License-Identifier: Apache-2.0
"""Fresh-process entry; do not run any device suite before these first users."""
from checks.native_tests.native_device_concurrency_suite import run_checks


def main() raises:
    run_checks()
