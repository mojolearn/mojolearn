# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the pooling and channel-dropout RUN suite."""
from checks.native_tests.xcnn_pool_run_suite import run_checks


def main() raises:
    run_checks()
