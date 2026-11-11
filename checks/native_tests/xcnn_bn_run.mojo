# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the BatchNorm RUN suite."""
from checks.native_tests.xcnn_bn_run_suite import run_checks


def main() raises:
    run_checks()
