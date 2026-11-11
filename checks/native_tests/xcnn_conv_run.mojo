# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the Conv2d / Conv1d / BasicBlock RUN suite."""
from checks.native_tests.xcnn_conv_run_suite import run_checks


def main() raises:
    run_checks()
