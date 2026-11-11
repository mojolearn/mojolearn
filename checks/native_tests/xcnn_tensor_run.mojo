# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the CNN DeviceTensor RUN suite."""
from checks.native_tests.xcnn_tensor_run_suite import run_checks


def main() raises:
    run_checks()
