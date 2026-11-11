# SPDX-License-Identifier: Apache-2.0
"""Standalone run of the CNNClassifier RUN suite."""
from checks.native_tests.xcnn_classifier_run_suite import run_checks


def main() raises:
    run_checks()
