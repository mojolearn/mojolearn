# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
from std.testing import TestSuite
from ensemble.checks.predict_check import run_checks


def test_predict_check() raises:
    """`ensemble/checks/predict_check.mojo`, whole."""
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
