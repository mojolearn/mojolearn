# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`checks/native_tests/fit_loop_neighbors_suite.mojo` as one TestSuite test:
LabelPropagation / LabelSpreading's loop ops report the capped n_iter_ and a
non-negative loop time (`fit_loop_`) on the device and the host column."""
from std.testing import TestSuite
from checks.native_tests.fit_loop_neighbors_suite import run_checks


def test_fit_loop_neighbors() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
