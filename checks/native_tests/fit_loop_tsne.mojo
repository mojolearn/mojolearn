# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""`checks/native_tests/fit_loop_tsne_suite.mojo` as one TestSuite test: TSNE's
gradient loop reports its time (`fit_loop_`) on the device and the host
column."""
from std.testing import TestSuite
from checks.native_tests.fit_loop_tsne_suite import run_checks


def test_fit_loop_tsne() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
