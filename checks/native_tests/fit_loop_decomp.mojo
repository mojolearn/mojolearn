# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""`checks/native_tests/fit_loop_decomp_suite.mojo` as one TestSuite test: the
fit loop clock (`fit_loop_`) of NMF, FastICA, FactorAnalysis and
AlternatingLeastSquares reports exactly the capped iteration count and a
non-negative time, on the host column and the device."""
from std.testing import TestSuite
from checks.native_tests.fit_loop_decomp_suite import run_checks


def test_fit_loop_decomp() raises:
    run_checks()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
