# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The SGD family's `fit_loop_` record on both columns (lane alm-left,
2026-11-24): the four capped fits of x_linear/checks/sgd_fit_loop_suite.mojo
(regression and one-vs-rest, per-sample and minibatch; max_iter K = 3, tol
None) each report K iterations == n_iter and a loop time >= 0. The host
column runs the four as independent units across the host cores; the device
column runs them in order (they share the linear device context and its
queue), through `xl_fit_device_run`, the GPU binding's fit body."""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD
from x_linear.checks.sgd_fit_loop_suite import SGD_FIT_LOOP_CASES, sgd_fit_loop_unit


def _host_unit(unit: Int) raises:
    sgd_fit_loop_unit(unit, False)


def _device_unit(unit: Int) raises:
    sgd_fit_loop_unit(unit, True)


def test_sgd_fit_loop_host() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("sgd_fit_loop host", SGD_FIT_LOOP_CASES, _host_unit)


def test_sgd_fit_loop_device() raises:
    comptime if not CPU_COLUMN_BUILD:
        check_cases_serial("sgd_fit_loop device", SGD_FIT_LOOP_CASES, _device_unit)


def main() raises:
    # Each column registers its own test by hand (as checks/native_tests/x_linear_ops.mojo): discovery would
    # elaborate the device fit in the host column's build.
    var suite = TestSuite()
    comptime if CPU_COLUMN_BUILD:
        suite.test[test_sgd_fit_loop_host]()
    else:
        suite.test[test_sgd_fit_loop_device]()
    suite^.run()
