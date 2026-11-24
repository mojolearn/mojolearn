# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The neural trainers' `fit_loop_` record (lane alm-neural, 2026-11-24):
MLP_FIT, RNN_FIT, the whole-model (Sequential) fit and CNN_FIT with the cap
K and early stopping off report K iterations and a loop time >= 0; MLP_FIT
without the record address writes none. The fixtures are in
`neural_fit_loop_suite.mojo`. The host column: the independent fixtures
across host cores, each on its own HostExec. The device column: in order on
one DeviceExec (they share the one device and its queue); the CNN fixture
on its own executor, which its program takes by value."""
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD
from checks.native_tests.neural_fit_loop_suite import NEURAL_FIT_LOOP_CASES, fixture, cnn_fixture


def test_neural_fit_loop() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_fixture(unit: Int) raises:
            if unit == 3:
                cnn_fixture[HostExec](HostExec())
            else:
                var ex = HostExec()
                fixture(ex, unit)
        check_cases("neural_fit_loop", NEURAL_FIT_LOOP_CASES, host_fixture)
    else:
        var ex = DeviceExec()
        def device_fixture(unit: Int) raises {mut ex}:
            if unit == 3:
                cnn_fixture[DeviceExec](DeviceExec())
            else:
                fixture(ex, unit)
        check_cases_serial("neural_fit_loop", NEURAL_FIT_LOOP_CASES, device_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
