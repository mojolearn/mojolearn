# SPDX-License-Identifier: Apache-2.0
"""Recurrent models against the float64 PyTorch cell equations, and their loss going down, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_rnn_cases import RNN_FIXTURES, run_rnn_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(RNN_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_rnn_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_rnn_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_rnn_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_rnn_gpu:", RNN_FIXTURES, "fixtures")
