# SPDX-License-Identifier: Apache-2.0
"""Croston classic and SBA against the float64 SES restatement, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_croston_cases import CROSTON_FIXTURES, run_croston_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(CROSTON_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_croston_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_croston_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_croston_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_croston_gpu:", CROSTON_FIXTURES, "fixtures")
