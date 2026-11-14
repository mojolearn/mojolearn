# SPDX-License-Identifier: Apache-2.0
"""GARCH(1, 1) recovery of the stored simulated process, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_garch_cases import GARCH_FIXTURES, run_garch_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(GARCH_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_garch_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_garch_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_garch_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_garch_gpu:", GARCH_FIXTURES, "fixtures")
