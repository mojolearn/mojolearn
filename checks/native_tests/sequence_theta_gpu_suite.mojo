# SPDX-License-Identifier: Apache-2.0
"""AutoTheta batch-equals-single and seasonal decomposition, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_theta_cases import THETA_FIXTURES, run_theta_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(THETA_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_theta_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_theta_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_theta_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_theta_gpu:", THETA_FIXTURES, "fixtures")
