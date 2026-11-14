# SPDX-License-Identifier: Apache-2.0
"""VAR(p) params and sigma_u against float64 least squares, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_var_cases import VAR_FIXTURES, run_var_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(VAR_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_var_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_var_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_var_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_var_gpu:", VAR_FIXTURES, "fixtures")
