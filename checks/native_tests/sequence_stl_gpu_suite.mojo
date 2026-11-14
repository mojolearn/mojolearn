# SPDX-License-Identifier: Apache-2.0
"""STL decomposition identities, robust weights and batch rows, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_stl_cases import STL_FIXTURES, run_stl_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(STL_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_stl_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_stl_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_stl_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_stl_gpu:", STL_FIXTURES, "fixtures")
