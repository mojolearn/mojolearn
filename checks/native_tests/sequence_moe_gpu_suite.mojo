# SPDX-License-Identifier: Apache-2.0
"""MoEBlock forward against the float64 Mixtral block and the routing tie rule, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_moe_cases import MOE_FIXTURES, run_moe_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(MOE_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_moe_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_moe_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_moe_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_moe_gpu:", MOE_FIXTURES, "fixtures")
