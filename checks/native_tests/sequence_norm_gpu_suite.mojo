# SPDX-License-Identifier: Apache-2.0
"""LayerNorm forward/backward against the float64 F.layer_norm restatement, on the device column. The fixtures run in order on one
DeviceExec: they share the one device and its queue, and each fixture's
kernels already spread its work across the device."""
from sequence.exec_device import DeviceExec
from checks.native_tests.sequence_port_common import require
from checks.native_tests.sequence_norm_cases import NORM_FIXTURES, run_norm_fixture


def run_checks() raises:
    var ex = DeviceExec()
    var failed = 0
    for unit in range(NORM_FIXTURES):  # driver-loop: over fixtures on the one device
        try:
            run_norm_fixture(ex, unit)
        except error:
            failed += 1
            print("FAIL sequence_norm_gpu fixture", unit, String(error))
    require(failed == 0, "sequence_norm_gpu failed " + String(failed) + " fixtures")
    print("PASS sequence_norm_gpu:", NORM_FIXTURES, "fixtures")
