# SPDX-License-Identifier: Apache-2.0
"""The device ragged passes, in order on one DeviceContext: each pass is one
thread per destination byte, and the four fixtures share the context's
stream, so ordering them costs nothing."""
from max.gpu.host import DeviceContext
from checks.native_tests.core_ragged_lengths_cases import (
    RAGGED_FIXTURES, require, ragged_fixture, check_ragged,
)
from core.input_device import device_ragged_rows


def run_checks() raises:
    var ctx = DeviceContext()
    var failed = 0
    for unit in range(RAGGED_FIXTURES):
        try:
            var f = ragged_fixture(unit)
            var src = Int(f.src.unsafe_ptr()) if f.mode != 1 else Int(f.dst.unsafe_ptr())
            var wrote = device_ragged_rows(ctx, src, Int(f.dst.unsafe_ptr()), Int(f.lens.unsafe_ptr()),
                                           f.n, f.row, f.pos, f.mode)
            require(wrote, "the device pass declined fixture " + String(unit))
            check_ragged(unit, f)
        except error:
            failed += 1
            print("FAIL core_ragged_lengths_gpu fixture", unit, String(error))
    require(failed == 0, "core_ragged_lengths_gpu failed " + String(failed) + " of " + String(RAGGED_FIXTURES))
    print("PASS core_ragged_lengths_gpu:", RAGGED_FIXTURES, "ragged byte-pass fixtures, ordered on one device")
