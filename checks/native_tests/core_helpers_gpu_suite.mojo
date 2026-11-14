# SPDX-License-Identifier: Apache-2.0
"""Row-gather fixtures on the device, in order on one DeviceContext.

The fixtures share one context and its stream; each gather is already one
thread per byte, so ordering the three submissions costs nothing.
"""
from max.gpu.host import DeviceContext
from checks.native_tests.core_helpers_cases import (
    CORE_HELPERS_FIXTURES, GatherFixture, require, gather_fixture, check_gather, OUT_ROWS, SOURCE_ROWS, WIDTH,
)
from core.rows_bytes_device import device_gather_rows_bytes


def gather_device(ctx: DeviceContext, mut f: GatherFixture) raises -> Bool:
    """The device gather; True when it reports an index out of bounds."""
    var ok = device_gather_rows_bytes(
        ctx, Int(f.source.unsafe_ptr()), Int(f.dst.unsafe_ptr()), Int(f.idx.unsafe_ptr()),
        SOURCE_ROWS, OUT_ROWS, WIDTH,
    )
    return not ok


def run_checks() raises:
    var ctx = DeviceContext()
    var failed = 0
    for unit in range(CORE_HELPERS_FIXTURES):
        try:
            var f = gather_fixture(unit)
            var bad = gather_device(ctx, f)
            check_gather(unit, f, bad)
        except error:
            failed += 1
            print("FAIL core_helpers_gpu fixture", unit, String(error))
    require(failed == 0, "core_helpers_gpu failed " + String(failed) + " fixtures")
    print("PASS core_helpers_gpu:", CORE_HELPERS_FIXTURES, "row-gather fixtures, ordered on one device")
