# SPDX-License-Identifier: Apache-2.0
"""Independent host PCS numerical graphs run in parallel."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from core.pcs_native_api import PCSNativeDriver
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.pcs_native_cases import PCS_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=PCS_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(PCSNativeDriver(XnDriver(xn_native_v1)), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL pcs_native_cpu fixture", unit, String(error))
    host_parallelize(task, PCS_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pcs_native_cpu failed " + String(failed) + " fixtures")
    print("PASS pcs_native_cpu:", PCS_NATIVE_FIXTURES, "native draw/sketch numerical fixtures")
