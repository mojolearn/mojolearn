# SPDX-License-Identifier: Apache-2.0
"""Independent typed host DBSCAN profiles run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.dbscan_native_api import DBSCANNativeDriver, dbscan_native_v1
from core.dbscan_native_host import DBSCANHostKernels
from checks.native_tests.dbscan_native_cases import DBSCAN_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=DBSCAN_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(DBSCANNativeDriver(dbscan_native_v1[DBSCANHostKernels]), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL dbscan_native_cpu fixture", unit, String(error))
    host_parallelize(task, DBSCAN_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "dbscan_native_cpu failed " + String(failed) + " fixtures")
    print("PASS dbscan_native_cpu:", DBSCAN_NATIVE_FIXTURES, "native fit/core/query fixtures")
