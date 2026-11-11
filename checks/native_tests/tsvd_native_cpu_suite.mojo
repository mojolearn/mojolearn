# SPDX-License-Identifier: Apache-2.0
"""Independent typed CPU TSVD query fixtures run in parallel."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.tsvd_native_api import TSVDNativeDriver
from core.tsvd_native_host import tsvd_host_native_v1
from checks.native_tests.tsvd_native_cases import TSVD_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=TSVD_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(TSVDNativeDriver(tsvd_host_native_v1), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsvd_native_cpu fixture", unit, String(error))
    host_parallelize(task, TSVD_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsvd_native_cpu failed " + String(failed) + " fixtures")
    print("PASS tsvd_native_cpu:", TSVD_NATIVE_FIXTURES, "native query/state/refusal fixtures")
