# SPDX-License-Identifier: Apache-2.0
"""Actual host TSVD fit graphs run independently through the native host pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.tsvd_native_api import TSVDNativeDriver
from core.tsvd_native_host import tsvd_host_native_v1
from core.tsvd_native_decomp import tsvd_decomp_native_v1
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from checks.native_tests.tsvd_fit_cases import TSVD_FIT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=TSVD_FIT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(TSVDNativeDriver(tsvd_host_native_v1),
                TSVDNativeDriver(tsvd_decomp_native_v1[HostExec, NoRDev, False]), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsvd_fit_cpu fixture", unit, String(error))
    host_parallelize(task, TSVD_FIT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsvd_fit_cpu failed " + String(failed) + " fixtures")
    print("PASS tsvd_fit_cpu:", TSVD_FIT_FIXTURES, "native fit/reference/refusal fixtures")
