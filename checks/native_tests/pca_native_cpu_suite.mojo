# SPDX-License-Identifier: Apache-2.0
"""Actual CPU PCA request/state fixtures, with independent graphs in parallel."""
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.pca_native_api import PCANativeDriver
from core.pca_native_host import pca_host_native_v1
from checks.native_tests.pca_native_cases import PCA_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=PCA_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(PCANativeDriver(pca_host_native_v1), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL pca_native_cpu fixture", unit, String(error))
    host_parallelize(task, PCA_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "pca_native_cpu failed " + String(failed) + " fixtures")
    print("PASS pca_native_cpu:", PCA_NATIVE_FIXTURES, "native fit/query/state fixtures")
