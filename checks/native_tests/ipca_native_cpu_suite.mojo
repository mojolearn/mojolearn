# SPDX-License-Identifier: Apache-2.0
"""Independent actual host IncrementalPCA graphs execute in parallel."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.ipca_native_api import IPCANativeDriver
from core.ipca_native_provider import ipca_native_v1
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.ipca_owner import NoIpcaDev
from checks.native_tests.ipca_native_cases import IPCA_NATIVE_FIXTURES, IPCA_RESIDENT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=IPCA_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(IPCANativeDriver(ipca_native_v1[HostExec, NoRDev, False, NoIpcaDev]), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL ipca_native_cpu fixture", unit, String(error))
    host_parallelize(task, IPCA_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "ipca_native_cpu failed " + String(failed) + " fixtures")
    print("PASS ipca_native_cpu:", IPCA_NATIVE_FIXTURES, "native fit/partial/query/refusal fixtures")
