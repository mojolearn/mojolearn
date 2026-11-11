# SPDX-License-Identifier: Apache-2.0
"""Independent host NMF graphs execute through the host worker pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.nmf_native_api import NMFNativeDriver
from core.nmf_native_provider import nmf_native_v1
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.decomp_owner_host import HostDecomp
from checks.native_tests.nmf_native_cases import NMF_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=NMF_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(NMFNativeDriver(nmf_native_v1[HostExec, NoRDev, False, HostDecomp]), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL nmf_native_cpu fixture", unit, String(error))
    host_parallelize(task, NMF_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "nmf_native_cpu failed " + String(failed) + " fixtures")
    print("PASS nmf_native_cpu:", NMF_NATIVE_FIXTURES, "native fit/inverse/refusal fixtures")
