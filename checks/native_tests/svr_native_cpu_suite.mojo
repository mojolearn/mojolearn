# SPDX-License-Identifier: Apache-2.0
from core.svr_native_api import SvrNativeDriver
from svm.host.svr_native_host import svr_host_native_v1
from core.host_parallel import host_parallelize
from checks.native_tests.svr_native_cases import fixture, require, SVR_NATIVE_CASES


def run_checks() raises:
    var failures = List[Int](length=SVR_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(SvrNativeDriver(svr_host_native_v1), unit)
        except problem:
            flags[unit] = 1
            print("FAIL svr_native_cpu fixture", unit, String(problem))
    host_parallelize(run, SVR_NATIVE_CASES)
    var failed = 0
    for flag in failures:  # independent fixture statuses
        failed += flag
    require(failed == 0, "SVR CPU failed " + String(failed) + " cases")
    print("PASS svr_native_cpu:", SVR_NATIVE_CASES)
