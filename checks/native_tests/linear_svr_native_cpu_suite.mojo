# SPDX-License-Identifier: Apache-2.0
from core.linear_svc_native_api import LinearSvcDriver
from glm.host.linear_svc_native_host import linear_svc_host_v1
from checks.native_tests.linear_svr_native_cases import fixture, require, LINEAR_SVR_NATIVE_CASES
from core.host_parallel import host_parallelize


def run_checks() raises:
    var failures = List[Int](length=LINEAR_SVR_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(LinearSvcDriver(linear_svc_host_v1), unit)
        except problem:
            flags[unit] = 1
            print("FAIL linear_svr_native_cpu fixture", unit, String(problem))
    host_parallelize(run, LINEAR_SVR_NATIVE_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    require(failed == 0, "LinearSVR CPU failed " + String(failed) + " cases")
    print("PASS linear_svr_native_cpu:", LINEAR_SVR_NATIVE_CASES)
