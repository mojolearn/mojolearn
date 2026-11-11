# SPDX-License-Identifier: Apache-2.0
from core.linear_svc_native_api import LinearSvcDriver
from glm.linear_svc_native_device import linear_svc_device_v1
from checks.native_tests.linear_svr_native_cases import fixture, require, LINEAR_SVR_NATIVE_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(LINEAR_SVR_NATIVE_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LinearSvcDriver(linear_svc_device_v1), unit)
        except problem:
            failed += 1
            print("FAIL linear_svr_native_gpu fixture", unit, String(problem))
    require(failed == 0, "LinearSVR GPU failed " + String(failed) + " cases")
    print("PASS linear_svr_native_gpu:", LINEAR_SVR_NATIVE_CASES)
