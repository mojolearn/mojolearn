# SPDX-License-Identifier: Apache-2.0
"""Ordered actual OVR numerical cases; shared-context concurrency stays parked."""
from core.linear_svc_native_api import LinearSvcDriver
from glm.linear_svc_native_device import linear_svc_device_v1
from checks.native_tests.linear_svc_native_cases import fixture, require, LINEAR_SVC_PROVIDER_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(LINEAR_SVC_PROVIDER_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LinearSvcDriver(linear_svc_device_v1), unit)
        except problem:
            failed += 1
            print("FAIL linear_svc_native_gpu fixture", unit, String(problem))
    require(failed == 0, "LinearSVC GPU failed " + String(failed) + " cases")
    print("PASS linear_svc_native_gpu:", LINEAR_SVC_PROVIDER_CASES)
