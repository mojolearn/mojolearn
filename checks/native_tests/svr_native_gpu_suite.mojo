# SPDX-License-Identifier: Apache-2.0
"""Ordered numerical cases; shared-DeviceContext concurrent submission stays parked."""
from core.svr_native_api import SvrNativeDriver
from svm.svr_native_device import svr_device_native_v1
from checks.native_tests.svr_native_cases import fixture, require, SVR_NATIVE_CASES, SVR_RESIDENT_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(SVR_NATIVE_CASES + SVR_RESIDENT_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(SvrNativeDriver(svr_device_native_v1), unit % SVR_NATIVE_CASES, unit >= SVR_NATIVE_CASES)
        except problem:
            failed += 1
            print("FAIL svr_native_gpu fixture", unit, String(problem))
    require(failed == 0, "SVR GPU failed " + String(failed) + " cases")
    print("PASS svr_native_gpu:", SVR_NATIVE_CASES + SVR_RESIDENT_CASES)
