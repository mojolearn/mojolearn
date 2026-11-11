# SPDX-License-Identifier: Apache-2.0
"""Ordered GPU numerical coverage; shared-DeviceContext concurrency stays parked."""
from core.hdbscan_fit_api import HdbscanFitDriver
from core.hdbscan_query_api import HdbscanQueryDriver
from hdbscan.native_fit_device import hdbscan_fit_device_v1
from hdbscan.native_query_device import hdbscan_query_device_v1
from checks.native_tests.hdbscan_native_cases import fixture, require, HDBSCAN_NATIVE_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(HDBSCAN_NATIVE_CASES):  # ordered shared-context numerical cases, not concurrency qualification
        try:
            fixture(HdbscanFitDriver(hdbscan_fit_device_v1), HdbscanQueryDriver(hdbscan_query_device_v1), unit)
        except problem:
            failed += 1
            print("FAIL hdbscan_native_gpu fixture", unit, String(problem))
    require(failed == 0, "HDBSCAN GPU failed " + String(failed) + " fixtures")
    print("PASS hdbscan_native_gpu:", HDBSCAN_NATIVE_CASES, "actual provider fixtures")
