# SPDX-License-Identifier: Apache-2.0
"""Ordered numerical submissions while upstream shared-context concurrency is parked."""
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentAPI, OcResidentDriver
from x_neighbors.native_descriptor_device import xn_native_v1
from x_neighbors.ocsvm_resident_native_device import OcDeviceResident
from checks.native_tests.ocsvm_native_cases import fixture, require, OCSVM_NATIVE_CASES


def run_checks() raises:
    var resident = OcResidentDriver(OcResidentAPI[OcDeviceResident].entry)
    var enabled = (resident.metadata(1) & 2) != 0
    var count = OCSVM_NATIVE_CASES + (10 if enabled else 0)
    var failed = 0
    for index in range(count):  # ordered shared-context numerical fixtures
        var use_resident = index >= OCSVM_NATIVE_CASES
        var unit = index
        if use_resident:
            unit -= OCSVM_NATIVE_CASES
            if unit >= 4:
                unit += 1  # precomputed keeps its original Gram route
        try:
            fixture(XnDriver(xn_native_v1), resident, unit, use_resident)
        except problem:
            failed += 1
            print("FAIL ocsvm_native_gpu fixture", index, String(problem))
    if not enabled:
        print("SKIP ocsvm_native_gpu resident: 10 cases; existing OCSVM_IDN_RES disabled")
    require(failed == 0, "OneClassSVM GPU failed " + String(failed) + " cases")
    print("PASS ocsvm_native_gpu:", count)
