# SPDX-License-Identifier: Apache-2.0
"""Ordered numerical coverage while shared-context GPU concurrency is parked."""
from core.labels_native_api import LabelsNativeDriver
from core.labels_native_device import labels_device_native_v2
from checks.native_tests.labels_threshold_cases import fixture, LABELS_THRESHOLD_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(LABELS_THRESHOLD_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LabelsNativeDriver(labels_device_native_v2), unit)
        except problem:
            failed += 1
            print("FAIL labels_threshold_gpu fixture", unit, String(problem))
    if failed:
        raise Error("labels threshold GPU failed " + String(failed) + " cases")
    print("PASS labels_threshold_gpu:", LABELS_THRESHOLD_CASES)
