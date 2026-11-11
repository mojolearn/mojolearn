# SPDX-License-Identifier: Apache-2.0
from core.labels_native_api import LabelsNativeDriver
from core.labels_native_device import labels_device_native_v2
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.svc_accuracy_native_cases import fixture, require, SVC_ACCURACY_NATIVE_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(SVC_ACCURACY_NATIVE_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LabelsNativeDriver(labels_device_native_v2), metrics_program_device_v1, unit)
        except problem:
            failed += 1
            print("FAIL svc_accuracy_native_gpu fixture", unit, String(problem))
    require(failed == 0, "SVM accuracy GPU failed " + String(failed) + " cases")
    print("PASS svc_accuracy_native_gpu:", SVC_ACCURACY_NATIVE_CASES)
