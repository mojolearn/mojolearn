# SPDX-License-Identifier: Apache-2.0
from core.labels_native_api import LabelsNativeDriver
from core.labels_native_host_provider import labels_host_native_v2
from x_metrics.descriptor_host import metrics_program_host_v1
from checks.native_tests.svc_accuracy_native_cases import fixture, require, SVC_ACCURACY_NATIVE_CASES
from core.host_parallel import host_parallelize


def run_checks() raises:
    var failures = List[Int](length=SVC_ACCURACY_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(LabelsNativeDriver(labels_host_native_v2), metrics_program_host_v1, unit)
        except problem:
            flags[unit] = 1
            print("FAIL svc_accuracy_native_cpu fixture", unit, String(problem))
    host_parallelize(run, SVC_ACCURACY_NATIVE_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    require(failed == 0, "SVM accuracy CPU failed " + String(failed) + " cases")
    print("PASS svc_accuracy_native_cpu:", SVC_ACCURACY_NATIVE_CASES)
