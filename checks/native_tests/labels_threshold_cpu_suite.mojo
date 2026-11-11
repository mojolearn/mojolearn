# SPDX-License-Identifier: Apache-2.0
from core.host_parallel import host_parallelize
from core.labels_native_api import LabelsNativeDriver
from core.labels_native_host_provider import labels_host_native_v2
from checks.native_tests.labels_threshold_cases import fixture, LABELS_THRESHOLD_CASES


def run_checks() raises:
    var failures = List[Int](length=LABELS_THRESHOLD_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(LabelsNativeDriver(labels_host_native_v2), unit)
        except problem:
            flags[unit] = 1
            print("FAIL labels_threshold_cpu fixture", unit, String(problem))
    host_parallelize(run, LABELS_THRESHOLD_CASES)
    var failed = 0
    for flag in failures:  # metadata: independent fixture statuses
        failed += flag
    if failed:
        raise Error("labels threshold CPU failed " + String(failed) + " cases")
    print("PASS labels_threshold_cpu:", LABELS_THRESHOLD_CASES)
