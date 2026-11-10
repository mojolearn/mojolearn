# SPDX-License-Identifier: Apache-2.0
"""Actual CPU class-weight descriptor; target define is mandatory."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.label_weights_native_api import LabelWeightsDriver
from xtrees.label_weights_native import label_weights_native_v1
from checks.native_tests.logistic_label_weights_cases import LABEL_WEIGHT_FIXTURES, fixture, require

comptime assert TARGET_COLUMN == COLUMN_CPU


def run_checks() raises:
    var failures = List[Int](length=LABEL_WEIGHT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(fixture_id: Int) {imm fp}:
        try:
            fixture(LabelWeightsDriver(label_weights_native_v1), fixture_id)
        except error:
            fp.unsafe_store(fixture_id, 1)
            print("FAIL logistic_label_weights_cpu fixture", fixture_id, String(error))
    host_parallelize(task, LABEL_WEIGHT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "logistic_label_weights_cpu failed " + String(failed) + " fixtures")
    print("PASS logistic_label_weights_cpu:", LABEL_WEIGHT_FIXTURES, "actual class-weight descriptor fixtures")
