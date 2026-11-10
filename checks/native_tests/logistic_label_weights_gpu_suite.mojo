# SPDX-License-Identifier: Apache-2.0
"""Actual GPU class-weight descriptor; shared context submissions are ordered."""
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.label_weights_native_api import LabelWeightsDriver
from xtrees.label_weights_native import label_weights_native_v1
from checks.native_tests.logistic_label_weights_cases import LABEL_WEIGHT_FIXTURES, fixture, require

def run_checks() raises:
    comptime assert TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for fixture_id in range(LABEL_WEIGHT_FIXTURES):
        try:
            fixture(LabelWeightsDriver(label_weights_native_v1), fixture_id)
        except error:
            failed += 1
            print("FAIL logistic_label_weights_gpu fixture", fixture_id, String(error))
    require(failed == 0, "logistic_label_weights_gpu failed " + String(failed) + " fixtures")
    print("PASS logistic_label_weights_gpu:", LABEL_WEIGHT_FIXTURES, "actual class-weight descriptor fixtures")
