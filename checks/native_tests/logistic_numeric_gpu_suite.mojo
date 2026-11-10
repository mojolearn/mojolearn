# SPDX-License-Identifier: Apache-2.0
"""Importable native GPU LogisticRegression numerical suite.

The production QN entries share a process DeviceContext. Fixture submissions
are ordered; input/oracle rows use host_parallelize and all production GPU
kernels/selected argmax paths are preserved. No GPU concurrency claim.
"""
from core.logistic_native_api import QnNativeDriver, QnQueryRequest, QN_NATIVE_MULTICLASS
from core.label_rows_device import device_argmax_rows
from core.neural_context import process_ctx
from glm.logistic_native_device import qn_device_native_v1, QnDeviceKernels, _CONTEXT
from checks.native_tests.logistic_numeric_cases import LOGISTIC_NUMERIC_FIXTURES, LogisticPrediction, fixture
from checks.native_tests.logistic_numeric_fixtures import require


struct DevicePrediction(LogisticPrediction):
    @staticmethod
    def codes(driver: QnNativeDriver, request: QnQueryRequest, scores: Int) raises:
        if QnDeviceKernels.multiclass():
            driver.query(QN_NATIVE_MULTICLASS, request)
        else:
            # Preserve the public native labels fallback when the measured
            # QN device argmax switch is disabled; no host data processing.
            var ctx = process_ctx[_CONTEXT]()
            device_argmax_rows(ctx, scores, False, request.rows, request.targets, request.dst)
            _ = ctx


def run_checks() raises:
    var failed = 0
    for fixture_id in range(LOGISTIC_NUMERIC_FIXTURES):
        try:
            fixture[DevicePrediction](QnNativeDriver(qn_device_native_v1), fixture_id)
        except error:
            failed += 1
            print("FAIL logistic_numeric_gpu fixture", fixture_id, String(error))
    require(failed == 0, "logistic_numeric_gpu failed " + String(failed) + " fixtures")
    print("PASS logistic_numeric_gpu:", LOGISTIC_NUMERIC_FIXTURES, "native fit/query/state fixtures")
