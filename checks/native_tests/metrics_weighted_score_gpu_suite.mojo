# SPDX-License-Identifier: Apache-2.0
"""Weighted accuracy and R^2 on the device column: the metrics device
binding's bodies (`accuracy_score_weighted_host`, `_r2_weighted_value`).

The units share the process's one metrics device context, so they run in
order; each reduction keeps its own chunked device parallelism.
"""
from metrics.estimator import accuracy_score_weighted_host
from metrics.r2_entry_device import _r2_weighted_value
from checks.native_tests.metrics_weighted_score_cases import WEIGHTED_UNITS, WeightedCalls, require, weighted_unit


struct DeviceWeighted(WeightedCalls):
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        return accuracy_score_weighted_host(y_true, y_pred, w, n)

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        return _r2_weighted_value(Int(y.unsafe_ptr()), Int(y_hat.unsafe_ptr()), Int(w.unsafe_ptr()), n)


def run_checks() raises:
    var failed = 0
    for unit in range(WEIGHTED_UNITS):  # driver-loop: over the units on the one device context
        try:
            weighted_unit[DeviceWeighted](unit)
        except error:
            failed += 1
            print("FAIL metrics_weighted_score_gpu unit", unit, String(error))
    require(failed == 0, "metrics_weighted_score_gpu failed " + String(failed) + " units")
    print("PASS metrics_weighted_score_gpu:", WEIGHTED_UNITS, "weighted-score units")
