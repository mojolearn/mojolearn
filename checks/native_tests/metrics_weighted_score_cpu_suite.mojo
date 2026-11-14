# SPDX-License-Identifier: Apache-2.0
"""Weighted accuracy and R^2 on the host column: the metrics host
binding's bodies (`host_weighted_accuracy_ptr`, `_r2_weighted_value`).

The four units are independent and run across host cores.
"""
from core.host_parallel import host_parallelize
from metrics.host.metrics_oracle import host_weighted_accuracy_ptr
from metrics.host.r2_entry_host import _r2_weighted_value
from checks.native_tests.metrics_weighted_score_cases import WEIGHTED_UNITS, WeightedCalls, require, weighted_unit


struct HostWeighted(WeightedCalls):
    @staticmethod
    def accuracy(y_true: List[Int32], y_pred: List[Int32], w: List[Float32], n: Int) raises -> Float32:
        return host_weighted_accuracy_ptr(
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(y_true.unsafe_ptr())),
            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(y_pred.unsafe_ptr())),
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(w.unsafe_ptr())), n)

    @staticmethod
    def r2(y: List[Float32], y_hat: List[Float32], w: List[Float32], n: Int) raises -> Float32:
        return _r2_weighted_value(Int(y.unsafe_ptr()), Int(y_hat.unsafe_ptr()), Int(w.unsafe_ptr()), n)


def run_checks() raises:
    var failures = List[Int](length=WEIGHTED_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            weighted_unit[HostWeighted](unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL metrics_weighted_score_cpu unit", unit, String(error))
    host_parallelize(task, WEIGHTED_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "metrics_weighted_score_cpu failed " + String(failed) + " units")
    print("PASS metrics_weighted_score_cpu:", WEIGHTED_UNITS, "parallel weighted-score units")
