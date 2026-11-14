# SPDX-License-Identifier: Apache-2.0
"""ARIMA host fit repeat, forecast/predict agreement and refusal.

The seven units (two fits per order, the refusal) are independent and run
across host cores, each into its own record; the refit comparison follows.
The original ran on the host route only (`GATE_BACKENDS = ("cpu",)`), so
there is no GPU column.
"""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from checks.native_tests.tsa_training_arima_cases import (
    ARIMA_UNITS, ARIMA_RECORD, require, arima_series, fit_unit, check_refits,
)


def run_checks() raises:
    var y = arima_series()
    var records = List[Float32](length=ARIMA_UNITS * ARIMA_RECORD, fill=Float32(0))
    var failures = List[Int](length=ARIMA_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    var base = Int(records.unsafe_ptr())
    var y_address = Int(y.unsafe_ptr())
    def task(unit: Int) {imm fp, imm base, imm y_address}:
        try:
            fit_unit(unit, y_address, f32_ptr(base + 4 * unit * ARIMA_RECORD))
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsa_training_arima_cpu unit", unit, String(error))
    host_parallelize(task, ARIMA_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsa_training_arima_cpu failed " + String(failed) + " units")
    check_refits(records)
    _ = y^
    print("PASS tsa_training_arima_cpu:", ARIMA_UNITS, "parallel host ARIMA units")
