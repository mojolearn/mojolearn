# SPDX-License-Identifier: Apache-2.0
"""Shuffle-split row order on the host metrics program entry
(x_metrics/descriptor_host.mojo), the independent fixtures across host
cores, each with a private failure slot."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from x_metrics.descriptor_host import metrics_program_host_v1
from checks.native_tests.resample_cv_row_order_cases import CV_ROW_ORDER_FIXTURES, require, run_row_order_fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=CV_ROW_ORDER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            run_row_order_fixture(metrics_program_host_v1, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL resample_cv_row_order_cpu fixture", unit, String(error))
    host_parallelize(task, CV_ROW_ORDER_FIXTURES)
    var failed = 0
    for value in failures:  # collect independent fixture status words
        failed += value
    _ = failures
    require(failed == 0, "resample_cv_row_order_cpu failed " + String(failed) + " fixtures")
    print("PASS resample_cv_row_order_cpu:", CV_ROW_ORDER_FIXTURES, "shuffle-split orders against the counter-key oracle")
