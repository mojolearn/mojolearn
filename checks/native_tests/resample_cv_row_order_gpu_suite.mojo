# SPDX-License-Identifier: Apache-2.0
"""Shuffle-split row order on the device metrics program entry
(x_metrics/descriptor_device.mojo). Fixtures run in order: every program is
a sequence of launches on the one shared process device context, and each
stage keeps its own kernel parallelism."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.resample_cv_row_order_cases import CV_ROW_ORDER_FIXTURES, require, run_row_order_fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(CV_ROW_ORDER_FIXTURES):  # ordered programs on the one shared device context
        try:
            run_row_order_fixture(metrics_program_device_v1, unit)
        except error:
            failed += 1
            print("FAIL resample_cv_row_order_gpu fixture", unit, String(error))
    require(failed == 0, "resample_cv_row_order_gpu failed " + String(failed) + " fixtures")
    print("PASS resample_cv_row_order_gpu:", CV_ROW_ORDER_FIXTURES, "shuffle-split orders against the counter-key oracle")
