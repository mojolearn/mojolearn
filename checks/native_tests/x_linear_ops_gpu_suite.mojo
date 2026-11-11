# SPDX-License-Identifier: Apache-2.0
"""The device column's x_linear operations entry. The entries share the
linear device context, so the cases submit in order; the kernels keep
their native parallelism (no concurrency claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_linear.ops_entry_device import x_linear_ops_device_v1
from checks.native_tests.x_linear_ops_cases import XL_OPS_CASES, require, run_x_linear_ops_case


def run_checks() raises:
    var fold = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for case_id in range(XL_OPS_CASES):  # driver-loop: over the ordered device cases
        try:
            run_x_linear_ops_case(x_linear_ops_device_v1, fold, case_id)
        except error:
            print("FAIL x_linear_ops_gpu case", case_id, String(error))
            failed += 1
    require(failed == 0, "x_linear_ops_gpu failed " + String(failed) + " cases")
    print("PASS x_linear_ops_gpu:", XL_OPS_CASES, "native x_linear table cases")
