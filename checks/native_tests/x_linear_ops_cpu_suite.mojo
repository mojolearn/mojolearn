# SPDX-License-Identifier: Apache-2.0
"""The CPU column's x_linear operations entry; independent cases run on
the existing host pool."""
from core.host_parallel import host_parallelize
from x_linear.ops_entry_host import x_linear_ops_host_v1
from checks.native_tests.x_linear_ops_cases import XL_OPS_CASES, require, run_x_linear_ops_case


def run_checks() raises:
    var failures = List[Int](length=XL_OPS_CASES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(case_id: Int) {imm fp}:
        try:
            run_x_linear_ops_case(x_linear_ops_host_v1, False, case_id)
        except error:
            print("FAIL x_linear_ops_cpu case", case_id, String(error))
            fp.unsafe_store(case_id, 1)
    host_parallelize(task, XL_OPS_CASES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "x_linear_ops_cpu failed " + String(failed) + " cases")
    print("PASS x_linear_ops_cpu:", XL_OPS_CASES, "native x_linear table cases, parallel")
