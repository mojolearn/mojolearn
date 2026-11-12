# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the ANN operations table: the table round trip
first (ordered metadata), then the independent entry cases on the existing
host pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.neighbors_entry_table import NeighborsEntryKey
from x_ann.ops_entry_host import ann_ops_host_v1, ANN_OPS_HOST_API
from checks.native_tests.ann_ops_cases import (
    ANN_SHARED_CASES, require, run_shared_case, case_host_refuses_resident, check_ann_table,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = ANN_OPS_HOST_API.get_or_create_ptr()
    check_ann_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), ann_ops_host_v1)
    var total = ANN_SHARED_CASES + 1
    var failures = List[Int](length=total, fill=0)
    var fp = failures.unsafe_ptr()
    def task(case_id: Int) {imm fp}:
        try:
            if case_id < ANN_SHARED_CASES:
                run_shared_case(ann_ops_host_v1, case_id)
            else:
                case_host_refuses_resident(ann_ops_host_v1)
        except error:
            print("FAIL ann_ops_cpu case", case_id, String(error))
            fp[case_id] = 1
    host_parallelize(task, total)
    var failed = 0
    for i in range(total):  # driver-loop: over the case verdicts
        failed += failures[i]
    _ = failures
    require(failed == 0, "ann_ops_cpu failed " + String(failed) + " cases")
    print("PASS ann_ops_cpu:", total, "native ANN table cases")
