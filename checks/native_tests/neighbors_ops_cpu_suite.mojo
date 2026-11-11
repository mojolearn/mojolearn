# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the neighbors operations table: the table round trip
first (ordered metadata), then the independent entry cases on the existing
host pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.neighbors_entry_table import NeighborsEntryKey
from neighbors.ops_entry_host import neighbors_ops_host_v1, NEIGHBORS_OPS_HOST_API
from checks.native_tests.neighbors_ops_cases import (
    NEIGHBORS_SHARED_CASES, require, run_shared_case, case_host_refuses_resident, check_neighbors_table,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = NEIGHBORS_OPS_HOST_API.get_or_create_ptr()
    check_neighbors_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), neighbors_ops_host_v1)
    var total = NEIGHBORS_SHARED_CASES + 1
    var failures = List[Int](length=total, fill=0)
    var fp = failures.unsafe_ptr()
    def task(case_id: Int) {imm fp}:
        try:
            if case_id < NEIGHBORS_SHARED_CASES:
                run_shared_case(neighbors_ops_host_v1, case_id)
            else:
                case_host_refuses_resident(neighbors_ops_host_v1)
        except error:
            print("FAIL neighbors_ops_cpu case", case_id, String(error))
            fp[case_id] = 1
    host_parallelize(task, total)
    var failed = 0
    for i in range(total):  # driver-loop: over the case verdicts
        failed += failures[i]
    _ = failures
    require(failed == 0, "neighbors_ops_cpu failed " + String(failed) + " cases")
    print("PASS neighbors_ops_cpu:", total, "native neighbors table cases")
