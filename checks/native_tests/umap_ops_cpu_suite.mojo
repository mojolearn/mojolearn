# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the UMAP operations table: the table round trip first
(ordered metadata), then the independent entry cases on the existing host
pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.neighbors_entry_table import NeighborsEntryKey
from umap.ops_entry_host import umap_ops_host_v1, UMAP_OPS_HOST_API
from checks.native_tests.umap_ops_cases import UMAP_SHARED_CASES, require, run_shared_case, check_umap_table


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = UMAP_OPS_HOST_API.get_or_create_ptr()
    check_umap_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), umap_ops_host_v1)
    var total = UMAP_SHARED_CASES
    var failures = List[Int](length=total, fill=0)
    var fp = failures.unsafe_ptr()
    def task(case_id: Int) {imm fp}:
        try:
            run_shared_case(umap_ops_host_v1, case_id)
        except error:
            print("FAIL umap_ops_cpu case", case_id, String(error))
            fp[case_id] = 1
    host_parallelize(task, total)
    var failed = 0
    for i in range(total):  # driver-loop: over the case verdicts
        failed += failures[i]
    _ = failures
    require(failed == 0, "umap_ops_cpu failed " + String(failed) + " cases")
    print("PASS umap_ops_cpu:", total, "native UMAP table cases")
