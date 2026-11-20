# SPDX-License-Identifier: Apache-2.0
"""Root-only combined native upgrade batch.

Independent groups use the full native host pool; each suite retains its own
parallel fixture scheduling. No product/test source or oracle is rewritten.
NET-P01 and cross-image publication proofs require separate fresh processes.
"""
from core.host_parallel import host_parallelize
from checks.native_tests.gpr_score_cpu_suite import run_checks as suite_0
from checks.native_tests.gpc_score_cpu_suite import run_checks as suite_1
from checks.native_tests.linear_svc_native_cpu_suite import run_checks as suite_2
from checks.native_tests.labels_threshold_cpu_suite import run_checks as suite_3
from checks.native_tests.hdbscan_plan_suite import run_checks as suite_4
from checks.native_tests.hdbscan_native_cpu_suite import run_checks as suite_5
from checks.native_tests.linear_svr_native_cpu_suite import run_checks as suite_6
from checks.native_tests.svc_accuracy_native_cpu_suite import run_checks as suite_7
from checks.native_tests.svr_native_cpu_suite import run_checks as suite_8
from checks.native_tests.entry_table_contract_suite import run_checks as suite_9
from checks.native_tests.ipca_native_cpu_suite import run_checks as suite_10
from checks.native_tests.dbscan_state_suite import run_checks as suite_11
from checks.native_tests.optics_native_cpu_suite import run_checks as suite_12
from checks.native_tests.ocsvm_native_cpu_suite import run_checks as suite_13


def main() raises:
    var failures = List[Int](length=14, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            if unit == 0:
                suite_0()
            elif unit == 1:
                suite_1()
            elif unit == 2:
                suite_2()
            elif unit == 3:
                suite_3()
            elif unit == 4:
                suite_4()
            elif unit == 5:
                suite_5()
            elif unit == 6:
                suite_6()
            elif unit == 7:
                suite_7()
            elif unit == 8:
                suite_8()
            elif unit == 9:
                suite_9()
            elif unit == 10:
                suite_10()
            elif unit == 11:
                suite_11()
            elif unit == 12:
                suite_12()
            elif unit == 13:
                suite_13()
        except problem:
            flags[unit] = 1
            print("FAIL consolidated_native_cpu group", unit, String(problem))
    host_parallelize(run, 14)
    var failed = 0
    for flag in failures:  # completed independent group status metadata
        failed += flag
    _ = failures
    if failed:
        raise Error("Consolidated native CPU groups failed: " + String(failed))
    print("PASS consolidated_native_cpu: 179 fixtures")
