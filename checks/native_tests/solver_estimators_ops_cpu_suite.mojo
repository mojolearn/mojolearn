# SPDX-License-Identifier: Apache-2.0
"""The CPU column of the solver operations table and the estimators QN
table: the table round trips first (ordered metadata), then the independent
entry cases on the existing host pool."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.solver_entry_table import SolverEntryKey
from core.estimators_entry_table import EstimatorsEntryKey, ESTIMATORS_TABLE_QN
from solver.ops_entry_host import solver_ops_host_v1, SOLVER_OPS_HOST_API
from glm.host.logistic_native_host import qn_host_native_v1, LOGISTIC_HOST_API
from checks.native_tests.solver_estimators_ops_cases import (
    SOLVER_CASES, QN_CASES, require, run_solver_case, run_qn_case, check_solver_table, check_estimators_table,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var solver_api = SOLVER_OPS_HOST_API.get_or_create_ptr()
    check_solver_table(SolverEntryKey(mode, 0), Int(solver_api[].entries.unsafe_ptr()), solver_ops_host_v1)
    var qn_api = LOGISTIC_HOST_API.get_or_create_ptr()
    check_estimators_table[ESTIMATORS_TABLE_QN](EstimatorsEntryKey(mode, 0), Int(qn_api[].entries.unsafe_ptr()),
                                                qn_host_native_v1)
    var total = SOLVER_CASES + QN_CASES
    var failures = List[Int](length=total, fill=0)
    var fp = failures.unsafe_ptr()
    def task(case_id: Int) {imm fp}:
        try:
            if case_id < SOLVER_CASES:
                run_solver_case(solver_ops_host_v1, case_id)
            else:
                run_qn_case(qn_host_native_v1, case_id - SOLVER_CASES)
        except error:
            print("FAIL solver_estimators_ops_cpu case", case_id, String(error))
            fp.unsafe_store(case_id, 1)
    host_parallelize(task, total)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "solver_estimators_ops_cpu failed " + String(failed) + " cases")
    print("PASS solver_estimators_ops_cpu:", total, "native solver / QN table cases, parallel")
