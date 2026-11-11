# SPDX-License-Identifier: Apache-2.0
"""The device column of the solver operations table and the estimators QN
and QN-fold tables. The entries share their bindings' device contexts, so
the cases submit in order; the kernels keep their native parallelism (no
concurrency claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.solver_entry_table import SolverEntryKey, solver_vendor_word
from core.estimators_entry_table import EstimatorsEntryKey, ESTIMATORS_TABLE_QN, ESTIMATORS_TABLE_QN_FOLD
from solver.ops_entry_device import solver_ops_device_v1, SOLVER_OPS_DEVICE_API
from glm.logistic_native_device import qn_device_native_v1, LOGISTIC_DEVICE_API
from glm.logistic_fold_native import logistic_fold_v1, LOGISTIC_FOLD_API
from checks.native_tests.solver_estimators_ops_cases import (
    SOLVER_CASES, QN_CASES, require, run_solver_case, run_qn_case, check_solver_table, check_estimators_table,
    case_qn_fold_refuses_bad_store,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = solver_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "solver_estimators_ops_gpu: this build has no GPU vendor")
    var solver_api = SOLVER_OPS_DEVICE_API.get_or_create_ptr()
    check_solver_table(SolverEntryKey(mode, vendor), Int(solver_api[].entries.unsafe_ptr()), solver_ops_device_v1)
    var qn_api = LOGISTIC_DEVICE_API.get_or_create_ptr()
    check_estimators_table[ESTIMATORS_TABLE_QN](EstimatorsEntryKey(mode, vendor),
                                                Int(qn_api[].entries.unsafe_ptr()), qn_device_native_v1)
    var fold_api = LOGISTIC_FOLD_API.get_or_create_ptr()
    check_estimators_table[ESTIMATORS_TABLE_QN_FOLD](EstimatorsEntryKey(mode, vendor),
                                                     Int(fold_api[].entries.unsafe_ptr()), logistic_fold_v1)
    var failed = 0
    for case_id in range(SOLVER_CASES):  # driver-loop: over the ordered device cases
        try:
            run_solver_case(solver_ops_device_v1, case_id)
        except error:
            print("FAIL solver_estimators_ops_gpu solver case", case_id, String(error))
            failed += 1
    for case_id in range(QN_CASES):  # driver-loop: over the ordered device cases
        try:
            run_qn_case(qn_device_native_v1, case_id)
        except error:
            print("FAIL solver_estimators_ops_gpu qn case", case_id, String(error))
            failed += 1
    try:
        case_qn_fold_refuses_bad_store(logistic_fold_v1)
    except error:
        print("FAIL solver_estimators_ops_gpu qn fold case", String(error))
        failed += 1
    require(failed == 0, "solver_estimators_ops_gpu failed " + String(failed) + " cases")
    print("PASS solver_estimators_ops_gpu:", SOLVER_CASES + QN_CASES + 1, "native solver / QN table cases")
