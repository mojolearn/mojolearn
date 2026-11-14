# SPDX-License-Identifier: Apache-2.0
"""The solver operations table (core/solver_entry_table.mojo) and the
estimators QN / QN-fold tables (core/estimators_entry_table.mojo), one case
per entry the linear owner (bindings/x_linear_native_door.mojo, ElasticNet /
Lasso) and QNRegressor's owner (bindings/qn_native.mojo) call, with
expectations known by construction (dyadic data, orthogonal designs), not by
another call into the implementation. No Python objects.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.entry_table import entry_table_same_call
from core.solver_entry_table import (
    SolverEntryKey,
    SV_OP_CD_FIT,
    solver_ops_publish,
    solver_ops_entry,
    solver_cd_fit_call,
    solver_cd_predict_call,
    solver_vendor_word,
)
from core.estimators_entry_table import (
    EstimatorsEntryKey,
    estimators_table_publish,
    estimators_table_entry,
    ESTIMATORS_TABLE_QN,
    ESTIMATORS_TABLE_QN_FOLD,
)
from core.logistic_native_api import (
    QnNativeDriver,
    QnFoldDriver,
    QnFitRequest,
    QnQueryRequest,
    QnFoldStore,
    QN_NATIVE_DECISION,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from solver.ops_entry_host import solver_ops_host_v1, SOLVER_OPS_HOST_API
from glm.host.logistic_native_host import qn_host_native_v1, LOGISTIC_HOST_API
from checks.vendor import COMPILED_VENDOR
from solver.ops_entry_device import solver_ops_device_v1, SOLVER_OPS_DEVICE_API
from glm.logistic_native_device import qn_device_native_v1, LOGISTIC_DEVICE_API
from glm.logistic_fold_native import logistic_fold_v1, LOGISTIC_FOLD_API
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SOLVER_CASES = 4
comptime QN_CASES = 3
comptime _QN_LOSS_SQUARED = 1


def close(actual: Float64, expected: Float64, atol: Float64, label: String) raises:
    require(abs(actual - expected) <= atol, label + ": " + String(actual) + " != " + String(expected))


def _addr(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def case_cd_predict(entry: NativeDescriptorCall) raises:
    # Column-major X = [[1, 2], [3, 4]], coef [1, 0.5], intercept 0.25:
    # [1 + 1 + 0.25, 3 + 2 + 0.25] = [2.25, 5.25], exact in float32.
    var x: List[Float32] = [1.0, 3.0, 2.0, 4.0]
    var coef: List[Float32] = [1.0, 0.5]
    var out = List[Float32](length=2, fill=-7.0)
    solver_cd_predict_call(entry, _addr(x), _addr(coef), _addr(out), 2, 2, 0.25)
    require(out[0] == 2.25 and out[1] == 5.25, "cd_predict: " + String(out[0]) + ", " + String(out[1]))
    _ = x
    _ = coef
    _ = out


def case_cd_fit(entry: NativeDescriptorCall) raises:
    # Orthogonal column-major X (x1 = [1,0,1,0], x2 = [0,1,0,1]), y = [2,4,2,4],
    # no intercept, alpha 0: each coordinate step is exact, coef = [2, 4].
    var x: List[Float32] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    var y: List[Float32] = [2.0, 4.0, 2.0, 4.0]
    var coef = List[Float32](length=2, fill=-7.0)
    var info = List[Float32](length=1, fill=-7.0)
    var n_iter = solver_cd_fit_call(entry, _addr(x), _addr(y), _addr(coef), _addr(info), 0, 4, 2, False, 100,
                                    False, False, 0.0, 1.0, 1e-6)
    require(n_iter >= 1, "cd_fit: n_iter " + String(n_iter))
    close(Float64(coef[0]), 2.0, 1e-4, "cd_fit: coef[0]")
    close(Float64(coef[1]), 4.0, 1e-4, "cd_fit: coef[1]")
    close(Float64(info[0]), 0.0, 1e-6, "cd_fit: intercept")
    _ = x
    _ = y
    _ = coef
    _ = info


def case_cd_fit_needs_weights(entry: NativeDescriptorCall) raises:
    # has_sample_weight with a null weight address refuses with the export's words.
    var x: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    var y: List[Float32] = [1.0, 1.0]
    var coef = List[Float32](length=2, fill=0.0)
    var info = List[Float32](length=1, fill=0.0)
    var refused = False
    try:
        _ = native_descriptor_call(entry, SV_OP_CD_FIT, [_addr(x), _addr(y), _addr(coef), _addr(info), 0],
                                   [2, 2, 0, 10, 0, 1, 0], [0.0, 1.0, 1e-4])
    except error:
        refused = "has_sample_weight" in String(error)
    require(refused, "cd_fit: a null weight address was not refused")
    _ = x
    _ = y
    _ = coef
    _ = info


def case_solver_spans(entry: NativeDescriptorCall) raises:
    var refused_op = False
    try:
        _ = native_descriptor_call(entry, 7, List[Int](), List[Int](), List[Float64]())
    except error:
        refused_op = "unknown operation" in String(error)
    require(refused_op, "solver ops: an unknown operation was not refused")
    var refused_span = False
    try:
        _ = native_descriptor_call(entry, SV_OP_CD_FIT, [1, 2, 3], [1], List[Float64]())
    except error:
        refused_span = "request spans" in String(error)
    require(refused_span, "solver ops: short spans were not refused")


def run_solver_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_cd_predict(entry)
    elif case_id == 1:
        case_cd_fit(entry)
    elif case_id == 2:
        case_cd_fit_needs_weights(entry)
    else:
        case_solver_spans(entry)


def case_qn_decision(entry: NativeDescriptorCall) raises:
    # QNRegressor.predict's request: row-major X . [2, -1] + 0.5 (bias last).
    var x: List[Float32] = [1.0, 2.0, 3.0, 0.5]
    var w: List[Float32] = [2.0, -1.0, 0.5]
    var out = List[Float32](length=2, fill=-7.0)
    QnNativeDriver(entry).query(QN_NATIVE_DECISION, QnQueryRequest(_addr(x), _addr(w), _addr(out), 0, 2, 2, 1,
                                                                   True, False))
    require(out[0] == 0.5 and out[1] == 6.0, "qn decision: " + String(out[0]) + ", " + String(out[1]))
    _ = x
    _ = w
    _ = out


def case_qn_fit(entry: NativeDescriptorCall) raises:
    # QNRegressor.fit's request: squared loss, no penalty, no intercept, on
    # the orthogonal row-major X = [[1,0],[0,1],[1,0],[0,1]], y = [2,4,2,4]:
    # the least-squares minimum is w = [2, 4].
    var x: List[Float32] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    var y: List[Float32] = [2.0, 4.0, 2.0, 4.0]
    var w = List[Float32](length=2, fill=-7.0)
    var info = List[Float32](length=2, fill=-7.0)
    var status = List[Int32](length=1, fill=0)
    var request = QnFitRequest(_addr(x), _addr(y), _addr(w), _addr(info), 0, Int(status.unsafe_ptr()),
        4, 2, 1, 200, 50, 5, False, True, _QN_LOSS_SQUARED, 0.0, 0.0, 1e-7, 1e-9, 0.0)
    var iterations = QnNativeDriver(entry).fit(request)
    require(iterations >= 0, "qn fit: iterations " + String(iterations))
    require(status[0] == 0, "qn fit: finite status " + String(status[0]))
    close(Float64(w[0]), 2.0, 1e-3, "qn fit: w[0]")
    close(Float64(w[1]), 4.0, 1e-3, "qn fit: w[1]")
    _ = x
    _ = y
    _ = w
    _ = info
    _ = status


def case_qn_spans(entry: NativeDescriptorCall) raises:
    var refused = False
    try:
        _ = native_descriptor_call(entry, 99, List[Int](), List[Int](), List[Float64]())
    except error:
        refused = "logistic native ABI" in String(error)
    require(refused, "qn entry: an unknown operation was not refused")


def run_qn_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_qn_decision(entry)
    elif case_id == 1:
        case_qn_fit(entry)
    else:
        case_qn_spans(entry)


def case_qn_fold_refuses_bad_store(entry: NativeDescriptorCall) raises:
    # The resident-fold entry (QNRegressor's DeviceRows fit): a store that
    # does not exist refuses; nothing is fitted. (A live store needs model
    # selection's device base; the integrator's public cases cover it.)
    var y: List[Float32] = [1.0, 2.0]
    var w = List[Float32](length=2, fill=0.0)
    var info = List[Float32](length=2, fill=0.0)
    var words = List[Int32](length=2, fill=0)
    var request = QnFitRequest(0, _addr(y), _addr(w), _addr(info), 0, 0,
        2, 1, 1, 10, 10, 5, True, True, _QN_LOSS_SQUARED, 0.0, 0.0, 1e-4, 1e-6, 0.0)
    var refused = False
    try:
        _ = QnFoldDriver(entry).fit(QnFoldStore(-5, 2, -5), request, Int(words.unsafe_ptr()))
    except:
        refused = True
    require(refused, "qn fold: a missing store was not refused")
    _ = y
    _ = w
    _ = info
    _ = words


def check_solver_table(key: SolverEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publication then lookup returns the published pointer; a repeated
    identical publication is accepted; another tier refuses. Run first, in
    a fresh process (it fills the process slot)."""
    var missing = False
    try:
        _ = solver_ops_entry(key)
    except:
        missing = True
    require(missing, "solver table: lookup before publication did not refuse")
    solver_ops_publish(key, publisher, entry)
    solver_ops_publish(key, publisher, entry)
    require(entry_table_same_call(solver_ops_entry(key), entry), "solver table: lookup returned another pointer")
    var other = False
    try:
        _ = solver_ops_entry(SolverEntryKey(1 - key.mode, key.vendor))
    except:
        other = True
    require(other, "solver table: the other tier's unpublished slot did not refuse")


def check_estimators_table[family: Int](key: EstimatorsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    var missing = False
    try:
        _ = estimators_table_entry[family](key)
    except:
        missing = True
    require(missing, "estimators table: lookup before publication did not refuse")
    estimators_table_publish[family](key, publisher, entry)
    estimators_table_publish[family](key, publisher, entry)
    require(entry_table_same_call(estimators_table_entry[family](key), entry),
            "estimators table: lookup returned another pointer")


# ---- The host column (was solver_estimators_ops_cpu_suite.mojo) ----
# The CPU column of the solver operations table and the estimators QN
# table: the table round trips first (ordered metadata), then the independent
# entry cases on the existing host pool.
def _run_cpu(mut report: Report) raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var solver_api = SOLVER_OPS_HOST_API.get_or_create_ptr()
    check_solver_table(SolverEntryKey(mode, 0), Int(solver_api[].entries.unsafe_ptr()), solver_ops_host_v1)
    var qn_api = LOGISTIC_HOST_API.get_or_create_ptr()
    check_estimators_table[ESTIMATORS_TABLE_QN](EstimatorsEntryKey(mode, 0), Int(qn_api[].entries.unsafe_ptr()),
                                                qn_host_native_v1)
    var total = SOLVER_CASES + QN_CASES
    var cases = report.cases(total)
    def task(case_id: Int) {imm cases}:
        try:
            if case_id < SOLVER_CASES:
                run_solver_case(solver_ops_host_v1, case_id)
            else:
                run_qn_case(qn_host_native_v1, case_id - SOLVER_CASES)
        except error:
            cases.fail(case_id, error)
    report.parallel(task, total)


# ---- The device column (was solver_estimators_ops_gpu_suite.mojo) ----
# The device column of the solver operations table and the estimators QN
# and QN-fold tables. The entries share their bindings' device contexts, so
# the cases submit in order; the kernels keep their native parallelism (no
# concurrency claim).
def _run_gpu(mut report: Report) raises:
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


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("solver_estimators_ops", cpu=True, gpu=True, serial="", run=run_cases)
