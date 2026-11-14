# SPDX-License-Identifier: Apache-2.0
"""The neighbors operations table (core/neighbors_entry_table.mojo), one case
per entry the k-NN owner (bindings/neighbors_knn_native.mojo) calls, with
expectations known by construction (one-feature points far apart, exact
float32 means), not by another call into the implementation. No Python
objects.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_table_same_call
from core.native_descriptor_call import native_descriptor_call
from core.neighbors_entry_table import (
    NeighborsEntryKey,
    NbFold,
    neighbors_ops_publish,
    neighbors_ops_entry,
    nb_index_prepare_call,
    nb_index_release_call,
    nb_fold_ready_call,
    nb_classify_call,
    nb_regress_call,
    nb_classify_resident_call,
    nb_regress_resident_call,
    NB_OP_CLASSIFY,
    neighbors_vendor_word,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from neighbors.ops_entry_host import neighbors_ops_host_v1, NEIGHBORS_OPS_HOST_API
from checks.vendor import COMPILED_VENDOR
from neighbors.ops_entry_device import neighbors_ops_device_v1, NEIGHBORS_OPS_DEVICE_API
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime NEIGHBORS_SHARED_CASES = 4
comptime NEIGHBORS_DEVICE_CASES = 3
comptime _METRIC_FROM_IS_SQRT = -1
comptime _UNIFORM = 0


def _f(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def _i(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr())


def _index() -> List[Float32]:
    # Four one-feature points in two far groups: {0, 1} and {10, 11}.
    var index: List[Float32] = [0.0, 1.0, 10.0, 11.0]
    return index^


def _queries() -> List[Float32]:
    var queries: List[Float32] = [0.25, 10.75]
    return queries^


def case_classify(entry: NativeDescriptorCall) raises:
    # k = 1: query 0.25 -> point 0 (class 0), query 10.75 -> point 11 (class 1).
    var index = _index()
    var queries = _queries()
    var y: List[Int32] = [0, 0, 1, 1]
    var labels = List[Int32](length=2, fill=-7)
    var proba = List[Float32](length=1, fill=0.0)
    var uniq = List[Int32](length=2, fill=-7)
    var n_classes: List[Int] = [2]
    var used = nb_classify_call(entry, _f(index), _f(queries), _i(y), _i(labels), _f(proba), _i(uniq), 4, 2, 1,
                                1, 64, False, n_classes, _METRIC_FROM_IS_SQRT, 2.0, _UNIFORM)
    require(used >= 1, "classify: no query tile reported")
    require(labels[0] == 0 and labels[1] == 1, "classify: labels " + String(labels[0]) + ", " + String(labels[1]))
    require(uniq[0] == 0 and uniq[1] == 1, "classify: class set " + String(uniq[0]) + ", " + String(uniq[1]))
    _ = index
    _ = queries
    _ = y
    _ = labels
    _ = proba
    _ = uniq
    _ = n_classes


def case_classify_proba(entry: NativeDescriptorCall) raises:
    # k = 2, uniform: each query's two neighbours share its class -> [1, 0], [0, 1].
    var index = _index()
    var queries = _queries()
    var y: List[Int32] = [0, 0, 1, 1]
    var labels = List[Int32](length=1, fill=-7)
    var proba = List[Float32](length=4, fill=-7.0)
    var uniq = List[Int32](length=2, fill=-7)
    var n_classes: List[Int] = [2]
    _ = nb_classify_call(entry, _f(index), _f(queries), _i(y), _i(labels), _f(proba), _i(uniq), 4, 2, 1, 2, 64,
                         True, n_classes, _METRIC_FROM_IS_SQRT, 2.0, _UNIFORM)
    require(proba[0] == 1.0 and proba[1] == 0.0 and proba[2] == 0.0 and proba[3] == 1.0,
            "classify proba: " + String(proba[0]) + ", " + String(proba[1]) + ", " + String(proba[2]) + ", "
            + String(proba[3]))
    _ = index
    _ = queries
    _ = y
    _ = labels
    _ = proba
    _ = uniq
    _ = n_classes


def case_regress(entry: NativeDescriptorCall) raises:
    # k = 2, uniform: (1 + 3) / 2 = 2 and (5 + 7) / 2 = 6, exact in float32.
    var index = _index()
    var queries = _queries()
    var y: List[Float32] = [1.0, 3.0, 5.0, 7.0]
    var dst = List[Float32](length=2, fill=-7.0)
    _ = nb_regress_call(entry, _f(index), _f(queries), _f(y), _f(dst), 4, 2, 1, 2, 64, 1,
                        _METRIC_FROM_IS_SQRT, 2.0, _UNIFORM)
    require(dst[0] == 2.0 and dst[1] == 6.0, "regress: " + String(dst[0]) + ", " + String(dst[1]))
    _ = index
    _ = queries
    _ = y
    _ = dst


def case_refuses_short_class_counts(entry: NativeDescriptorCall) raises:
    # n_outputs 2 with one class count: refused before any word is read.
    var index = _index()
    var queries = _queries()
    var y: List[Int32] = [0, 0, 1, 1]
    var labels = List[Int32](length=4, fill=-7)
    var proba = List[Float32](length=1, fill=0.0)
    var uniq = List[Int32](length=4, fill=-7)
    var n_classes: List[Int] = [2]
    var refused = False
    try:
        var addresses: List[Int] = [_f(index), _f(queries), _i(y), _i(labels), _f(proba), _i(uniq)]
        var ints: List[Int] = [4, 2, 1, 1, 64, 2, 0, _METRIC_FROM_IS_SQRT, _UNIFORM, 2]
        var floats: List[Float64] = [2.0]
        _ = native_descriptor_call(entry, NB_OP_CLASSIFY, addresses, ints, floats)
    except:
        refused = True
    require(refused, "classify: a short class-count tail was not refused")
    require(labels[0] == -7, "classify: a refused request wrote labels")
    _ = index
    _ = queries
    _ = y
    _ = labels
    _ = proba
    _ = uniq
    _ = n_classes


def run_shared_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_classify(entry)
    elif case_id == 1:
        case_classify_proba(entry)
    elif case_id == 2:
        case_regress(entry)
    else:
        case_refuses_short_class_counts(entry)


def case_host_refuses_resident(entry: NativeDescriptorCall) raises:
    var refused = False
    try:
        _ = nb_fold_ready_call(entry)
    except:
        refused = True
    require(refused, "host column: the fold probe was not refused")


def case_resident_classify(entry: NativeDescriptorCall) raises:
    # prepare (with the finite status word), classify against the device
    # copy, release.
    var index = _index()
    var queries = _queries()
    var y: List[Int32] = [0, 0, 1, 1]
    var labels = List[Int32](length=2, fill=-7)
    var proba = List[Float32](length=1, fill=0.0)
    var uniq = List[Int32](length=2, fill=-7)
    var status = List[Int32](length=2, fill=0)
    var n_classes: List[Int] = [2]
    var handle = nb_index_prepare_call(entry, _f(index), _i(status), 4, 1)
    require(handle != 0 and status[0] == 0, "prepare: handle " + String(handle) + ", status " + String(status[0]))
    _ = nb_classify_resident_call(entry, handle, _f(index), _f(queries), _i(y), _i(labels), _f(proba), _i(uniq),
                                  _i(status), 4, 2, 1, 1, 64, False, n_classes, _METRIC_FROM_IS_SQRT, 2.0,
                                  _UNIFORM, NbFold.none())
    nb_index_release_call(entry, handle)
    require(status[0] == 0, "resident classify: finite queries reported " + String(status[0]))
    require(labels[0] == 0 and labels[1] == 1,
            "resident classify: labels " + String(labels[0]) + ", " + String(labels[1]))
    _ = index
    _ = queries
    _ = y
    _ = labels
    _ = proba
    _ = uniq
    _ = status
    _ = n_classes


def case_resident_regress(entry: NativeDescriptorCall) raises:
    var index = _index()
    var queries = _queries()
    var y: List[Float32] = [1.0, 3.0, 5.0, 7.0]
    var dst = List[Float32](length=2, fill=-7.0)
    var handle = nb_index_prepare_call(entry, _f(index), 0, 4, 1)
    require(handle != 0, "prepare: no handle")
    _ = nb_regress_resident_call(entry, handle, _f(index), _f(queries), _f(y), _f(dst), 0, 4, 2, 1, 2, 64, 1,
                                 _METRIC_FROM_IS_SQRT, 2.0, _UNIFORM, NbFold.none())
    nb_index_release_call(entry, handle)
    require(dst[0] == 2.0 and dst[1] == 6.0, "resident regress: " + String(dst[0]) + ", " + String(dst[1]))
    _ = index
    _ = queries
    _ = y
    _ = dst


def case_fold_ready_answers(entry: NativeDescriptorCall) raises:
    # The probe answers (0 or 1) without a model-selection store: no refusal.
    _ = nb_fold_ready_call(entry)


def run_device_case(entry: NativeDescriptorCall, case_id: Int) raises:
    if case_id == 0:
        case_resident_classify(entry)
    elif case_id == 1:
        case_resident_regress(entry)
    else:
        case_fold_ready_answers(entry)


def check_neighbors_table(key: NeighborsEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publication then lookup returns the published pointer; a repeated
    identical publication is accepted; another tier refuses. Run first, in
    a fresh process (it fills the process slot)."""
    var missing = False
    try:
        _ = neighbors_ops_entry(key)
    except:
        missing = True
    require(missing, "neighbors table: lookup before publication did not refuse")
    neighbors_ops_publish(key, publisher, entry)
    neighbors_ops_publish(key, publisher, entry)
    require(entry_table_same_call(neighbors_ops_entry(key), entry), "neighbors table: lookup returned another pointer")
    var other = False
    try:
        _ = neighbors_ops_entry(NeighborsEntryKey(1 - key.mode, key.vendor))
    except:
        other = True
    require(other, "neighbors table: the other tier's unpublished slot did not refuse")


# ---- The host column (was neighbors_ops_cpu_suite.mojo) ----
# The CPU column of the neighbors operations table: the table round trip
# first (ordered metadata), then the independent entry cases on the existing
# host pool.
def _run_cpu(mut report: Report) raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = NEIGHBORS_OPS_HOST_API.get_or_create_ptr()
    check_neighbors_table(NeighborsEntryKey(mode, 0), Int(api[].entries.unsafe_ptr()), neighbors_ops_host_v1)
    var total = NEIGHBORS_SHARED_CASES + 1
    var cases = report.cases(total)
    def task(case_id: Int) {imm cases}:
        try:
            if case_id < NEIGHBORS_SHARED_CASES:
                run_shared_case(neighbors_ops_host_v1, case_id)
            else:
                case_host_refuses_resident(neighbors_ops_host_v1)
        except error:
            cases.fail(case_id, error)
    report.parallel(task, total)


# ---- The device column (was neighbors_ops_gpu_suite.mojo) ----
# The device column of the neighbors operations table. The entries share
# the base binding's device context and resident registry, so the cases
# submit in order; the kernels keep their native parallelism (no concurrency
# claim).
def _run_gpu(mut report: Report) raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "neighbors_ops_gpu: this build has no GPU vendor")
    var api = NEIGHBORS_OPS_DEVICE_API.get_or_create_ptr()
    check_neighbors_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), neighbors_ops_device_v1)
    var failed = 0
    for case_id in range(NEIGHBORS_SHARED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_shared_case(neighbors_ops_device_v1, case_id)
        except error:
            print("FAIL neighbors_ops_gpu shared case", case_id, String(error))
            failed += 1
    for case_id in range(NEIGHBORS_DEVICE_CASES):  # driver-loop: over the ordered device cases
        try:
            run_device_case(neighbors_ops_device_v1, case_id)
        except error:
            print("FAIL neighbors_ops_gpu device case", case_id, String(error))
            failed += 1
    require(failed == 0, "neighbors_ops_gpu failed " + String(failed) + " cases")
    print("PASS neighbors_ops_gpu:", NEIGHBORS_SHARED_CASES + NEIGHBORS_DEVICE_CASES, "native neighbors table cases")


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("neighbors_ops", cpu=True, gpu=True, serial="", run=run_cases)
