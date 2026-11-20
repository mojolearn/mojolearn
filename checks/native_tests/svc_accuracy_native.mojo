# SPDX-License-Identifier: Apache-2.0
"""Original labels prepare/gather providers plus the shared COUNT program.

Each four-row block has exactly two matches. The true and predicted class
sets differ, so independently encoding without union/remap would be wrong.
Three range settings exercise the existing device ranges/out/full routes.
"""
from core.host_parallel import host_parallelize
from core.labels_native_api import LabelsNativeDriver, LabelsPrepareRequest, LabelsGatherRequest
from core.labels_score_keys import score_integer_keys, score_key_join, ScoreKeyIndex, score_key_codes
from core.gp_native_accuracy import GPCAccuracyProgram
from core.svc_score_codes import svc_score_codes_valid, svc_used_classes, svc_gather_words
from core.native_descriptor_abi import NativeDescriptorCall
from core.labels_native_host_provider import labels_host_native_v2
from x_metrics.descriptor_host import metrics_program_host_v1
from core.labels_native_device import labels_device_native_v2
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime SVC_ACCURACY_NATIVE_CASES = 7


def numerical(labels: LabelsNativeDriver, metrics: NativeDescriptorCall, unit: Int) raises:
    var count = 12 if unit == 0 else (1032 if unit == 1 else 8196)
    var truth = List[Int64](length=count, fill=0)
    var prediction = List[Int64](length=count, fill=0)
    var tp = truth.unsafe_ptr()
    var pp = prediction.unsafe_ptr()
    def fill(row: Int) {imm tp, imm pp}:
        var at = row % 4
        tp[row] = Int64(-4 if at == 1 else (50 if at == 3 else 10))
        pp[row] = Int64(-4 if at < 2 else (999 if at == 3 else 10))
    host_parallelize(fill, count)
    var true_values = List[Int64](length=count, fill=0)
    var pred_values = List[Int64](length=count, fill=0)
    var true_encoded = List[Int32](length=count, fill=0)
    var pred_encoded = List[Int32](length=count, fill=0)
    var true_count = labels.prepare(LabelsPrepareRequest(Int(tp), Int(true_values.unsafe_ptr()),
        Int(true_encoded.unsafe_ptr()), 0, 3, count, 0, 0))
    var pred_count = labels.prepare(LabelsPrepareRequest(Int(pp), Int(pred_values.unsafe_ptr()),
        Int(pred_encoded.unsafe_ptr()), 0, 3, count, 0, 0))
    require(true_count == 3 and pred_count == 3, "labels provider changed distinct count")
    require(true_values[0] == -4 and true_values[1] == 10 and true_values[2] == 50,
        "truth ORDER RULE changed")
    require(pred_values[0] == -4 and pred_values[1] == 10 and pred_values[2] == 999,
        "prediction ORDER RULE changed")
    var true_keys = score_integer_keys(Int(true_values.unsafe_ptr()), true_count)
    var pred_keys = score_integer_keys(Int(pred_values.unsafe_ptr()), pred_count)
    var joined = score_key_join(true_keys, pred_keys)
    var index = ScoreKeyIndex(Int(joined.unsafe_ptr()), len(joined))
    require(len(index.sorted_rows) == 4, "score union count changed")
    var true_table = List[Int32](length=true_count, fill=0)
    var pred_table = List[Int32](length=pred_count, fill=0)
    score_key_codes(index, true_keys, Int(true_table.unsafe_ptr()))
    score_key_codes(index, pred_keys, Int(pred_table.unsafe_ptr()))
    var true_codes = List[Int32](length=count, fill=0)
    var pred_codes = List[Int32](length=count, fill=0)
    labels.gather(0, LabelsGatherRequest(Int(true_table.unsafe_ptr()), Int(true_encoded.unsafe_ptr()),
        Int(true_codes.unsafe_ptr()), true_count, count))
    labels.gather(0, LabelsGatherRequest(Int(pred_table.unsafe_ptr()), Int(pred_encoded.unsafe_ptr()),
        Int(pred_codes.unsafe_ptr()), pred_count, count))
    var program = GPCAccuracyProgram(count, 4, Int(true_codes.unsafe_ptr()), Int(pred_codes.unsafe_ptr()))
    _ = truth
    _ = prediction
    _ = true_values
    _ = pred_values
    _ = true_encoded
    _ = pred_encoded
    _ = true_keys
    _ = pred_keys
    _ = joined
    _ = index
    _ = true_table
    _ = pred_table
    _ = true_codes
    _ = pred_codes
    var value = program.run(metrics, unit == 0, unit != 2)
    require(value == 0.5, "COUNT accuracy differs from independent exact half")
    _ = program


def fixture(labels: LabelsNativeDriver, metrics: NativeDescriptorCall, unit: Int) raises:
    if unit < 3:
        numerical(labels, metrics, unit)
    elif unit == 3:
        var empty = List[Int64]()
        require(svc_score_codes_valid(empty, 0), "empty code vector refused")
        var codes: List[Int64] = [-3, -1, 0, 2]
        require(svc_score_codes_valid(codes, 3), "existing Python negative indexing narrowed")
    elif unit == 4:
        var high: List[Int64] = [0, 3]
        var low: List[Int64] = [-4, 0]
        var no_classes: List[Int64] = [0]
        require(not svc_score_codes_valid(high, 3), "upper code bound accepted")
        require(not svc_score_codes_valid(low, 3), "lower code bound accepted")
        require(not svc_score_codes_valid(no_classes, 0), "code accepted without classes")
    elif unit == 5:
        used_classes()
    else:
        used_classes_large()


def used_classes() raises:
    # Rows use classes 3, 0 (also as -5) and 4 of five; class 1 and 2 never.
    # First-appearance order is 3, 0, 4; slots index that order.
    var codes: List[Int64] = [3, 0, -5, 4, 3, -1]
    var used = svc_used_classes(codes, 5)
    require(len(used.order) == 3, "used class count changed")
    require(used.order[0] == 3 and used.order[1] == 0 and used.order[2] == 4,
        "used classes not in first-appearance order")
    var expected: List[Int32] = [0, 1, 1, 2, 0, 2]
    for row in range(len(expected)):  # small-loop(six fixture rows)
        require(used.slots[row] == expected[row], "row slot differs at " + String(row))
    var class_words: List[UInt64] = [30, 0, 40]
    var words = svc_gather_words(class_words, used.slots)
    var gathered: List[UInt64] = [30, 0, 0, 40, 30, 40]
    for row in range(len(gathered)):  # small-loop(six fixture rows)
        require(words[row] == gathered[row], "gathered word differs at " + String(row))
    var empty = List[Int64]()
    var none = svc_used_classes(empty, 0)
    require(len(none.order) == 0 and len(none.slots) == 0, "empty codes produced used classes")


def used_classes_large() raises:
    # Enough rows to split across the host pool: the first appearance of
    # class 6 is late (row count - 1), so the cross-unit reduce must find it.
    var count = 100003
    var codes = List[Int64](length=count, fill=0)
    var cp = codes.unsafe_ptr()
    def fill(row: Int) {imm cp, imm count}:
        cp[row] = Int64(6 if row == count - 1 else (row % 3) * 2)
    host_parallelize(fill, count)
    var used = svc_used_classes(codes, 7)
    require(len(used.order) == 4, "large used class count changed")
    require(used.order[0] == 0 and used.order[1] == 2 and used.order[2] == 4 and used.order[3] == 6,
        "large first-appearance order changed")
    var class_words: List[UInt64] = [10, 12, 14, 16]
    var words = svc_gather_words(class_words, used.slots)
    var bad = List[Int](length=1, fill=0)
    var bp = bad.unsafe_ptr()
    var wp = words.unsafe_ptr()
    def check(row: Int) {imm cp, imm wp, imm bp}:
        if wp[row] != UInt64(10 + 2 * (Int(cp[row]) // 2)):
            bp[0] = 1
    host_parallelize(check, count)
    require(bad[0] == 0, "large gather differs from the class words")
    _ = codes
    _ = words


# ---- The host column (was svc_accuracy_native_cpu_suite.mojo) ----
def _run_cpu(mut report: Report) raises:
    var cases = report.cases(SVC_ACCURACY_NATIVE_CASES)
    def run(unit: Int) {imm cases}:
        try:
            fixture(LabelsNativeDriver(labels_host_native_v2), metrics_program_host_v1, unit)
        except problem:
            cases.fail(unit, problem)
    report.parallel(run, SVC_ACCURACY_NATIVE_CASES)


# ---- The device column (was svc_accuracy_native_gpu_suite.mojo) ----
def _run_gpu(mut report: Report) raises:
    var cases = report.cases(SVC_ACCURACY_NATIVE_CASES)
    for unit in range(SVC_ACCURACY_NATIVE_CASES):  # ordered shared-context numerical fixtures
        try:
            fixture(LabelsNativeDriver(labels_device_native_v2), metrics_program_device_v1, unit)
        except problem:
            cases.fail(unit, problem)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("svc_accuracy_native", cpu=True, gpu=True, serial="", run=run_cases)
