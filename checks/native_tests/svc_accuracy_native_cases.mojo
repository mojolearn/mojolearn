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
from core.svc_score_codes import svc_score_codes_valid
from core.native_descriptor_abi import NativeDescriptorCall

comptime SVC_ACCURACY_NATIVE_CASES = 5


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
    else:
        var high: List[Int64] = [0, 3]
        var low: List[Int64] = [-4, 0]
        var no_classes: List[Int64] = [0]
        require(not svc_score_codes_valid(high, 3), "upper code bound accepted")
        require(not svc_score_codes_valid(low, 3), "lower code bound accepted")
        require(not svc_score_codes_valid(no_classes, 0), "code accepted without classes")
