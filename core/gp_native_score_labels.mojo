# SPDX-License-Identifier: Apache-2.0
"""GP metric label codes and sorted union over existing native encoders."""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.labels_score_encode import score_encode_signed
from core.labels_score_keys import ScoreLabelKey, ScoreKeyIndex, score_integer_keys, score_key_join, score_key_codes
from core.metrics_exp_impl import mexp_gather_i32


@fieldwise_init
struct GPScoreLabelCodes(Movable):
    var rows: Int
    var classes: List[ScoreLabelKey]
    var codes: List[Int32]


def gp_score_integer_labels(source: Int, rows: Int) raises -> GPScoreLabelCodes:
    var values = List[Int64](length=max(rows, 1), fill=0)
    var codes = List[Int32](length=rows, fill=0)
    var count = score_encode_signed(source, 8, rows, Int(values.unsafe_ptr()), Int(codes.unsafe_ptr()))
    var keys = score_integer_keys(Int(values.unsafe_ptr()), count)
    _ = values
    return GPScoreLabelCodes(rows, keys^, codes^)


def gp_score_key_labels(keys: List[ScoreLabelKey]) -> GPScoreLabelCodes:
    var rows = len(keys)
    var index = ScoreKeyIndex(Int(keys.unsafe_ptr()), rows)
    var codes = List[Int32](length=rows, fill=0)
    score_key_codes(index, keys, Int(codes.unsafe_ptr()))
    var count = len(index.sorted_rows)
    var classes = List[ScoreLabelKey](unsafe_uninit_length=count)
    var source = Pointer[ScoreLabelKey, ImmUntrackedOrigin](unsafe_from_address=Int(keys.unsafe_ptr()))
    var selected = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=Int(index.sorted_rows.unsafe_ptr()))
    var dst = MutPointer[ScoreLabelKey, MutUntrackedOrigin](unsafe_from_address=Int(classes.unsafe_ptr()))
    var units = host_predict_task_count_work(max(1, count), max(1, count))
    var chunk = host_predict_chunk(max(1, count), units)
    def task(unit: Int) {imm source, imm selected, imm dst, imm count, imm chunk}:
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):
            (dst + i).unsafe_write(source[selected[i]].copy())
    host_parallelize(task, units)
    _ = keys
    _ = index
    return GPScoreLabelCodes(rows, classes^, codes^)


@fieldwise_init
struct GPScorePair(Movable):
    var rows: Int
    var classes: Int
    var truth: List[Int32]
    var prediction: List[Int32]


def _gp_score_remap(labels: GPScoreLabelCodes, index: ScoreKeyIndex) -> List[Int32]:
    var table = List[Int32](length=len(labels.classes), fill=0)
    score_key_codes(index, labels.classes, Int(table.unsafe_ptr()))
    var result = List[Int32](length=labels.rows, fill=0)
    # Retain the metrics owner's existing gather implementation and route.
    _ = mexp_gather_i32(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(table.unsafe_ptr())), len(table),
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(labels.codes.unsafe_ptr())), labels.rows,
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(result.unsafe_ptr())))
    _ = table
    _ = labels
    _ = index
    return result^


def gp_score_pair(truth: GPScoreLabelCodes, prediction: GPScoreLabelCodes) -> GPScorePair:
    var joined = score_key_join(truth.classes, prediction.classes)
    var index = ScoreKeyIndex(Int(joined.unsafe_ptr()), len(joined))
    var true_codes = _gp_score_remap(truth, index)
    var pred_codes = _gp_score_remap(prediction, index)
    var classes = len(index.sorted_rows)
    _ = index
    _ = joined
    _ = truth
    _ = prediction
    return GPScorePair(truth.rows, classes, true_codes^, pred_codes^)
