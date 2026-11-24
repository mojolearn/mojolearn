# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of a gbdt image's entry (lane calltable-gbdt-forest-dart,
2026-11-11; ABI gbdt/fit_request.mojo, table core/gbdt_entry_table.mojo).
No Python: the caller passes plain values and keeps its buffers alive until
the call returns. The caller resolves the table once per fit (`xg_ref`; a
missing or mismatched publication refuses there with the table's error)
and every call is one indirect call through the `EntryRef` (lane
entry-table-zero-alloc); a refusal of the fit comes back in the words the
fit raised.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_op, entry_ref_word
from core.trees_entry_table import TreesEntryKey
from core.gbdt_entry_table import gbdt_table_entry
from core.fit_loop_clock import FitLoop
from gbdt.fit_request import (
    GbdtFitArgs, GbdtFitOutcome, GBDT_ENTRY_ABI, XG_OP_COUNT, XG_CAPS, XG_NUMERIC_MODE, XG_FIT, XG_FIT_ROWMAJOR,
    XG_OUTCOME_SIZES, XG_OUTCOME_COPY, XG_OUTCOME_FREE, XG_MODEL_DIM, XG_CAP_ROW_MAJOR, gbdt_string_at,
    XG_PREDICT, XG_PREDICT_MULTI, XG_BINARY, XG_SIGMOID_PAIR, XG_RESIDENT_PREPARE, XG_RESIDENT_PREDICT,
    XG_CAP_RESIDENT, XG_ORDERED_FIT, XG_FEATURE_FREQ_FIT, XG_IMPORTANCES, XG_OUTCOME_WORDS,
)


def xg_ref(key: TreesEntryKey) raises -> EntryRef:
    """The gbdt entry of the image the key names, resolved once."""
    return EntryRef(gbdt_table_entry(key, 0, String(GBDT_ENTRY_ABI), XG_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, XG_OP_COUNT))


def _none_f() -> List[Float64]:
    return List[Float64]()


def xg_row_major(entry: EntryRef) raises -> Bool:
    """The former `gbdt_fit_row_major_available` probe (absent = False)."""
    var none = List[Int]()
    return (entry_ref_call(entry, entry_ref_op(entry, XG_CAPS), none, none, _none_f()) & XG_CAP_ROW_MAJOR) != 0


def xg_numeric_mode(entry: EntryRef) raises -> Int:
    """The image's compiled tier (the former `gbdt_numeric_mode`)."""
    var none = List[Int]()
    return entry_ref_call(entry, entry_ref_op(entry, XG_NUMERIC_MODE), none, none, _none_f())


def xg_model_dim(entry: EntryRef, text: String) raises -> Int:
    """The model's approx dimension (the former `gbdt_model_dim`)."""
    var addrs: List[Int] = [Int(text.unsafe_ptr())]
    var ints: List[Int] = [text.byte_length()]
    var floats: List[Float64] = [Float64(text.byte_length())]
    var dim = entry_ref_call(entry, entry_ref_op(entry, XG_MODEL_DIM), addrs, ints, floats)
    _ = text
    return dim


# ---- The predict operations (lane dw-trees-boost, 2026-11-21): plain words
# on this frame (no heap), one indirect call each. `text` / `text_bytes` are
# a UTF-8 span the caller keeps alive (the model text is copied once, by the
# image, as its Python entry copied it). The image releases the GIL around
# its work.


def xg_resident(entry: EntryRef) raises -> Bool:
    """Whether the image holds device-resident parsed models (the former
    `gbdt_resident_prepare` attribute probe)."""
    var none = List[Int]()
    return (entry_ref_call(entry, entry_ref_op(entry, XG_CAPS), none, none, _none_f()) & XG_CAP_RESIDENT) != 0


def xg_predict(entry: EntryRef, text: Int, text_bytes: Int, x: Int, n_rows: Int, dst: Int) raises -> Int:
    """Raw approxes of a one-dimensional model; the rows written."""
    var a: Array[Int, 3] = [text, x, dst]
    var p: Array[Int, 2] = [text_bytes, n_rows]
    var f: Array[Float64, 2] = [0.0, 0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_PREDICT), a, p, f)


def xg_predict_multi(entry: EntryRef, text: Int, text_bytes: Int, x: Int, n_rows: Int, dst: Int,
                     mode: Int) raises -> Int:
    """A multi-dimensional model in `mode`; the width written per row."""
    var a: Array[Int, 3] = [text, x, dst]
    var p: Array[Int, 3] = [text_bytes, n_rows, mode]
    var f: Array[Float64, 3] = [0.0, 0.0, 0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_PREDICT_MULTI), a, p, f)


def xg_binary(entry: EntryRef, probabilities: Bool, raw: Int, dst: Int, n: Int) raises -> Int:
    """The adapters' binary transform of n float32 margins; the values
    written (2 n probabilities or n class codes)."""
    var a: Array[Int, 2] = [raw, dst]
    var p: Array[Int, 2] = [n, 1 if probabilities else 0]
    var f: Array[Float64, 2] = [0.0, 0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_BINARY), a, p, f)


def xg_sigmoid_pair(entry: EntryRef, raw: Int, dst: Int, n: Int) raises -> Int:
    """Float64 raw scores to the float64 pair `[1 - p, p]`; n."""
    var a: Array[Int, 2] = [raw, dst]
    var p: Array[Int, 1] = [n]
    var f: Array[Float64, 1] = [0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_SIGMOID_PAIR), a, p, f)


def xg_resident_prepare(entry: EntryRef, text: Int, text_bytes: Int) raises -> Int:
    """Parse, pack and upload the model once; the image's handle."""
    var a: Array[Int, 1] = [text]
    var p: Array[Int, 1] = [text_bytes]
    var f: Array[Float64, 1] = [0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_RESIDENT_PREPARE), a, p, f)


def xg_resident_predict(entry: EntryRef, handle: Int, x: Int, n_rows: Int, dst: Int, mode: Int,
                        row_major: Bool) raises -> Int:
    """Apply a prepared model; the width written per row."""
    var a: Array[Int, 2] = [x, dst]
    var p: Array[Int, 4] = [handle, n_rows, mode, 1 if row_major else 0]
    var f: Array[Float64, 4] = [0.0, 0.0, 0.0, 0.0]
    return entry_ref_call(entry, entry_ref_op(entry, XG_RESIDENT_PREDICT), a, p, f)


def xg_importances(entry: EntryRef, text: String, x: Int, n_rows: Int, n_input: Int, row_major: Bool,
                   weights: Int, n_weights: Int) raises -> List[Float32]:
    """`feature_importances_` of the model `text` fitted on the `n_rows` x
    `n_input` float32 rows at `x` (C order when `row_major`; the fit's own
    block, categorical columns as its dense codes) with `n_weights` (0 or
    n_rows) float32 sample weights at `weights`: n_input float32 (lane
    mf-imp6; the image computes it on its own column). The caller keeps its
    buffers alive until this returns."""
    var dst = List[Float32](length=max(n_input, 1), fill=Float32(0.0))
    var a: Array[Int, 4] = [Int(text.unsafe_ptr()), x, weights, Int(dst.unsafe_ptr())]
    var p: Array[Int, 5] = [text.byte_length(), n_rows, n_input, 1 if row_major else 0, n_weights]
    var f: Array[Float64, 5] = [0.0, 0.0, 0.0, 0.0, 0.0]
    var wrote = entry_ref_call(entry, entry_ref_op(entry, XG_IMPORTANCES), a, p, f)
    _ = text  # the image read the text through its address
    if wrote != n_input:
        raise Error("gbdt importances: the image wrote " + String(wrote) + " values for " + String(n_input)
                    + " columns")
    dst.resize(n_input, Float32(0.0))
    return dst^


def _xg_outcome_text(entry: EntryRef, handle: Int) raises -> String:
    """The model text of a text-only fit's outcome, copied once from the
    image's String, the outcome freed in that image."""
    var sizes = List[Int](length=XG_OUTCOME_WORDS, fill=0)
    var outcome_addrs: List[Int] = [handle, Int(sizes.unsafe_ptr())]
    var none = List[Int]()
    try:
        _ = entry_ref_call(entry, entry_ref_op(entry, XG_OUTCOME_SIZES), outcome_addrs, none, _none_f())
        return gbdt_string_at(sizes[5], sizes[0])
    finally:
        var free_addrs: List[Int] = [handle]
        _ = entry_ref_call(entry, entry_ref_op(entry, XG_OUTCOME_FREE), free_addrs, none, _none_f())
        _ = sizes


def xg_ordered_fit(entry: EntryRef, x: Int, y: Int, weights: Int, permutation: Int, n_rows: Int, n_features: Int,
                   n_weights: Int, n_estimators: Int, max_depth: Int, border_count: Int, learning_rate: Float64,
                   l2_leaf_reg: Float64) raises -> String:
    """The OrderedRMSE fit (the former `gbdt_fit_ordered_rmse` entry): the
    model text. The permutation holds one UInt32 row id per row."""
    var a: Array[Int, 4] = [x, y, weights, permutation]
    var p: Array[Int, 9] = [n_rows, n_features, n_weights, n_rows, n_estimators, max_depth, border_count, 0, 0]
    var f: Array[Float64, 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, learning_rate, l2_leaf_reg]
    return _xg_outcome_text(entry, entry_ref_call(entry, entry_ref_op(entry, XG_ORDERED_FIT), a, p, f))


def xg_feature_freq_fit(entry: EntryRef, x: Int, y: Int, weights: Int, sources: Int, n_rows: Int, n_features: Int,
                        n_weights: Int, n_sources: Int, learning_rate: Float64, l2_leaf_reg: Float64,
                        random_seed: Int) raises -> String:
    """The two-level FeatureFreq fit (the former
    `gbdt_fit_two_level_feature_freq` entry): the model text."""
    var a: Array[Int, 4] = [x, y, weights, sources]
    var p: Array[Int, 7] = [n_rows, n_features, n_weights, n_sources, 0, 0, random_seed]
    var f: Array[Float64, 7] = [0.0, 0.0, 0.0, 0.0, learning_rate, l2_leaf_reg, 0.0]
    return _xg_outcome_text(entry, entry_ref_call(entry, entry_ref_op(entry, XG_FEATURE_FREQ_FIT), a, p, f))


def xg_fit(entry: EntryRef, row_major: Bool, args: GbdtFitArgs) raises -> GbdtFitOutcome:
    """The fit in the image the key names, its outcome copied out and freed
    in that image (the former `gbdt_fit` / `gbdt_fit_rowmajor` entries)."""
    var addrs: List[Int] = [args.x, args.y, args.weights, args.cat_flags, args.eval_x, args.eval_y,
                            Int(args.bad.unsafe_ptr()) if len(args.bad) else 0]
    for k in range(len(args.strs)):  # small-loop(strs: four to eight option names)
        addrs.append(Int(args.strs[k].unsafe_ptr()))
        addrs.append(args.strs[k].byte_length())
    var handle = entry_ref_call(entry, entry_ref_op(entry, XG_FIT_ROWMAJOR if row_major else XG_FIT), addrs, args.ints,
                                        args.floats)
    _ = args  # the strings, views and flags stay alive through the call above
    var sizes = List[Int](length=XG_OUTCOME_WORDS, fill=0)
    var outcome_addrs: List[Int] = [handle, Int(sizes.unsafe_ptr())]
    var none = List[Int]()
    try:
        _ = entry_ref_call(entry, entry_ref_op(entry, XG_OUTCOME_SIZES), outcome_addrs, none, _none_f())
        # The model text is copied ONCE, straight from the image's outcome
        # String (sizes[5], valid until XG_OUTCOME_FREE below) into ours; the
        # former path copied it into a byte List here, then into a second
        # List and into the String in `gbdt_string_at` (round2-fixes,
        # 2026-11-11; perf audit 2 A pattern 7).
        var text = gbdt_string_at(sizes[5], sizes[0])
        var learn = List[Float64](length=sizes[3], fill=0.0)
        var test = List[Float64](length=sizes[4], fill=0.0)
        var copy_addrs: List[Int] = [handle, 0, Int(learn.unsafe_ptr()), Int(test.unsafe_ptr())]
        _ = entry_ref_call(entry, entry_ref_op(entry, XG_OUTCOME_COPY), copy_addrs, none, _none_f())
        var outcome = GbdtFitOutcome(text^, sizes[1], sizes[2] != 0, learn^, test^)
        # the fit's boosting loop (`fit_loop_`, lane ab-loop-mode): three words
        outcome.fit_loop = FitLoop(sizes[6], sizes[7], sizes[8] != 0)
        return outcome^
    finally:
        var free_addrs: List[Int] = [handle]
        _ = entry_ref_call(entry, entry_ref_op(entry, XG_OUTCOME_FREE), free_addrs, none, _none_f())
        _ = sizes
