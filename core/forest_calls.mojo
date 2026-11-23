# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of a forest image's entry (lane calltable-forest,
2026-11-11; ABI core/forest_fit_request.mojo, table
core/forest_entry_table.mojo). No Python: the caller passes plain values and
keeps its buffers alive until the call returns. Each call reads the
published table (an acquire load and a schema compare) and calls the
pointer; a missing or mismatched publication refuses with the table's
error, and a refusal of the image comes back in the words it raised.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_op, entry_ref_word
from core.forest_entry_table import ForestEntryKey, forest_table_entry
from core.forest_fit_request import (
    ForestFitArgs, FOREST_ENTRY_ABI, XF_OP_COUNT, XF_CAPS, XF_NUMERIC_MODE, XF_DT_DEFAULT_BINS, XF_FIT, XF_FIT_DART,
    XF_EXPORT_SIZES, XF_EXPORT_INTO, XF_EXPORT_VERIFY, XF_EXPORT_RELEASE, XF_PREDICT, XF_RESIDENT_PREPARE,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_RESIDENT_RELEASE, XF_STRICT_PREPARE, XF_STRICT_PREDICT,
    XF_STRICT_RELEASE, XF_SESSION_OPEN, XF_SESSION_CLOSE, XF_FIT_SESSION, XF_ADA_OPEN, XF_FIT_ADA_ROUND,
    XF_STAGE_INPUT, XF_UNSTAGE_INPUT, XF_META_MAX, XF_SHAPE_CLASSIFIER, XF_SHAPE_ROWMAJOR, forest_fit_words,
)


@fieldwise_init
struct ForestExport(Copyable, Movable):
    """An export handle's sizes and metadata words (XF_EXPORT_SIZES)."""
    var handle: Int
    var trees: Int
    var nodes: Int
    var outputs: Int
    var meta: List[Int]


def xf_ref(key: ForestEntryKey) raises -> EntryRef:
    """The forest entry of the image the key names, resolved once (lane
    entry-table-zero-alloc); the table refusals raise here. Every call below
    takes the resolved ref."""
    return EntryRef(forest_table_entry(key, 0, String(FOREST_ENTRY_ABI), XF_OP_COUNT),
                    entry_ref_word(key.key.mode, key.key.vendor, XF_OP_COUNT))


def _none_f(n: Int = 0) -> List[Float64]:
    return List[Float64](length=n, fill=0.0)


def _call(key: EntryRef, op: Int, addrs: List[Int], ints: List[Int]) raises -> Int:
    # `entry_ref_op`: a ref marked GIL-free (a member's worker) says so with
    # the operation (core/entry_table.mojo, lane fx-gil); the image then
    # runs its fits and walks with no release (`forest_entry_dispatch`). An
    # unmarked ref (every door) adds 0.
    return entry_ref_call(key, entry_ref_op(key, op), addrs, ints, _none_f(len(ints)))


def xf_caps(key: EntryRef) raises -> Int:
    var none = List[Int]()
    return _call(key, XF_CAPS, none, none)


def xf_numeric_mode(key: EntryRef) raises -> Int:
    var none = List[Int]()
    return _call(key, XF_NUMERIC_MODE, none, none)


def xf_dt_default_bins(key: EntryRef) raises -> Int:
    """The former `rf_dt_default_bins` (128 in an image without one)."""
    var none = List[Int]()
    return _call(key, XF_DT_DEFAULT_BINS, none, none)


def _fit(key: EntryRef, op: Int, shape: Int, args: ForestFitArgs) raises -> Int:
    var words = forest_fit_words(args, shape)
    var handle = entry_ref_call(key, entry_ref_op(key, op), words, args.ints, args.floats)
    _ = args  # the views and flags stay alive through the call above
    return handle


def xf_fit(key: EntryRef, classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
    """One fit in the image the key names; its export handle (the former
    `rf_*_fit*_export` / `et_*_fit*` entries)."""
    var shape = (XF_SHAPE_CLASSIFIER if classifier else 0) | (XF_SHAPE_ROWMAJOR if row_major else 0)
    return _fit(key, XF_FIT, shape, args)


def xf_fit_dart(key: EntryRef, args: ForestFitArgs) raises -> Int:
    """A DART member fit on a DART session (the former
    `rf_regressor_fit_dart_export`)."""
    return _fit(key, XF_FIT_DART, 0, args)


def xf_export_sizes(key: EntryRef, handle: Int) raises -> ForestExport:
    var out = List[Int](length=4 + XF_META_MAX, fill=0)
    var addrs: List[Int] = [handle, Int(out.unsafe_ptr())]
    var none = List[Int]()
    _ = _call(key, XF_EXPORT_SIZES, addrs, none)
    var meta = List[Int]()
    for k in range(out[3]):  # small-loop(meta: at most XF_META_MAX export metadata words)
        meta.append(out[4 + k])
    var sizes = ForestExport(handle, out[0], out[1], out[2], meta^)
    _ = out
    return sizes^


def xf_export_into(key: EntryRef, export: ForestExport, dst: List[Int]) raises:
    """The owned fit into the caller's five arrays (the former `forest_export`)."""
    var addrs: List[Int] = [export.handle, dst[0], dst[1], dst[2], dst[3], dst[4]]
    var ints: List[Int] = [export.trees, export.nodes, export.outputs]
    _ = _call(key, XF_EXPORT_INTO, addrs, ints)


def xf_export_verify(key: EntryRef, handle: Int, dst: List[Int]) raises -> Int:
    """The former `forest_export_legacy` comparison, in the image."""
    var addrs: List[Int] = [handle, dst[0], dst[1], dst[2], dst[3], dst[4]]
    var none = List[Int]()
    return _call(key, XF_EXPORT_VERIFY, addrs, none)


def xf_export_release(key: EntryRef, handle: Int) raises:
    var addrs: List[Int] = [handle]
    var none = List[Int]()
    _ = _call(key, XF_EXPORT_RELEASE, addrs, none)


def xf_predict(key: EntryRef, model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int,
               outputs: Int, regression: Bool) raises -> Int:
    """The sequential vote (the former `rf_predict_proba` / `rf_predict_reg`
    / `et_predict`); rows written."""
    var addrs: List[Int] = [model[0], model[1], model[2], model[3], model[4], x, out_addr]
    var ints: List[Int] = [rows, cols, trees, outputs, 1 if regression else 0]
    return _call(key, XF_PREDICT, addrs, ints)


def xf_resident_prepare(key: EntryRef, model: List[Int], features: Int, trees: Int, outputs: Int,
                        ordered: Int) raises -> Int:
    """The former `forest_prepare_gpu` (ordered -1: the image's default)."""
    var ints: List[Int] = [features, trees, outputs, ordered]
    return _call(key, XF_RESIDENT_PREPARE, model, ints)


def _query(key: EntryRef, op: Int, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
           outputs: Int) raises -> Int:
    var addrs: List[Int] = [handle, x, out_addr]
    var ints: List[Int] = [rows, cols, outputs]
    return _call(key, op, addrs, ints)


def xf_resident_predict(key: EntryRef, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                        outputs: Int) raises -> Int:
    """The former `forest_predict_resident_reuse_gpu`."""
    return _query(key, XF_RESIDENT_PREDICT, handle, x, out_addr, rows, cols, outputs)


def xf_resident_labels(key: EntryRef, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                       outputs: Int) raises -> Int:
    """The former `forest_predict_resident_labels_gpu`."""
    return _query(key, XF_RESIDENT_LABELS, handle, x, out_addr, rows, cols, outputs)


def xf_strict_prepare(key: EntryRef, model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
    """The former `forest_strict_prepare`."""
    var ints: List[Int] = [features, trees, outputs]
    return _call(key, XF_STRICT_PREPARE, model, ints)


def xf_strict_predict(key: EntryRef, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                      outputs: Int) raises -> Int:
    """The former `forest_strict_predict`."""
    return _query(key, XF_STRICT_PREDICT, handle, x, out_addr, rows, cols, outputs)


def xf_snapshot_release(key: EntryRef, op: Int, handle: Int) raises:
    """XF_RESIDENT_RELEASE / XF_STRICT_RELEASE of a snapshot the call owns
    alone (a kept one is released by the resident table)."""
    var none = List[Int]()
    var ints: List[Int] = [handle]
    _ = _call(key, op, none, ints)


def xf_stage_input(key: EntryRef, x: Int, rows: Int, cols: Int) raises -> Bool:
    """XF_STAGE_INPUT: stage the C-order float32 rows at `x` once as the
    image's shared resident input for the member predictions that follow on
    this thread (lane mf-meta3c). False: nothing staged (a host image,
    several devices, no live snapshot); the members upload as before. The
    caller keeps the rows alive and unchanged until `xf_unstage_input`."""
    var addrs: List[Int] = [x]
    var ints: List[Int] = [rows, cols]
    return _call(key, XF_STAGE_INPUT, addrs, ints) == 1


def xf_unstage_input(key: EntryRef):
    """XF_UNSTAGE_INPUT: end `xf_stage_input`'s staging. Never raises (a
    failed unstage is ignored, as the door's `trees_unstage` ignores one)."""
    var none = List[Int]()
    try:
        _ = _call(key, XF_UNSTAGE_INPUT, none, none)
    except:
        pass


def xf_session_open(key: EntryRef, x: Int, rows: Int, cols: Int, row_major: Bool, share_tables: Bool) raises -> Int:
    """The former `rf_data_session_open`: X (float32, row-major or
    column-major) kept in the image for the member fits of one ensemble; the
    session handle. An image without XF_CAP_SESSION refuses."""
    var addrs: List[Int] = [x]
    var ints: List[Int] = [rows, cols, 1 if row_major else 0, 1 if share_tables else 0]
    return _call(key, XF_SESSION_OPEN, addrs, ints)


def xf_session_close(key: EntryRef, handle: Int) raises:
    """The former `rf_data_session_close`."""
    var none = List[Int]()
    var ints: List[Int] = [handle]
    _ = _call(key, XF_SESSION_CLOSE, none, ints)


def xf_fit_session(key: EntryRef, classifier: Bool, args: ForestFitArgs) raises -> Int:
    """A member fit on the data session `args.session`; its export handle
    (the former `rf_classifier_fit_weighted_session_export`: y = class
    codes, `weights` the row weights; and `rf_regressor_fit_session_rows_export`:
    x = the int32 row ids, y = their targets)."""
    return _fit(key, XF_FIT_SESSION, XF_SHAPE_CLASSIFIER if classifier else 0, args)


def xf_ada_open(key: EntryRef, handle: Int, w: Int, y: Int, cw: Int, n: Int, k: Int) raises:
    """The former `rf_ada_open`: an AdaBoost fit's weights, labels or
    targets and class_weight rows kept on the session's device."""
    var addrs: List[Int] = [w, y, cw]
    var ints: List[Int] = [handle, n, k]
    _ = _call(key, XF_ADA_OPEN, addrs, ints)


def xf_fit_ada_round(key: EntryRef, classifier: Bool, args: ForestFitArgs) raises -> Int:
    """One AdaBoost round on the data session `args.session` (the former
    `rf_classifier_fit_ada_round_export` / `rf_regressor_fit_ada_round_export`):
    the member's export handle; the step's four words are written at
    `args.stats`."""
    return _fit(key, XF_FIT_ADA_ROUND, XF_SHAPE_CLASSIFIER if classifier else 0, args)
