# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of a forest image's entry (lane calltable-forest,
2026-11-11; ABI core/forest_fit_request.mojo, table
core/forest_entry_table.mojo). No Python: the caller passes plain values and
keeps its buffers alive until the call returns. Each call reads the
published table (an acquire load and a schema compare) and calls the
pointer; a missing or mismatched publication refuses with the table's
error, and a refusal of the image comes back in the words it raised.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.forest_entry_table import ForestEntryKey, forest_table_entry
from core.forest_fit_request import (
    ForestFitArgs, FOREST_ENTRY_ABI, XF_OP_COUNT, XF_CAPS, XF_NUMERIC_MODE, XF_DT_DEFAULT_BINS, XF_FIT, XF_FIT_DART,
    XF_EXPORT_SIZES, XF_EXPORT_INTO, XF_EXPORT_VERIFY, XF_EXPORT_RELEASE, XF_PREDICT, XF_RESIDENT_PREPARE,
    XF_RESIDENT_PREDICT, XF_RESIDENT_LABELS, XF_RESIDENT_RELEASE, XF_STRICT_PREPARE, XF_STRICT_PREDICT,
    XF_STRICT_RELEASE, XF_META_MAX, XF_SHAPE_CLASSIFIER, XF_SHAPE_ROWMAJOR, forest_fit_words,
)


@fieldwise_init
struct ForestExport(Copyable, Movable):
    """An export handle's sizes and metadata words (XF_EXPORT_SIZES)."""
    var handle: Int
    var trees: Int
    var nodes: Int
    var outputs: Int
    var meta: List[Int]


def xf_entry(key: ForestEntryKey) raises -> NativeDescriptorCall:
    return forest_table_entry(key, 0, String(FOREST_ENTRY_ABI), XF_OP_COUNT)


def _none_f(n: Int = 0) -> List[Float64]:
    return List[Float64](length=n, fill=0.0)


def _call(key: ForestEntryKey, op: Int, addrs: List[Int], ints: List[Int]) raises -> Int:
    return native_descriptor_call(xf_entry(key), op, addrs, ints, _none_f(len(ints)))


def xf_caps(key: ForestEntryKey) raises -> Int:
    var none = List[Int]()
    return _call(key, XF_CAPS, none, none)


def xf_numeric_mode(key: ForestEntryKey) raises -> Int:
    var none = List[Int]()
    return _call(key, XF_NUMERIC_MODE, none, none)


def xf_dt_default_bins(key: ForestEntryKey) raises -> Int:
    """The former `rf_dt_default_bins` (128 in an image without one)."""
    var none = List[Int]()
    return _call(key, XF_DT_DEFAULT_BINS, none, none)


def _fit(key: ForestEntryKey, op: Int, shape: Int, args: ForestFitArgs) raises -> Int:
    var words = forest_fit_words(args, shape)
    var handle = native_descriptor_call(xf_entry(key), op, words, args.ints, args.floats)
    _ = args  # the views and flags stay alive through the call above
    return handle


def xf_fit(key: ForestEntryKey, classifier: Bool, row_major: Bool, args: ForestFitArgs) raises -> Int:
    """One fit in the image the key names; its export handle (the former
    `rf_*_fit*_export` / `et_*_fit*` entries)."""
    var shape = (XF_SHAPE_CLASSIFIER if classifier else 0) | (XF_SHAPE_ROWMAJOR if row_major else 0)
    return _fit(key, XF_FIT, shape, args)


def xf_fit_dart(key: ForestEntryKey, args: ForestFitArgs) raises -> Int:
    """A DART member fit on a DART session (the former
    `rf_regressor_fit_dart_export`)."""
    return _fit(key, XF_FIT_DART, 0, args)


def xf_export_sizes(key: ForestEntryKey, handle: Int) raises -> ForestExport:
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


def xf_export_into(key: ForestEntryKey, export: ForestExport, dst: List[Int]) raises:
    """The owned fit into the caller's five arrays (the former `forest_export`)."""
    var addrs: List[Int] = [export.handle, dst[0], dst[1], dst[2], dst[3], dst[4]]
    var ints: List[Int] = [export.trees, export.nodes, export.outputs]
    _ = _call(key, XF_EXPORT_INTO, addrs, ints)


def xf_export_verify(key: ForestEntryKey, handle: Int, dst: List[Int]) raises -> Int:
    """The former `forest_export_legacy` comparison, in the image."""
    var addrs: List[Int] = [handle, dst[0], dst[1], dst[2], dst[3], dst[4]]
    var none = List[Int]()
    return _call(key, XF_EXPORT_VERIFY, addrs, none)


def xf_export_release(key: ForestEntryKey, handle: Int) raises:
    var addrs: List[Int] = [handle]
    var none = List[Int]()
    _ = _call(key, XF_EXPORT_RELEASE, addrs, none)


def xf_predict(key: ForestEntryKey, model: List[Int], x: Int, out_addr: Int, rows: Int, cols: Int, trees: Int,
               outputs: Int, regression: Bool) raises -> Int:
    """The sequential vote (the former `rf_predict_proba` / `rf_predict_reg`
    / `et_predict`); rows written."""
    var addrs: List[Int] = [model[0], model[1], model[2], model[3], model[4], x, out_addr]
    var ints: List[Int] = [rows, cols, trees, outputs, 1 if regression else 0]
    return _call(key, XF_PREDICT, addrs, ints)


def xf_resident_prepare(key: ForestEntryKey, model: List[Int], features: Int, trees: Int, outputs: Int,
                        ordered: Int) raises -> Int:
    """The former `forest_prepare_gpu` (ordered -1: the image's default)."""
    var ints: List[Int] = [features, trees, outputs, ordered]
    return _call(key, XF_RESIDENT_PREPARE, model, ints)


def _query(key: ForestEntryKey, op: Int, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
           outputs: Int) raises -> Int:
    var addrs: List[Int] = [handle, x, out_addr]
    var ints: List[Int] = [rows, cols, outputs]
    return _call(key, op, addrs, ints)


def xf_resident_predict(key: ForestEntryKey, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                        outputs: Int) raises -> Int:
    """The former `forest_predict_resident_reuse_gpu`."""
    return _query(key, XF_RESIDENT_PREDICT, handle, x, out_addr, rows, cols, outputs)


def xf_resident_labels(key: ForestEntryKey, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                       outputs: Int) raises -> Int:
    """The former `forest_predict_resident_labels_gpu`."""
    return _query(key, XF_RESIDENT_LABELS, handle, x, out_addr, rows, cols, outputs)


def xf_strict_prepare(key: ForestEntryKey, model: List[Int], features: Int, trees: Int, outputs: Int) raises -> Int:
    """The former `forest_strict_prepare`."""
    var ints: List[Int] = [features, trees, outputs]
    return _call(key, XF_STRICT_PREPARE, model, ints)


def xf_strict_predict(key: ForestEntryKey, handle: Int, x: Int, out_addr: Int, rows: Int, cols: Int,
                      outputs: Int) raises -> Int:
    """The former `forest_strict_predict`."""
    return _query(key, XF_STRICT_PREDICT, handle, x, out_addr, rows, cols, outputs)


def xf_snapshot_release(key: ForestEntryKey, op: Int, handle: Int) raises:
    """XF_RESIDENT_RELEASE / XF_STRICT_RELEASE of a snapshot the call owns
    alone (a kept one is released by the resident table)."""
    var none = List[Int]()
    var ints: List[Int] = [handle]
    _ = _call(key, op, none, ints)
