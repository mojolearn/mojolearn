# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_inference_binding.mojo` (wave 8 run split): the
typed prepare and predict of the owned RF/ET GPU inference snapshots and the
resident capability bits. Imports no Python; the binding keeps the UNPACK
doors. Nothing here relies on the GIL: the registry has its own lock word and
a prediction holds its snapshot for the call (core/forest_inference_model.mojo),
so a handle released on another thread stays alive until that call returns."""
from core.hostptr import list_f32, list_i32
from core.forest_inference_model import resident_prepare, resident_predict_into, resident_predict_labels, FOREST_ORDERED_RESIDENT
from core.forest_inference_model import (
    resident_reads_device_rows, resident_predict_rows_into, resident_labels_rows_into,
)
from core.device_store import DeviceStore
from max.gpu.host import DeviceContext


def _i32_ptr(address: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Int32 pointer")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)


def _f32_ptr(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Float32 pointer")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def forest_prepare_gpu_native[RF_INPUT: Bool](model: List[Int], features: Int, trees: Int, outputs: Int,
                                             ordered_word: Int) raises -> Int:
    """Prepare an owned immutable GPU snapshot (XF_RESIDENT_PREPARE; lane
    calltable-forest). `model` holds the five array addresses. The registry
    insert takes the registry's own lock word.

    `ordered_word` picks the snapshot's aggregation: 1 the strict
    increasing-tree kernel (the sequential route's bits), 0 the 32-grove
    graph, -1 the compiled default `FOREST_ORDERED_RESIDENT`."""
    var ordered = FOREST_ORDERED_RESIDENT
    if ordered_word != -1:
        if ordered_word != 0 and ordered_word != 1:
            raise Error("forest_prepare_gpu ordered must be 0 or 1")
        ordered = ordered_word == 1
    if features < 1 or trees < 1 or trees >= 2147483647 or outputs < 1:
        raise Error("invalid resident forest dimensions")
    var op = _i32_ptr(model[0])
    var cp = _i32_ptr(model[1])
    var tp = _f32_ptr(model[2])
    var lp = _i32_ptr(model[3])
    var vp = _f32_ptr(model[4])
    var nodes = Int(op[trees])
    if nodes < 1 or nodes > 2147483647 // outputs:
        raise Error("invalid resident forest node count")
    # one memcpy per array (no per-node host loop); the snapshot uploads once
    var offsets = list_i32(op, trees + 1)
    var columns = list_i32(cp, nodes)
    var thresholds = list_f32(tp, nodes)
    var left = list_i32(lp, nodes)
    var leaves = list_f32(vp, nodes * outputs)
    return resident_prepare[RF_INPUT](offsets, columns, thresholds, left, leaves, features, outputs, ordered)


def forest_predict_resident_into_gpu_native[RF_INPUT: Bool, REUSE_IO: Bool = False](
    handle: Int, x_addr: Int, out_addr: Int, rows: Int, features: Int, outputs: Int) raises -> Int:
    """Borrowed synchronous prediction (XF_RESIDENT_PREDICT with
    REUSE_IO=True, the public `forest_predict_resident_reuse_gpu`).

    DEVIATION 2483: parallel_groves already selects this pointer-through body
    through forest_predict_resident_reuse_gpu. Output may change on a
    nonfinite-result error; caller ownership lasts until return.
    """
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid resident prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident prediction dimensions exceed Int32")
    var xp = _f32_ptr(x_addr)
    var op = _f32_ptr(out_addr)
    resident_predict_into[RF_INPUT](handle,
        xp.unsafe_origin_cast[MutAnyOrigin](), op.unsafe_origin_cast[MutAnyOrigin](),
        rows, features, outputs, REUSE_IO)
    return rows


def forest_predict_resident_labels_gpu_native[RF_INPUT: Bool](
    handle: Int, x_addr: Int, out_addr: Int, rows: Int, features: Int, outputs: Int) raises -> Int:
    """XF_RESIDENT_LABELS: the resident classifier's int32 codes."""
    if rows < 0 or features < 1 or outputs < 2:
        raise Error("invalid resident label prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident label prediction dimensions exceed Int32")
    var xp = _f32_ptr(x_addr)
    var op = _i32_ptr(out_addr)
    resident_predict_labels[RF_INPUT](handle,
        xp.unsafe_origin_cast[MutAnyOrigin](), op.unsafe_origin_cast[MutAnyOrigin](),
        rows, features, outputs)
    return rows


def forest_resident_rows_native[RF_INPUT: Bool](mut store: DeviceStore, ctx: DeviceContext, labels: Bool,
    handle: Int, store_id: Int, idx_addr: Int, out_addr: Int, rows: Int, features: Int, outputs: Int) raises -> Int:
    """XF_RESIDENT_ROWS (lane vdf-scope-trees2a): the resident vote
    (float32 at `out_addr`) or, with `labels`, the int32 codes of rows
    `idx_addr` (`rows` host int64 words) of the image's device store slot
    `store_id` (row-major, `features` words a row). The rows are taken on
    the device (`DeviceStore.take_rows`, on `ctx`: the image's one context,
    which is the snapshot's) and the snapshot reads them there: no upload of
    X, the same scans, traversal and one read of the output as
    XF_RESIDENT_PREDICT / XF_RESIDENT_LABELS on the host-gathered rows, so
    the same bytes. A snapshot on several devices (a device pool) reads
    the taken rows through the host, then predicts as XF_RESIDENT_*: the
    one host read left, outside the one-device route."""
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid resident prediction dimensions")
    if labels and outputs < 2:
        raise Error("invalid resident label prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident prediction dimensions exceed Int32")
    if rows == 0:
        return 0
    var taken = store.take_rows(ctx, store_id, features, idx_addr, rows)
    try:
        if resident_reads_device_rows[RF_INPUT](handle):
            var dx = store._held(taken, rows * features)
            if labels:
                resident_labels_rows_into[RF_INPUT](handle, dx,
                    _i32_ptr(out_addr).unsafe_origin_cast[MutAnyOrigin](), rows, features, outputs)
            else:
                resident_predict_rows_into[RF_INPUT](handle, dx,
                    _f32_ptr(out_addr).unsafe_origin_cast[MutAnyOrigin](), rows, features, outputs)
            _ = dx^  # the prediction above read the taken rows (synchronous)
        else:
            var host = List[Float32](length=rows * features, fill=0.0)
            store.read(ctx, taken, Int(host.unsafe_ptr()), rows * features)
            var xp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(host.unsafe_ptr()))
            if labels:
                resident_predict_labels[RF_INPUT](handle, xp,
                    _i32_ptr(out_addr).unsafe_origin_cast[MutAnyOrigin](), rows, features, outputs)
            else:
                resident_predict_into[RF_INPUT](handle, xp,
                    _f32_ptr(out_addr).unsafe_origin_cast[MutAnyOrigin](), rows, features, outputs, True)
            _ = len(host)  # the prediction above read the rows
    except problem:
        store.free(ctx, taken)
        raise problem^
    store.free(ctx, taken)
    return rows


def forest_gpu_resident_caps() -> Int:
    """The resident capability bits both GPU forest images report through
    XF_CAPS (core/forest_fit_request.mojo; lane calltable-forest): the
    ordered word (`forest_ordered_resident`), the resident label query
    (`forest_predict_resident_labels_gpu`)."""
    from core.forest_fit_request import XF_CAP_ORDERED_RESIDENT, XF_CAP_RESIDENT_LABELS
    # TOMBSTONE: MOJOLEARN_TREES_T38_FUSED_LABELS (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 1caa34cfb3.
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_T38_FUSED_LABELS.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
    return XF_CAP_ORDERED_RESIDENT | XF_CAP_RESIDENT_LABELS
