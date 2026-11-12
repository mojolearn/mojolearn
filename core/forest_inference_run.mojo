# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_inference_binding.mojo` (wave 8 run split): the
typed prepare and predict of the owned RF/ET GPU inference snapshots and the
resident capability bits. Imports no Python; the binding keeps the UNPACK
doors. The calling extension holds the GIL throughout prepare, predict and
release, so no native handle is released while another call uses it."""
from core.hostptr import list_f32, list_i32
from core.forest_inference_model import resident_prepare, resident_predict_into, resident_predict_labels, FOREST_ORDERED_RESIDENT


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
    calltable-forest). `model` holds the five array addresses. The caller
    holds the GIL, which serializes registry access.

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


def forest_gpu_resident_caps() -> Int:
    """The resident capability bits both GPU forest images report through
    XF_CAPS (core/forest_fit_request.mojo; lane calltable-forest): the
    ordered word (`forest_ordered_resident`), the resident label query
    (`forest_predict_resident_labels_gpu`) and, when compiled, the T38 fused
    labels (`forest_identical_fused_labels`)."""
    from core.forest_experiments import T38_FUSED_LABELS
    from core.forest_fit_request import XF_CAP_ORDERED_RESIDENT, XF_CAP_RESIDENT_LABELS, XF_CAP_FUSED_LABELS
    var caps = XF_CAP_ORDERED_RESIDENT | XF_CAP_RESIDENT_LABELS
    comptime if T38_FUSED_LABELS:
        caps |= XF_CAP_FUSED_LABELS
    return caps
