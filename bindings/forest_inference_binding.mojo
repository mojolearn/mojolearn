# SPDX-License-Identifier: Apache-2.0
"""Shared borrowed-pointer boundary for owned RF/ET GPU inference snapshots.

The calling extension holds the GIL throughout prepare/predict/release, so no
native handle can be released while another call is using it.
"""
from std.python import PythonObject
from core.hostptr import list_f32, list_i32
from core.forest_inference import FOREST_PACKED_NODES
from core.forest_inference_model import resident_prepare, resident_predict, resident_release, resident_predict_into, resident_predict_labels, resident_stage_input, resident_unstage_input, FOREST_ORDERED_RESIDENT


# TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


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


def _model_py(offsets_addr: PythonObject, colid_addr: PythonObject, quesval_addr: PythonObject,
              left_child_addr: PythonObject, leaves_addr: PythonObject) raises -> List[Int]:
    """The five array address words (UNPACK)."""
    var model: List[Int] = [Int(py=offsets_addr), Int(py=colid_addr), Int(py=quesval_addr),
                            Int(py=left_child_addr), Int(py=leaves_addr)]
    return model^


def forest_prepare_gpu_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu`: UNPACK of `forest_prepare_gpu_native`; an
    optional fourth params entry is the ordered word."""
    if len(params) != 3 and len(params) != 4:
        raise Error("forest_prepare_gpu requires features, trees, outputs[, ordered]")
    var ordered = Int(py=params[3]) if len(params) == 4 else -1
    if len(params) == 4 and ordered != 0 and ordered != 1:
        raise Error("forest_prepare_gpu ordered must be 0 or 1")
    return PythonObject(forest_prepare_gpu_native[RF_INPUT](
        _model_py(offsets_addr, colid_addr, quesval_addr, left_child_addr, leaves_addr),
        Int(py=params[0]), Int(py=params[1]), Int(py=params[2]), ordered))


def forest_release_gpu_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    resident_release[RF_INPUT](Int(py=handle))
    return PythonObject(None)


def forest_ordered_resident_binding() raises -> PythonObject:
    """Read the compiled experimental resident aggregation route."""
    return PythonObject(1 if FOREST_ORDERED_RESIDENT else 0)


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


def forest_predict_resident_into_gpu_binding[RF_INPUT: Bool, REUSE_IO: Bool = False](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject) raises -> PythonObject:
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    return PythonObject(forest_predict_resident_into_gpu_native[RF_INPUT, REUSE_IO](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


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


def forest_predict_resident_labels_gpu_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject) raises -> PythonObject:
    if len(params) != 3:
        raise Error("resident label prediction requires rows, features, outputs")
    return PythonObject(forest_predict_resident_labels_gpu_native[RF_INPUT](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


def forest_stage_input_gpu_binding[RF_INPUT: Bool](
    x_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """`forest_stage_input_gpu(x_addr, [rows, features])`: upload the
    borrowed C-order float32 rows once as this registry's shared input, for
    the members of one ensemble call (lane perf-sweep-trees, 2026-11-05).
    Returns 1 when staged, 0 when not (each member then uploads as before).
    The caller keeps the rows alive and unchanged until it calls
    `forest_unstage_input_gpu`."""
    if len(params) != 2:
        raise Error("input staging requires rows, features")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    if rows < 1 or features < 1 or rows > 2147483647 // features:
        return PythonObject(0)
    var xp = _f32_ptr(Int(py=x_addr))
    return PythonObject(1 if resident_stage_input[RF_INPUT](xp.unsafe_origin_cast[MutAnyOrigin](), rows * features) else 0)


def forest_unstage_input_gpu_binding[RF_INPUT: Bool]() raises -> PythonObject:
    """`forest_unstage_input_gpu()`: end the staged input. Returns 0."""
    resident_unstage_input[RF_INPUT]()
    return PythonObject(0)


def forest_resident_layout_binding() raises -> PythonObject:
    comptime if FOREST_PACKED_NODES:
        return PythonObject("packed_siblings")
    return PythonObject("separate_arrays")


def forest_identical_fused_labels_binding() raises -> PythonObject:
    """T38 capability; absent define preserves incumbent classification."""
    from core.forest_experiments import T38_FUSED_LABELS
    return PythonObject(T38_FUSED_LABELS)


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
