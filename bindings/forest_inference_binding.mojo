# SPDX-License-Identifier: Apache-2.0
"""Shared borrowed-pointer boundary for owned RF/ET GPU inference snapshots.

The calling extension holds the GIL throughout prepare/predict/release, so no
native handle can be released while another call is using it.
"""
from std.python import PythonObject
from core.forest_inference import FOREST_PACKED_NODES
from core.forest_inference_model import resident_release, resident_stage_input, resident_unstage_input, FOREST_ORDERED_RESIDENT
from core.forest_inference_run import (
    _f32_ptr,
    forest_prepare_gpu_native,
    forest_predict_resident_into_gpu_native,
    forest_predict_resident_labels_gpu_native,
)


# TOMBSTONE: the legacy-arm rf fits, rf_regressor_fit_session_export, the forest comparison arms and probes, rf_*_gpu_parallel, rf_fused_bootstrap_gather (dead native entries: no live caller) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_trees.patch


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


def forest_predict_resident_into_gpu_binding[RF_INPUT: Bool, REUSE_IO: Bool = False](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject) raises -> PythonObject:
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    return PythonObject(forest_predict_resident_into_gpu_native[RF_INPUT, REUSE_IO](
        Int(py=handle), Int(py=x_addr), Int(py=out_addr), Int(py=params[0]), Int(py=params[1]), Int(py=params[2])))


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


