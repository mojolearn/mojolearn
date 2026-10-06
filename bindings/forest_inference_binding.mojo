# SPDX-License-Identifier: Apache-2.0
"""Shared borrowed-pointer boundary for owned RF/ET GPU inference snapshots.

The calling extension holds the GIL throughout prepare/predict/release, so no
native handle can be released while another call is using it.
"""
from std.python import PythonObject
from std.sys.compile import is_defined
from hostptr import list_f32, list_i32
from core.forest_inference import vector_groves_for, FOREST_PACKED_NODES
from core.forest_inference_model import resident_prepare, resident_predict, resident_release, resident_predict_into, resident_predict_labels, FOREST_ORDERED_RESIDENT


def forest_pool_available() raises -> PythonObject:
    return PythonObject(1)


def forest_pool_fault_available() raises -> PythonObject:
    return PythonObject(1 if is_defined["MOJOLEARN_FOREST_POOL_FAULT"]() else 0)


def _i32_ptr(address: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Int32 pointer")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)


def _f32_ptr(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Float32 pointer")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def forest_prepare_gpu_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """Prepare an owned immutable GPU snapshot. GIL serializes registry access.

    An optional fourth entry picks the snapshot's aggregation: 1 the strict
    increasing-tree kernel (the sequential route's bits), 0 the 32-grove
    graph. Absent, the compiled default `FOREST_ORDERED_RESIDENT` applies."""
    if len(params) != 3 and len(params) != 4:
        raise Error("forest_prepare_gpu requires features, trees, outputs[, ordered]")
    var features = Int(py=params[0])
    var trees = Int(py=params[1])
    var outputs = Int(py=params[2])
    var ordered = FOREST_ORDERED_RESIDENT
    if len(params) == 4:
        var flag = Int(py=params[3])
        if flag != 0 and flag != 1:
            raise Error("forest_prepare_gpu ordered must be 0 or 1")
        ordered = flag == 1
    if features < 1 or trees < 1 or trees >= 2147483647 or outputs < 1:
        raise Error("invalid resident forest dimensions")
    var op = _i32_ptr(Int(py=offsets_addr))
    var cp = _i32_ptr(Int(py=colid_addr))
    var tp = _f32_ptr(Int(py=quesval_addr))
    var lp = _i32_ptr(Int(py=left_child_addr))
    var vp = _f32_ptr(Int(py=leaves_addr))
    var nodes = Int(op[trees])
    if nodes < 1 or nodes > 2147483647 // outputs:
        raise Error("invalid resident forest node count")
    # one memcpy per array (no per-node host loop); the snapshot uploads once
    var offsets = list_i32(op, trees + 1)
    var columns = list_i32(cp, nodes)
    var thresholds = list_f32(tp, nodes)
    var left = list_i32(lp, nodes)
    var leaves = list_f32(vp, nodes * outputs)
    return PythonObject(resident_prepare[RF_INPUT](
        offsets, columns, thresholds, left, leaves, features, outputs, ordered))


def forest_predict_resident_gpu_binding[RF_INPUT: Bool](handle: PythonObject, x_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    var outputs = Int(py=params[2])
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid resident prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident prediction dimensions exceed Int32")
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    # borrowed rows go straight to the device and the result straight back
    # into the borrowed output (no host List copies of X or of the result);
    # same kernel and fold as the List route, fresh per-call buffers
    resident_predict_into[RF_INPUT](Int(py=handle),
        xp.unsafe_origin_cast[MutAnyOrigin](), op.unsafe_origin_cast[MutAnyOrigin](),
        rows, features, outputs, False)
    return PythonObject(rows)


def forest_release_gpu_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    resident_release[RF_INPUT](Int(py=handle))
    return PythonObject(None)



def forest_vector_groves_binding(outputs: PythonObject) raises -> PythonObject:
    """Read actual compiled vector dispatch, not environment or filenames."""
    return PythonObject(vector_groves_for(Int(py=outputs)))


def forest_ordered_resident_binding() raises -> PythonObject:
    """Read the compiled experimental resident aggregation route."""
    return PythonObject(1 if FOREST_ORDERED_RESIDENT else 0)


def forest_predict_resident_into_gpu_binding[RF_INPUT: Bool, REUSE_IO: Bool = False](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject) raises -> PythonObject:
    """Borrowed synchronous prediction; REUSE_IO=True is the public default.

    DEVIATION 2483: parallel_groves already selects this pointer-through body
    through forest_predict_resident_reuse_gpu. The REUSE_IO=False export and
    List-based forest_predict_resident_gpu remain comparison arms. Output may
    change on a nonfinite-result error; caller ownership lasts until return.
    """
    if len(params) != 3:
        raise Error("resident prediction requires rows, features, outputs")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    var outputs = Int(py=params[2])
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid resident prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident prediction dimensions exceed Int32")
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _f32_ptr(Int(py=out_addr))
    resident_predict_into[RF_INPUT](Int(py=handle),
        xp.unsafe_origin_cast[MutAnyOrigin](), op.unsafe_origin_cast[MutAnyOrigin](),
        rows, features, outputs, REUSE_IO)
    return PythonObject(rows)


def forest_predict_resident_labels_gpu_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject) raises -> PythonObject:
    if len(params) != 3:
        raise Error("resident label prediction requires rows, features, outputs")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    var outputs = Int(py=params[2])
    if rows < 0 or features < 1 or outputs < 2:
        raise Error("invalid resident label prediction dimensions")
    if rows > 2147483647 // features or rows > 2147483647 // outputs:
        raise Error("resident label prediction dimensions exceed Int32")
    var xp = _f32_ptr(Int(py=x_addr))
    var op = _i32_ptr(Int(py=out_addr))
    resident_predict_labels[RF_INPUT](Int(py=handle),
        xp.unsafe_origin_cast[MutAnyOrigin](), op.unsafe_origin_cast[MutAnyOrigin](),
        rows, features, outputs)
    return PythonObject(rows)


def forest_resident_layout_binding() raises -> PythonObject:
    comptime if FOREST_PACKED_NODES:
        return PythonObject("packed_siblings")
    return PythonObject("separate_arrays")


def forest_identical_fused_labels_binding() raises -> PythonObject:
    """T38 capability; absent define preserves incumbent classification."""
    from core.forest_experiments import T38_FUSED_LABELS
    return PythonObject(T38_FUSED_LABELS)
