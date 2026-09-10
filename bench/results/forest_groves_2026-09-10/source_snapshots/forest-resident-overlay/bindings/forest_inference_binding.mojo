# SPDX-License-Identifier: Apache-2.0
"""Shared borrowed-pointer boundary for owned RF/ET GPU inference snapshots.

The calling extension holds the GIL throughout prepare/predict/release, so no
native handle can be released while another call is using it.
"""
from std.python import PythonObject
from core.forest_inference import vector_groves_for
from core.forest_inference_model import resident_prepare, resident_predict, resident_release


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
    """Prepare an owned immutable GPU snapshot. GIL serializes registry access."""
    if len(params) != 3:
        raise Error("forest_prepare_gpu requires features, trees, outputs")
    var features = Int(py=params[0])
    var trees = Int(py=params[1])
    var outputs = Int(py=params[2])
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
    var offsets = List[Int32](capacity=trees + 1)
    var columns = List[Int32](capacity=nodes)
    var thresholds = List[Float32](capacity=nodes)
    var left = List[Int32](capacity=nodes)
    var leaves = List[Float32](capacity=nodes * outputs)
    for i in range(trees + 1):
        offsets.append(op[i])
    for i in range(nodes):
        columns.append(cp[i])
        thresholds.append(tp[i])
        left.append(lp[i])
    for i in range(nodes * outputs):
        leaves.append(vp[i])
    return PythonObject(resident_prepare[RF_INPUT](
        offsets, columns, thresholds, left, leaves, features, outputs))


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
    var x = List[Float32](capacity=rows * features)
    for i in range(rows * features):
        x.append(xp[i])
    var result = resident_predict[RF_INPUT](Int(py=handle), x, rows, features, outputs)
    for i in range(rows * outputs):
        op[i] = result[i]
    return PythonObject(rows)


def forest_release_gpu_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    resident_release[RF_INPUT](Int(py=handle))
    return PythonObject(None)



def forest_vector_groves_binding(outputs: PythonObject) raises -> PythonObject:
    """Read actual compiled vector dispatch, not environment or filenames."""
    return PythonObject(vector_groves_for(Int(py=outputs)))
