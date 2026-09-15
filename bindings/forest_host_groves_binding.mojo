# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The resident `parallel_groves` entries of a CPU-only install, under the GPU
bindings' names and address contract (`bindings/forest_inference_binding.mojo`:
`forest_prepare_gpu`, `forest_predict_resident_reuse_gpu`,
`forest_release_gpu`), over `core/forest_host_groves.mojo`.

HOST ONLY. The `rf` host binding specializes these with `RF_INPUT=True` and
the `trees` host binding with `RF_INPUT=False`, each over its own registry,
as `core/forest_inference_model.mojo` keeps one registry per specialization.
The caller holds the GIL through prepare, predict and release, as the GPU
binding's caller does.
"""
from std.ffi import _Global
from std.python import PythonObject

from core.forest_host_groves import HostGroveForest, HostGroveRegistry


comptime RF_GROVE_REGISTRY = _Global[
    StorageType=HostGroveRegistry,
    name="MojoRFResidentForestHost",
    init_fn=HostGroveRegistry.__init__,
]
comptime ET_GROVE_REGISTRY = _Global[
    StorageType=HostGroveRegistry,
    name="MojoETResidentForestHost",
    init_fn=HostGroveRegistry.__init__,
]


def _i32_ptr(address: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Int32 pointer")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)


def _f32_ptr(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Float32 pointer")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def forest_prepare_host_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`forest_prepare_gpu_binding` (`bindings/forest_inference_binding.mojo:33-68`):
    the same checks in the same words, then the host snapshot."""
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
    var model = HostGroveForest(
        offsets^, columns^, thresholds^, left^, leaves^, features, outputs
    )
    comptime if RF_INPUT:
        return PythonObject(RF_GROVE_REGISTRY.get_or_create_ptr()[].prepare(model^))
    else:
        return PythonObject(ET_GROVE_REGISTRY.get_or_create_ptr()[].prepare(model^))


def forest_predict_resident_host_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`forest_predict_resident_into_gpu_binding[RF_INPUT, True]`
    (`bindings/forest_inference_binding.mojo:104-128`), exported as
    `forest_predict_resident_reuse_gpu`: borrowed ROW-major `x`, `params`
    is `[rows, features, outputs]`, returns rows."""
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
    var id = Int(py=handle)
    var state = RF_GROVE_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_GROVE_REGISTRY.get_or_create_ptr()
    if id not in state[].entries:
        raise Error("unknown or released resident forest handle")
    state[].entries[id].predict_into[RF_INPUT](xp, op, rows, features, outputs)
    return PythonObject(rows)


def forest_release_host_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    """`forest_release_gpu_binding` (`bindings/forest_inference_binding.mojo:93-96`)."""
    comptime if RF_INPUT:
        RF_GROVE_REGISTRY.get_or_create_ptr()[].release(Int(py=handle))
    else:
        ET_GROVE_REGISTRY.get_or_create_ptr()[].release(Int(py=handle))
    return PythonObject(None)
