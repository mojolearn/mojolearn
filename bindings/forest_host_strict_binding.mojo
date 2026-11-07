# SPDX-License-Identifier: Apache-2.0
"""The host (CPU column) strict snapshot of a fitted forest (N044, lane
ne1-a-trees, 2026-11-06).

`rf_predict_proba`, `rf_predict_reg` and `et_predict` of the host bindings
rebuild every tree from the five flat model arrays (`rf_host_trees`,
`et_host_trees`) on every call before the walk. These entries rebuild them
once per fitted model and keep them under a handle the tree door's resident
table holds (bindings/trees_resident_table.mojo), so a predict walks the kept
trees. The walk is the same function over the same trees and the same rows
(`rf_host_predict` / `et_host_predict`, the sequential vote), so no result
changes.

Entries (host bindings only; the door looks for `forest_strict_prepare`):
- `forest_strict_prepare(offsets, colid, quesval, left_child, leaves, [features, trees, outputs])` -> handle
- `forest_strict_predict(handle, x, out, [rows, features, outputs])` -> rows (ROW-major x)
- `forest_strict_release(handle)`

The caller (the door) pins the handle across a predict, so a release from
another thread never frees trees a walk reads; the walk itself runs with the
GIL released, as the per-call entries do. The query X is still copied into a
List before the walk: `DecisionTree.predict_one` reads a `List[Float32]` row
store (a borrowed-pointer walk changes ensemble/decisiontree, not this file;
the copy is one memcpy, the rebuild it replaces was per node).
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.forest_host_predict import rf_host_predict, rf_host_trees, et_host_predict, et_host_trees
from ensemble.decisiontree.decisiontree import TreeMetaDataNode as RfTree
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode as EtTree


struct HostStrictForest(Movable):
    var rf: List[RfTree[DType.float32]]
    var et: List[EtTree[DType.float32]]
    var features: Int
    var trees: Int
    var outputs: Int

    def __init__(out self, var rf: List[RfTree[DType.float32]], var et: List[EtTree[DType.float32]],
                 features: Int, trees: Int, outputs: Int):
        self.rf = rf^
        self.et = et^
        self.features = features
        self.trees = trees
        self.outputs = outputs


struct HostStrictRegistry(Defaultable, Movable):
    var entries: Dict[Int, HostStrictForest]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, HostStrictForest]()
        self.next_id = 1


comptime RF_STRICT_REGISTRY = _Global[
    StorageType=HostStrictRegistry, name="MojoRFStrictForestHostV1", init_fn=HostStrictRegistry.__init__,
]
comptime ET_STRICT_REGISTRY = _Global[
    StorageType=HostStrictRegistry, name="MojoETStrictForestHostV1", init_fn=HostStrictRegistry.__init__,
]


def forest_strict_prepare_host_binding[RF_INPUT: Bool](
    offsets_addr: PythonObject, colid_addr: PythonObject,
    quesval_addr: PythonObject, left_child_addr: PythonObject,
    leaves_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """The trees of one fitted forest, rebuilt once (the same checks
    `rf_host_trees` / `et_host_trees` make per call)."""
    if len(params) != 3:
        raise Error("forest_strict_prepare requires features, trees, outputs")
    var features = Int(py=params[0])
    var trees = Int(py=params[1])
    var outputs = Int(py=params[2])
    if features < 1 or trees < 1 or trees >= 2147483647 or outputs < 1:
        raise Error("invalid strict forest dimensions")
    var offsets_p = i32_ptr(Int(py=offsets_addr))
    var colid_p = i32_ptr(Int(py=colid_addr))
    var quesval_p = f32_ptr(Int(py=quesval_addr))
    var left_p = i32_ptr(Int(py=left_child_addr))
    var leaves_p = f32_ptr(Int(py=leaves_addr))
    var n_nodes = Int(offsets_p[trees])
    if Int(offsets_p[0]) != 0 or n_nodes < trees:
        raise Error("forest_strict_prepare: tree_offsets must start at 0 and hold at least one node per tree")
    var rf = List[RfTree[DType.float32]]()
    var et = List[EtTree[DType.float32]]()
    comptime if RF_INPUT:
        rf = rf_host_trees(offsets_p, colid_p, quesval_p, left_p, leaves_p, trees, n_nodes, features, outputs)
    else:
        et = et_host_trees(offsets_p, colid_p, quesval_p, left_p, leaves_p, trees, n_nodes, features, outputs)
    var state = RF_STRICT_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_STRICT_REGISTRY.get_or_create_ptr()
    var id = state[].next_id
    state[].next_id += 1
    state[].entries[id] = HostStrictForest(rf^, et^, features, trees, outputs)
    return PythonObject(id)


def forest_strict_predict_host_binding[RF_INPUT: Bool](
    handle: PythonObject, x_addr: PythonObject, out_addr: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """The sequential vote over the kept trees: `rows * outputs` float32
    values into `out`, exactly what `rf_predict_*` / `et_predict` write."""
    if len(params) != 3:
        raise Error("strict prediction requires rows, features, outputs")
    var rows = Int(py=params[0])
    var features = Int(py=params[1])
    var outputs = Int(py=params[2])
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid strict prediction dimensions")
    var id = Int(py=handle)
    var state = RF_STRICT_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_STRICT_REGISTRY.get_or_create_ptr()
    if id not in state[].entries:
        raise Error("unknown or released strict forest handle")
    if state[].entries[id].features != features or state[].entries[id].outputs != outputs:
        raise Error("strict prediction dimensions do not match the snapshot")
    if rows == 0:
        return PythonObject(0)
    var x_address = Int(py=x_addr)
    var op = f32_ptr(Int(py=out_addr))
    var trees = state[].entries[id].trees
    # The kept trees are read in place for the walk (the door's pin keeps
    # the entry alive across it).
    with GILReleased(Python()):
        var x = read_f32(x_address, rows * features)
        var out = List[Float32](length=rows * outputs, fill=Float32(0.0))
        comptime if RF_INPUT:
            rf_host_predict(state[].entries[id].rf, x, rows, features, trees, outputs, out)
        else:
            et_host_predict(state[].entries[id].et, x, rows, features, trees, outputs, out)
        for i in range(rows * outputs):  # small-loop(output copy): the same copy forest_host_vote_into makes
            op[i] = out[i]
    return PythonObject(rows)


def forest_strict_release_host_binding[RF_INPUT: Bool](handle: PythonObject) raises -> PythonObject:
    var id = Int(py=handle)
    comptime if RF_INPUT:
        var state = RF_STRICT_REGISTRY.get_or_create_ptr()
        if id in state[].entries:
            _ = state[].entries.pop(id)
    else:
        var state = ET_STRICT_REGISTRY.get_or_create_ptr()
        if id in state[].entries:
            _ = state[].entries.pop(id)
    return PythonObject(None)
