# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_host_strict_binding.mojo` (wave 8 run split): the
host strict snapshot registries, their typed prepare and release, and the
two typed halves of a strict predict (the checks the caller runs holding the
GIL, and the walk it runs with the GIL released). Imports no Python; the
binding keeps the UNPACK doors and the GIL release."""
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.forest_host_predict import rf_host_predict, rf_host_trees, et_host_predict, et_host_trees
from ensemble.decisiontree.decisiontree import TreeMetaDataNode as RfTree
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode as EtTree
from core.shared_slot import SharedSlot


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


comptime RF_STRICT_REGISTRY = SharedSlot[HostStrictRegistry, "MojolearnRFStrictForestHostV2"]


comptime ET_STRICT_REGISTRY = SharedSlot[HostStrictRegistry, "MojolearnETStrictForestHostV2"]


def forest_strict_prepare_host_native[RF_INPUT: Bool](model: List[Int], features: Int, trees: Int,
                                                     outputs: Int) raises -> Int:
    """The trees of one fitted forest, rebuilt once (the same checks
    `rf_host_trees` / `et_host_trees` make per call; XF_STRICT_PREPARE, lane
    calltable-forest). `model` holds the five array addresses."""
    if features < 1 or trees < 1 or trees >= 2147483647 or outputs < 1:
        raise Error("invalid strict forest dimensions")
    var offsets_p = i32_ptr(model[0])
    var colid_p = i32_ptr(model[1])
    var quesval_p = f32_ptr(model[2])
    var left_p = i32_ptr(model[3])
    var leaves_p = f32_ptr(model[4])
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
    return id


def forest_strict_release_host_native[RF_INPUT: Bool](handle: Int) raises:
    """XF_STRICT_RELEASE: an unknown handle is already released."""
    comptime if RF_INPUT:
        var state = RF_STRICT_REGISTRY.get_or_create_ptr()
        if handle in state[].entries:
            _ = state[].entries.pop(handle)
    else:
        var state = ET_STRICT_REGISTRY.get_or_create_ptr()
        if handle in state[].entries:
            _ = state[].entries.pop(handle)


def forest_strict_predict_host_check[RF_INPUT: Bool](handle: Int, rows: Int, features: Int,
                                                    outputs: Int) raises -> Int:
    """The checks of a strict predict (XF_STRICT_PREDICT), run by the door
    holding the GIL: dimensions, the handle, and the snapshot's dimensions.
    Returns the snapshot's tree count."""
    if rows < 0 or features < 1 or outputs < 1:
        raise Error("invalid strict prediction dimensions")
    var state = RF_STRICT_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_STRICT_REGISTRY.get_or_create_ptr()
    if handle not in state[].entries:
        raise Error("unknown or released strict forest handle")
    if state[].entries[handle].features != features or state[].entries[handle].outputs != outputs:
        raise Error("strict prediction dimensions do not match the snapshot")
    return state[].entries[handle].trees


def forest_strict_predict_host_walk[RF_INPUT: Bool](handle: Int, x_address: Int, out_addr: Int, rows: Int,
                                                   features: Int, trees: Int, outputs: Int) raises:
    """The sequential vote over the kept trees, run by the door with the GIL
    released after `forest_strict_predict_host_check` (rows > 0): `rows *
    outputs` float32 values into `out`, exactly what `rf_predict_*` /
    `et_predict` write. The door pins the handle across the walk."""
    var state = RF_STRICT_REGISTRY.get_or_create_ptr()
    comptime if not RF_INPUT:
        state = ET_STRICT_REGISTRY.get_or_create_ptr()
    var op = f32_ptr(out_addr)
    var x = read_f32(x_address, rows * features)
    var out = List[Float32](length=rows * outputs, fill=Float32(0.0))
    comptime if RF_INPUT:
        rf_host_predict(state[].entries[handle].rf, x, rows, features, trees, outputs, out)
    else:
        et_host_predict(state[].entries[handle].et, x, rows, features, trees, outputs, out)
    for i in range(rows * outputs):  # small-loop(output copy): the same copy forest_host_vote_into makes
        op[i] = out[i]
