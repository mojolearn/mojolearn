# SPDX-License-Identifier: Apache-2.0
"""Typed snapshot copy moved word for word from bindings/forest_host_groves_binding.mojo."""
from core.forest_host_groves import HostGroveForest


def _i32_ptr(address: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Int32 pointer")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)


def _f32_ptr(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address == 0:
        raise Error("null resident forest Float32 pointer")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


def host_grove_forest_from(
    offsets_addr: Int, colid_addr: Int, quesval_addr: Int, left_child_addr: Int,
    leaves_addr: Int, features: Int, trees: Int, outputs: Int,
) raises -> HostGroveForest:
    """Copies the caller's flattened forest arrays into one host snapshot."""
    var op = _i32_ptr(offsets_addr)
    var cp = _i32_ptr(colid_addr)
    var tp = _f32_ptr(quesval_addr)
    var lp = _i32_ptr(left_child_addr)
    var vp = _f32_ptr(leaves_addr)
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
    return HostGroveForest(
        offsets^, columns^, thresholds^, left^, leaves^, features, outputs
    )
