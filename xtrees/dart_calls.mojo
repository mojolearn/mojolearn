# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the x_trees DART operations through the trees DART
table (lane calltable-gbdt-forest-dart, 2026-11-11). No Python: a caller
passes buffer addresses and plain scalars and keeps the buffers alive until
the call returns. Positions follow `xtrees/dart_abi.mojo`. `xd_dart` reads the
published table once per call (an acquire load and a schema compare) and
calls the pointer; a missing or mismatched publication refuses with the
table's error. The refusal text of an operation is its former Python
entry's.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.trees_entry_table import TreesEntryKey
from core.trees_entry_table_dart import trees_dart_table_entry
from xtrees.dart_abi import (
    XD_DART_ABI, XD_OP_COUNT, XD_CAPS, XD_OPEN, XD_STEP, XD_ADD, XD_CLOSE, XD_PREDICT, XD_FOREST_PREPARE,
    XD_FOREST_PREDICT, XD_FOREST_RELEASE,
)


def xd_entry(key: TreesEntryKey) raises -> NativeDescriptorCall:
    """The DART entry of the x_trees image the key names."""
    return trees_dart_table_entry(key, 0, String(XD_DART_ABI), XD_OP_COUNT)


def xd_dart(key: TreesEntryKey, op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    return native_descriptor_call(xd_entry(key), op, addrs, ints, floats)


def _floats(ints: List[Int]) -> List[Float64]:
    var out = List[Float64](capacity=len(ints))
    for i in range(len(ints)):  # small-loop(the call's few integer parameters)
        out.append(Float64(ints[i]))
    return out^


def xd_caps(key: TreesEntryKey) raises -> Int:
    """XD_CAP_ROUND | XD_CAP_RESIDENT bits of the image."""
    var addrs = List[Int]()
    var ints = List[Int]()
    var floats = List[Float64]()
    return xd_dart(key, XD_CAPS, addrs, ints, floats)


def xd_open(key: TreesEntryKey, x: Int, y: Int, inits: Int, n: Int, d: Int, k: Int, kind: Int, n_iters: Int,
            node_cap: Int) raises -> Int:
    var addrs: List[Int] = [x, y, inits]
    var ints: List[Int] = [n, d, k, kind, n_iters, node_cap]
    return xd_dart(key, XD_OPEN, addrs, ints, _floats(ints))


def xd_step(key: TreesEntryKey, handle: Int, coef: Int, thr: Int, flags: Int, bad: Int, t: Int, drop_seed: Int,
            iteration: Int, skip_thr: Int) raises -> Int:
    """The round's step with no host targets (they stay in the session)."""
    var addrs: List[Int] = [coef, thr, flags, bad]
    var ints: List[Int] = [handle, t, drop_seed, iteration, skip_thr]
    return xd_dart(key, XD_STEP, addrs, ints, _floats(ints))


def xd_add(key: TreesEntryKey, handle: Int, colid: Int, quesval: Int, left: Int, values: Int, rows: Int, tree: Int,
           cls: Int, lo: Int, n_nodes: Int, shrink: Float64, factor: Float64, lam: Float64, l1: Float64,
           mds: Float64, m: Int) raises -> Int:
    var addrs: List[Int] = [colid, quesval, left, values, rows]
    var ints: List[Int] = [handle, tree, cls, lo, n_nodes, 0, 0, 0, 0, 0, m]
    var floats: List[Float64] = [Float64(handle), Float64(tree), Float64(cls), Float64(lo), Float64(n_nodes),
                                 shrink, factor, lam, l1, mds, Float64(m)]
    return xd_dart(key, XD_ADD, addrs, ints, floats)


def xd_close(key: TreesEntryKey, handle: Int, bad: Int) raises -> Int:
    var addrs: List[Int] = [bad]
    var ints: List[Int] = [handle]
    return xd_dart(key, XD_CLOSE, addrs, ints, _floats(ints))


def xd_predict(key: TreesEntryKey, x: Int, dst: Int, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int,
               sizes: Int, coefs: Int, inits: Int, n: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    var addrs: List[Int] = [x, dst, colids, quesvals, lefts, leaf_values, sizes, coefs, inits]
    var ints: List[Int] = [n, d, k, n_trees]
    return xd_dart(key, XD_PREDICT, addrs, ints, _floats(ints))


def xd_forest_prepare(key: TreesEntryKey, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int, sizes: Int,
                      coefs: Int, inits: Int, out2: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    """Writes [handle, device bytes] to out2; returns the handle."""
    var addrs: List[Int] = [colids, quesvals, lefts, leaf_values, sizes, coefs, inits, out2]
    var ints: List[Int] = [d, k, n_trees]
    return xd_dart(key, XD_FOREST_PREPARE, addrs, ints, _floats(ints))


def xd_forest_predict(key: TreesEntryKey, handle: Int, x: Int, dst: Int, n: Int, d: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x, dst]
    var ints: List[Int] = [handle, n, d, k]
    return xd_dart(key, XD_FOREST_PREDICT, addrs, ints, _floats(ints))


def xd_forest_release(key: TreesEntryKey, handle: Int) raises -> Int:
    var addrs = List[Int]()
    var ints: List[Int] = [handle]
    return xd_dart(key, XD_FOREST_RELEASE, addrs, ints, _floats(ints))
