# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the x_trees DART operations through the trees DART
table (lane calltable-gbdt-forest-dart, 2026-11-11). No Python: a caller
passes buffer addresses and plain scalars and keeps the buffers alive until
the call returns. Positions follow `xtrees/dart_abi.mojo`.

Lane entry-table-zero-alloc (2026-11-11): the caller resolves the table once
(`xd_ref`, where a missing or mismatched publication refuses with the table's
error) and passes the `EntryRef` to every call; a call is one indirect call.
Every call builds its request words in inline `Array` envelopes on its
frame, so a round allocates nothing. The refusal text of an operation is its
former Python entry's.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word
from core.trees_entry_table import TreesEntryKey
from core.trees_entry_table_dart import trees_dart_table_entry
from xtrees.dart_abi import (
    XD_DART_ABI, XD_OP_COUNT, XD_CAPS, XD_OPEN, XD_STEP, XD_ADD, XD_CLOSE, XD_PREDICT, XD_FOREST_PREPARE,
    XD_FOREST_PREDICT, XD_FOREST_RELEASE,
)


def xd_ref(key: TreesEntryKey) raises -> EntryRef:
    """The DART entry of the x_trees image the key names, resolved once."""
    return EntryRef(trees_dart_table_entry(key, 0, String(XD_DART_ABI), XD_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, XD_OP_COUNT))


def xd_dart(dart: EntryRef, op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    return entry_ref_call(dart, op, addrs, ints, floats)


def xd_caps(dart: EntryRef) raises -> Int:
    """XD_CAP_ROUND | XD_CAP_RESIDENT bits of the image."""
    return entry_ref_call(dart, XD_CAPS, 0, 0, 0, 0, 0, 0)


def xd_open(dart: EntryRef, x: Int, y: Int, inits: Int, n: Int, d: Int, k: Int, kind: Int, n_iters: Int,
            node_cap: Int) raises -> Int:
    var addrs: Array[Int, 3] = [x, y, inits]
    var ints: Array[Int, 6] = [n, d, k, kind, n_iters, node_cap]
    var floats: Array[Float64, 6] = [Float64(n), Float64(d), Float64(k), Float64(kind), Float64(n_iters), Float64(node_cap)]
    return entry_ref_call(dart, XD_OPEN, addrs, ints, floats)


def xd_step(dart: EntryRef, handle: Int, coef: Int, thr: Int, flags: Int, bad: Int, t: Int, drop_seed: Int,
            iteration: Int, skip_thr: Int) raises -> Int:
    """The round's step with no host targets (they stay in the session).
    Request words inline on the frame: no allocation per round."""
    var addrs: Array[Int, 4] = [coef, thr, flags, bad]
    var ints: Array[Int, 5] = [handle, t, drop_seed, iteration, skip_thr]
    var floats: Array[Float64, 5] = [Float64(handle), Float64(t), Float64(drop_seed), Float64(iteration),
                                     Float64(skip_thr)]
    return entry_ref_call(dart, XD_STEP, addrs, ints, floats)


def xd_add(dart: EntryRef, handle: Int, colid: Int, quesval: Int, left: Int, values: Int, rows: Int, tree: Int,
           cls: Int, lo: Int, n_nodes: Int, shrink: Float64, factor: Float64, lam: Float64, l1: Float64,
           mds: Float64, m: Int) raises -> Int:
    """One member's add. Request words inline on the frame: no allocation per
    member."""
    var addrs: Array[Int, 5] = [colid, quesval, left, values, rows]
    var ints: Array[Int, 11] = [handle, tree, cls, lo, n_nodes, 0, 0, 0, 0, 0, m]
    var floats: Array[Float64, 11] = [Float64(handle), Float64(tree), Float64(cls), Float64(lo), Float64(n_nodes),
                                      shrink, factor, lam, l1, mds, Float64(m)]
    return entry_ref_call(dart, XD_ADD, addrs, ints, floats)


def xd_close(dart: EntryRef, handle: Int, bad: Int) raises -> Int:
    var addrs: Array[Int, 1] = [bad]
    var ints: Array[Int, 1] = [handle]
    var floats: Array[Float64, 1] = [Float64(handle)]
    return entry_ref_call(dart, XD_CLOSE, addrs, ints, floats)


def xd_predict(dart: EntryRef, x: Int, dst: Int, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int,
               sizes: Int, coefs: Int, inits: Int, n: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    var addrs: Array[Int, 9] = [x, dst, colids, quesvals, lefts, leaf_values, sizes, coefs, inits]
    var ints: Array[Int, 4] = [n, d, k, n_trees]
    var floats: Array[Float64, 4] = [Float64(n), Float64(d), Float64(k), Float64(n_trees)]
    return entry_ref_call(dart, XD_PREDICT, addrs, ints, floats)


def xd_forest_prepare(dart: EntryRef, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int, sizes: Int,
                      coefs: Int, inits: Int, out2: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    """Writes [handle, device bytes] to out2; returns the handle."""
    var addrs: Array[Int, 8] = [colids, quesvals, lefts, leaf_values, sizes, coefs, inits, out2]
    var ints: Array[Int, 3] = [d, k, n_trees]
    var floats: Array[Float64, 3] = [Float64(d), Float64(k), Float64(n_trees)]
    return entry_ref_call(dart, XD_FOREST_PREPARE, addrs, ints, floats)


def xd_forest_predict(dart: EntryRef, handle: Int, x: Int, dst: Int, n: Int, d: Int, k: Int) raises -> Int:
    var addrs: Array[Int, 2] = [x, dst]
    var ints: Array[Int, 4] = [handle, n, d, k]
    var floats: Array[Float64, 4] = [Float64(handle), Float64(n), Float64(d), Float64(k)]
    return entry_ref_call(dart, XD_FOREST_PREDICT, addrs, ints, floats)


def xd_forest_release(dart: EntryRef, handle: Int) raises -> Int:
    var addrs: Array[Int, 0] = []
    var ints: Array[Int, 1] = [handle]
    var floats: Array[Float64, 1] = [Float64(handle)]
    return entry_ref_call(dart, XD_FOREST_RELEASE, addrs, ints, floats)
