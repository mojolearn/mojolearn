# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the x_trees DART operations through the trees DART
table (lane calltable-gbdt-forest-dart, 2026-11-11). No Python: a caller
passes buffer addresses and plain scalars and keeps the buffers alive until
the call returns. Positions follow `xtrees/dart_abi.mojo`.

Lane entry-table-zero-alloc (2026-11-11): the caller resolves the table once
(`xd_ref`, where a missing or mismatched publication refuses with the table's
error) and passes the `EntryRef` to every call; a call is one indirect call.
The per-round calls (`xd_step`, `xd_add`) build their request words on the
stack, so a round allocates nothing. The refusal text of an operation is its
former Python entry's.
"""
from std.memory import stack_allocation
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


def _floats(ints: List[Int]) -> List[Float64]:
    var out = List[Float64](capacity=len(ints))
    for i in range(len(ints)):  # small-loop(the call's few integer parameters)
        out.append(Float64(ints[i]))
    return out^


def xd_caps(dart: EntryRef) raises -> Int:
    """XD_CAP_ROUND | XD_CAP_RESIDENT bits of the image."""
    return entry_ref_call(dart, XD_CAPS, 0, 0, 0, 0, 0, 0)


def xd_open(dart: EntryRef, x: Int, y: Int, inits: Int, n: Int, d: Int, k: Int, kind: Int, n_iters: Int,
            node_cap: Int) raises -> Int:
    var addrs: List[Int] = [x, y, inits]
    var ints: List[Int] = [n, d, k, kind, n_iters, node_cap]
    return xd_dart(dart, XD_OPEN, addrs, ints, _floats(ints))


def xd_step(dart: EntryRef, handle: Int, coef: Int, thr: Int, flags: Int, bad: Int, t: Int, drop_seed: Int,
            iteration: Int, skip_thr: Int) raises -> Int:
    """The round's step with no host targets (they stay in the session).
    Request words on the stack: no allocation per round."""
    var a = stack_allocation[4, Int]()
    a[0] = coef
    a[1] = thr
    a[2] = flags
    a[3] = bad
    var p = stack_allocation[5, Int]()
    p[0] = handle
    p[1] = t
    p[2] = drop_seed
    p[3] = iteration
    p[4] = skip_thr
    var f = stack_allocation[5, Float64]()
    for i in range(5):  # small-loop(the call's five integer parameters)
        f[i] = Float64(p[i])
    return entry_ref_call(dart, XD_STEP, Int(a), 4, Int(p), 5, Int(f), 5)


def xd_add(dart: EntryRef, handle: Int, colid: Int, quesval: Int, left: Int, values: Int, rows: Int, tree: Int,
           cls: Int, lo: Int, n_nodes: Int, shrink: Float64, factor: Float64, lam: Float64, l1: Float64,
           mds: Float64, m: Int) raises -> Int:
    """One member's add. Request words on the stack: no allocation per member."""
    var a = stack_allocation[5, Int]()
    a[0] = colid
    a[1] = quesval
    a[2] = left
    a[3] = values
    a[4] = rows
    var p = stack_allocation[11, Int]()
    p[0] = handle
    p[1] = tree
    p[2] = cls
    p[3] = lo
    p[4] = n_nodes
    for i in range(5, 10):  # small-loop(five reserved zero words)
        p[i] = 0
    p[10] = m
    var f = stack_allocation[11, Float64]()
    f[0] = Float64(handle)
    f[1] = Float64(tree)
    f[2] = Float64(cls)
    f[3] = Float64(lo)
    f[4] = Float64(n_nodes)
    f[5] = shrink
    f[6] = factor
    f[7] = lam
    f[8] = l1
    f[9] = mds
    f[10] = Float64(m)
    return entry_ref_call(dart, XD_ADD, Int(a), 5, Int(p), 11, Int(f), 11)


def xd_close(dart: EntryRef, handle: Int, bad: Int) raises -> Int:
    var addrs: List[Int] = [bad]
    var ints: List[Int] = [handle]
    return xd_dart(dart, XD_CLOSE, addrs, ints, _floats(ints))


def xd_predict(dart: EntryRef, x: Int, dst: Int, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int,
               sizes: Int, coefs: Int, inits: Int, n: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    var addrs: List[Int] = [x, dst, colids, quesvals, lefts, leaf_values, sizes, coefs, inits]
    var ints: List[Int] = [n, d, k, n_trees]
    return xd_dart(dart, XD_PREDICT, addrs, ints, _floats(ints))


def xd_forest_prepare(dart: EntryRef, colids: Int, quesvals: Int, lefts: Int, leaf_values: Int, sizes: Int,
                      coefs: Int, inits: Int, out2: Int, d: Int, k: Int, n_trees: Int) raises -> Int:
    """Writes [handle, device bytes] to out2; returns the handle."""
    var addrs: List[Int] = [colids, quesvals, lefts, leaf_values, sizes, coefs, inits, out2]
    var ints: List[Int] = [d, k, n_trees]
    return xd_dart(dart, XD_FOREST_PREPARE, addrs, ints, _floats(ints))


def xd_forest_predict(dart: EntryRef, handle: Int, x: Int, dst: Int, n: Int, d: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x, dst]
    var ints: List[Int] = [handle, n, d, k]
    return xd_dart(dart, XD_FOREST_PREDICT, addrs, ints, _floats(ints))


def xd_forest_release(dart: EntryRef, handle: Int) raises -> Int:
    var addrs = List[Int]()
    var ints: List[Int] = [handle]
    return xd_dart(dart, XD_FOREST_RELEASE, addrs, ints, _floats(ints))
