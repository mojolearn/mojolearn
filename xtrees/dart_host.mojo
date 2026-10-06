# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DART's boosting round on the host, for the CPU-only binding and the CPU
verification column (lane fam2-forests, 2026-10-04, `IDN_DART_DEVICE`):
xtrees/dart_units.mojo's units in xtrees/dart_device.mojo's stage order,
each stage's units in ascending index, so the words are the GPU columns'
words (float32 state, the same chunked leaf-sum fold, the same pinned
float operations, the same counter draws). GPU installs never run this
file. The four entries take the device entries' arguments and are
registered under the same names by bindings/_mojolearn_x_trees_host.mojo,
so the Python glue takes `_boost_loop_device` on both columns.

Serial loops on purpose: this column is a verification digest and a
CPU-only install's fit; it is never timed."""
from gbdt.trees_identical_switches import T30
from std.ffi import _Global
from xtrees.ops import stream_base, draw
from xtrees.dart_units import (
    IDN_DART_DEVICE, DART_CHUNK, F32P, I32P, I64P, U16P, dart_init_unit, dart_drop_unit, dart_row_unit,
    dart_apply_unit, dart_leaf_sum_unit, dart_leaf_sum_rows_unit, dart_newton_unit, dart_add_unit, U64P,
    dart_predict_unit,
)
from std.memory import bitcast

#: The host twin exists exactly where the IDENTICAL device round does.
comptime DART_HOST = IDN_DART_DEVICE


@always_inline
def _hf(mut v: List[Float32]) -> F32P:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _hi(mut v: List[Int32]) -> I32P:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _hu16(mut v: List[UInt16]) -> U16P:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct DartHostSession(Movable):
    var id: Int
    var n: Int
    var d: Int
    var k: Int
    var kind: Int
    var cap_iters: Int
    var node_cap: Int
    var n_chunks: Int
    var x_addr: Int
    var y: List[Float32]
    var score: List[Float32]
    var dsum: List[Float32]
    var target: List[Float32]
    var h: List[Float32]
    var nodes: List[UInt16]
    var values: List[Float32]
    var cached: List[Float32]
    var flags: List[Int32]
    var bad: List[Int32]
    var part: List[Float32]

    def __init__(
        out self, id: Int, n: Int, d: Int, k: Int, kind: Int, cap_iters: Int, node_cap: Int, n_chunks: Int,
        x_addr: Int,
    ):
        self.id = id
        self.n = n
        self.d = d
        self.k = k
        self.kind = kind
        self.cap_iters = cap_iters
        self.node_cap = node_cap
        self.n_chunks = n_chunks
        # X is read in place: the caller's array outlives the session (the
        # Python loop holds `Xa` until `x_trees_dart_close`).
        self.x_addr = x_addr
        self.y = List[Float32](length=n, fill=0.0)
        self.score = List[Float32](length=k * n, fill=0.0)
        self.dsum = List[Float32](length=k * n, fill=0.0)
        self.target = List[Float32](length=k * n, fill=0.0)
        self.h = List[Float32](length=k * n, fill=0.0)
        self.nodes = List[UInt16](length=cap_iters * k * n, fill=0)
        self.values = List[Float32](length=cap_iters * k * node_cap, fill=0.0)
        self.cached = List[Float32](length=cap_iters * k * n if T30 else 0, fill=0.0)
        self.flags = List[Int32](length=cap_iters, fill=0)
        self.bad = List[Int32](length=1, fill=0)
        self.part = List[Float32](length=n_chunks * node_cap * 2, fill=0.0)


struct DartHostRegistry(Defaultable, Movable):
    var sessions: List[DartHostSession]
    var next_id: Int

    def __init__(out self):
        self.sessions = List[DartHostSession]()
        self.next_id = 1

    def find(self, id: Int) raises -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].id == id:
                return i
        raise Error("x_trees dart: unknown or closed session handle")


comptime DART_HOST_SESSIONS = _Global[
    StorageType=DartHostRegistry, name="MojoXTreesDartHostSessions", init_fn=DartHostRegistry.__init__
]


def dart_open(
    x_addr: Int, y_addr: Int, inits_addr: Int, n: Int, d: Int, k: Int, kind: Int, cap_iters: Int, node_cap: Int,
) raises -> Int:
    """The device `dart_open` on host memory: y copied, the k x n score set
    to the class starts, the per-tree leaf index rows and leaf values
    allocated. Returns the handle."""
    comptime if DART_HOST:
        if n <= 0 or d <= 0 or k <= 0 or cap_iters <= 0 or node_cap <= 0 or node_cap > 65535:
            raise Error("x_trees dart_open: needs rows, features, classes, iterations and 1 <= node_cap <= 65535")
        var n_chunks = (n + DART_CHUNK - 1) // DART_CHUNK
        var reg = DART_HOST_SESSIONS.get_or_create_ptr()
        var id = reg[].next_id
        reg[].next_id += 1
        reg[].sessions.append(DartHostSession(id, n, d, k, kind, cap_iters, node_cap, n_chunks, x_addr))
        var idx = reg[].find(id)
        var ysrc = F32P(unsafe_from_address=y_addr)
        var yp = _hf(reg[].sessions[idx].y)
        for i in range(n):
            yp[i] = ysrc[i]
        var inits = F32P(unsafe_from_address=inits_addr)
        var sp = _hf(reg[].sessions[idx].score)
        for e in range(k * n):
            dart_init_unit(e, n, inits, sp)
        return id
    else:
        raise Error("x_trees dart_open: built without the host DART round (MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_step(
    id: Int, coef_addr: Int, thr_addr: Int, flags_out: Int, bad_out: Int, targets: List[Int], t: Int, seed: Int,
    stream: Int, skip_thr: Int,
) raises:
    """The device `dart_step`: drop flags of iterations 0 .. t, the dropped
    trees off the score, the gradients; the fit target of each class stays
    in the session's `target` plane (read there by the member fits, the host
    rf binding's `rf_regressor_fit_dart_export`), the flags and the bad word
    to the caller."""
    comptime if DART_HOST:
        var reg = DART_HOST_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var n = reg[].sessions[idx].n
        var k = reg[].sessions[idx].k
        # box-run-2-dart-host (2026-10-05): cpu4-forest (820fd9928) moved the
        # device protocol to "targets stay in the session's target plane"
        # (xtrees/dart_device.mojo dart_step: `targets` must be empty) and the
        # Python loop now passes []; the host twin required len(targets) == k
        # and raised. Same protocol here: `targets` must be empty (kept in
        # the signature only), the K planes stay in this session for
        # `rf_regressor_fit_dart_export` (bindings/_mojolearn_rf_host.mojo).
        if t < 0 or t > reg[].sessions[idx].cap_iters or len(targets) != 0:
            raise Error("x_trees dart_step: iteration count or targets out of range")
        var flags = _hi(reg[].sessions[idx].flags)
        var coef = F32P(unsafe_from_address=coef_addr)
        if t > 0:
            var thr = I64P(unsafe_from_address=thr_addr)
            var base = stream_base(seed, stream)
            var skip = (draw(base, 0) >> 11) < UInt64(skip_thr)
            for e in range(t):
                dart_drop_unit(e, base, skip, thr, flags)
        var nodes = _hu16(reg[].sessions[idx].nodes)
        var values = _hf(reg[].sessions[idx].values)
        var cached = _hf(reg[].sessions[idx].cached)
        var y = _hf(reg[].sessions[idx].y)
        var score = _hf(reg[].sessions[idx].score)
        var dsum = _hf(reg[].sessions[idx].dsum)
        var target = _hf(reg[].sessions[idx].target)
        var h = _hf(reg[].sessions[idx].h)
        var kind = Int32(reg[].sessions[idx].kind)
        var node_cap = reg[].sessions[idx].node_cap
        for i in range(n):
            dart_row_unit(i, n, k, t, kind, node_cap, flags, coef, nodes, values, y, score, dsum, target, h, cached)
        if t > 0:
            var fo = I32P(unsafe_from_address=flags_out)
            for e in range(t):
                fo[e] = flags[e]
        var bo = I32P(unsafe_from_address=bad_out)
        var bad = _hi(reg[].sessions[idx].bad)
        bo[0] = bad[0]
    else:
        raise Error("x_trees dart_step: built without the host DART round (MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_add(
    id: Int, colid_addr: Int, quesval_addr: Int, left_addr: Int, values_out: Int, j: Int, c: Int, lo: Int,
    n_nodes: Int, shrink: Float64, factor: Float64, lam: Float64, l1: Float64, mds: Float64, rows_addr: Int, m: Int,
) raises:
    """The device `dart_add` for class c's new tree j: the leaf index row,
    the chunked leaf sums, the Newton leaf values (to values_out) and the
    score update. m > 0: the leaf sums over the bag rows rows_addr[0 .. m)
    only, chunked over the list positions in list order (the device's
    `dart_leaf_sum_rows_unit`, the same units in the same order)."""
    comptime if DART_HOST:
        var reg = DART_HOST_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var n = reg[].sessions[idx].n
        var k = reg[].sessions[idx].k
        var d = reg[].sessions[idx].d
        var node_cap = reg[].sessions[idx].node_cap
        var n_chunks = reg[].sessions[idx].n_chunks
        if j < 0 or j >= reg[].sessions[idx].cap_iters * k or c < 0 or c >= k:
            raise Error("x_trees dart_add: tree or class index out of range")
        if n_nodes < 1 or n_nodes > node_cap or lo < 0:
            raise Error("x_trees dart_add: a tree with more nodes than 2 * num_leaves - 1")
        if m < 0 or m > n:
            raise Error("x_trees dart_add: bag row count out of range")
        var colid = I32P(unsafe_from_address=colid_addr + 4 * lo)
        var quesval = F32P(unsafe_from_address=quesval_addr + 4 * lo)
        var left = I32P(unsafe_from_address=left_addr + 4 * lo)
        var x = F32P(unsafe_from_address=reg[].sessions[idx].x_addr)
        var nodes = _hu16(reg[].sessions[idx].nodes)
        var bad = _hi(reg[].sessions[idx].bad)
        var target = _hf(reg[].sessions[idx].target)
        var h = _hf(reg[].sessions[idx].h)
        var part = _hf(reg[].sessions[idx].part)
        var values = _hf(reg[].sessions[idx].values)
        var cached = _hf(reg[].sessions[idx].cached)
        var dsum = _hf(reg[].sessions[idx].dsum)
        var score = _hf(reg[].sessions[idx].score)
        var row_off = j * n
        var class_off = c * n
        var voff = j * node_cap
        for i in range(n):
            dart_apply_unit(i, d, n_nodes, colid, quesval, left, x, row_off, nodes, bad)
        if m > 0:
            n_chunks = (m + DART_CHUNK - 1) // DART_CHUNK
            var rows = I32P(unsafe_from_address=rows_addr)
            for e in range(n_nodes * n_chunks):
                dart_leaf_sum_rows_unit(e, n, m, n_nodes, row_off, class_off, rows, nodes, target, h, part, bad)
        else:
            for e in range(n_nodes * n_chunks):
                dart_leaf_sum_unit(e, n, n_nodes, row_off, class_off, nodes, target, h, part)
        for e in range(n_nodes):
            dart_newton_unit(e, n_nodes, n_chunks, part, Float32(lam), Float32(l1), Float32(mds), voff, values)
        for e in range(n):
            dart_add_unit(e, class_off, row_off, voff, Float32(factor), Float32(shrink), nodes, values, dsum, score, cached)
        var vo = F32P(unsafe_from_address=values_out)
        for e in range(n_nodes):
            vo[e] = values[voff + e]
    else:
        raise Error("x_trees dart_add: built without the host DART round (MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_close(id: Int, bad_out: Int) raises:
    """Write the bad word and free the session."""
    comptime if DART_HOST:
        var reg = DART_HOST_SESSIONS.get_or_create_ptr()
        var idx = reg[].find(id)
        var bo = I32P(unsafe_from_address=bad_out)
        var bad = _hi(reg[].sessions[idx].bad)
        bo[0] = bad[0]
        var gone = reg[].sessions.pop(idx)
        _ = gone^
    else:
        raise Error("x_trees dart_close: built without the host DART round (MOJOLEARN_IDN_DART_DEVICE_OFF)")


def dart_predict(
    x_addr: Int, colids: List[Int], quesvals: List[Int], lefts: List[Int], leaf_values: List[Int], sizes: List[Int],
    coefs: List[Float64], inits: List[Float64], out_addr: Int, n: Int, d: Int, k: Int,
) raises:
    """The device `dart_predict` on the host column (lane cpu2-l5-trees,
    registered on every host build): the forest concatenated in tree order,
    then `dart_predict_unit` over e = 0 .. k * n - 1 in ascending order, so
    the float64 raw score is the device's word for word (soft binary64, the
    words of main's per-tree host adds under IDENTICAL)."""
    var nt = len(sizes)
    if n <= 0:
        return
    if (
        d <= 0 or k <= 0 or nt < 1 or len(colids) != nt or len(quesvals) != nt or len(lefts) != nt
        or len(leaf_values) != nt or len(coefs) != nt or len(inits) != k
    ):
        raise Error("x_trees dart_predict: needs features, classes, trees and one address per tree and class")
    var toff = List[Int32](length=nt + 1, fill=0)
    var total = 0
    for j in range(nt):
        if sizes[j] < 1:
            raise Error("x_trees dart_predict: empty tree")
        total += sizes[j]
        if total >= (1 << 31):
            raise Error("x_trees dart_predict: more than 2^31 forest nodes")
        toff[j + 1] = Int32(total)
    var colid = List[Int32](length=total, fill=0)
    var quesval = List[Float32](length=total, fill=0.0)
    var left = List[Int32](length=total, fill=0)
    var values = List[Float32](length=total, fill=0.0)
    for j in range(nt):
        var lo = Int(toff[j])
        var cs = I32P(unsafe_from_address=colids[j])
        var qs = F32P(unsafe_from_address=quesvals[j])
        var ls = I32P(unsafe_from_address=lefts[j])
        var vs = F32P(unsafe_from_address=leaf_values[j])
        for q in range(sizes[j]):
            colid[lo + q] = cs[q]
            quesval[lo + q] = qs[q]
            left[lo + q] = ls[q]
            values[lo + q] = vs[q]
    var cw = List[UInt64](length=nt, fill=0)
    for j in range(nt):
        cw[j] = bitcast[DType.uint64](coefs[j])
    var iw = List[UInt64](length=k, fill=0)
    for c in range(k):
        iw[c] = bitcast[DType.uint64](inits[c])
    var bad = List[Int32](length=1, fill=0)
    var x = F32P(unsafe_from_address=x_addr)
    var out = U64P(unsafe_from_address=out_addr)
    var tp = _hi(toff)
    var cp = _hi(colid)
    var qp = _hf(quesval)
    var lp = _hi(left)
    var vp = _hf(values)
    var cwp = cw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var iwp = iw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var bp = _hi(bad)
    for e in range(k * n):
        dart_predict_unit(e, n, d, k, nt, tp, cp, qp, lp, vp, cwp, iwp, x, out, bp)
    # the lists live past the loop (their pointers were taken above)
    _ = toff^
    _ = colid^
    _ = quesval^
    _ = left^
    _ = values^
    _ = cw^
    _ = iw^
    if bad[0] != Int32(0):
        raise Error("x_trees dart_predict: a tree walk left its tree (child or column out of range)")
