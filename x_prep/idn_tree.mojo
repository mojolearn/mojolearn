# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE ml-prep-nb (2026-10-04): the device's spelling of the lane-tree
order (x_prep/idn_fold.mojo) for `te_global`, `te_enc` and `ii_gram` under
IDENTICAL, on every vendor. x_prep/device.mojo calls `idn_tree_stage` for
every stage; it enqueues the stage here when its switch is on and returns
False otherwise (the units then run, already in the same order).

  te_global  one TREE_W-thread group per unit (fold, target column): thread
             l strides the rows l, l + TREE_W, ..., then the shared tree;
             the count is an exact integer tree; then the squared
             deviations the same way.
  te_enc     one group per unit (fold, feature, category, target column)
             over the category's bucket (BK > 0), the rank k - lo striding
             the threads; gathered words (GB > 0) or the bucket's row
             indices. BK == 0 (no buckets) keeps the unit.
  ii_gram    a chunk kernel, one thread per (chunk, cell a <= b), the
             chunk's IIG_ROWS rows serially (`ii_gram_chunk`, the unit's
             call), into the dw scratch; then one group per output cell,
             thread l striding the chunks, then the tree.

The trees are `add` in shared memory with the halving widths of `lt_tree`;
TREE_W = 256 threads, 2 KiB of shared words per group (fits every vendor).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_prep.common import FP, IP, STAGE_INTS, p, ld, ldi, st
from x_prep.prims import add, sub, mul, div
from x_prep.target import te_value
from x_prep.iterative import ii_gram_chunk
from x_prep.idn_fold import IDN_TE_GLOBAL_TREE, IDN_TE_ENC_TREE, IDN_II_GRAM_TILE, TREE_W, IIG_ROWS, iig_chunks

comptime OP_TE_GLOBAL = 20
comptime OP_TE_ENC = 21
comptime OP_II_GRAM = 54
comptime IDN_TREE_ANY = IDN_TE_GLOBAL_TREE or IDN_TE_ENC_TREE or IDN_II_GRAM_TILE


@always_inline
def _tree_f(sh: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], tid: Int) -> Float32:
    """`lt_tree` over the group's shared lanes (every thread calls it; all
    threads return lane 0's word)."""
    barrier()
    var w = TREE_W // 2
    while w >= 1:
        if tid < w:
            sh[tid] = add(sh[tid], sh[tid + w])
        barrier()
        w //= 2
    var r = sh[0]
    barrier()
    return r


@always_inline
def _tree_i(sh: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], tid: Int) -> Int:
    """The exact integer count over the group's shared lanes."""
    barrier()
    var w = TREE_W // 2
    while w >= 1:
        if tid < w:
            sh[tid] = sh[tid] + sh[tid + w]
        barrier()
        w //= 2
    var r = Int(sh[0])
    barrier()
    return r


# ------------------------------------------------------------- TargetEncoder
def te_global_lt_kernel(f: FP, q: IP):
    """`te_global_lt` (x_prep/target.mojo) for t = block_idx.x = fi*T + tt."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var Y = p(q, 0)
    var n = p(q, 1)
    var T = p(q, 2)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var c = 0
    for i in range(tid, n, TREE_W):
        if Int(ld(f, FO + i)) == fi:
            continue
        s = add(s, ld(f, Y + i * T + tt))
        c += 1
    sh[tid] = s
    shc[tid] = Int32(c)
    var sum_ = _tree_f(sh, tid)
    var cnt = _tree_i(shc, tid)
    var mean = Float32(0)
    var ss = Float32(0)
    if cnt > 0:
        mean = div(sum_, Float32(cnt))
        var e2 = Float32(0)
        for i in range(tid, n, TREE_W):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            e2 = add(e2, mul(e, e))
        sh[tid] = e2
        ss = div(_tree_f(sh, tid), Float32(cnt))
    if tid == 0:
        st(f, p(q, 4) + 2 * t, mean)
        st(f, p(q, 4) + 2 * t + 1, ss)


def te_enc_lt_kernel(f: FP, q: IP):
    """`te_enc_lt` (x_prep/target.mojo) with BK > 0 for t = block_idx.x =
    ((fi*d + j)*CMAX + cat)*T + tt."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var tt = t % T
    var r = t // T
    var cat = r % cmax
    var r2 = r // cmax
    var j = r2 % d
    var fi = r2 // d
    # a whole group returns together: no thread reaches a barrier
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
    var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
    var smooth = ld(f, p(q, 9))
    var S = p(q, 11) - 1 + j * (cmax + 1)
    var lo = ldi(f, S + cat)
    var hi = ldi(f, S + cat + 1)
    var gb = p(q, 13)
    var BF = -1
    var BY = 0
    if gb > 0:
        BF = gb - 1 + j * n
        BY = gb - 1 + d * n + (j * T + tt) * n
    var R = p(q, 12) + j * n
    var Y = p(q, 3)
    var FO = p(q, 5)
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var c = 0
    for k in range(lo + tid, hi, TREE_W):
        var fo: Int
        var y: Float32
        if BF >= 0:
            fo = Int(ld(f, BF + k))
            y = ld(f, BY + k)
        else:
            var i = ldi(f, R + k)
            fo = Int(ld(f, FO + i))
            y = ld(f, Y + i * T + tt)
        if fo == fi:
            continue
        s = add(s, y)
        c += 1
    sh[tid] = s
    shc[tid] = Int32(c)
    var sum_ = _tree_f(sh, tid)
    var cnt = _tree_i(shc, tid)
    var mean = Float32(0)
    var ssd = Float32(0)
    if smooth < Float32(0) and cnt > 0:
        mean = div(sum_, Float32(cnt))
        var e2 = Float32(0)
        for k in range(lo + tid, hi, TREE_W):
            var fo: Int
            var y: Float32
            if BF >= 0:
                fo = Int(ld(f, BF + k))
                y = ld(f, BY + k)
            else:
                var i = ldi(f, R + k)
                fo = Int(ld(f, FO + i))
                y = ld(f, Y + i * T + tt)
            if fo == fi:
                continue
            var e = sub(y, mean)
            e2 = add(e2, mul(e, e))
        sh[tid] = e2
        ssd = _tree_f(sh, tid)
    if tid == 0:
        st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, sum_, cnt, mean, ssd))


# ---------------------------------------------------------- IterativeImputer
def ii_gram_part_kernel(f: FP, q: IP, w: FP, chunks: Int32):
    """Chunk c = block_idx.x's partial of cell a*d + b, b >= a (cell =
    block_idx.y * TREE_W + thread_idx.x), into w[c*d*d + cell]."""
    if ld(f, p(q, 7)) != Float32(0):
        return
    var d = p(q, 2)
    var cell = Int(block_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= d * d:
        return
    var a = cell // d
    var b = cell % d
    if b < a:
        return
    var c = Int(block_idx.x)
    var n = p(q, 1)
    var lo = c * IIG_ROWS
    var hi = min(n, lo + IIG_ROWS)
    var ma = ld(f, p(q, 5) + a)
    var mb = ld(f, p(q, 5) + b)
    st(w, c * d * d + cell, ii_gram_chunk(f, p(q, 0), d, p(q, 3), p(q, 4), a, b, ma, mb, lo, hi))


def ii_gram_fold_kernel(f: FP, q: IP, w: FP, chunks: Int32):
    """G[t] for t = block_idx.x = a*d + b: the partials of (min, max), thread
    l striding the chunks ascending, then the tree (`ii_gram_lt`)."""
    if ld(f, p(q, 7)) != Float32(0):
        return
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var d = p(q, 2)
    var cell = min(t // d, t % d) * d + max(t // d, t % d)
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    for c in range(tid, Int(chunks), TREE_W):
        s = add(s, ld(w, c * d * d + cell))
    sh[tid] = s
    var g = _tree_f(sh, tid)
    if tid == 0:
        st(f, p(q, 6) + t, g)


# ------------------------------------------------------------------ dispatch
def idn_tree_scratch_words(host_q: IP, stages: Int) -> Int:
    """The float words of x_prep/device.mojo's `dw` scratch the program's
    tiled ii_gram stages need (0: none)."""
    var need = 0
    comptime if IDN_II_GRAM_TILE:
        for s in range(stages):
            if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_II_GRAM:
                var hq = host_q + (s * STAGE_INTS + 2)
                need = max(need, iig_chunks(Int(hq[1])) * Int(hq[2]) * Int(hq[2]))
    return need


def idn_tree_stage(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                   host_q: IP, s: Int, op: Int, total: Int, qp: IP) raises -> Bool:
    """Enqueue stage s in the lane-tree form when its switch is on and the
    stage fits; False leaves it to the units (the same order)."""
    var hq = host_q + (s * STAGE_INTS + 2)
    var f = FP(unsafe_from_address=Int(df.unsafe_ptr()))
    comptime if IDN_TE_GLOBAL_TREE:
        if op == OP_TE_GLOBAL:
            ctx.enqueue_function[te_global_lt_kernel](f, qp, grid_dim=total, block_dim=TREE_W)
            return True
    comptime if IDN_TE_ENC_TREE:
        if op == OP_TE_ENC and Int(hq[11]) > 0:
            ctx.enqueue_function[te_enc_lt_kernel](f, qp, grid_dim=total, block_dim=TREE_W)
            return True
    comptime if IDN_II_GRAM_TILE:
        if op == OP_II_GRAM:
            var d = Int(hq[2])
            var chunks = iig_chunks(Int(hq[1]))
            if d > 0:
                var w = FP(unsafe_from_address=Int(dw.unsafe_ptr()))
                ctx.enqueue_function[ii_gram_part_kernel](
                    f, qp, w, Int32(chunks), grid_dim=(chunks, (d * d + TREE_W - 1) // TREE_W), block_dim=TREE_W,
                )
                ctx.enqueue_function[ii_gram_fold_kernel](f, qp, w, Int32(chunks), grid_dim=total, block_dim=TREE_W)
                return True
    return False
