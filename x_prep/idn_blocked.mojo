# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE classical-te-gmm (2026-10-07): the device's spelling of the BLOCKED
order `BLT` (x_prep/idn_fold.mojo) for `te_global` and `te_enc` under
IDENTICAL, on every vendor; default OFF, the A/B define
-D MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD. x_prep/device.mojo calls
`te_blocked_stage` for every stage; it enqueues the stage here when the
switch is on and the stage fits, and returns False otherwise (the units
then run, already in the same order: x_prep/target.mojo `te_global_blt`,
`te_enc_blt`).

  te_global  part kernel, grid (n / TEB row blocks, units): group (m, t)
             folds block m's rows outside fold fi, thread l the rows
             i = m*TEB + l, + TREE_W, .. (lane i mod TREE_W, TEB a multiple
             of TREE_W), then the shared tree -> w[t*nb + m]; the exact
             count -> w[PC + t*nb + m]. fold kernel, one group per unit:
             thread l the blocks m = l, l + TREE_W, .. (lane m mod TREE_W),
             then the tree. Pass 0: the sum and count, the mean to META[2t];
             pass 1: the squared deviations from that mean, META[2t+1].
  te_enc     per fold fi (a loop over the program's folds on the host; the
             scratch holds d*T planes of n words twice): part kernel, grid
             (n / TEB position blocks, d*T (column, target) planes): group
             (m, g) walks block m of column j's bucket array one category
             SEGMENT at a time (START's lower bound of the block's first
             position, then the following categories while they start in
             the block): the segment's tree -> w[g*n + its first position],
             its count beside it; the words are the gathered ones (GB > 0)
             or the bucket's row indices, as `te_enc_lt_kernel` reads them.
             fold kernel, one group per (column, category, target) unit of
             fold fi: thread l the blocks m = l mod TREE_W over the
             category's [lo, hi), each partial at position max(lo, m*TEB),
             then the tree. Pass 0 writes the encoding (smooth >= 0) or the
             mean (smooth < 0) into ENC[t]; pass 1 (smooth < 0 only) the
             encoding from the count, the mean and the squared deviations.

The trees are x_prep/idn_tree.mojo's `_tree_f` / `_tree_i` (`add` in shared
memory, the halving widths of `lt_tree`); every group returns as a whole
before its first barrier, or not at all. Scratch words: te_global 2 * units
* (n / TEB); te_enc 2 * d * T * n (`te_blocked_scratch_words`).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from x_prep.common import FP, IP, STAGE_INTS, p, ld, ldi, st, sti
from x_prep.prims import add, sub, mul, div
from x_prep.target import te_value, _te_pos
from x_prep.idn_fold import IDN_TE_BLOCKED, TREE_W, TEB
from x_prep.idn_tree import _tree_f, _tree_i

comptime OP_TE_GLOBAL = 20
comptime OP_TE_ENC = 21
#: the largest grid.y extent every vendor's launch accepts (hardware, not a shape rule)
comptime _GRID_Y_MAX = 65535


@always_inline
def _nblocks(n: Int) -> Int:
    """The TEB-blocks of n positions (at least one)."""
    return max(1, (n + TEB - 1) // TEB)


@always_inline
def _blt_first(lo: Int, tid: Int) -> Int:
    """The first value at or after lo whose lane (value mod TREE_W) is tid."""
    var k = lo - lo % TREE_W + tid
    if k < lo:
        k += TREE_W
    return k


# ------------------------------------------------------------- te_global
def te_global_blt_part_kernel(f: FP, q: IP, w: FP, nb: Int32, units: Int32, pass_: Int32):
    """`te_global_blt`'s block m = block_idx.x of unit t = block_idx.y: the
    block's tree of the targets (pass 0; the count beside it at PC) or of
    the squared deviations from META[2t] (pass 1), to w[t*nb + m]."""
    var m = Int(block_idx.x)
    var t = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var Y = p(q, 0)
    var n = p(q, 1)
    var T = p(q, 2)
    var FO = p(q, 3)
    var fi = t // T
    var tt = t % T
    var nbi = Int(nb)
    var hi = min(n, (m + 1) * TEB)
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var c = 0
    if Int(pass_) == 0:
        for i in range(m * TEB + tid, hi, TREE_W):
            if Int(ld(f, FO + i)) == fi:
                continue
            s = add(s, ld(f, Y + i * T + tt))
            c += 1
    else:
        var mean = ld(f, p(q, 4) + 2 * t)
        for i in range(m * TEB + tid, hi, TREE_W):
            if Int(ld(f, FO + i)) == fi:
                continue
            var e = sub(ld(f, Y + i * T + tt), mean)
            s = add(s, mul(e, e))
    sh[tid] = s
    var part = _tree_f(sh, tid)
    if Int(pass_) == 0:
        shc[tid] = Int32(c)
        var cnt = _tree_i(shc, tid)
        if tid == 0:
            sti(w, Int(units) * nbi + t * nbi + m, cnt)
    if tid == 0:
        st(w, t * nbi + m, part)


def te_global_blt_fold_kernel(f: FP, q: IP, w: FP, nb: Int32, units: Int32, pass_: Int32):
    """`te_global_blt`'s outer tree for unit t = block_idx.x: thread l the
    blocks m = l mod TREE_W ascending, the count exact; pass 0 the mean to
    META[2t], pass 1 the variance to META[2t+1]."""
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nbi = Int(nb)
    var PC = Int(units) * nbi
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var c = 0
    for m in range(tid, nbi, TREE_W):
        s = add(s, ld(w, t * nbi + m))
        c += ldi(w, PC + t * nbi + m)
    sh[tid] = s
    shc[tid] = Int32(c)
    var sum_ = _tree_f(sh, tid)
    var cnt = _tree_i(shc, tid)
    if tid == 0:
        var v = Float32(0)
        if cnt > 0:
            v = div(sum_, Float32(cnt))
        if Int(pass_) == 0:
            st(f, p(q, 4) + 2 * t, v)
        else:
            st(f, p(q, 4) + 2 * t + 1, v)


# ---------------------------------------------------------------- te_enc
def te_enc_blt_part_kernel(f: FP, q: IP, w: FP, fi_in: Int32, pass_: Int32):
    """Block m = block_idx.x of column j's bucket array for plane
    g = block_idx.y = j*T + tt and fold fi: every category segment the block
    holds, its tree (pass 0: the targets, the count beside it; pass 1: the
    squared deviations from the category's mean in ENC[t]) to
    w[g*n + the segment's first position]."""
    var m = Int(block_idx.x)
    var g = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var fi = Int(fi_in)
    var tt = g % T
    var j = g // T
    var smooth = ld(f, p(q, 9))
    # a whole group returns together: no thread reaches a barrier
    if Int(pass_) != 0 and smooth >= Float32(0):
        return
    var S = p(q, 11) - 1 + j * (cmax + 1)
    var blo = m * TEB
    var bhi = min(ldi(f, S + cmax), blo + TEB)
    if bhi <= blo:
        return
    # the first category whose end is past the block's first position
    var lo_c = 0
    var hi_c = cmax
    while lo_c < hi_c:
        var mid = (lo_c + hi_c) // 2
        if ldi(f, S + mid + 1) > blo:
            hi_c = mid
        else:
            lo_c = mid + 1
    var cat = lo_c
    var gb = p(q, 13)
    var BF = -1
    var BY = 0
    if gb > 0:
        BF = gb - 1 + j * n
        BY = gb - 1 + d * n + (j * T + tt) * n
    var R = p(q, 12) + j * n
    var Y = p(q, 3)
    var FO = p(q, 5)
    var PS = g * n
    var PC = d * T * n + g * n
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    while cat < cmax:
        var clo = ldi(f, S + cat)
        if clo >= bhi:
            break
        var slo = max(clo, blo)
        var shi = min(ldi(f, S + cat + 1), bhi)
        if shi > slo:
            var s = Float32(0)
            var c = 0
            if Int(pass_) == 0:
                for k in range(_blt_first(slo, tid), shi, TREE_W):
                    var v = _te_pos(f, f, BF, BY, R, Y, T, tt, FO, k)
                    if v[0] == fi:
                        continue
                    s = add(s, v[1])
                    c += 1
            else:
                var mean = ld(f, p(q, 10) + ((fi * d + j) * cmax + cat) * T + tt)
                for k in range(_blt_first(slo, tid), shi, TREE_W):
                    var v = _te_pos(f, f, BF, BY, R, Y, T, tt, FO, k)
                    if v[0] == fi:
                        continue
                    var e = sub(v[1], mean)
                    s = add(s, mul(e, e))
            sh[tid] = s
            var part = _tree_f(sh, tid)
            if Int(pass_) == 0:
                shc[tid] = Int32(c)
                var cnt = _tree_i(shc, tid)
                if tid == 0:
                    sti(w, PC + slo, cnt)
            if tid == 0:
                st(w, PS + slo, part)
        cat += 1


def te_enc_blt_fold_kernel(f: FP, q: IP, w: FP, fi_in: Int32, pass_: Int32):
    """Unit u = block_idx.x = (j*CMAX + cat)*T + tt of fold fi: the outer
    tree over the category's blocks (thread l the blocks m = l mod TREE_W),
    the count exact; pass 0 writes ENC[t] (the encoding when smooth >= 0,
    else the mean), pass 1 the encoding from the mean and the squared
    deviations."""
    var u = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var n = p(q, 1)
    var d = p(q, 2)
    var T = p(q, 4)
    var cmax = p(q, 6)
    var fi = Int(fi_in)
    var tt = u % T
    var r = u // T
    var cat = r % cmax
    var j = r // cmax
    # a whole group returns together: no thread reaches a barrier
    if cat >= Int(ld(f, p(q, 7) + j)):
        return
    var smooth = ld(f, p(q, 9))
    if Int(pass_) != 0 and smooth >= Float32(0):
        return
    var t = ((fi * d + j) * cmax + cat) * T + tt
    var S = p(q, 11) - 1 + j * (cmax + 1)
    var lo = ldi(f, S + cat)
    var hi = ldi(f, S + cat + 1)
    var g = j * T + tt
    var PS = g * n
    var PC = d * T * n + g * n
    var sh = stack_allocation[TREE_W, Float32, address_space = AddressSpace.SHARED]()
    var shc = stack_allocation[TREE_W, Int32, address_space = AddressSpace.SHARED]()
    var s = Float32(0)
    var c = 0
    if hi > lo:
        var mlo = lo // TEB
        var mhi = (hi - 1) // TEB
        for m in range(_blt_first(mlo, tid), mhi + 1, TREE_W):
            var at = max(lo, m * TEB)
            s = add(s, ld(w, PS + at))
            c += ldi(w, PC + at)
    sh[tid] = s
    shc[tid] = Int32(c)
    var sum_ = _tree_f(sh, tid)
    var cnt = _tree_i(shc, tid)
    if tid == 0:
        var ymean = ld(f, p(q, 8) + 2 * (fi * T + tt))
        var yvar = ld(f, p(q, 8) + 2 * (fi * T + tt) + 1)
        if Int(pass_) == 0:
            if smooth >= Float32(0):
                st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, sum_, cnt, Float32(0), Float32(0)))
            else:
                var mean = Float32(0)
                if cnt > 0:
                    mean = div(sum_, Float32(cnt))
                st(f, p(q, 10) + t, mean)
        else:
            var mean = ld(f, p(q, 10) + t)
            st(f, p(q, 10) + t, te_value(ymean, yvar, smooth, Float32(0), cnt, mean, sum_))


# ------------------------------------------------------------------ dispatch
def te_blocked_scratch_words(host_q: IP, stages: Int) -> Int:
    """The words of x_prep/device.mojo's `dw` scratch the program's blocked
    te_global / te_enc stages need (0: none)."""
    var need = 0
    comptime if IDN_TE_BLOCKED:
        for s in range(stages):  # small-loop(stages: program stages): reads the op words of one program, a plan list, never data
            var op = Int(host_q.unsafe_load(s * STAGE_INTS))
            var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
            var hq = host_q + (s * STAGE_INTS + 2)
            if op == OP_TE_GLOBAL:
                need = max(need, 2 * total * _nblocks(Int(hq[1])))
            elif op == OP_TE_ENC and Int(hq[11]) > 0:
                need = max(need, 2 * Int(hq[2]) * Int(hq[4]) * Int(hq[1]))
    return need


def te_blocked_stage(ctx: DeviceContext, mut df: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.uint32],
                     host_q: IP, s: Int, op: Int, total: Int, qp: IP) raises -> Bool:
    """Enqueue stage s in the blocked form when the switch is on and the
    stage fits the launch; False leaves it to the units (the same order)."""
    comptime if IDN_TE_BLOCKED:
        var hq = host_q + (s * STAGE_INTS + 2)
        var f = FP(unsafe_from_address=Int(df.unsafe_ptr()))
        var w = FP(unsafe_from_address=Int(dw.unsafe_ptr()))
        if op == OP_TE_GLOBAL and total > 0 and total <= _GRID_Y_MAX:
            var nb = _nblocks(Int(hq[1]))
            for pass_ in range(2):  # small-loop(2: the sum pass, then the squared-deviation pass)
                ctx.enqueue_function[te_global_blt_part_kernel](
                    f, qp, w, Int32(nb), Int32(total), Int32(pass_), grid_dim=(nb, total), block_dim=TREE_W,
                )
                ctx.enqueue_function[te_global_blt_fold_kernel](
                    f, qp, w, Int32(nb), Int32(total), Int32(pass_), grid_dim=total, block_dim=TREE_W,
                )
            return True
        if op == OP_TE_ENC and Int(hq[11]) > 0:
            var planes = Int(hq[2]) * Int(hq[4])
            var units = planes * Int(hq[6])
            if units <= 0 or planes > _GRID_Y_MAX:
                return False
            var nb = _nblocks(Int(hq[1]))
            for fi in range(total // units):  # small-loop(folds: the program's fold count, a plan number, never data)
                for pass_ in range(2):  # small-loop(2: the sum pass, then the squared-deviation pass)
                    ctx.enqueue_function[te_enc_blt_part_kernel](
                        f, qp, w, Int32(fi), Int32(pass_), grid_dim=(nb, planes), block_dim=TREE_W,
                    )
                    ctx.enqueue_function[te_enc_blt_fold_kernel](
                        f, qp, w, Int32(fi), Int32(pass_), grid_dim=units, block_dim=TREE_W,
                    )
            return True
    return False
