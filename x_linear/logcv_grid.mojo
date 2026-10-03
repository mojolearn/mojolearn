# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegressionCV wholly on the device (lane/neural-pass127,
2026-10-02: the row passes; cgr-linear: the folds, rows and scores; lane
cgr4-device-optim, 2026-10-03: the minimizer and the driver).

`logcv_fit_device` runs `logcv_fit`'s schedule (x_linear/logcv.mojo) with
every value on the device:
  * the objective `LcvObjective` (an `LbObjective`): the rows' terms
    (`logcv_map_row`, a thread per training row), every (block, cell)
    partial (`logcv_part_rows`), the partials folded per cell in block
    order (`fold_parts`), then one block finishing (`logcv_finish_t`: the
    scale, the penalty in the vfold order, the loss);
  * the minimizer x_linear/lbfgs_device.mojo (warm starts across Cs are
    the device theta carried over; a fold starts from zeros);
  * the held-out score into the device score table (`_predict_code` a
    thread a row, the hits folded per row block, the blocks folded);
  * the fold's training rows compacted on the grid (block counts, a block
    scan, the writes); the count comes home, a scalar per fold;
  * the best C (`lcv_best_kernel`: the fold means and the first best, the
    host column's statements) and the result words.
The StratifiedKFold ids are built on the grid from the labels
(`lcv_fold_ids_device`). Per line-search trial two flag words come home,
per fold one count, and the result once. The host column runs `logcv_fit`:
the same statements on the same rows in the same order: the same words.
"""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, ldi, i2f, fa, fd
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.scan import SC_NT, _sc_block_excl
from x_linear.tops import FOLD_BLOCK
from x_linear.team import team_at
from x_linear.tops import fold_blocks, fold_parts
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.vfold import vscratch
from x_linear.lbfgs_device import LbObjective, lbfgs_device, lbd_words, lbd_th, lbd_witness_words, LBD_TPB
from x_linear.logcv import (
    logcv_map_row, logcv_part_rows, logcv_finish_t, _predict_code,
    lcv_fold_table, lcv_fold_of, lcv_score_part, lcv_score_final,
)

comptime LCV_TPB = 256


def _blocks(count: Int) -> Int:
    return max((count + LCV_TPB - 1) // LCV_TPB, 1)


def lcv_map_kernel(x: FP, y: FP, n: Int32, d: Int32, kp: Int32, fi: Int32, sw: Int32, th: FP, ix: IP,
                   cnt: Int32, fold: Int32, rows: FP, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    if q < Int(cnt):
        var i = Int(ix.unsafe_load(q)) if Int(fold) >= 0 else q
        logcv_map_row(i, x, y, Int(n), Int(d), Int(kp), fi != 0, sw != 0, th, 0, rows)
    witness_end(wf, woff, nonce)

def lcv_part_kernel(x: FP, y: FP, n: Int32, d: Int32, kp: Int32, fi: Int32, sw: Int32, fold: Int32, ix: IP,
                    cnt: Int32, rows: FP, scr: FP, nbk: Int32, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var p = Int(kp) * (Int(d) + 1)
    var nb = Int(nbk)
    if q < (p + 2) * nb:
        var bk = q // (p + 2)
        var o = q - bk * (p + 2)
        st(scr, o * nb + bk, logcv_part_rows(o, x, y, Int(n), Int(d), Int(kp), fi != 0, sw != 0, Int(fold), ix,
                                             Int(cnt), bk, rows))
    witness_end(wf, woff, nonce)

def lcv_fold_kernel(scr: FP, nbk: Int32, p: Int32, g: FP, sums: FP, wf: IP, woff: Int32, nonce: Int32):
    var o = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var pp = Int(p)
    if o < pp + 2:
        var v = fold_parts(scr, o * Int(nbk), Int(nbk))
        if o < pp:
            st(g, o, v)
        else:
            st(sums, o - pp, v)
    witness_end(wf, woff, nonce)

def lcv_hit_kernel(x: FP, y: FP, n: Int32, d: Int32, kpp: Int32, fi: Int32, th: FP, f: Int32, hit: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        if Int(ld(y, nn + i)) == Int(f):
            var h = _predict_code(x, i, Int(d), Int(kpp), fi != 0, th, 0) == Int(ld(y, i))
            st(hit, i, Float32(1) if h else Float32(0))
    witness_end(wf, woff, nonce)

# ------------------------------------------------ StratifiedKFold ids (cgr-linear)
def lcv_class_count_kernel(y: FP, n: Int32, nk: Int32, bc: IP, bf: IP, wf: IP, woff: Int32, nonce: Int32):
    """Block b: each class's count and first row among its LCV_TPB rows."""
    var nn = Int(n)
    var nbk = _blocks(nn)
    var b = Int(block_idx.x)
    var lo = b * LCV_TPB
    var hi = min(lo + LCV_TPB, nn)
    for c in range(Int(thread_idx.x), Int(nk), LCV_TPB):
        var cnt = 0
        var fst = nn
        for i in range(lo, hi):
            if Int(ld(y, i)) == c:
                if cnt == 0:
                    fst = i
                cnt += 1
        bc.unsafe_store(c * nbk + b, Int32(cnt))
        bf.unsafe_store(c * nbk + b, Int32(fst))
    witness_end(wf, woff, nonce)


def lcv_class_scan_kernel(bc: IP, bf: IP, nbk: Int32, counts: IP, first: IP, wf: IP, woff: Int32, nonce: Int32):
    """Block c: class c's block counts to exclusive offsets (a block scan),
    its total and its first row."""
    var part = stack_allocation[SC_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var c = Int(block_idx.x)
    var nb = Int(nbk)
    var tid = Int(thread_idx.x)
    var total = _sc_block_excl(bc, bc, c * nb, nb, Int32(0), part)
    var mn = Int32(0x7FFFFFFF)
    for b in range(tid, nb, SC_NT):
        mn = min(mn, bf.unsafe_load(c * nb + b))
    part[tid] = mn
    barrier()
    var h = SC_NT // 2
    while h > 0:
        if tid < h:
            part[tid] = min(part[tid], part[tid + h])
        barrier()
        h //= 2
    if tid == 0:
        counts.unsafe_store(c, total)
        first.unsafe_store(c, part[0])
    witness_end(wf, woff, nonce)


def lcv_fold_table_kernel(counts: IP, first: IP, nk: Int32, k: Int32, order: IP, table: IP,
                          wf: IP, woff: Int32, nonce: Int32):
    """One thread: the K x k fold table (`lcv_fold_table`)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        lcv_fold_table(counts, first, Int(nk), Int(k), order, table)
    witness_end(wf, woff, nonce)


def lcv_fold_id_kernel(y: FP, n: Int32, k: Int32, bc: IP, table: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per row: its rank in its class (the class's block offset plus
    the earlier rows of its block), then its fold into y[n + i]."""
    var nn = Int(n)
    var i = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    if i < nn:
        var nbk = _blocks(nn)
        var c = Int(ld(y, i))
        var b = i // LCV_TPB
        var r = Int(bc.unsafe_load(c * nbk + b))
        for j in range(b * LCV_TPB, i):
            if Int(ld(y, j)) == c:
                r += 1
        st(y, nn + i, i2f(lcv_fold_of(table, c, Int(k), r)))
    witness_end(wf, woff, nonce)


def lcv_fold_ids_device(var ctx: DeviceContext, y: FP, n: Int, nk: Int, k: Int) raises:
    """The fold ids into the device labels' y[n, 2n) (classes 0..nk-1, k folds)."""
    var nbk = _blocks(n)
    var dbc = ctx.enqueue_create_buffer[DType.int32](max(nk * nbk, 1))
    var dbf = ctx.enqueue_create_buffer[DType.int32](max(nk * nbk, 1))
    var dcn = ctx.enqueue_create_buffer[DType.int32](max(nk, 1))
    var dfs = ctx.enqueue_create_buffer[DType.int32](max(nk, 1))
    var dord = ctx.enqueue_create_buffer[DType.int32](max(nk, 1))
    var dtab = ctx.enqueue_create_buffer[DType.int32](max(nk * (k + 1), 1))
    var wit = Witness(ctx, 2 * nbk + nk + 2)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[lcv_class_count_kernel](
            y, Int32(n), Int32(nk), dbc.unsafe_ptr(), dbf.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=nbk, block_dim=LCV_TPB,
        )
        wo += nbk
        ctx.enqueue_function[lcv_class_scan_kernel](
            dbc.unsafe_ptr(), dbf.unsafe_ptr(), Int32(nbk), dcn.unsafe_ptr(), dfs.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=nk, block_dim=SC_NT,
        )
        wo += nk
        ctx.enqueue_function[lcv_fold_table_kernel](
            dcn.unsafe_ptr(), dfs.unsafe_ptr(), Int32(nk), Int32(k), dord.unsafe_ptr(), dtab.unsafe_ptr(),
            wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1,
        )
        wo += 1
        ctx.enqueue_function[lcv_fold_id_kernel](
            y, Int32(n), Int32(k), dbc.unsafe_ptr(), dtab.unsafe_ptr(), wit.p(), Int32(wo), nonce,
            grid_dim=nbk, block_dim=LCV_TPB,
        )
        wo += nbk
        if wit.ok(ctx, wo, "LogisticRegressionCV folds"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = dbc^
    _ = dbf^
    _ = dcn^
    _ = dfs^
    _ = dord^
    _ = dtab^
    _ = wit^


# ------------------------------------------------ a fold's training rows (cgr-linear)
@always_inline
def _lcv_train(y: FP, n: Int, fold: Int, i: Int) -> Int:
    return 1 if (i < n and Int(ld(y, n + i)) != fold) else 0


def lcv_rows_count_kernel(y: FP, n: Int32, fold: Int32, bcnt: IP, wf: IP, woff: Int32, nonce: Int32):
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * LCV_TPB + tid
    var sh = stack_allocation[LCV_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(_lcv_train(y, Int(n), Int(fold), i))
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(LCV_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)
    witness_end(wf, woff, nonce)


def lcv_rows_scan_kernel(bcnt: IP, nbk: Int32, tot: IP, wf: IP, woff: Int32, nonce: Int32):
    var part = stack_allocation[SC_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var total = _sc_block_excl(bcnt, bcnt, 0, Int(nbk), Int32(0), part)
    if Int(thread_idx.x) == 0:
        tot.unsafe_store(0, total)
    witness_end(wf, woff, nonce)


def lcv_rows_write_kernel(y: FP, n: Int32, fold: Int32, bcnt: IP, ix: IP, wf: IP, woff: Int32, nonce: Int32):
    """The training rows ascending: row i at its block's offset plus the
    training rows before it in the block."""
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * LCV_TPB + tid
    var sh = stack_allocation[LCV_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var f = _lcv_train(y, Int(n), Int(fold), i)
    sh[tid] = Int32(f)
    barrier()
    if f != 0:
        var r = Int32(0)
        for u in range(tid):
            r += sh[u]
        ix.unsafe_store(Int(bcnt.unsafe_load(Int(block_idx.x)) + r), Int32(i))
    witness_end(wf, woff, nonce)


# ------------------------------------------------ the held-out score (cgr-linear)
def lcv_score_parts_kernel(y: FP, n: Int32, f: Int32, sw: Int32, hit: FP, sp: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per FOLD_BLOCK rows: the fold's (weighted) hits and rows from zero."""
    var nn = Int(n)
    var nb = fold_blocks(nn)
    var b = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    if b < nb:
        var lo = b * FOLD_BLOCK
        var p = lcv_score_part(y, hit, nn, Int(f), sw != 0, lo, min(FOLD_BLOCK, nn - lo))
        st(sp, b, p[0])
        st(sp, nb + b, p[1])
    witness_end(wf, woff, nonce)


def lcv_score_fin_kernel(sp: FP, nb: Int32, res: FP, wf: IP, woff: Int32, nonce: Int32):
    """One thread: the nb block partials folded blocks ascending, the score."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var b = Int(nb)
        st(res, 0, lcv_score_final(fold_parts(sp, 0, b), fold_parts(sp, b, b)))
    witness_end(wf, woff, nonce)


def lcv_finish_kernel(th: FP, g: FP, f: FP, outs: FP, cs: FP, kp: Int32, d: Int32, sw: Int32, cnt: Int32,
                      parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """One block: `logcv_finish_t` on the folded sums; f[0] by the lead."""
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), parts, 0, 0, 0)
    var swb = sw != 0
    var fv = logcv_finish_t(t, g, 0, th, 0, Int(kp), Int(d), swb, ld(cs, 0), Int(cnt), ld(outs, 0),
                            ld(outs, 1) if swb else Float32(0), parts)
    if t.lead():
        st(f, 0, fv)
    witness_end(wf, woff, nonce)


def lcv_theta_kernel(lw: FP, p: Int32, c: Int32):
    """theta[0] = theta[c] (c = 1), or zeros (c < 0): a thread an entry."""
    var j = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var pp = Int(p)
    if j < pp:
        if Int(c) < 0:
            st(lw, lbd_th(pp, 0) + j, Float32(0))
        elif Int(c) == 1:
            st(lw, lbd_th(pp, 0) + j, ld(lw, lbd_th(pp, 1) + j))


def lcv_setc_kernel(cs: FP, cvals: FP, ci: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        st(cs, 0, ld(cvals, Int(ci)))


def lcv_best_kernel(scores: FP, nf: Int32, nc: Int32, cvals: FP, cs: FP):
    """`logcv_fit`'s choice: each C's fold mean, the first best (DEVIATION
    5005), C_ = Cs[best] into cs[0] and best into cs[1] (bits). nc x nf
    scalars: the one thread is the choice, not a pass over rows."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var best = 0
        var bs = Float32(0)
        var f = Int(nf)
        var c = Int(nc)
        for ci in range(c):
            var acc = Float32(0)
            for k in range(f):
                acc = fa(acc, ld(scores, k * c + ci))
            var m = fd(acc, i2f(f))
            if ci == 0 or m > bs:
                best = ci
                bs = m
        st(cs, 0, ld(cvals, best))


def lcv_result_kernel(lw: FP, c: Int32, res: FP, kp: Int32, d: Int32, fi: Int32, cs: FP, iters: Int32,
                      scores: FP, nsc: Int32):
    """res: coef K'*d | intercept K' | C_ | n_iter | scores F*nC."""
    var q = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var k1 = Int(kp)
    var dd = Int(d)
    var stride = dd + 1
    var p = k1 * stride
    var th = lw + lbd_th(p, Int(c))
    if q < k1 * dd:
        var k = q // dd
        st(res, q, ld(th, k * stride + (q - k * dd)))
    elif q < k1 * dd + k1:
        var k = q - k1 * dd
        st(res, q, ld(th, k * stride + dd) if fi != 0 else Float32(0))
    elif q == k1 * dd + k1:
        st(res, q, ld(cs, 0))
    elif q == k1 * dd + k1 + 1:
        var it = Int(iters)
        st(res, q, i2f(it if it >= 0 else -it))
    elif q < k1 * dd + k1 + 2 + Int(nsc):
        st(res, q, ld(scores, q - (k1 * dd + k1 + 2)))


struct LcvObjective(LbObjective):
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var rows: DeviceBuffer[DType.float32]
    var scr: DeviceBuffer[DType.float32]
    var outs: DeviceBuffer[DType.float32]
    var parts: DeviceBuffer[DType.float32]
    var cs: DeviceBuffer[DType.float32]
    var ix: DeviceBuffer[DType.int32]
    var n: Int
    var d: Int
    var kp: Int
    var fi: Bool
    var sw: Bool
    var fold: Int
    var cnt: Int

    def __init__(out self, mut ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, kp: Int,
                 fi: Bool, sw: Bool) raises:
        var p = kp * (d + 1)
        self.x = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
        self.y = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
        self.rows = ctx.enqueue_create_buffer[DType.float32](max((kp + 1) * n, 1))
        self.scr = ctx.enqueue_create_buffer[DType.float32](max((p + 2) * fold_blocks(n), 1))
        self.outs = ctx.enqueue_create_buffer[DType.float32](2)
        self.parts = ctx.enqueue_create_buffer[DType.float32](vscratch(kp * d) + 16)
        self.cs = ctx.enqueue_create_buffer[DType.float32](2)
        self.ix = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
        if n_x > 0:
            ctx.enqueue_copy(dst_buf=self.x, src_ptr=x)
        if n_y > 0:
            ctx.enqueue_copy(dst_buf=self.y, src_ptr=y)
        self.n = n
        self.d = d
        self.kp = kp
        self.fi = fi
        self.sw = sw
        self.fold = -1
        self.cnt = n

    def _p(self) -> Int:
        return self.kp * (self.d + 1)

    def blocks_at(self, cnt: Int) -> Int:
        var p = self._p()
        return _blocks(cnt) + _blocks((p + 2) * fold_blocks(cnt)) + _blocks(p + 2) + 1

    def blocks(self) -> Int:
        return self.blocks_at(self.cnt)

    def enqueue(mut self, mut ctx: DeviceContext, th: FP, g: FP, f: FP, wf: IP, woff: Int, nonce: Int32) raises:
        var n = self.n
        var d = self.d
        var kp = self.kp
        var p = self._p()
        var cnt = self.cnt
        var nbk = fold_blocks(cnt)
        var fi = Int32(1 if self.fi else 0)
        var sw = Int32(1 if self.sw else 0)
        var b1 = _blocks(cnt)
        var b2 = _blocks((p + 2) * nbk)
        var b3 = _blocks(p + 2)
        ctx.enqueue_function[lcv_map_kernel](
            self.x.unsafe_ptr(), self.y.unsafe_ptr(), Int32(n), Int32(d), Int32(kp), fi, sw, th,
            self.ix.unsafe_ptr(), Int32(cnt), Int32(self.fold), self.rows.unsafe_ptr(), wf, Int32(woff), nonce,
            grid_dim=b1, block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_part_kernel](
            self.x.unsafe_ptr(), self.y.unsafe_ptr(), Int32(n), Int32(d), Int32(kp), fi, sw, Int32(self.fold),
            self.ix.unsafe_ptr(), Int32(cnt), self.rows.unsafe_ptr(), self.scr.unsafe_ptr(), Int32(nbk), wf,
            Int32(woff + b1), nonce, grid_dim=b2, block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_fold_kernel](
            self.scr.unsafe_ptr(), Int32(nbk), Int32(p), g, self.outs.unsafe_ptr(), wf, Int32(woff + b1 + b2), nonce,
            grid_dim=b3, block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_finish_kernel](  # small-launch(kp: classes): one block strides the K-prime by d+1 gradient cells and one vfold sum
            th, g, f, self.outs.unsafe_ptr(), self.cs.unsafe_ptr(), Int32(kp), Int32(d), sw, Int32(cnt),
            self.parts.unsafe_ptr(), wf, Int32(woff + b1 + b2 + b3), nonce, grid_dim=1, block_dim=LBD_TPB,
        )


def _lcv_rows(mut ctx: DeviceContext, mut obj: LcvObjective, mut wit: Witness, mut bcnt: DeviceBuffer[DType.int32],
              mut tot: DeviceBuffer[DType.int32], fold: Int) raises -> Int:
    """The fold's training rows ascending into obj.ix; the count home."""
    var n = obj.n
    var nbk = _blocks(n)
    var cnt = List[Int32](length=1, fill=Int32(0))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wf = wit.p()
        ctx.enqueue_function[lcv_rows_count_kernel](
            obj.y.unsafe_ptr(), Int32(n), Int32(fold), bcnt.unsafe_ptr(), wf, Int32(0), nonce,
            grid_dim=nbk, block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_rows_scan_kernel](
            bcnt.unsafe_ptr(), Int32(nbk), tot.unsafe_ptr(), wf, Int32(nbk), nonce, grid_dim=1, block_dim=SC_NT,
        )
        ctx.enqueue_function[lcv_rows_write_kernel](
            obj.y.unsafe_ptr(), Int32(n), Int32(fold), bcnt.unsafe_ptr(), obj.ix.unsafe_ptr(), wf, Int32(nbk + 1),
            nonce, grid_dim=nbk, block_dim=LCV_TPB,
        )
        ctx.enqueue_copy(dst_ptr=cnt.unsafe_ptr(), src_buf=tot)
        if wit.ok(ctx, 2 * nbk + 1, "LogisticRegressionCV rows"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    var c = Int(cnt[0])
    _ = cnt^
    return c


def _lcv_score(mut ctx: DeviceContext, mut obj: LcvObjective, mut wit: Witness, lw: FP, c: Int,
               mut hit: DeviceBuffer[DType.float32], mut sp: DeviceBuffer[DType.float32], f: Int, weighted: Bool,
               dst: FP) raises:
    """The held-out score of theta[c] on fold f into dst[0] (device)."""
    var n = obj.n
    var d = obj.d
    var kpp = obj.kp if obj.kp > 1 else 1
    var nb = fold_blocks(n)
    var g = _blocks(nb)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wf = wit.p()
        ctx.enqueue_function[lcv_hit_kernel](
            obj.x.unsafe_ptr(), obj.y.unsafe_ptr(), Int32(n), Int32(d), Int32(kpp), Int32(1 if obj.fi else 0),
            lw + lbd_th(obj._p(), c), Int32(f), hit.unsafe_ptr(), wf, Int32(0), nonce, grid_dim=_blocks(n),
            block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_score_parts_kernel](
            obj.y.unsafe_ptr(), Int32(n), Int32(f), Int32(1 if weighted else 0), hit.unsafe_ptr(), sp.unsafe_ptr(),
            wf, Int32(_blocks(n)), nonce, grid_dim=g, block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_score_fin_kernel](
            sp.unsafe_ptr(), Int32(nb), dst, wf, Int32(_blocks(n) + g), nonce, grid_dim=1, block_dim=1,
        )
        if wit.ok(ctx, _blocks(n) + g + 1, "LogisticRegressionCV score"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()


def logcv_fit_grid(
    ctx: DeviceContext, algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """`logcv_fit` on the device (see the module notes)."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var kp = Int(ip[2])
    var nc = Int(ip[3])
    var nf = Int(ip[4])
    var sw = Int(ip[5]) != 0
    var tol = fp[0]
    var p = kp * (d + 1)
    var c = ctx.copy()
    var obj = LcvObjective(c, x, n_x, y, n_y, n, d, kp, fi, sw)
    var wcap = max(lbd_witness_words(obj.blocks_at(n), p), 2 * _blocks(n) + _blocks(fold_blocks(n)) + 2)
    var wit = Witness(c, wcap)
    var lw = c.enqueue_create_buffer[DType.float32](lbd_words(p))
    lw.enqueue_fill(Float32(0))
    var hit = c.enqueue_create_buffer[DType.float32](max(n, 1))
    var sp = c.enqueue_create_buffer[DType.float32](max(2 * fold_blocks(n), 1))
    var bcnt = c.enqueue_create_buffer[DType.int32](_blocks(n) + 1)
    var tot = c.enqueue_create_buffer[DType.int32](1)
    var scores = c.enqueue_create_buffer[DType.float32](max(nf * nc, 1))
    scores.enqueue_fill(Float32(0))
    var cvals = c.enqueue_create_buffer[DType.float32](max(nc, 1))
    var hc = List[Float32](length=max(nc, 1), fill=Float32(0))
    for ci in range(nc):
        hc[ci] = fp[1 + ci]
    c.enqueue_copy(dst_buf=cvals, src_ptr=hc.unsafe_ptr())
    # the StratifiedKFold ids from the device labels (the caller sent zeros)
    lcv_fold_ids_device(c.copy(), FP(unsafe_from_address=Int(obj.y.unsafe_ptr())), n, max(kp, 2), nf)
    c.synchronize()
    _ = hc^
    var lp = FP(unsafe_from_address=Int(lw.unsafe_ptr()))
    var bt = _blocks(p)
    for f in range(nf):
        obj.fold = f
        obj.cnt = _lcv_rows(c, obj, wit, bcnt, tot, f)
        c.enqueue_function[lcv_theta_kernel](lp, Int32(p), Int32(-1), grid_dim=bt, block_dim=LCV_TPB)
        for ci in range(nc):
            c.enqueue_function[lcv_setc_kernel](obj.cs.unsafe_ptr(), cvals.unsafe_ptr(), Int32(ci), grid_dim=1,  # small-launch(ci: the C index): one thread copies one scalar C into the objective slot
                                                block_dim=1)
            var r = lbfgs_device(c, obj, wit, lw, p, max_iter, tol, "LogisticRegressionCV fit")
            _lcv_score(c, obj, wit, lp, r[1], hit, sp, f, sw, FP(unsafe_from_address=Int(scores.unsafe_ptr())) + f * nc + ci)
            # the warm start: the next C starts from this theta
            c.enqueue_function[lcv_theta_kernel](lp, Int32(p), Int32(r[1]), grid_dim=bt, block_dim=LCV_TPB)
    c.enqueue_function[lcv_best_kernel](scores.unsafe_ptr(), Int32(nf), Int32(nc), cvals.unsafe_ptr(),  # small-launch(nc: the C grid): one thread folds the nf by nc score table and picks the first best C
                                        obj.cs.unsafe_ptr(), grid_dim=1, block_dim=1)
    obj.fold = -1
    obj.cnt = n
    c.enqueue_function[lcv_theta_kernel](lp, Int32(p), Int32(-1), grid_dim=bt, block_dim=LCV_TPB)
    var r = lbfgs_device(c, obj, wit, lw, p, max_iter, tol, "LogisticRegressionCV fit")
    var dres = c.enqueue_create_buffer[DType.float32](max(n_out, 1))
    dres.enqueue_fill(Float32(0))
    var nsc = nf * nc
    c.enqueue_function[lcv_result_kernel](lp, Int32(r[1]), dres.unsafe_ptr(), Int32(kp), Int32(d),
                                          Int32(1 if fi else 0), obj.cs.unsafe_ptr(), Int32(r[0]),
                                          scores.unsafe_ptr(), Int32(nsc), grid_dim=_blocks(kp * d + kp + 2 + nsc),
                                          block_dim=LCV_TPB)
    c.enqueue_copy(dst_ptr=res, src_buf=dres)
    c.synchronize()
    _ = obj^
    _ = wit^
    _ = lw^
    _ = hit^
    _ = sp^
    _ = bcnt^
    _ = tot^
    _ = scores^
    _ = cvals^
    _ = dres^
