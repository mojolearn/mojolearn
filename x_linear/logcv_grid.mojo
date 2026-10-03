# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LogisticRegressionCV with every row pass on the whole device
(lane/neural-pass127, 2026-10-02).

The team form (x_linear/logcv.mojo) ran the whole fit on ONE block: every
L-BFGS objective evaluation (each training row's residuals and loss term,
then each gradient cell's FOLD_BLOCK-row partials folded in block order)
and every held-out scoring pass on 256 threads (board: taxi 9.35 s on the
MI325X, 3.3 s on the L40S, sklearn 0.76). Here `logcv_fit` runs on a host
team with two device steps in place of its team ones:
  * `logistic_objective_grid`: the rows' terms (`logcv_map_row`, a thread
    per row), every (block, cell) partial (`logcv_part_rows`, a thread per
    pair), the partials folded per cell in block order (`fold_parts`, a
    thread per cell); the folded sums come home and `logcv_finish` runs
    the team lead's last statements;
  * `lcv_score_grid` (cgr-linear): a thread per held-out row
    (`_predict_code`), the hits folded per row block and the blocks
    folded (`lcv_score_part`, the host column's order); one word home;
  * `lcv_rows_grid` (cgr-linear): the fold's training rows compacted on
    the grid (block counts, a block scan, the writes); the count home.
The StratifiedKFold ids are built on the grid from the labels
(`lcv_fold_ids_device`: per-class block counts, a scan per class, the
fold table, a thread per row). The L-BFGS iterations are logcv_fit's own
code on the host (the x_linear host and device columns already share
every word). The same statements on the same rows in the same order: the
same words.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, ldi, i2f
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.scan import SC_NT, _sc_block_excl
from x_linear.tops import FOLD_BLOCK
from x_linear.team import Team, team_work, solo
from x_linear.tops import fold_blocks, fold_parts
from x_linear.witness import Witness, witness_end, WITNESS_TRIES, WITNESS_ABORT
from x_linear.logcv import (
    logcv_fit, logcv_map_row, logcv_part_rows, logcv_finish, logcv_team_rows, _predict_code,
    lcv_fold_table, lcv_fold_of, lcv_score_part, lcv_score_final,
)

comptime LCV_TPB = 256

# the state the two thin callbacks reach (one fit at a time per process)
comptime LCV_X = 0
comptime LCV_Y = 1
comptime LCV_TH = 2
comptime LCV_ROWS = 3
comptime LCV_SCR = 4
comptime LCV_G = 5
comptime LCV_OUT = 6
comptime LCV_HIT = 7
comptime LCV_SP = 8
# int buffers
comptime LCV_IX = 0
comptime LCV_BCNT = 1
comptime LCV_TOT = 2


struct _LcvState(Defaultable, Movable):
    var ctx: Optional[DeviceContext]
    var bf: List[DeviceBuffer[DType.float32]]
    var ix: List[DeviceBuffer[DType.int32]]
    var cur_fold: Int
    var err: Bool
    var aborted: Bool
    var wit: Optional[Witness]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.bf = List[DeviceBuffer[DType.float32]]()
        self.ix = List[DeviceBuffer[DType.int32]]()
        self.cur_fold = -2
        self.err = False
        self.aborted = False
        self.wit = Optional[Witness]()


comptime LCV_STATE = _Global[StorageType=_LcvState, name="MojoXLinearLogcvGrid", init_fn=_LcvState.__init__]


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

def _lcv_objective(t: Team, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) raises -> Float32:
    var stp = LCV_STATE.get_or_create_ptr()
    var ctx = stp[].ctx.value().copy()
    var kp = ldi(ip, 0)
    var fi = ldi(ip, 1) != 0
    var fold = ldi(ip, 2)
    var c = ld(fp, 0)
    var sw = ldi(ip, 3) != 0
    var p = kp * (d + 1)
    var cnt = ldi(ip, 4) if fold >= 0 else n
    # the fold's training rows: listed on the device by `lcv_rows_grid`
    ctx.enqueue_copy(dst_buf=stp[].bf[LCV_TH], src_ptr=th + toff)
    var nbk = fold_blocks(cnt)
    # the map, partials and fold rebuild from theta: rerun as one unit when
    # the Metal witness finds a launch cut (x_linear/witness.mojo)
    var outs = List[Float32](length=2, fill=Float32(0))
    var b1 = _blocks(cnt)
    var b2 = _blocks((p + 2) * nbk)
    var b3 = _blocks(p + 2)
    var tries = 0
    while True:
        var nonce = stp[].wit.value().begin()
        var wf = stp[].wit.value().p()
        ctx.enqueue_function[lcv_map_kernel](
            stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kp),
            Int32(1 if fi else 0), Int32(1 if sw else 0), stp[].bf[LCV_TH].unsafe_ptr(), stp[].ix[0].unsafe_ptr(),
            Int32(cnt), Int32(fold), stp[].bf[LCV_ROWS].unsafe_ptr(), wf, Int32(0), nonce, grid_dim=_blocks(cnt), block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_part_kernel](
            stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kp),
            Int32(1 if fi else 0), Int32(1 if sw else 0), Int32(fold), stp[].ix[0].unsafe_ptr(), Int32(cnt),
            stp[].bf[LCV_ROWS].unsafe_ptr(), stp[].bf[LCV_SCR].unsafe_ptr(), Int32(nbk), wf, Int32(b1), nonce,
            grid_dim=_blocks((p + 2) * nbk), block_dim=LCV_TPB,
        )
        ctx.enqueue_function[lcv_fold_kernel](
            stp[].bf[LCV_SCR].unsafe_ptr(), Int32(nbk), Int32(p), stp[].bf[LCV_G].unsafe_ptr(),
            stp[].bf[LCV_OUT].unsafe_ptr(), wf, Int32(b1 + b2), nonce, grid_dim=_blocks(p + 2), block_dim=LCV_TPB,
        )
        # the copies home ride before the witness read (one in-order queue):
        # a good read means they ran
        ctx.enqueue_copy(dst_ptr=g + goff, src_buf=stp[].bf[LCV_G])
        ctx.enqueue_copy(dst_ptr=outs.unsafe_ptr(), src_buf=stp[].bf[LCV_OUT])
        if stp[].wit.value().ok(ctx, b1 + b2 + b3, "LogisticRegressionCV objective"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            stp[].aborted = True
            raise Error(WITNESS_ABORT)
    ctx.synchronize()
    var acc = outs[0]
    var wrows = outs[1] if sw else Float32(0)
    _ = outs^
    return logcv_finish(g, goff, th, toff, kp, d, sw, c, cnt, acc, wrows)


def logistic_objective_grid(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP,
                            goff: Int, sc: FP) -> Float32:
    try:
        return _lcv_objective(t, n, d, ip, fp, th, toff, g, goff)
    except:
        try:
            LCV_STATE.get_or_create_ptr()[].err = True
        except:
            pass
        return Float32(0)


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


def lcv_rows_grid(t: Team, y: FP, n: Int, fold: Int, kp: Int) -> Int:
    """`logcv_rows` on the grid: the list stays on the device (the objective
    reads it there), the count comes home."""
    if fold < 0:
        return n
    try:
        var stp = LCV_STATE.get_or_create_ptr()
        var ctx = stp[].ctx.value().copy()
        var nbk = _blocks(n)
        var cnt = List[Int32](length=1, fill=Int32(0))
        var tries = 0
        while True:
            var nonce = stp[].wit.value().begin()
            var wf = stp[].wit.value().p()
            ctx.enqueue_function[lcv_rows_count_kernel](
                stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(fold), stp[].ix[LCV_BCNT].unsafe_ptr(), wf, Int32(0), nonce,
                grid_dim=nbk, block_dim=LCV_TPB,
            )
            ctx.enqueue_function[lcv_rows_scan_kernel](
                stp[].ix[LCV_BCNT].unsafe_ptr(), Int32(nbk), stp[].ix[LCV_TOT].unsafe_ptr(), wf, Int32(nbk), nonce,
                grid_dim=1, block_dim=SC_NT,
            )
            ctx.enqueue_function[lcv_rows_write_kernel](
                stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(fold), stp[].ix[LCV_BCNT].unsafe_ptr(),
                stp[].ix[LCV_IX].unsafe_ptr(), wf, Int32(nbk + 1), nonce, grid_dim=nbk, block_dim=LCV_TPB,
            )
            ctx.enqueue_copy(dst_ptr=cnt.unsafe_ptr(), src_buf=stp[].ix[LCV_TOT])
            if stp[].wit.value().ok(ctx, 2 * nbk + 1, "LogisticRegressionCV rows"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                stp[].aborted = True
                raise Error(WITNESS_ABORT)
        ctx.synchronize()
        stp[].cur_fold = fold
        var c = Int(cnt[0])
        _ = cnt^
        return c
    except:
        try:
            LCV_STATE.get_or_create_ptr()[].err = True
        except:
            pass
        return 0


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


def lcv_score_grid(t: Team, x: FP, y: FP, n: Int, d: Int, kpp: Int, fi: Bool, fw: FP, th: Int, f: Int,
                   hitr: FP, weighted: Bool) -> Float32:
    try:
        var stp = LCV_STATE.get_or_create_ptr()
        var ctx = stp[].ctx.value().copy()
        var nb = fold_blocks(n)
        var g = _blocks(nb)
        var out = List[Float32](length=1, fill=Float32(0))
        var tries = 0
        while True:
            var nonce = stp[].wit.value().begin()
            var wf = stp[].wit.value().p()
            ctx.enqueue_copy(dst_buf=stp[].bf[LCV_TH], src_ptr=fw + th)
            ctx.enqueue_function[lcv_hit_kernel](
                stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kpp),
                Int32(1 if fi else 0), stp[].bf[LCV_TH].unsafe_ptr(), Int32(f), stp[].bf[LCV_HIT].unsafe_ptr(),
                wf, Int32(0), nonce, grid_dim=_blocks(n), block_dim=LCV_TPB,
            )
            ctx.enqueue_function[lcv_score_parts_kernel](
                stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(f), Int32(1 if weighted else 0),
                stp[].bf[LCV_HIT].unsafe_ptr(), stp[].bf[LCV_SP].unsafe_ptr(), wf, Int32(_blocks(n)), nonce,
                grid_dim=g, block_dim=LCV_TPB,
            )
            ctx.enqueue_function[lcv_score_fin_kernel](
                stp[].bf[LCV_SP].unsafe_ptr(), Int32(nb), stp[].bf[LCV_OUT].unsafe_ptr(), wf, Int32(_blocks(n) + g), nonce,
                grid_dim=1, block_dim=1,
            )
            ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=stp[].bf[LCV_OUT].create_sub_buffer[DType.float32](0, 1))
            if stp[].wit.value().ok(ctx, _blocks(n) + g + 1, "LogisticRegressionCV score"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                stp[].aborted = True
                raise Error(WITNESS_ABORT)
        ctx.synchronize()
        var v = out[0]
        _ = out^
        return v
    except:
        try:
            LCV_STATE.get_or_create_ptr()[].err = True
        except:
            pass
        return Float32(0)


def logcv_fit_grid(
    ctx: DeviceContext, algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """`logcv_fit` on a host team with the device objective and hits."""
    var kp = Int(ip[2])
    var p = kp * (d + 1)
    var stp = LCV_STATE.get_or_create_ptr()
    stp[].ctx = ctx.copy()
    stp[].bf = List[DeviceBuffer[DType.float32]]()
    stp[].ix = List[DeviceBuffer[DType.int32]]()
    stp[].cur_fold = -2
    stp[].err = False
    stp[].aborted = False
    var wcap = max(_blocks(n) + _blocks((p + 2) * fold_blocks(n)) + _blocks(p + 2), 2 * _blocks(n) + _blocks(fold_blocks(n)) + 2)
    var wctx = ctx.copy()
    stp[].wit = Witness(wctx, wcap)
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_x, 1)))  # X
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_y, 1)))  # Y
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(p, 1)))  # TH
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max((kp + 1) * n, 1)))  # ROWS
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max((p + 2) * fold_blocks(n), 1)))  # SCR
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(p, 1)))  # G
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](2))  # OUT
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n, 1)))  # HIT
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(2 * fold_blocks(n), 1)))  # SP
    stp[].ix.append(ctx.enqueue_create_buffer[DType.int32](max(n, 1)))  # IX
    stp[].ix.append(ctx.enqueue_create_buffer[DType.int32](_blocks(n) + 1))  # BCNT
    stp[].ix.append(ctx.enqueue_create_buffer[DType.int32](1))  # TOT
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[LCV_X], src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[LCV_Y], src_ptr=y)
    # the StratifiedKFold ids from the device labels (the caller sent zeros)
    lcv_fold_ids_device(ctx.copy(), FP(unsafe_from_address=Int(stp[].bf[LCV_Y].unsafe_ptr())), n, max(Int(ip[2]), 2), Int(ip[4]))
    ctx.synchronize()
    # the host team, as `_fit_on_host` builds it
    var hip = ip.copy()
    var hfp = fp.copy()
    var fw = List[Float32](length=max(n_fw, 1), fill=Float32(0))
    var iw = List[Int32](length=max(n_iw, 1), fill=Int32(0))
    var bufs = logcv_team_rows(IP(unsafe_from_address=Int(hip.unsafe_ptr())))
    var tw = List[Float32](length=team_work(n, bufs, 0), fill=Float32(0))
    for i in range(n_out):
        res.unsafe_store(i, Float32(0))
    logcv_fit[logistic_objective_grid, lcv_rows_grid, lcv_score_grid](
        solo(FP(unsafe_from_address=Int(tw.unsafe_ptr())), n, bufs, 0), x, y, n, d,
        IP(unsafe_from_address=Int(hip.unsafe_ptr())), FP(unsafe_from_address=Int(hfp.unsafe_ptr())),
        res, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
    )
    _ = hip^
    _ = hfp^
    _ = fw^
    _ = iw^
    _ = tw^
    var failed = stp[].err
    var aborted = stp[].aborted
    stp[].wit = Optional[Witness]()
    stp[].bf = List[DeviceBuffer[DType.float32]]()
    stp[].ix = List[DeviceBuffer[DType.int32]]()
    stp[].ctx = Optional[DeviceContext]()
    if aborted:
        raise Error(WITNESS_ABORT)
    if failed:
        raise Error("LogisticRegressionCV: a device step of the grid fit failed")
