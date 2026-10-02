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
  * `lcv_hits_grid`: a thread per held-out row (`_predict_code`), the hits
    home for the lead's ascending count.
The L-BFGS iterations, the fold lists and the counts are logcv_fit's own
code on the host (the x_linear host and device columns already share
every word). The same statements on the same rows in the same order: the
same words. `MOJOLEARN_X_LINEAR_LOGCV_GRID=0` restores the team fit.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, ldi
from x_linear.team import Team, team_work, solo
from x_linear.tops import fold_blocks, fold_parts
from x_linear.logcv import (
    logcv_fit, logcv_map_row, logcv_part_rows, logcv_finish, logcv_team_rows, _predict_code,
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


struct _LcvState(Defaultable, Movable):
    var ctx: Optional[DeviceContext]
    var bf: List[DeviceBuffer[DType.float32]]
    var ix: List[DeviceBuffer[DType.int32]]
    var cur_fold: Int
    var err: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.bf = List[DeviceBuffer[DType.float32]]()
        self.ix = List[DeviceBuffer[DType.int32]]()
        self.cur_fold = -2
        self.err = False


comptime LCV_STATE = _Global[StorageType=_LcvState, name="MojoXLinearLogcvGrid", init_fn=_LcvState.__init__]


def _blocks(count: Int) -> Int:
    return max((count + LCV_TPB - 1) // LCV_TPB, 1)


def lcv_map_kernel(x: FP, y: FP, n: Int32, d: Int32, kp: Int32, fi: Int32, sw: Int32, th: FP, ix: IP,
                   cnt: Int32, fold: Int32, rows: FP):
    var q = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    if q < Int(cnt):
        var i = Int(ix.unsafe_load(q)) if Int(fold) >= 0 else q
        logcv_map_row(i, x, y, Int(n), Int(d), Int(kp), fi != 0, sw != 0, th, 0, rows)


def lcv_part_kernel(x: FP, y: FP, n: Int32, d: Int32, kp: Int32, fi: Int32, sw: Int32, fold: Int32, ix: IP,
                    cnt: Int32, rows: FP, scr: FP, nbk: Int32):
    var q = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var p = Int(kp) * (Int(d) + 1)
    var nb = Int(nbk)
    if q < (p + 2) * nb:
        var bk = q // (p + 2)
        var o = q - bk * (p + 2)
        st(scr, o * nb + bk, logcv_part_rows(o, x, y, Int(n), Int(d), Int(kp), fi != 0, sw != 0, Int(fold), ix,
                                             Int(cnt), bk, rows))


def lcv_fold_kernel(scr: FP, nbk: Int32, p: Int32, g: FP, sums: FP):
    var o = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var pp = Int(p)
    if o < pp + 2:
        var v = fold_parts(scr, o * Int(nbk), Int(nbk))
        if o < pp:
            st(g, o, v)
        else:
            st(sums, o - pp, v)


def lcv_hit_kernel(x: FP, y: FP, n: Int32, d: Int32, kpp: Int32, fi: Int32, th: FP, f: Int32, hit: FP):
    var i = Int(block_idx.x) * LCV_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        if Int(ld(y, nn + i)) == Int(f):
            var h = _predict_code(x, i, Int(d), Int(kpp), fi != 0, th, 0) == Int(ld(y, i))
            st(hit, i, Float32(1) if h else Float32(0))


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
    if fold >= 0 and fold != stp[].cur_fold:
        # the fold's training rows, as logcv_rows listed them in team row K' + 1
        ctx.enqueue_copy(dst_buf=stp[].ix[0], src_ptr=t.row(kp + 1).bitcast[Int32]())
        stp[].cur_fold = fold
    ctx.enqueue_copy(dst_buf=stp[].bf[LCV_TH], src_ptr=th + toff)
    var nbk = fold_blocks(cnt)
    ctx.enqueue_function[lcv_map_kernel](
        stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kp),
        Int32(1 if fi else 0), Int32(1 if sw else 0), stp[].bf[LCV_TH].unsafe_ptr(), stp[].ix[0].unsafe_ptr(),
        Int32(cnt), Int32(fold), stp[].bf[LCV_ROWS].unsafe_ptr(), grid_dim=_blocks(cnt), block_dim=LCV_TPB,
    )
    ctx.enqueue_function[lcv_part_kernel](
        stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kp),
        Int32(1 if fi else 0), Int32(1 if sw else 0), Int32(fold), stp[].ix[0].unsafe_ptr(), Int32(cnt),
        stp[].bf[LCV_ROWS].unsafe_ptr(), stp[].bf[LCV_SCR].unsafe_ptr(), Int32(nbk),
        grid_dim=_blocks((p + 2) * nbk), block_dim=LCV_TPB,
    )
    ctx.enqueue_function[lcv_fold_kernel](
        stp[].bf[LCV_SCR].unsafe_ptr(), Int32(nbk), Int32(p), stp[].bf[LCV_G].unsafe_ptr(),
        stp[].bf[LCV_OUT].unsafe_ptr(), grid_dim=_blocks(p + 2), block_dim=LCV_TPB,
    )
    var outs = List[Float32](length=2, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=g + goff, src_buf=stp[].bf[LCV_G])
    ctx.enqueue_copy(dst_ptr=outs.unsafe_ptr(), src_buf=stp[].bf[LCV_OUT])
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


def lcv_hits_grid(t: Team, x: FP, y: FP, n: Int, d: Int, kpp: Int, fi: Bool, fw: FP, th: Int, f: Int, hitr: FP):
    try:
        var stp = LCV_STATE.get_or_create_ptr()
        var ctx = stp[].ctx.value().copy()
        ctx.enqueue_copy(dst_buf=stp[].bf[LCV_TH], src_ptr=fw + th)
        ctx.enqueue_function[lcv_hit_kernel](
            stp[].bf[LCV_X].unsafe_ptr(), stp[].bf[LCV_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(kpp),
            Int32(1 if fi else 0), stp[].bf[LCV_TH].unsafe_ptr(), Int32(f), stp[].bf[LCV_HIT].unsafe_ptr(),
            grid_dim=_blocks(n), block_dim=LCV_TPB,
        )
        ctx.enqueue_copy(dst_ptr=hitr, src_buf=stp[].bf[LCV_HIT])
        ctx.synchronize()
    except:
        try:
            LCV_STATE.get_or_create_ptr()[].err = True
        except:
            pass


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
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_x, 1)))  # X
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_y, 1)))  # Y
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(p, 1)))  # TH
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max((kp + 1) * n, 1)))  # ROWS
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max((p + 2) * fold_blocks(n), 1)))  # SCR
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(p, 1)))  # G
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](2))  # OUT
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n, 1)))  # HIT
    stp[].ix.append(ctx.enqueue_create_buffer[DType.int32](max(n, 1)))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[LCV_X], src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[LCV_Y], src_ptr=y)
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
    logcv_fit[logistic_objective_grid, lcv_hits_grid](
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
    stp[].bf = List[DeviceBuffer[DType.float32]]()
    stp[].ix = List[DeviceBuffer[DType.int32]]()
    stp[].ctx = Optional[DeviceContext]()
    if failed:
        raise Error("LogisticRegressionCV: a device step of the grid fit failed")
