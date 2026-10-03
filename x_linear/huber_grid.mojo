# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HuberRegressor with every row pass on the whole device
(lane/neural-pass129, 2026-10-02): x_linear/logcv_grid.mojo's pattern.
`huber_fit` runs on a host team with `huber_objective_grid`: a thread per
row (`huber_map_row`), a thread per (block, task) (`_huber_part`: the
gradient cells, the inlier squares, the outlier |r|, the outlier weight,
the weight total, the outlier count), a thread per task folding its
blocks in order (`fold_parts`; the count as an integer sum); the folded
sums home and `huber_finish`, the team lead's last statements. The same
words. Board (team form): taxi 4.2 s MI325X vs sklearn 2.7.
"""
from std.gpu import block_idx, thread_idx
from std.ffi import _Global
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, ldi, fexp, fm, fd
from x_linear.team import Team, team_work, solo, TEAM_ROW_BUFS
from x_linear.tops import fold_blocks, fold_parts, FOLD_BLOCK
from x_linear.witness import Witness, witness_end, WITNESS_TRIES, WITNESS_ABORT
from x_linear.huber import huber_fit, huber_map_row, huber_finish, _huber_part

comptime HG_TPB = 256
comptime HG_X = 0
comptime HG_Y = 1
comptime HG_TH = 2
comptime HG_ROWS = 3
comptime HG_SCR = 4
comptime HG_G = 5
comptime HG_SUMS = 6


struct _HgState(Defaultable, Movable):
    var ctx: Optional[DeviceContext]
    var bf: List[DeviceBuffer[DType.float32]]
    var err: Bool
    var aborted: Bool
    var wit: Optional[Witness]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.bf = List[DeviceBuffer[DType.float32]]()
        self.err = False
        self.aborted = False
        self.wit = Optional[Witness]()


comptime HG_STATE = _Global[StorageType=_HgState, name="MojoXLinearHuberGrid", init_fn=_HgState.__init__]


def _blocks(count: Int) -> Int:
    return max((count + HG_TPB - 1) // HG_TPB, 1)


def hg_map_kernel(x: FP, y: FP, n: Int32, d: Int32, th: FP, b: Float32, thr: Float32, tos: Float32,
                  two_eps: Float32, sw: Int32, rows: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        huber_map_row(i, x, y, nn, Int(d), th, 0, b, thr, tos, two_eps, sw != 0, rows, rows + nn)
    witness_end(wf, woff, nonce)

def hg_part_kernel(x: FP, y: FP, n: Int32, d: Int32, cells: Int32, rows: FP, thr: Float32, sw: Int32,
                   scr: FP, nbk: Int32, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nt = Int(cells) + 5
    var nb = Int(nbk)
    if q < nt * nb:
        var bk = q // nt
        var o = q - bk * nt
        var lo = bk * FOLD_BLOCK
        st(scr, o * nb + bk, _huber_part(o, Int(cells), Int(d), x, y, nn, rows, rows + nn, thr, sw != 0, lo,
                                         min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)

def hg_fold_kernel(scr: FP, nbk: Int32, cells: Int32, g: FP, sums: FP, wf: IP, woff: Int32, nonce: Int32):
    var o = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var c = Int(cells)
    var nb = Int(nbk)
    if o < c + 4:
        var v = fold_parts(scr, o * nb, nb)
        if o < c:
            st(g, o, v)
        else:
            st(sums, o - c, v)
    elif o == c + 4:
        var k = 0
        for bk in range(nb):
            k += Int(bitcast[DType.int32](ld(scr, o * nb + bk)))
        st(sums, 4, bitcast[DType.float32](Int32(k)))
    witness_end(wf, woff, nonce)

def _hg_objective(n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP, goff: Int) raises -> Float32:
    var stp = HG_STATE.get_or_create_ptr()
    var ctx = stp[].ctx.value().copy()
    var fi = ldi(ip, 1) != 0
    var eps = ld(fp, 0)
    var alpha = ld(fp, 1)
    var p = d + 2 if fi else d + 1
    var s = ld(th, toff + p - 1)
    var sigma = fexp(s)
    var b = ld(th, toff + d) if fi else Float32(0)
    var sw = ldi(ip, 2) != 0
    var thr = fm(eps, sigma)
    var two_over_sigma = fd(Float32(2), sigma)
    var two_eps = fm(Float32(2), eps)
    var cells = d + 1 if fi else d
    var nbk = fold_blocks(n)
    # map, partials, fold (they rebuild from theta) and the copies home run
    # as one guarded unit (x_linear/witness.mojo); a good read means all ran
    var sums = List[Float32](length=5, fill=Float32(0))
    var b1 = _blocks(n)
    var b2 = _blocks((cells + 5) * nbk)
    var b3 = _blocks(cells + 5)
    var tries = 0
    while True:
        var nonce = stp[].wit.value().begin()
        var wf = stp[].wit.value().p()
        ctx.enqueue_copy(dst_buf=stp[].bf[HG_TH], src_ptr=th + toff)
        ctx.enqueue_function[hg_map_kernel](
            stp[].bf[HG_X].unsafe_ptr(), stp[].bf[HG_Y].unsafe_ptr(), Int32(n), Int32(d), stp[].bf[HG_TH].unsafe_ptr(),
            b, thr, two_over_sigma, two_eps, Int32(1 if sw else 0), stp[].bf[HG_ROWS].unsafe_ptr(),
            wf, Int32(0), nonce, grid_dim=_blocks(n), block_dim=HG_TPB,
        )
        ctx.enqueue_function[hg_part_kernel](
            stp[].bf[HG_X].unsafe_ptr(), stp[].bf[HG_Y].unsafe_ptr(), Int32(n), Int32(d), Int32(cells),
            stp[].bf[HG_ROWS].unsafe_ptr(), thr, Int32(1 if sw else 0), stp[].bf[HG_SCR].unsafe_ptr(), Int32(nbk),
            wf, Int32(b1), nonce, grid_dim=_blocks((cells + 5) * nbk), block_dim=HG_TPB,
        )
        ctx.enqueue_function[hg_fold_kernel](
            stp[].bf[HG_SCR].unsafe_ptr(), Int32(nbk), Int32(cells), stp[].bf[HG_G].unsafe_ptr(),
            stp[].bf[HG_SUMS].unsafe_ptr(), wf, Int32(b1 + b2), nonce, grid_dim=_blocks(cells + 5), block_dim=HG_TPB,
        )
        ctx.enqueue_copy(dst_ptr=g + goff, src_buf=stp[].bf[HG_G])
        ctx.enqueue_copy(dst_ptr=sums.unsafe_ptr(), src_buf=stp[].bf[HG_SUMS])
        if stp[].wit.value().ok(ctx, b1 + b2 + b3, "HuberRegressor objective"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            stp[].aborted = True
            raise Error(WITNESS_ABORT)
    ctx.synchronize()
    var sq = sums[0]
    var out_abs = sums[1]
    var w_out = sums[2]
    var w_all = sums[3] if sw else Float32(0)
    var n_out = Int(bitcast[DType.int32](sums[4]))
    _ = sums^
    return huber_finish(g, goff, th, toff, d, p, n, eps, alpha, sigma, two_eps, sw, sq, out_abs, n_out, w_out, w_all)


def huber_objective_grid(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, th: FP, toff: Int, g: FP,
                         goff: Int, sc: FP) -> Float32:
    try:
        return _hg_objective(n, d, ip, fp, th, toff, g, goff)
    except:
        try:
            HG_STATE.get_or_create_ptr()[].err = True
        except:
            pass
        return Float32(0)


def huber_fit_grid(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """`huber_fit` on a host team with `huber_objective_grid`."""
    var fi = Int(ip[1]) != 0
    var p = d + 2 if fi else d + 1
    var cells = d + 1 if fi else d
    var stp = HG_STATE.get_or_create_ptr()
    stp[].ctx = ctx.copy()
    stp[].bf = List[DeviceBuffer[DType.float32]]()
    stp[].err = False
    stp[].aborted = False
    var wctx = ctx.copy()
    stp[].wit = Witness(wctx, _blocks(n) + _blocks((cells + 5) * fold_blocks(n)) + _blocks(cells + 5))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_x, 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(n_y, 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(p, 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(2 * n, 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max((cells + 5) * fold_blocks(n), 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](max(cells, 1)))
    stp[].bf.append(ctx.enqueue_create_buffer[DType.float32](5))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[HG_X], src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=stp[].bf[HG_Y], src_ptr=y)
    ctx.synchronize()
    var hip = ip.copy()
    var hfp = fp.copy()
    var fw = List[Float32](length=max(n_fw, 1), fill=Float32(0))
    var iw = List[Int32](length=max(n_iw, 1), fill=Int32(0))
    var tw = List[Float32](length=team_work(n, TEAM_ROW_BUFS, 0), fill=Float32(0))
    for i in range(n_out):
        res.unsafe_store(i, Float32(0))
    huber_fit[huber_objective_grid](
        solo(FP(unsafe_from_address=Int(tw.unsafe_ptr())), n, TEAM_ROW_BUFS, 0), x, y, n, d,
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
    stp[].ctx = Optional[DeviceContext]()
    if aborted:
        raise Error(WITNESS_ABORT)
    if failed:
        raise Error("HuberRegressor: a device step of the grid fit failed")
