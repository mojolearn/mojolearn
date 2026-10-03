# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HuberRegressor wholly on the device (lane/neural-pass129, 2026-10-02: the
row passes; lane cgr4-device-optim, 2026-10-03: the minimizer).
The objective is `HuberObjective` (an `LbObjective`): a thread per row
(`huber_map_row`), a thread per (block, task) (`_huber_part`: the gradient
cells, the inlier squares, the outlier |r|, the outlier weight, the weight
total, the outlier count), a thread per task folding its blocks in order
(`fold_parts`; the count as an integer sum), then one block finishing
(`huber_finish_t`: the penalty, the sigma cell, f). Every scalar of theta
(b, sigma, the threshold) is read on the device. The minimizer is
x_linear/lbfgs_device.mojo; the result is assembled on the device and
comes home once. The host column runs `huber_fit` (x_linear/huber.mojo):
the same statements on the same values in the same order.
"""
from std.gpu import block_idx, thread_idx, block_dim
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, fexp, fm, fd, i2f
from x_linear.team import team_at
from x_linear.tops import fold_blocks, fold_parts, FOLD_BLOCK
from x_linear.witness import Witness, witness_end
from x_linear.huber import huber_map_row, huber_finish_t, _huber_part
from x_linear.vfold import vscratch
from x_linear.lbfgs_device import (
    LbObjective, lbfgs_device, lbd_words, lbd_th, lbd_witness_words, LBD_TPB,
)

comptime HG_TPB = 256


def _blocks(count: Int) -> Int:
    return max((count + HG_TPB - 1) // HG_TPB, 1)


@always_inline
def _hg_sigma(th: FP, p: Int) -> Float32:
    return fexp(ld(th, p - 1))


def hg_map_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, eps: Float32, th: FP, sw: Int32, rows: FP,
                  wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    if i < nn:
        var p = dd + 2 if fi != 0 else dd + 1
        var sigma = _hg_sigma(th, p)
        var b = ld(th, dd) if fi != 0 else Float32(0)
        huber_map_row(i, x, y, nn, dd, th, 0, b, fm(eps, sigma), fd(Float32(2), sigma), fm(Float32(2), eps),
                      sw != 0, rows, rows + nn)
    witness_end(wf, woff, nonce)


def hg_part_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, eps: Float32, th: FP, cells: Int32, rows: FP,
                   sw: Int32, scr: FP, nbk: Int32, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nt = Int(cells) + 5
    var nb = Int(nbk)
    if q < nt * nb:
        var p = Int(d) + 2 if fi != 0 else Int(d) + 1
        var thr = fm(eps, _hg_sigma(th, p))
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


def hg_finish_kernel(th: FP, g: FP, f: FP, sums: FP, nrows: Float32, d: Int32, fi: Int32, eps: Float32, alpha: Float32,
                     sw: Int32, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """One block: `huber_finish_t` on the folded sums; f[0] by the lead."""
    var t = team_at(Int(thread_idx.x), Int(block_dim.x), parts, 0, 0, 0)
    var dd = Int(d)
    var p = dd + 2 if fi != 0 else dd + 1
    var sigma = _hg_sigma(th, p)
    var swb = sw != 0
    var w_all = ld(sums, 3) if swb else Float32(0)
    var n_out = Int(bitcast[DType.int32](ld(sums, 4)))
    var fv = huber_finish_t(t, g, 0, th, 0, dd, p, nrows, eps, alpha, sigma, fm(Float32(2), eps), swb,
                            ld(sums, 0), ld(sums, 1), n_out, ld(sums, 2), w_all, parts)
    if t.lead():
        st(f, 0, fv)
    witness_end(wf, woff, nonce)


def hg_result_kernel(lw: FP, c: Int32, res: FP, d: Int32, fi: Int32, iters: Int32):
    """res: coef d | intercept | scale | n_iter | theta P (huber_fit's words)."""
    var j = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var p = dd + 2 if fi != 0 else dd + 1
    var th = lw + lbd_th(p, Int(c))
    if j < p:
        st(res, dd + 4 + j, ld(th, j))
    if j < dd:
        st(res, j, ld(th, j))
    elif j == dd:
        st(res, dd, ld(th, dd) if fi != 0 else Float32(0))
    elif j == dd + 1:
        st(res, dd + 1, fexp(ld(th, p - 1)))
    elif j == dd + 2:
        var it = Int(iters)
        st(res, dd + 2, i2f(it if it >= 0 else -it))


struct HuberObjective(LbObjective):
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var rows: DeviceBuffer[DType.float32]
    var scr: DeviceBuffer[DType.float32]
    var sums: DeviceBuffer[DType.float32]
    var parts: DeviceBuffer[DType.float32]
    var n: Int
    var d: Int
    var fi: Bool
    var sw: Bool
    var eps: Float32
    var alpha: Float32

    def __init__(out self, mut ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, fi: Bool,
                 sw: Bool, eps: Float32, alpha: Float32) raises:
        var cells = d + 1 if fi else d
        self.x = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
        self.y = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
        self.rows = ctx.enqueue_create_buffer[DType.float32](max(2 * n, 1))
        self.scr = ctx.enqueue_create_buffer[DType.float32](max((cells + 5) * fold_blocks(n), 1))
        self.sums = ctx.enqueue_create_buffer[DType.float32](5)
        self.parts = ctx.enqueue_create_buffer[DType.float32](vscratch(d) + 16)
        if n_x > 0:
            ctx.enqueue_copy(dst_buf=self.x, src_ptr=x)
        if n_y > 0:
            ctx.enqueue_copy(dst_buf=self.y, src_ptr=y)
        self.n = n
        self.d = d
        self.fi = fi
        self.sw = sw
        self.eps = eps
        self.alpha = alpha

    def _cells(self) -> Int:
        return self.d + 1 if self.fi else self.d

    def blocks(self) -> Int:
        var cells = self._cells()
        return _blocks(self.n) + _blocks((cells + 5) * fold_blocks(self.n)) + _blocks(cells + 5) + 1

    def enqueue(mut self, mut ctx: DeviceContext, th: FP, g: FP, f: FP, wf: IP, woff: Int, nonce: Int32) raises:
        var n = self.n
        var d = self.d
        var cells = self._cells()
        var nbk = fold_blocks(n)
        var fi = Int32(1 if self.fi else 0)
        var sw = Int32(1 if self.sw else 0)
        var b1 = _blocks(n)
        var b2 = _blocks((cells + 5) * nbk)
        var b3 = _blocks(cells + 5)
        ctx.enqueue_function[hg_map_kernel](
            self.x.unsafe_ptr(), self.y.unsafe_ptr(), Int32(n), Int32(d), fi, self.eps, th, sw,
            self.rows.unsafe_ptr(), wf, Int32(woff), nonce, grid_dim=b1, block_dim=HG_TPB,
        )
        ctx.enqueue_function[hg_part_kernel](
            self.x.unsafe_ptr(), self.y.unsafe_ptr(), Int32(n), Int32(d), fi, self.eps, th, Int32(cells),
            self.rows.unsafe_ptr(), sw, self.scr.unsafe_ptr(), Int32(nbk), wf, Int32(woff + b1), nonce,
            grid_dim=b2, block_dim=HG_TPB,
        )
        ctx.enqueue_function[hg_fold_kernel](
            self.scr.unsafe_ptr(), Int32(nbk), Int32(cells), g, self.sums.unsafe_ptr(), wf, Int32(woff + b1 + b2),
            nonce, grid_dim=b3, block_dim=HG_TPB,
        )
        ctx.enqueue_function[hg_finish_kernel](
            th, g, f, self.sums.unsafe_ptr(), i2f(n), Int32(d), fi, self.eps, self.alpha, sw,
            self.parts.unsafe_ptr(), wf, Int32(woff + b1 + b2 + b3), nonce, grid_dim=1, block_dim=LBD_TPB,
        )


def huber_fit_grid(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    """`huber_fit` on the device: theta = 0, the device L-BFGS, the result
    words assembled on the device and copied home once."""
    var fi = Int(ip[1]) != 0
    var sw = Int(ip[2]) != 0
    var p = d + 2 if fi else d + 1
    var c = ctx.copy()
    var obj = HuberObjective(c, x, n_x, y, n_y, n, d, fi, sw, fp[0], fp[1])
    var wit = Witness(c, lbd_witness_words(obj.blocks(), p))
    var lw = c.enqueue_create_buffer[DType.float32](lbd_words(p))
    lw.enqueue_fill(Float32(0))
    var r = lbfgs_device(c, obj, wit, lw, p, Int(ip[0]), fp[2], "HuberRegressor fit")
    var dres = c.enqueue_create_buffer[DType.float32](max(n_out, 1))
    dres.enqueue_fill(Float32(0))
    c.enqueue_function[hg_result_kernel](lw.unsafe_ptr(), Int32(r[1]), dres.unsafe_ptr(), Int32(d),
                                         Int32(1 if fi else 0), Int32(r[0]), grid_dim=_blocks(d + 4 + p),
                                         block_dim=HG_TPB)
    c.enqueue_copy(dst_ptr=res, src_buf=dres)
    c.synchronize()
    _ = obj^
    _ = wit^
    _ = lw^
    _ = dres^
