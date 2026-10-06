# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BLOCKED TSQR ON THE DEVICE (lane neural-pass140, 2026-10-02): the
kernels of x_decomp/tsqr_core.mojo's order. x_decomp/tsqr_host.mojo is
their host replay, statement for statement.

WHY. `linalg.qr(mode='reduced')`, `linalg.svd(full_matrices=False)`,
`lstsq` and `LinearRegression` ran on `geqrf_serial` / `orgqr_col` (one
dependent chain of m multiply-adds per column per step: 4 n launches of one
block each), on `qr_factor` (at most 64 slices of 32 threads) twice more for
U's orthonormalization, or on the normal equations. At 1,000,000 x 220 on
the L40S that was 8.4 s (qr), 15.1 s (svd) and 4.4 s (lstsq) against
0.22-0.24 s for torch and CuPy.

THE LAUNCHES. One threadgroup of TS_TPB = TS_P x TS_NB threads per leaf
block, thread (g, l): row group g (rows == g mod TS_P, the chain of the
core's inner products) and column lane l (a coalesced row segment of TS_NB
words). Per panel ONE launch per slice of blocks factors the panel (the
reflectors one at a time), builds T, and applies (I - Y T^T Y^T) to every
trailing column in chunks of TS_NB columns: each thread accumulates the
TS_NB inner products of its rows with one pass over the column (the
panel's reflectors share the loads), the TS_P partials of each are folded
in threadgroup memory, w = T^T z, and one more pass updates the column. The
tree: one threadgroup per pair, one thread per column. Q C: the tree top
down (one thread per column of C), then every block's panels last to first
with the same chunk routine and w = T z.

Every launch is sliced to about TS_LAUNCH_CELLS multiply-adds (blocks or
pairs per launch), and on the Apple column the host waits after each one:
macOS cuts a long command buffer silently. The slicing never changes a bit
(the blocks and pairs of one launch are independent).

Threadgroup memory: TS_SMEM_BYTES (20 KB), under every column's limit
(`lib_smem_page_fits_for`, checked before any launch). The barriers that
hand DEVICE words between threads are `dev_barrier` (Apple's
`air.wg.barrier(3, 1)`; `barrier()` there orders threadgroup memory only).
"""
from std.atomic import Atomic
from std.ffi import _Global
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_sqrt
from core.device_zero import enqueue_fill
from glm.impl.center_items import center_cell
from x_decomp.cells import F32Ptr
from x_decomp.jacobi2 import dev_barrier
from x_decomp.tsqr_core import (
    TS_LAUNCH_CELLS,
    TS_LAUNCH_MIN_BLOCKS,
    TS_NB,
    TS_P,
    TS_ROWS,
    TS_TPB,
    ts_block_hi,
    ts_block_lo,
    ts_blocks,
    ts_first_row,
    ts_fma,
    ts_fold,
    ts_panels,
    ts_reflector,
    ts_scale,
)

comptime _SH = UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime _TT = TS_NB * TS_NB
comptime TS_PART = TS_NB * TS_P * TS_NB
comptime TS_SMEM_BYTES = 4 * (TS_PART + 3 * _TT + TS_TPB + TS_P)

# lane idn-dense-linalg (2026-10-04), IDENTICAL speed on NVIDIA and AMD; the
# same words on every column (the host replay x_decomp/tsqr_host.mojo is
# unchanged):
#
# TS_GRID_UPDATE (-D MOJOLEARN_IDN_TSQR_GRID_OFF restores the old form): a
# panel's trailing update (and Q C's panel apply) is its own launch of one
# threadgroup per (block, chunk of TS_NB columns), not a serial walk over the
# chunks inside the block's one threadgroup. The chunks of one panel are
# independent (each reads the panel's reflectors and T, and reads and writes
# only its own columns), so no bit moves.
#
# TS_NORM_FUSED (-D MOJOLEARN_IDN_TSQR_NORM_OFF restores the old form): the
# norm chain of panel column j + 1 is accumulated by the threads that update
# that column in step j (chain g walks the same rows in the same order and
# squares the very words it stores), so the separate norm pass (TS_P live
# threads of TS_TPB) runs only for a panel's first column and after a zero
# reflector. The same fmas in the same order: no bit moves.
# IDENTICAL builds only: a FAST build keeps its launches as they were.
comptime _TS_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime TS_GRID_UPDATE = _TS_IDN and not (is_defined["MOJOLEARN_IDN_TSQR_GRID_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime TS_NORM_FUSED = _TS_IDN and not (is_defined["MOJOLEARN_IDN_TSQR_NORM_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime TS_SMEM_OK = lib_smem_page_fits_for[TARGET_COLUMN, TS_SMEM_BYTES]()


@always_inline
def _fold_sh(sh: _SH, base: Int, stride: Int) -> Float32:
    """`ts_fold` of sh[base + g * stride], g = 0 .. TS_P - 1."""
    var p = InlineArray[Float32, TS_P](fill=Float32(0.0))
    comptime for gg in range(TS_P):
        p[gg] = sh[base + gg * stride]
    return ts_fold(p)


@always_inline
def _wy_chunk_kern[transpose: Bool](
    y: F32Ptr, ldy: Int, x: F32Ptr, ldx: Int, c0: Int, c_hi: Int, mb: Int, j0: Int, pw: Int,
    tsh: _SH, part: _SH, zsh: _SH, wsh: _SH, g: Int, l: Int,
):
    """`ts_wy_host` for the columns c0 .. c0 + TS_NB - 1 (< c_hi) of x, the
    whole threadgroup: thread (g, l) runs chain g of every reflector's inner
    product with column c0 + l, folds reflector g's, and updates its rows."""
    var c = c0 + l
    var live = c < c_hi
    var acc = InlineArray[Float32, TS_NB](fill=Float32(0.0))
    if live:
        var i = ts_first_row(j0 + 1, g)
        while i < mb:
            var xv = ftz(x.unsafe_load(i * ldx + c))
            comptime for p in range(TS_NB):
                if p < pw and i > j0 + p:
                    acc[p] = ts_fma(ftz(y.unsafe_load(i * ldy + j0 + p)), xv, acc[p])
            i += TS_P
    comptime for p in range(TS_NB):
        part[(p * TS_P + g) * TS_NB + l] = acc[p]
    barrier()
    if g < pw and live:
        var tail = _fold_sh(part, (g * TS_P) * TS_NB + l, TS_NB)
        zsh[g * TS_NB + l] = ftz(ftz(x.unsafe_load((j0 + g) * ldx + c)) + tail)
    barrier()
    if g < pw and live:
        var w = Float32(0.0)
        comptime if transpose:
            for q in range(g + 1):
                w = ts_fma(tsh[q * TS_NB + g], zsh[q * TS_NB + l], w)
        else:
            for q in range(g, pw):
                w = ts_fma(tsh[g * TS_NB + q], zsh[q * TS_NB + l], w)
        wsh[g * TS_NB + l] = w
    barrier()
    if live:
        var i = ts_first_row(j0, g)
        while i < mb:
            var a2 = ftz(x.unsafe_load(i * ldx + c))
            var top = min(pw, i - j0 + 1)
            for p in range(top):
                var yv = Float32(1.0) if i == j0 + p else ftz(y.unsafe_load(i * ldy + j0 + p))
                a2 = ts_fma(-wsh[p * TS_NB + l], yv, a2)
            x.unsafe_store(i * ldx + c, a2)
            i += TS_P
    barrier()


def ts_pack_kernel(a: F32Ptr, b: F32Ptr, dst: F32Ptr, m_in: Int32, d_in: Int32, r_in: Int32):
    """dst (m x (d + r)) = [a | b], one thread per word (data movement)."""
    var d = Int(d_in)
    var r = Int(r_in)
    var n = d + r
    var t = Int(block_idx.x) * TS_TPB + Int(thread_idx.x)
    if t >= Int(m_in) * n:
        return
    var i = t // n
    var c = t - i * n
    dst.unsafe_store(t, a.unsafe_load(i * d + c) if c < d else b.unsafe_load(i * r + c - d))


def ts_pack_center_kernel(a: F32Ptr, b: F32Ptr, mx: F32Ptr, my: F32Ptr, dst: F32Ptr, m_in: Int32, d_in: Int32):
    """dst (m x (d + 1)) = [a - mx | b - my], one thread per word: `ts_pack_kernel`
    of `center_cell`'s words (glm/impl/center_items.mojo), which is what the
    two lm_center calls and the pack wrote."""
    var d = Int(d_in)
    var n = d + 1
    var t = Int(block_idx.x) * TS_TPB + Int(thread_idx.x)
    if t >= Int(m_in) * n:
        return
    var i = t // n
    var c = t - i * n
    if c < d:
        dst.unsafe_store(t, center_cell(a.unsafe_load(i * d + c), mx.unsafe_load(c)))
    else:
        dst.unsafe_store(t, center_cell(b.unsafe_load(i), my.unsafe_load(0)))


def ts_leaf_panel_kernel(a: F32Ptr, tst: F32Ptr, m_in: Int32, n_in: Int32, nb_in: Int32, b0_in: Int32, pan_in: Int32):
    """`ts_factor_block_host`'s panel `pan` for block b0 + block_idx.x."""
    var part = stack_allocation[TS_PART, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var zsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var red = stack_allocation[TS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nrm = stack_allocation[TS_P, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var m = Int(m_in)
    var n = Int(n_in)
    var nb = Int(nb_in)
    var pan = Int(pan_in)
    var b = Int(b0_in) + Int(block_idx.x)
    if b >= nb:
        return
    var lo = ts_block_lo(b)
    var mb = ts_block_hi(b, nb, m) - lo
    var blk = a + lo * n
    var tid = Int(thread_idx.x)
    var g = tid // TS_NB
    var l = tid - g * TS_NB
    var j0 = pan * TS_NB
    var pw = min(TS_NB, n - j0)
    var npan = ts_panels(n)
    tsh[tid] = Float32(0.0)
    barrier()
    # (F)
    # have_norm: nrm[] already holds column j's TS_P norm chains (TS_NORM_FUSED:
    # step j - 1's update of column j accumulated them); the same on every thread
    var have_norm = False
    for p in range(pw):
        var j = j0 + p
        if not have_norm:
            var acc = Float32(0.0)
            if l == 0:
                var i = ts_first_row(j, g)
                while i < mb:
                    var v = ftz(blk.unsafe_load(i * n + j))
                    acc = ts_fma(v, v, acc)
                    i += TS_P
                nrm[g] = acc
        barrier()
        have_norm = False
        var sigma = _fold_sh(nrm, 0, 1)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(blk.unsafe_load(j * n + j))
        barrier()
        var zero = normx == Float32(0.0)
        var tau = Float32(0.0)
        var u1 = Float32(1.0)
        var rjj = Float32(0.0)
        if not zero:
            var rf = ts_reflector(ajj, normx)
            rjj = rf[0]
            u1 = rf[1]
            tau = rf[2]
        var i2 = j + 1 + tid
        while i2 < mb:
            if zero:
                blk.unsafe_store(i2 * n + j, Float32(0.0))
            else:
                blk.unsafe_store(i2 * n + j, ftz(identical_div(ftz(blk.unsafe_load(i2 * n + j)), u1)))
            i2 += TS_TPB
        if tid == 0:
            blk.unsafe_store(j * n + j, rjj)
            tsh[p * TS_NB + p] = tau
        dev_barrier()
        if tau != Float32(0.0):
            var c = j0 + l
            var live = c > j and c < j0 + pw
            var ajc = Float32(0.0)
            var acc2 = Float32(0.0)
            if live:
                ajc = ftz(blk.unsafe_load(j * n + c))
                var i = ts_first_row(j + 1, g)
                while i < mb:
                    acc2 = ts_fma(ftz(blk.unsafe_load(i * n + j)), ftz(blk.unsafe_load(i * n + c)), acc2)
                    i += TS_P
            red[g * TS_NB + l] = acc2
            barrier()
            if live:
                var tail = _fold_sh(red, l, TS_NB)
                var td = ts_scale(tau, ftz(ajc + tail))
                if g == 0:
                    blk.unsafe_store(j * n + c, ftz(ajc - td))
                var i = ts_first_row(j + 1, g)
                comptime if TS_NORM_FUSED:
                    # column j + 1's thread also runs that column's norm chain g
                    # over the words it stores (rows ts_first_row(j + 1, g), +TS_P, ...)
                    var nacc = Float32(0.0)
                    while i < mb:
                        var nv = ts_fma(-td, ftz(blk.unsafe_load(i * n + j)), ftz(blk.unsafe_load(i * n + c)))
                        blk.unsafe_store(i * n + c, nv)
                        if c == j + 1:
                            var fv = ftz(nv)
                            nacc = ts_fma(fv, fv, nacc)
                        i += TS_P
                    if c == j + 1:
                        nrm[g] = nacc
                else:
                    while i < mb:
                        blk.unsafe_store(i * n + c, ts_fma(-td, ftz(blk.unsafe_load(i * n + j)), ftz(blk.unsafe_load(i * n + c))))
                        i += TS_P
            dev_barrier()
            comptime if TS_NORM_FUSED:
                have_norm = p + 1 < pw
    # (T)
    for p in range(1, pw):
        var acc = Float32(0.0)
        if l < p:
            var i = ts_first_row(j0 + p + 1, g)
            while i < mb:
                acc = ts_fma(ftz(blk.unsafe_load(i * n + j0 + l)), ftz(blk.unsafe_load(i * n + j0 + p)), acc)
                i += TS_P
        red[g * TS_NB + l] = acc
        barrier()
        if g == 0 and l < p:
            zsh[l] = ftz(ftz(blk.unsafe_load((j0 + p) * n + j0 + l)) + _fold_sh(red, l, TS_NB))
        barrier()
        if g == 0 and l < p:
            var s = Float32(0.0)
            for kk in range(l, p):
                s = ts_fma(tsh[l * TS_NB + kk], zsh[kk], s)
            tsh[l * TS_NB + p] = ts_scale(-tsh[p * TS_NB + p], s)
        barrier()
    tst.unsafe_store((b * npan + pan) * _TT + tid, tsh[tid])
    # (U): TS_GRID_UPDATE runs it as `ts_leaf_update_kernel`'s launch
    comptime if not TS_GRID_UPDATE:
        var c0 = j0 + pw
        while c0 < n:
            _wy_chunk_kern[True](blk, n, blk, n, c0, n, mb, j0, pw, tsh, part, zsh, wsh, g, l)
            c0 += TS_NB


def ts_leaf_update_kernel(
    a: F32Ptr, tst: F32Ptr, m_in: Int32, n_in: Int32, nb_in: Int32, b0_in: Int32, pan_in: Int32, nch_in: Int32
):
    """`ts_leaf_panel_kernel`'s (U) for panel `pan`, one threadgroup per
    (block, chunk): threadgroup t is block b0 + t // nch and the chunk of
    TS_NB trailing columns t % nch. T comes from `tst` (the words the panel
    launch held in threadgroup memory)."""
    var part = stack_allocation[TS_PART, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var zsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var m = Int(m_in)
    var n = Int(n_in)
    var nb = Int(nb_in)
    var pan = Int(pan_in)
    var nch = Int(nch_in)
    var bq = Int(block_idx.x) // nch
    var ch = Int(block_idx.x) - bq * nch
    var b = Int(b0_in) + bq
    if b >= nb:
        return
    var lo = ts_block_lo(b)
    var mb = ts_block_hi(b, nb, m) - lo
    var blk = a + lo * n
    var tid = Int(thread_idx.x)
    var g = tid // TS_NB
    var l = tid - g * TS_NB
    var j0 = pan * TS_NB
    var pw = min(TS_NB, n - j0)
    var npan = ts_panels(n)
    var c0 = j0 + pw + ch * TS_NB
    if c0 >= n:
        return
    tsh[tid] = tst.unsafe_load((b * npan + pan) * _TT + tid)
    barrier()
    _wy_chunk_kern[True](blk, n, blk, n, c0, n, mb, j0, pw, tsh, part, zsh, wsh, g, l)


def ts_rtile_kernel(a: F32Ptr, tiles: F32Ptr, m_in: Int32, n_in: Int32, nb_in: Int32):
    """Every block's R (the upper triangle of its first n rows) into its tile."""
    var n = Int(n_in)
    var nb = Int(nb_in)
    var t = Int(block_idx.x) * TS_TPB + Int(thread_idx.x)
    if t >= nb * n * n:
        return
    var b = t // (n * n)
    var e = t - b * n * n
    var i = e // n
    var c = e - i * n
    var lo = ts_block_lo(b)
    tiles.unsafe_store(t, ftz(a.unsafe_load((lo + i) * n + c)) if c >= i else Float32(0.0))


def ts_combine_kernel(tiles: F32Ptr, taus: F32Ptr, n_in: Int32, nb_in: Int32, s_in: Int32, p0_in: Int32):
    """`ts_combine_host` for pair p0 + block_idx.x of the level with stride s."""
    var n = Int(n_in)
    var nb = Int(nb_in)
    var s = Int(s_in)
    var ia = 2 * s * (Int(p0_in) + Int(block_idx.x))
    var ib = ia + s
    if ib >= nb:
        return
    var top = tiles + ia * n * n
    var bot = tiles + ib * n * n
    var tp = taus + ib * n
    var tid = Int(thread_idx.x)
    for j in range(n):
        # every thread runs the one norm chain (the same loads, the same bits)
        var alpha = ftz(top.unsafe_load(j * n + j))
        var sg = ts_fma(alpha, alpha, Float32(0.0))
        for i in range(j + 1):
            var x = ftz(bot.unsafe_load(i * n + j))
            sg = ts_fma(x, x, sg)
        var normx = ftz(identical_sqrt(sg))
        dev_barrier()
        var zero = normx == Float32(0.0)
        var tau = Float32(0.0)
        var u1 = Float32(1.0)
        var rjj = Float32(0.0)
        if not zero:
            var rf = ts_reflector(alpha, normx)
            rjj = rf[0]
            u1 = rf[1]
            tau = rf[2]
        var i2 = tid
        while i2 <= j:
            if zero:
                bot.unsafe_store(i2 * n + j, Float32(0.0))
            else:
                bot.unsafe_store(i2 * n + j, ftz(identical_div(ftz(bot.unsafe_load(i2 * n + j)), u1)))
            i2 += TS_TPB
        if tid == 0:
            top.unsafe_store(j * n + j, rjj)
            tp.unsafe_store(j, tau)
        dev_barrier()
        if tau != Float32(0.0):
            var c = j + 1 + tid
            while c < n:
                var tail = Float32(0.0)
                for i in range(j + 1):
                    tail = ts_fma(ftz(bot.unsafe_load(i * n + j)), ftz(bot.unsafe_load(i * n + c)), tail)
                var tc = ftz(top.unsafe_load(j * n + c))
                var td = ts_scale(tau, ftz(tc + tail))
                top.unsafe_store(j * n + c, ftz(tc - td))
                for i in range(j + 1):
                    bot.unsafe_store(i * n + c, ts_fma(-td, ftz(bot.unsafe_load(i * n + j)), ftz(bot.unsafe_load(i * n + c))))
                c += TS_TPB
        dev_barrier()


def ts_capply_kernel(
    cbuf: F32Ptr, tiles: F32Ptr, taus: F32Ptr, n_in: Int32, k_in: Int32, nb_in: Int32, s_in: Int32, p0_in: Int32
):
    """`ts_capply_host` for pair p0 + block_idx.x of the level with stride s:
    one thread per column of C (each column's chain is its own)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var nb = Int(nb_in)
    var s = Int(s_in)
    var ia = 2 * s * (Int(p0_in) + Int(block_idx.x))
    var ib = ia + s
    if ib >= nb:
        return
    var xt = cbuf + ia * n * k
    var xb = cbuf + ib * n * k
    var v = tiles + ib * n * n
    var tp = taus + ib * n
    var c = Int(thread_idx.x)
    while c < k:
        for i in range(n):
            xb.unsafe_store(i * k + c, Float32(0.0))
        for jj in range(n):
            var j = n - 1 - jj
            var tau = tp.unsafe_load(j)
            if tau == Float32(0.0):
                continue
            var tail = Float32(0.0)
            for i in range(j + 1):
                tail = ts_fma(ftz(v.unsafe_load(i * n + j)), ftz(xb.unsafe_load(i * k + c)), tail)
            var xv = ftz(xt.unsafe_load(j * k + c))
            var td = ts_scale(tau, ftz(xv + tail))
            xt.unsafe_store(j * k + c, ftz(xv - td))
            for i in range(j + 1):
                xb.unsafe_store(i * k + c, ts_fma(-td, ftz(v.unsafe_load(i * n + j)), ftz(xb.unsafe_load(i * k + c))))
        c += TS_TPB


def ts_qinit_kernel(cbuf: F32Ptr, q: F32Ptr, m_in: Int32, n_in: Int32, k_in: Int32, nb_in: Int32):
    """q's rows of block b: [C_b; 0] (data movement)."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var nb = Int(nb_in)
    var t = Int(block_idx.x) * TS_TPB + Int(thread_idx.x)
    if t >= m * k:
        return
    var row = t // k
    var c = t - row * k
    var b = min(row // TS_ROWS, nb - 1)
    var li = row - ts_block_lo(b)
    q.unsafe_store(t, cbuf.unsafe_load(b * n * k + li * k + c) if li < n else Float32(0.0))


def ts_leaf_apply_kernel(
    a: F32Ptr, tst: F32Ptr, q: F32Ptr, m_in: Int32, n_in: Int32, k_in: Int32, nb_in: Int32, b0_in: Int32, pan_in: Int32,
    nch_in: Int32,
):
    """`ts_apply_block_host`'s panel `pan`: with nch == 0 for block b0 +
    block_idx.x, every chunk of TS_NB columns of C in turn; with nch > 0
    (TS_GRID_UPDATE) threadgroup t is block b0 + t // nch and chunk t % nch
    (the chunks are independent: the same words)."""
    var part = stack_allocation[TS_PART, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var zsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wsh = stack_allocation[_TT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var nb = Int(nb_in)
    var pan = Int(pan_in)
    var nch = Int(nch_in)
    var bq = Int(block_idx.x)
    var ch = 0
    if nch > 0:
        bq = Int(block_idx.x) // nch
        ch = Int(block_idx.x) - bq * nch
    var b = Int(b0_in) + bq
    if b >= nb:
        return
    var lo = ts_block_lo(b)
    var mb = ts_block_hi(b, nb, m) - lo
    var tid = Int(thread_idx.x)
    var g = tid // TS_NB
    var l = tid - g * TS_NB
    var j0 = pan * TS_NB
    var pw = min(TS_NB, n - j0)
    var npan = ts_panels(n)
    if ch * TS_NB >= k:
        return
    tsh[tid] = tst.unsafe_load((b * npan + pan) * _TT + tid)
    barrier()
    if nch > 0:
        _wy_chunk_kern[False](a + lo * n, n, q + lo * k, k, ch * TS_NB, k, mb, j0, pw, tsh, part, zsh, wsh, g, l)
    else:
        var c0 = 0
        while c0 < k:
            _wy_chunk_kern[False](a + lo * n, n, q + lo * k, k, c0, k, mb, j0, pw, tsh, part, zsh, wsh, g, l)
            c0 += TS_NB


# ---- the factored state between `ts_factor_device` and `ts_apply_device`
struct _TsDev(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    var m: Int
    var n: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.m = 0
        self.n = 0


comptime TS_DEV_STATE = _Global[StorageType=_TsDev, name="MojoXDecompTsqrDevice", init_fn=_TsDev.__init__]


def ts_free_device() raises:
    var st = TS_DEV_STATE.get_or_create_ptr()
    st[].bufs = List[DeviceBuffer[DType.float32]]()
    st[].m = 0
    st[].n = 0


@always_inline
def _wait_apple(ctx: DeviceContext) raises:
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        ctx.synchronize()


def _per_launch(cells: Int) -> Int:
    """Blocks (or pairs) per sliced launch: TS_LAUNCH_CELLS of work, never
    fewer than TS_LAUNCH_MIN_BLOCKS. Bit-neutral: the blocks of one panel
    launch and the pairs of one tree level are independent, and the launch
    order (panel by panel, level by level) is unchanged."""
    var u = TS_LAUNCH_CELLS // (cells if cells > 0 else 1)
    return u if u >= TS_LAUNCH_MIN_BLOCKS else TS_LAUNCH_MIN_BLOCKS


comptime TS_NO_NAN = Int32(2147483647)


def ts_first_nan_kernel(r: MutPointer[Float32, MutAnyOrigin], count: Int32, first: MutPointer[Int32, MutAnyOrigin]):
    """first[0] = the lowest t < count with r[t] NaN (left at TS_NO_NAN
    when there is none)."""
    var t = Int(block_idx.x) * TS_TPB + Int(thread_idx.x)
    if t < Int(count):
        var v = r.unsafe_load(t)
        if v != v:
            _ = Atomic.min(first, Int32(t))


def _grid(count: Int) -> Int:
    return (count + TS_TPB - 1) // TS_TPB if count > 0 else 1


def ts_pack_device(
    ctx: DeviceContext, mut ta: DeviceBuffer[DType.float32], mut tb: DeviceBuffer[DType.float32], m: Int, d: Int, nrhs: Int
) raises -> DeviceBuffer[DType.float32]:
    var n = d + nrhs
    var out = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_function[ts_pack_kernel](
        ta.unsafe_ptr(), tb.unsafe_ptr(), out.unsafe_ptr(), Int32(m), Int32(d), Int32(nrhs),
        grid_dim=_grid(m * n), block_dim=TS_TPB,
    )
    return out^


def ts_pack_center_device(
    ctx: DeviceContext,
    mut ta: DeviceBuffer[DType.float32],
    mut tb: DeviceBuffer[DType.float32],
    mut mx: DeviceBuffer[DType.float32],
    mut my: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
) raises -> DeviceBuffer[DType.float32]:
    """[ta - mx | tb - my] (m x (d + 1)) on the device, enqueued."""
    var n = d + 1
    var out = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_function[ts_pack_center_kernel](
        ta.unsafe_ptr(), tb.unsafe_ptr(), mx.unsafe_ptr(), my.unsafe_ptr(), out.unsafe_ptr(), Int32(m), Int32(d),
        grid_dim=_grid(m * n), block_dim=TS_TPB,
    )
    return out^


def ts_factor_device(ctx: DeviceContext, mut da: DeviceBuffer[DType.float32], m: Int, n: Int, r: F32Ptr, keep: Bool) raises:
    """R (n x n) of the m x n row-major `da` (factored in place) into the host
    r; with `keep` the factored state stays for `ts_apply_device`."""
    comptime if not TS_SMEM_OK:
        raise Error("x_decomp tsqr: the panel kernels' threadgroup memory does not fit this column")
    ts_free_device()
    var nb = ts_blocks(m)
    var npan = ts_panels(n)
    var dt = ctx.enqueue_create_buffer[DType.float32](nb * npan * _TT)
    var dtl = ctx.enqueue_create_buffer[DType.float32](nb * n * n)
    var dtau = ctx.enqueue_create_buffer[DType.float32](nb * n)
    enqueue_fill(ctx, dtau, Float32(0.0))
    var mbmax = m - ts_block_lo(nb - 1)
    var bpl = _per_launch(3 * mbmax * n * TS_NB)
    for pan in range(npan):
        var b0 = 0
        while b0 < nb:
            var cnt = min(bpl, nb - b0)
            ctx.enqueue_function[ts_leaf_panel_kernel](
                da.unsafe_ptr(), dt.unsafe_ptr(), Int32(m), Int32(n), Int32(nb), Int32(b0), Int32(pan),
                grid_dim=cnt, block_dim=TS_TPB,
            )
            _wait_apple(ctx)
            comptime if TS_GRID_UPDATE:
                # the trailing chunks of this slice of blocks, one threadgroup each
                var j1 = pan * TS_NB + min(TS_NB, n - pan * TS_NB)
                var nch = (n - j1 + TS_NB - 1) // TS_NB
                if nch > 0:
                    ctx.enqueue_function[ts_leaf_update_kernel](
                        da.unsafe_ptr(), dt.unsafe_ptr(), Int32(m), Int32(n), Int32(nb), Int32(b0), Int32(pan),
                        Int32(nch), grid_dim=cnt * nch, block_dim=TS_TPB,
                    )
                    _wait_apple(ctx)
            b0 += cnt
    ctx.enqueue_function[ts_rtile_kernel](
        da.unsafe_ptr(), dtl.unsafe_ptr(), Int32(m), Int32(n), Int32(nb), grid_dim=_grid(nb * n * n), block_dim=TS_TPB
    )
    var ppl = _per_launch(n * n * n)
    var s = 1
    while s < nb:
        var pairs = (nb + 2 * s - 1) // (2 * s)
        var p0 = 0
        while p0 < pairs:
            var cnt = min(ppl, pairs - p0)
            ctx.enqueue_function[ts_combine_kernel](
                dtl.unsafe_ptr(), dtau.unsafe_ptr(), Int32(n), Int32(nb), Int32(s), Int32(p0),
                grid_dim=cnt, block_dim=TS_TPB,
            )
            _wait_apple(ctx)
            p0 += cnt
        s *= 2
    # lane/apple-fast-purity2: the NaN refusal's scan runs on the device
    # (the first NaN entry by an atomic min), not as a host loop over R.
    var dnan = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_fill(ctx, dnan, TS_NO_NAN)
    ctx.enqueue_function[ts_first_nan_kernel](
        dtl.unsafe_ptr(), Int32(n * n), dnan.unsafe_ptr(), grid_dim=_grid(n * n), block_dim=TS_TPB
    )
    var hnan = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hnan, src_buf=dnan)
    ctx.enqueue_copy(dst_ptr=r, src_buf=dtl.create_sub_buffer[DType.float32](0, n * n))
    ctx.synchronize()
    var first_nan = hnan.unsafe_ptr().unsafe_load(0)
    _ = hnan^
    _ = dnan^
    if first_nan != TS_NO_NAN:
        raise Error("x_decomp tsqr: R entry " + String(first_nan) + " is NaN (a NaN input, or a device launch cut short): refused")
    if keep:
        var st = TS_DEV_STATE.get_or_create_ptr()
        st[].bufs.append(da)
        st[].bufs.append(dt)
        st[].bufs.append(dtl)
        st[].bufs.append(dtau)
        st[].m = m
        st[].n = n
    _ = dt^
    _ = dtl^
    _ = dtau^


def ts_apply_device(ctx: DeviceContext, c: F32Ptr, m: Int, n: Int, k: Int, keep: Bool = False) raises -> DeviceBuffer[DType.float32]:
    """Q c (m x k) on the device for the kept factorization (c: host n x k);
    the state is released by default. Explicit guarded keep=True retains
    the same immutable factor words for another RHS; caller must free that
    state before replacing/closing the factor context. No content cache is
    inferred and no different factorization is reused."""
    var st = TS_DEV_STATE.get_or_create_ptr()
    if len(st[].bufs) != 4 or st[].m != m or st[].n != n:
        ts_free_device()
        raise Error("x_decomp tsqr: no kept factorization of this shape (tsqr_r with keep first)")
    if keep:
        comptime if not is_defined["MOJOLEARN_IDN_TSQR_REUSE"]():
            raise Error("x_decomp tsqr: retained apply requires the explicit reuse experiment")
        var retained_bytes = 0
        for i in range(len(st[].bufs)):
            retained_bytes += len(st[].bufs[i])*4
        if retained_bytes>16*1024*1024:
            raise Error("x_decomp tsqr: retained factor state exceeds16 MiB experiment budget")
    var da = st[].bufs[0]
    var dt = st[].bufs[1]
    var dtl = st[].bufs[2]
    var dtau = st[].bufs[3]
    var nb = ts_blocks(m)
    var npan = ts_panels(n)
    var dcb = ctx.enqueue_create_buffer[DType.float32](nb * n * k)
    ctx.enqueue_copy(dst_buf=dcb.create_sub_buffer[DType.float32](0, n * k), src_ptr=c)
    var strides = List[Int]()
    var s = 1
    while s < nb:
        strides.append(s)
        s *= 2
    var ppl = _per_launch(n * n * k)
    for li in range(len(strides)):
        var sl = strides[len(strides) - 1 - li]
        var pairs = (nb + 2 * sl - 1) // (2 * sl)
        var p0 = 0
        while p0 < pairs:
            var cnt = min(ppl, pairs - p0)
            ctx.enqueue_function[ts_capply_kernel](
                dcb.unsafe_ptr(), dtl.unsafe_ptr(), dtau.unsafe_ptr(), Int32(n), Int32(k), Int32(nb), Int32(sl), Int32(p0),
                grid_dim=cnt, block_dim=TS_TPB,
            )
            _wait_apple(ctx)
            p0 += cnt
    var dq = ctx.enqueue_create_buffer[DType.float32](m * k)
    ctx.enqueue_function[ts_qinit_kernel](
        dcb.unsafe_ptr(), dq.unsafe_ptr(), Int32(m), Int32(n), Int32(k), Int32(nb), grid_dim=_grid(m * k), block_dim=TS_TPB
    )
    var mbmax = m - ts_block_lo(nb - 1)
    var bpl = _per_launch(3 * mbmax * k * TS_NB)
    var ach = 0
    comptime if TS_GRID_UPDATE:
        ach = (k + TS_NB - 1) // TS_NB
    for pp in range(npan):
        var pan = npan - 1 - pp
        var b0 = 0
        while b0 < nb:
            var cnt = min(bpl, nb - b0)
            ctx.enqueue_function[ts_leaf_apply_kernel](
                da.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), Int32(m), Int32(n), Int32(k), Int32(nb), Int32(b0), Int32(pan),
                Int32(ach), grid_dim=cnt * (ach if ach > 0 else 1), block_dim=TS_TPB,
            )
            _wait_apple(ctx)
            b0 += cnt
    ctx.synchronize()
    _ = dcb^
    _ = da^
    _ = dt^
    _ = dtl^
    _ = dtau^
    if not keep:
        ts_free_device()
    return dq^
