# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CNN backward the tape calls: the conv block and linear head backward
launchers (conv_block_backward_into, linear_backward_into), the conv2d
backward stack under them and their bias-gradient fold. Moved unchanged out
of x_cnn/device.mojo; the per-cell bodies stay in x_cnn/ops.mojo and the
shared device plumbing (context, workspaces, GEMM) in x_cnn/device.mojo."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz
from checks.rtf_seam import rtf_mul_add
from gemm.neural_dispatch import NEURAL_GEMM_EXPERIMENT_ENABLED
from gemm.checks.gemm_identical import (
    GEMM_FOLD_SLOTS,
    _fold_drain,
    _fold_push,
    _rtf_leaf_partial,
    contract_partition,
    gemm_operand_strides,
    SPLIT_BLOCK_FOLD_MAX_CELLS,
    SPLITK_FOLD_TPB,
    FLAT_TPB,
    identical_gemm_fold_kernel,
    identical_gemm_fold_stack_kernel,
)
from gemm.contract import OP_NN, OP_TN
from x_cnn.neural_col2im import NI14_TILED_COL2IM, neural_col2im_tiled
from x_cnn.ops import (
    NN14_BOUNDED_IM2COL,
    nn14_conv_rows,
    nn14_wgrad_at,
    nn14_input_grad_at,
    FP,
    IP,
    CP_N,
    CP_C,
    CP_H,
    CP_W,
    CP_OC,
    CP_KH,
    CP_KW,
    CP_OH,
    CP_OW,
    CP_SH,
    CP_SW,
    CP_PH,
    CP_PW,
    CP_DH,
    CP_DW,
    dout_rows_at,
    col2im_at,
    col2im_bounded_at,
    fill_one_at,
    pool_relu_rows_bwd_at,
    pool_relu_row_val,
    relu_bwd_at,
    relu_rows_bwd_at,
)
from x_cnn.device import (
    BIAS_TPB,
    BIAS_WS_SLOT,
    NI12_IMPLICIT_CONV,
    NI12_KS,
    NI12_NCELL,
    NI12_RC,
    NI12_RS,
    NI12_THREADS,
    NI12_TM,
    NI12_TMP,
    NI12_TN,
    NI12_TNP,
    NI14_BOUNDED_COL2IM,
    NI15_BIAS_NO_ONES,
    NI16_CONV_RELU_FUSED,
    ONES_CACHE,
    ONES_WS_SLOT,
    TILED_LAYOUT,
    TILED_ROWS,
    XCNN_BIAS_FOLD,
    _LR,
    _LT,
    _a,
    _conv_relu_on_device,
    _gp,
    _im2col,
    _ni12_udiv,
    _pool_sizes,
    _slots,
    _tiled_grid,
    cnn_ctx,
    device_gemm,
    fetch,
    fp,
    ip,
    isdev,
    launch,
    lp,
    m_fetch,
    m_in,
    m_out,
    outb,
    put,
    put_i,
    put_prm,
    view,
    ws,
)


def rows_bwd_tiled_kernel(dpool: FP, yconv: FP, grow: FP, idx: IP, p: IP, S: Int32, OC: Int32):
    """grow[n*S + s, oc] = pool_relu_row_val(NCHW (n, oc, s)); block (32, 8),
    grid (ceil(S/32), ceil(OC/32), N)."""
    var t = stack_allocation[_LT * (_LT + 1), Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var s0 = Int(block_idx.x) * _LT
    var c0 = Int(block_idx.y) * _LT
    var n = Int(block_idx.z)
    var ss = Int(S)
    var cc = Int(OC)
    comptime for j in range(_LT // _LR):
        var oc = c0 + ty + j * _LR
        var s = s0 + tx
        if s < ss and oc < cc:
            t[(ty + j * _LR) * (_LT + 1) + tx] = pool_relu_row_val((n * cc + oc) * ss + s, dpool, yconv, idx, p)
    barrier()
    comptime for j in range(_LT // _LR):
        var s = s0 + ty + j * _LR
        var oc = c0 + tx
        if s < ss and oc < cc:
            grow.unsafe_store((n * ss + s) * cc + oc, t[tx * (_LT + 1) + ty + j * _LR])


def dout_rows_tiled_kernel(dout: FP, g: FP, S: Int32, OC: Int32):
    """g[n*S + s, oc] = ftz(dout[n, oc, s]) (`dout_rows_at`'s word); block
    (32, 8), grid (ceil(S/32), ceil(OC/32), N)."""
    var t = stack_allocation[_LT * (_LT + 1), Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var s0 = Int(block_idx.x) * _LT
    var c0 = Int(block_idx.y) * _LT
    var n = Int(block_idx.z)
    var ss = Int(S)
    var cc = Int(OC)
    comptime for j in range(_LT // _LR):
        var oc = c0 + ty + j * _LR
        var s = s0 + tx
        if s < ss and oc < cc:
            t[(ty + j * _LR) * (_LT + 1) + tx] = ftz(dout.unsafe_load((n * cc + oc) * ss + s))
    barrier()
    comptime for j in range(_LT // _LR):
        var s = s0 + ty + j * _LR
        var oc = c0 + tx
        if s < ss and oc < cc:
            g.unsafe_store((n * ss + s) * cc + oc, t[tx * (_LT + 1) + ty + j * _LR])


def _col2im(
    ctx: DeviceContext, mut dcols: DeviceBuffer[DType.float32],
    mut gx: DeviceBuffer[DType.float32], mut dp: DeviceBuffer[DType.int32], nx: Int,
) raises:
    comptime if NI14_TILED_COL2IM:
        neural_col2im_tiled(ctx, dcols, gx, dp, nx)
        return
    comptime if NI14_BOUNDED_COL2IM:
        launch[col2im_bounded_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    else:
        launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)


def implicit_wgrad_leaf_kernel(
    g: FP, x: FP, wsp: FP, p: IP, rows_in: Int32, leaf_in: Int32,
):
    """Leaf `t = block_idx.z` of `dW[oc, kk] = sum over rows r of g[r, oc] *
    cols[r, kk]` (the GEMM's OP_TN, A = g [rows x OC], B = cols gathered):
    `ws[t * OC*ckk + oc*ckk + kk] = ftz(acc)`, the chain over the leaf's rows
    ascending from +0.0. A block 64 channels x 64 taps, 4 x 4 cells a thread,
    the rows staged 16 at a time."""
    var atile = stack_allocation[NI12_KS * NI12_TMP, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var btile = stack_allocation[NI12_KS * NI12_TNP, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var tr = tid // NI12_RS
    var tc = tid - tr * NI12_RS
    var row0 = Int(block_idx.x) * NI12_TM  # output channels
    var col0 = Int(block_idx.y) * NI12_TN  # taps
    var t = Int(block_idx.z)
    var rows = Int(rows_in)
    var C = _gp(p, CP_C); var H = _gp(p, CP_H); var W = _gp(p, CP_W)
    var KH = _gp(p, CP_KH); var KW = _gp(p, CP_KW)
    var OH = _gp(p, CP_OH); var OW = _gp(p, CP_OW); var OC = _gp(p, CP_OC)
    var SH = _gp(p, CP_SH); var SW = _gp(p, CP_SW)
    var PH = _gp(p, CP_PH); var PW = _gp(p, CP_PW)
    var DH = _gp(p, CP_DH); var DW = _gp(p, CP_DW)
    var KK = KH * KW
    var ckk = C * KK
    var S = OH * OW
    var mn = OC * ckk
    # A staging (g^T): this thread owns channel `soc` for 4 rows `page + sr + 4 l`
    # (q = tid + 256 l over the 16 x 64 page: 64 consecutive g words of one row)
    var soc = row0 + tid - (tid // NI12_TM) * NI12_TM
    var sr = tid // NI12_TM
    var slive_a = soc < OC
    # B staging (cols gathered): this thread owns tap `skk` for the same 4 rows;
    # the tap's (c, kh, kw) is decoded once
    var skk = col0 + tid - (tid // NI12_TN) * NI12_TN
    var slive_b = skk < ckk
    var sc = 0
    var skh = 0
    var skw = 0
    if slive_b:
        sc = _ni12_udiv(skk, KK)
        var t2 = skk - sc * KK
        skh = _ni12_udiv(t2, KW)
        skw = t2 - skh * KW
    var leaf = Int(leaf_in)
    var lo = t * leaf
    var hi = min(lo + leaf, rows)
    var acc = SIMD[DType.float32, NI12_NCELL](0.0)
    var page = lo
    while page < hi:
        var cnt = min(NI12_KS, hi - page)
        comptime for l in range(NI12_RC):
            var r_l = sr + NI12_RC * l
            var r = page + r_l
            var av = Float32(0.0)
            if slive_a and r_l < cnt:
                av = ftz(g.unsafe_load(r * OC + soc))
            atile[r_l * NI12_TMP + soc - row0] = av
            var bv = Float32(0.0)
            if slive_b and r_l < cnt:
                var n = _ni12_udiv(r, S)
                var rem = r - n * S
                var oh = _ni12_udiv(rem, OW)
                var ow = rem - oh * OW
                var h = oh * SH - PH + skh * DH
                var wv = ow * SW - PW + skw * DW
                if h >= 0 and h < H and wv >= 0 and wv < W:
                    bv = ftz(x.unsafe_load(((n * C + sc) * H + h) * W + wv))
            btile[r_l * NI12_TNP + skk - col0] = bv
        barrier()
        if cnt == NI12_KS:
            comptime for i in range(NI12_KS):
                var av = InlineArray[Float32, NI12_RC](fill=Float32(0.0))
                var bv = InlineArray[Float32, NI12_RC](fill=Float32(0.0))
                comptime for c in range(NI12_RC):
                    av[c] = atile[i * NI12_TMP + tr + NI12_RS * c]
                    bv[c] = btile[i * NI12_TNP + tc + NI12_RS * c]
                comptime for a in range(NI12_RC):
                    comptime for b in range(NI12_RC):
                        acc[a * NI12_RC + b] = rtf_mul_add(av[a], bv[b], acc[a * NI12_RC + b])
        else:
            for i in range(cnt):
                var av = InlineArray[Float32, NI12_RC](fill=Float32(0.0))
                var bv = InlineArray[Float32, NI12_RC](fill=Float32(0.0))
                comptime for c in range(NI12_RC):
                    av[c] = atile[i * NI12_TMP + tr + NI12_RS * c]
                    bv[c] = btile[i * NI12_TNP + tc + NI12_RS * c]
                comptime for a in range(NI12_RC):
                    comptime for b in range(NI12_RC):
                        acc[a * NI12_RC + b] = rtf_mul_add(av[a], bv[b], acc[a * NI12_RC + b])
        barrier()
        page += NI12_KS
    comptime for a in range(NI12_RC):
        var oc = row0 + tr + NI12_RS * a
        if oc < OC:
            comptime for b in range(NI12_RC):
                var kk = col0 + tc + NI12_RS * b
                if kk < ckk:
                    wsp.unsafe_store(t * mn + oc * ckk + kk, ftz(acc[a * NI12_RC + b]))


# TOMBSTONE: MOJOLEARN_IDN_CNN_DX_IMPLICIT (noise 0.980) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_DX_IMPLICIT.patch
# TOMBSTONE: MOJOLEARN_IDN_CNN_WGRAD_LEAF1024 (noise 0.972) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_WGRAD_LEAF1024.patch (the same patch: x_cnn/cnn_dx_wgrad.mojo held both)
def _implicit_wgrad(
    ctx: DeviceContext, mut gw: DeviceBuffer[DType.float32], mut g: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32], mut dp: DeviceBuffer[DType.int32], slot: Int, rows: Int, OC: Int, ckk: Int,
) raises:
    """dW = g^T . im2col(x) through `implicit_wgrad_leaf_kernel` and the
    GEMM's fold kernels (`_launch_split`'s two launches), the partials in
    workspace slot `slot` (P * OC * ckk floats)."""
    if rows <= 0 or OC <= 0 or ckk <= 0:
        return
    var lp = contract_partition(rows)
    var mn = OC * ckk
    var wsp = ws(ctx, slot, lp[1] * mn)
    ctx.enqueue_function[implicit_wgrad_leaf_kernel](
        fp(g), fp(dx), fp(wsp), ip(dp), Int32(rows), Int32(lp[0]),
        grid_dim=((OC + NI12_TM - 1) // NI12_TM, (ckk + NI12_TN - 1) // NI12_TN, lp[1]),
        block_dim=(NI12_THREADS, 1, 1),
    )
    if mn <= SPLIT_BLOCK_FOLD_MAX_CELLS:
        ctx.enqueue_function[identical_gemm_fold_kernel[True]](
            fp(gw), fp(wsp), Int32(mn), Int32(lp[1]), Int32(lp[1]),
            grid_dim=(mn, 1, 1), block_dim=(SPLITK_FOLD_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[identical_gemm_fold_stack_kernel](
            fp(gw), fp(wsp), Int32(mn), Int32(lp[1]),
            grid_dim=((mn + FLAT_TPB - 1) // FLAT_TPB, 1, 1), block_dim=(FLAT_TPB, 1, 1),
        )
    _ = wsp^


def bias_leaf_kernel(g: FP, ones: FP, part: FP, m_oc: Int32, k_rows: Int32, leaf: Int32, p_count: Int32):
    """Thread t = l OC + oc: part[t] = leaf l of cell (oc, 0) of G^T ones."""
    var OC = Int(m_oc)
    var P = Int(p_count)
    var t = Int(block_idx.x) * BIAS_TPB + Int(thread_idx.x)
    if t >= OC * P:
        return
    var l = t // OC
    var oc = t - l * OC
    var k = Int(k_rows)
    var sd = gemm_operand_strides(OP_TN, OC, 1, k)
    part.unsafe_store(t, _rtf_leaf_partial(g, ones, oc, 0, l, P, Int(leaf), k, sd[0], sd[1], sd[2], sd[3]))


def bias_fold_kernel(part: FP, gb: FP, m_oc: Int32, p_count: Int32):
    """Thread oc: gb[oc] = ftz(the contract tree over part[l OC + oc], l ascending)."""
    var OC = Int(m_oc)
    var oc = Int(block_idx.x) * BIAS_TPB + Int(thread_idx.x)
    if oc >= OC:
        return
    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    for l in range(Int(p_count)):
        _ = _fold_push(stack, occ, part.unsafe_load(l * OC + oc))
    gb.unsafe_store(oc, ftz(_fold_drain(stack, occ)))


def bias_leaf_no_ones_kernel(g: FP, part: FP, m_oc: Int32, k_rows: Int32, leaf: Int32, p_count: Int32):
    """NI15: the exact G^T*1 leaf without loading a ones operand.

    Keep the explicit rtf FMA by +1 (not a freely contracted or reordered
    add), the leaf FTZ and the existing canonical fold. Host GEMM remains
    the reference spelling of the same arithmetic on every column.
    """
    var OC = Int(m_oc)
    var t = Int(block_idx.x) * BIAS_TPB + Int(thread_idx.x)
    if t >= OC * Int(p_count):
        return
    var l = t // OC; var oc = t - l * OC
    var lo = l * Int(leaf)
    var hi = min(lo + Int(leaf), Int(k_rows))
    var acc = Float32(0)
    for row in range(lo, hi):
        acc = rtf_mul_add(ftz(g.unsafe_load(row * OC + oc)), Float32(1), acc)
    part.unsafe_store(t, ftz(acc))


def bias_grad_gemm(
    ctx: DeviceContext, mut gb: DeviceBuffer[DType.float32], mut g: DeviceBuffer[DType.float32],
    mut ones: DeviceBuffer[DType.float32], OC: Int, rows: Int,
) raises:
    """gb = G^T ones (OC x 1 over `rows`): the pinned GEMM TN's words, by
    XCNN_BIAS_FOLD's two launches when it is on."""
    comptime if NEURAL_GEMM_EXPERIMENT_ENABLED:
        device_gemm(ctx, gb, g, ones, OC, 1, rows, OP_TN)
        return
    comptime if XCNN_BIAS_FOLD or NI15_BIAS_NO_ONES:
        if rows > 0 and OC > 0:
            var lp = contract_partition(rows)
            var leaf = lp[0]
            var P = lp[1]
            var part = ws(ctx, BIAS_WS_SLOT, OC * P)
            comptime if NI15_BIAS_NO_ONES:
                ctx.enqueue_function[bias_leaf_no_ones_kernel](
                    fp(g), fp(part), Int32(OC), Int32(rows), Int32(leaf), Int32(P),
                    grid_dim=((OC * P + BIAS_TPB - 1) // BIAS_TPB, 1, 1), block_dim=(BIAS_TPB, 1, 1),
                )
            else:
                ctx.enqueue_function[bias_leaf_kernel](
                    fp(g), fp(ones), fp(part), Int32(OC), Int32(rows), Int32(leaf), Int32(P),
                    grid_dim=((OC * P + BIAS_TPB - 1) // BIAS_TPB, 1, 1), block_dim=(BIAS_TPB, 1, 1),
                )
            ctx.enqueue_function[bias_fold_kernel](
                fp(part), fp(gb), Int32(OC), Int32(P),
                grid_dim=((OC + BIAS_TPB - 1) // BIAS_TPB, 1, 1), block_dim=(BIAS_TPB, 1, 1),
            )
            _ = part^
            return
    device_gemm(ctx, gb, g, ones, OC, 1, rows, OP_TN)


def ones_buf(ctx: DeviceContext, slot: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """`n` floats for a backward's ones operand. With `ONES_CACHE` they are
    the resident ones (already 1.0f; filled here, in order, when the vector
    grows); without it they are workspace slot `slot`, which the caller
    fills as it always did."""
    comptime if NI15_BIAS_NO_ONES:
        # Only a valid placeholder address is required: the candidate leaf
        # materializes +1 directly and never reads this allocation.
        return ws(ctx, slot, 1)
    comptime if ONES_CACHE:
        var need = n if n > 0 else 1
        var b = ws(ctx, ONES_WS_SLOT, need)
        var s = _slots()
        if s[].ones_n < need:
            # a grown slot is a new buffer (`ws`), so every word is filled
            b.enqueue_fill(Float32(1))
            s[].ones_n = need
        return b^
    return ws(ctx, slot, n)


def conv2d_backward_m(a: List[Int], dev: Int, prm: List[Int32]) raises:
    """a = [x, w, dout, dx (N*C*H*W), dW (OC*C*KH*KW), db (OC)]."""
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var ctx = cnn_ctx()
    var dxin = m_in(ctx, 0, a[0], nx, isdev(dev, 0))
    var dw = m_in(ctx, 1, a[1], OC * ckk, isdev(dev, 1))
    var ddout = m_in(ctx, 2, a[2], rows * OC, isdev(dev, 2))
    var dp = put_prm(ctx, 3, prm)
    var cols = ws(ctx, 4, 1 if (NN14_BOUNDED_IM2COL or NI12_IMPLICIT_CONV) else rows * ckk)
    var g = ws(ctx, 5, rows * OC)
    var ones = ones_buf(ctx, 6, rows)
    var gw = m_out(ctx, 7, a[4], OC * ckk, isdev(dev, 4))
    var gb = m_out(ctx, 8, a[5], OC, isdev(dev, 5))
    # TOMBSTONE: MOJOLEARN_IDN_CNN_DX_IMPLICIT (noise 0.980) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_DX_IMPLICIT.patch
    # TOMBSTONE: MOJOLEARN_IDN_CNN_WGRAD_LEAF1024 (noise 0.972) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_WGRAD_LEAF1024.patch (the same patch: x_cnn/cnn_dx_wgrad.mojo held both)
    var dcols = ws(ctx, 9, 1 if NN14_BOUNDED_IM2COL else rows * ckk)
    var gx = m_out(ctx, 10, a[3], nx, isdev(dev, 3))
    comptime if not NN14_BOUNDED_IM2COL and not NI12_IMPLICIT_CONV:
        _im2col(ctx, dxin, cols, dp, rows, ckk, C)
    comptime if TILED_LAYOUT:
        var tg = _tiled_grid(N, rows // N, OC)
        ctx.enqueue_function[dout_rows_tiled_kernel](
            fp(ddout), fp(g), Int32(rows // N), Int32(OC),
            grid_dim=(tg[0], tg[1], tg[2]), block_dim=(_LT, _LR, 1),
        )
    else:
        launch[dout_rows_at](ctx, fp(ddout), fp(g), fp(g), fp(g), ip(dp), ip(dp), rows * OC)
    comptime if not ONES_CACHE and not NI15_BIAS_NO_ONES:
        launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the weight gradient's reduction over the N*OH*OW rows is
    # the pinned GEMM's (leaves + balanced fold), never an atomic accumulation.
    comptime if NN14_BOUNDED_IM2COL:
        launch[nn14_wgrad_at](ctx, fp(dxin), fp(g), fp(gw), fp(gw), ip(dp), ip(dp), OC * ckk)
    elif NI12_IMPLICIT_CONV:
        # lane gap-gemm-layers: the cols operand gathered from x, the same
        # leaf partials and fold launches as the GEMM's split plan
        _implicit_wgrad(ctx, gw, g, dxin, dp, 18, rows, OC, ckk)
    else:
        device_gemm(ctx, gw, g, cols, OC, ckk, rows, OP_TN)
    bias_grad_gemm(ctx, gb, g, ones, OC, rows)
    comptime if NN14_BOUNDED_IM2COL:
        launch[nn14_input_grad_at](ctx, fp(g), fp(dw), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    else:
        device_gemm(ctx, dcols, g, dw, rows, ckk, OC, OP_NN)
        _col2im(ctx, dcols, gx, dp, nx)
    m_fetch(ctx, gx, a[3], nx, isdev(dev, 3))
    m_fetch(ctx, gw, a[4], OC * ckk, isdev(dev, 4))
    m_fetch(ctx, gb, a[5], OC, isdev(dev, 5))
    ctx.synchronize()
    _ = dxin^
    _ = dw^
    _ = ddout^
    _ = dp^
    _ = cols^
    _ = g^
    _ = ones^
    _ = gw^
    _ = gb^
    _ = dcols^
    _ = gx^
    _ = ctx^


def conv2d_backward_into(x: FP, w: FP, dout: FP, prm: List[Int32], gx_out: FP, gw_out: FP, gb_out: FP) raises:
    """dx (N*C*H*W), dW (OC*C*KH*KW), db (OC)."""
    conv2d_backward_m([_a(x), _a(w), _a(dout), _a(gx_out), _a(gw_out), _a(gb_out)], 0, prm)


def conv2d_backward_device(
    x: List[Float32], w: List[Float32], dout: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    """[dx | dW | db], concatenated (the List form, for the seam check)."""
    var OC = Int(prm[CP_OC])
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])
    var sx = x.copy()
    var sw = w.copy()
    var sd = dout.copy()
    var out = List[Float32](length=nx + OC * ckk + OC, fill=Float32(0))
    var base = lp(out)
    conv2d_backward_into(lp(sx), lp(sw), lp(sd), prm, base, base + nx, base + nx + OC * ckk)
    _ = sx^
    _ = sw^
    _ = sd^
    return out^


def linear_backward_m(a: List[Int], dev: Int, n: Int, d_in: Int, d_out: Int) raises:
    """dW = G^T X and db = G^T 1 (GEMM TN over the rows, the pinned fold),
    dx = G W (GEMM NN); a = [x, w, g, dx, dW, db]."""
    var ctx = cnn_ctx()
    var dx = m_in(ctx, 0, a[0], n * d_in, isdev(dev, 0))
    var dw = m_in(ctx, 1, a[1], d_out * d_in, isdev(dev, 1))
    var dg = m_in(ctx, 2, a[2], n * d_out, isdev(dev, 2))
    var dones = ones_buf(ctx, 3, n)
    comptime if not ONES_CACHE and not NI15_BIAS_NO_ONES:
        dones.enqueue_fill(Float32(1))
    var gx = m_out(ctx, 4, a[3], n * d_in, isdev(dev, 3))
    var gw = m_out(ctx, 5, a[4], d_out * d_in, isdev(dev, 4))
    var gb = m_out(ctx, 6, a[5], d_out, isdev(dev, 5))
    device_gemm(ctx, gw, dg, dx, d_out, d_in, n, OP_TN)
    bias_grad_gemm(ctx, gb, dg, dones, d_out, n)
    device_gemm(ctx, gx, dg, dw, n, d_in, d_out, OP_NN)
    m_fetch(ctx, gx, a[3], n * d_in, isdev(dev, 3))
    m_fetch(ctx, gw, a[4], d_out * d_in, isdev(dev, 4))
    m_fetch(ctx, gb, a[5], d_out, isdev(dev, 5))
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dg^
    _ = dones^
    _ = gx^
    _ = gw^
    _ = gb^
    _ = ctx^


def linear_backward_into[resident: Bool = False](
    x: FP, w: FP, g: FP, n: Int, d_in: Int, d_out: Int, gx_out: FP, gw_out: FP, gb_out: FP
) raises:
    """`linear_backward_m` with every argument resident, or every one host."""
    linear_backward_m([_a(x), _a(w), _a(g), _a(gx_out), _a(gw_out), _a(gb_out)], 63 if resident else 0, n, d_in, d_out)


def conv_block_backward_into[resident: Bool = False](
    x: FP, w: FP, bias: FP, g: FP, idx: IP, cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    gx_out: FP, gw_out: FP, gb_out: FP, save_cols: Int = 0, save_y: Int = 0,
) raises:
    """From the gradient of the block's output `g`: dx (when `need_dx`), dW, db.
    Resident with `save_cols` and `save_y`: the forward's im2col matrix and
    conv output, read instead of recomputed (the same kernels on the same
    inputs produced them, so the values are the recomputation's)."""
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = N * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ny = rows * OC
    var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
    var no = _pool_sizes(pprm)[1] if pool else ny
    var ctx = cnn_ctx()
    var dx = put[resident](ctx, 0, x, nx)
    var dw = put[resident](ctx, 1, w, OC * ckk)
    var dbias = put[resident](ctx, 2, bias, OC)
    var dgo = put[resident](ctx, 3, g, no)
    var dp = put_prm(ctx, 5, cprm)
    var saved = resident and save_y != 0 and (NN14_BOUNDED_IM2COL or save_cols != 0)
    var tile_rows = nn14_conv_rows(rows, ckk, OC) if NN14_BOUNDED_IM2COL else rows
    var cols = view(ctx, FP(unsafe_from_address=save_cols), rows * ckk) if saved and not NN14_BOUNDED_IM2COL else ws(ctx, 7, 1 if NI12_IMPLICIT_CONV else tile_rows * ckk)
    var y2 = ws(ctx, 8, 1 if NI12_IMPLICIT_CONV else tile_rows * OC)
    var yconv = view(ctx, FP(unsafe_from_address=save_y), ny) if saved else ws(ctx, 9, ny)
    var gy = ws(ctx, 11, ny)
    if not saved:
        _conv_relu_on_device(ctx, dx, dw, dbias, dp, cols, y2, yconv, rows, OC, ckk, N, C)
    # conv2d_backward_into from here, on the device-resident gy and cols
    var grow = ws(ctx, 12, ny)
    # the conv block followed by the pool block: a host array the upload
    # reads, kept alive past the entry's synchronize
    var both = cprm.copy()
    both.extend(pprm.copy())
    if pool:
        # DEVIATION 5720: the max pool's backward, the ReLU's and the row
        # layout in one launch (the same values)
        var di = put_i[resident](ctx, 4, idx, no)
        var dpb = put_prm(ctx, 6, both)
        comptime if TILED_ROWS:
            var S = rows // N
            var g = _tiled_grid(N, S, OC)
            ctx.enqueue_function[rows_bwd_tiled_kernel](
                fp(dgo), fp(yconv), fp(grow), ip(di), ip(dpb), Int32(S), Int32(OC),
                grid_dim=(g[0], g[1], g[2]), block_dim=(_LT, _LR, 1),
            )
        else:
            launch[pool_relu_rows_bwd_at](ctx, fp(dgo), fp(yconv), fp(grow), fp(grow), ip(di), ip(dpb), ny)
        _ = di^
        _ = dpb^
    else:
        comptime if NI16_CONV_RELU_FUSED:
            launch[relu_rows_bwd_at](ctx, fp(dgo), fp(yconv), fp(grow), fp(grow), ip(dp), ip(dp), ny)
        else:
            launch[relu_bwd_at](ctx, fp(yconv), fp(dgo), fp(gy), fp(gy), ip(dp), ip(dp), ny)
            launch[dout_rows_at](ctx, fp(gy), fp(grow), fp(grow), fp(grow), ip(dp), ip(dp), ny)
    var ones = ones_buf(ctx, 13, rows)
    var gw = outb[resident](ctx, 14, gw_out, OC * ckk)
    var gb = outb[resident](ctx, 15, gb_out, OC)
    comptime if not ONES_CACHE and not NI15_BIAS_NO_ONES:
        launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the pinned GEMM's fold over the rows, never an atomic.
    comptime if NN14_BOUNDED_IM2COL:
        launch[nn14_wgrad_at](ctx, fp(dx), fp(grow), fp(gw), fp(gw), ip(dp), ip(dp), OC * ckk)
    elif NI12_IMPLICIT_CONV:
        # lane gap-gemm-layers: cols never materialized (the forward wrote
        # none); the same words as the GEMM on the saved cols
        _implicit_wgrad(ctx, gw, grow, dx, dp, 18, rows, OC, ckk)
    else:
        device_gemm(ctx, gw, grow, cols, OC, ckk, rows, OP_TN)
    bias_grad_gemm(ctx, gb, grow, ones, OC, rows)
    if need_dx:
        # TOMBSTONE: MOJOLEARN_IDN_CNN_DX_IMPLICIT (noise 0.980) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_DX_IMPLICIT.patch
        # TOMBSTONE: MOJOLEARN_IDN_CNN_WGRAD_LEAF1024 (noise 0.972) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_WGRAD_LEAF1024.patch (the same patch: x_cnn/cnn_dx_wgrad.mojo held both)
        var dcols = ws(ctx, 16, 1 if NN14_BOUNDED_IM2COL else rows * ckk)
        var gx = outb[resident](ctx, 17, gx_out, nx)
        comptime if NN14_BOUNDED_IM2COL:
            launch[nn14_input_grad_at](ctx, fp(grow), fp(dw), fp(gx), fp(gx), ip(dp), ip(dp), nx)
        else:
            device_gemm(ctx, dcols, grow, dw, rows, ckk, OC, OP_NN)
            _col2im(ctx, dcols, gx, dp, nx)
        fetch[resident](ctx, gx, gx_out, nx)
        _ = dcols^
        _ = gx^
    fetch[resident](ctx, gw, gw_out, OC * ckk)
    fetch[resident](ctx, gb, gb_out, OC)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dgo^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = yconv^
    _ = gy^
    _ = both^
    _ = grow^
    _ = ones^
    _ = gw^
    _ = gb^
    _ = ctx^
