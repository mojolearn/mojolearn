# SPDX-License-Identifier: Apache-2.0
"""lane/cnn-apple2 (2026-09-28): every launch of CNNClassifier's conv blocks
((32,64) on 3x32x32, batch 256; the fit step's shapes), timed alone on this
device: R launches queued back to back, one wait, median of 5 (GPU time, not
the wait). The GEMMs go through x_cnn's `device_gemm` (the plan it measures
and caches), so the numbers are the shipped step's. Timing only: synthetic
operands.
Lines: CNN-STAGE <block> <stage> <ms>."""
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN
from x_cnn.ops import (
    FP, IP, conv_params, pool_params, CP_LEN, im2col_at, conv_out_at, relu_maxpool_fwd_at,
    pool_relu_rows_bwd_at, fill_one_at, col2im_at,
)
from x_cnn.device import (
    cnn_ctx, launch, fp, ip, device_gemm, _conv_out, _im2col, _conv_relu_on_device, TILED_ROWS, rows_bwd_tiled_kernel, _tiled_grid, _LT, _LR,
)


def fill_kernel(p: FP, n: Int32, salt: UInt32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var x = UInt32(i) * UInt32(1664525) + salt
        x = x ^ (x >> 13)
        x = x * UInt32(2246822519)
        p.unsafe_store(i, Float32(Int(x & UInt32(65535)) - 32767) / Float32(16384))


def rnd(ctx: DeviceContext, n: Int, salt: Int) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_function[fill_kernel](b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n), UInt32(salt),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    return b^


def med(mut xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]; xs[j] = xs[j - 1]; xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def block(ctx: DeviceContext, name: String, N: Int, C: Int, H: Int, OC: Int) raises:
    var raw: List[Int] = [N, C, H, H, OC, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    var cp = conv_params(raw)
    var OH = Int(cp[13])
    var praw: List[Int] = [N, OC, OH, OH, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    var pp = pool_params(praw)
    var PH = Int(pp[12])
    var ckk = C * 9
    var rows = N * OH * OH
    var ny = rows * OC
    var nx = N * C * H * H
    var no = N * OC * PH * PH
    var x = rnd(ctx, nx, 1)
    var w = rnd(ctx, OC * ckk, 2)
    var bias = rnd(ctx, OC, 3)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var y2 = ctx.enqueue_create_buffer[DType.float32](ny)
    var yconv = ctx.enqueue_create_buffer[DType.float32](ny)
    var pout = ctx.enqueue_create_buffer[DType.float32](no)
    var di = ctx.enqueue_create_buffer[DType.int32](no)
    var g = rnd(ctx, no, 4)
    var grow = ctx.enqueue_create_buffer[DType.float32](ny)
    var ones = ctx.enqueue_create_buffer[DType.float32](rows)
    var gw = ctx.enqueue_create_buffer[DType.float32](OC * ckk)
    var gb = ctx.enqueue_create_buffer[DType.float32](OC)
    var dcols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    var dp = ctx.enqueue_create_buffer[DType.int32](CP_LEN)
    ctx.enqueue_copy(dst_buf=dp, src_ptr=cp.unsafe_ptr())
    var both = cp.copy()
    both.extend(pp.copy())
    var dpb = ctx.enqueue_create_buffer[DType.int32](len(both))
    ctx.enqueue_copy(dst_buf=dpb, src_ptr=both.unsafe_ptr())
    var dpp = ctx.enqueue_create_buffer[DType.int32](len(pp))
    ctx.enqueue_copy(dst_buf=dpp, src_ptr=pp.unsafe_ptr())
    ctx.synchronize()
    var names: List[String] = [
        "im2col", "gemm_fwd_NT", "conv_out", "relu_maxpool", "pool_relu_rows_bwd", "fill_one",
        "gemm_dW_TN", "gemm_db_TN", "gemm_dx_NN", "col2im", "conv_fwd_shipped",
    ]
    var total = Float64(0)  # the shipped forward (last stage) is not in it
    for s in range(len(names)):
        var xs = List[Float64]()
        for rep in range(6):
            var t0 = perf_counter_ns()
            for _ in range(5):
                if s == 0:
                    _im2col(ctx, x, cols, dp, rows, ckk, C)
                elif s == 1:
                    device_gemm(ctx, y2, cols, w, rows, OC, ckk, OP_NT)
                elif s == 2:
                    _conv_out(ctx, y2, bias, yconv, dp, N, OH * OH, OC)
                elif s == 3:
                    launch[relu_maxpool_fwd_at](ctx, fp(yconv), fp(pout), fp(pout), fp(pout), ip(di), ip(dpp), no)
                elif s == 4:
                    comptime if TILED_ROWS:
                        var tg = _tiled_grid(N, OH * OH, OC)
                        ctx.enqueue_function[rows_bwd_tiled_kernel](
                            fp(g), fp(yconv), fp(grow), ip(di), ip(dpb), Int32(OH * OH), Int32(OC),
                            grid_dim=(tg[0], tg[1], tg[2]), block_dim=(_LT, _LR, 1),
                        )
                    else:
                        launch[pool_relu_rows_bwd_at](ctx, fp(g), fp(yconv), fp(grow), fp(grow), ip(di), ip(dpb), ny)
                elif s == 5:
                    launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
                elif s == 6:
                    device_gemm(ctx, gw, grow, cols, OC, ckk, rows, OP_TN)
                elif s == 7:
                    device_gemm(ctx, gb, grow, ones, OC, 1, rows, OP_TN)
                elif s == 8:
                    device_gemm(ctx, dcols, grow, w, rows, ckk, OC, OP_NN)
                elif s == 9:
                    launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
                else:
                    # im2col + GEMM + conv_out as the block forward runs them
                    # (the direct kernel where it applies), cols written
                    _conv_relu_on_device(ctx, x, w, bias, dp, cols, y2, yconv, rows, OC, ckk, N, C, True)
            ctx.synchronize()
            if rep > 0:  # the first includes any one-time plan measurement
                xs.append(Float64(perf_counter_ns() - t0) / 1e6 / 5.0)
        var ms = med(xs)
        if s < 10:
            total += ms
        print("CNN-STAGE", name, names[s], String(ms), flush=True)
    print("CNN-STAGE", name, "total", String(total), flush=True)
    _ = x^; _ = w^; _ = bias^; _ = cols^; _ = y2^; _ = yconv^; _ = pout^; _ = di^; _ = g^
    _ = grow^; _ = ones^; _ = gw^; _ = gb^; _ = dcols^; _ = gx^; _ = dp^; _ = dpb^; _ = dpp^
    _ = both^
    _ = cp^
    _ = pp^


def main() raises:
    var ctx = cnn_ctx()
    print("CNN-STAGE-DEVICE", ctx.name(), flush=True)
    block(ctx, "b1", 256, 3, 32, 32)
    block(ctx, "b2", 256, 32, 16, 64)
    block(ctx, "c64", 256, 64, 32, 64)
