# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE ON THE DEVICE: x_cnn/ops.mojo's element functions, one
thread per element, and the pinned GEMM (`identical_gemm`, full FP32, never
the vendor route) for every contraction. Host lists in, host lists out; each
entry owns its DeviceContext and synchronizes before it returns.

The host twin is x_cnn/host/ops_host.mojo: the same element functions in a
loop and `gemm_oracle` for the contractions."""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import identical_gemm
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN
from metrics.checks.device_io import upload_f32, upload_i32, download_f32, download_i32
from x_cnn.ops import (
    FP, IP, ElemFn, CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW,
    im2col_at, conv_out_at, dout_rows_at, col2im_at, fill_one_at,
    PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW,
    maxpool_fwd_at, maxpool_bwd_at, avgpool_fwd_at, avgpool_bwd_at,
    relu_fwd_at, relu_bwd_at, add_at, bias_rows_at, softmax_xent_row_at, seq_mean, sgd_at,
    bn_stats_at, bn_eval_stats_at, bn_apply_at, bn_running_at, bn_bwd_red_at, bn_bwd_dx_at, bn_bwd_eval_dx_at,
    dropout2d_at, mul_at, spmm_at, gcn_deg_at, gcn_norm_at,
)

comptime TPB = 256


def elem_kernel[f: ElemFn](a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(total):
        f(i, a, b, c, d, q, p)


@always_inline
def fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def ip(mut buf: DeviceBuffer[DType.int32]) -> IP:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def launch[f: ElemFn](ctx: DeviceContext, a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int) raises:
    if total <= 0:
        return
    comptime k = elem_kernel[f]
    ctx.enqueue_function[k](a, b, c, d, q, p, Int32(total), grid_dim=(total + TPB - 1) // TPB, block_dim=TPB)


def device_gemm(
    ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int, op: Int,
) raises:
    """`C = op(A) . op(B)` under mojolearn.identical.gemm.fp32.v1, full FP32
    in both tiers (allow_vendor=False: the NVIDIA vendor route is TF32)."""
    identical_gemm[False](ctx, c, a, b, m, n, k, op)


def gemm_device(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, op: Int) raises -> List[Float32]:
    var ctx = DeviceContext()
    var da = upload_f32(ctx, a)
    var db = upload_f32(ctx, b)
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    device_gemm(ctx, dc, da, db, m, n, k, op)
    var out = download_f32(ctx, dc, m * n)
    _ = da^
    _ = db^
    _ = dc^
    _ = ctx^
    return out^


def conv2d_forward_device(
    x: List[Float32], w: List[Float32], bias: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dw = upload_f32(ctx, w)
    var dbias = upload_f32(ctx, bias)
    var dp = upload_i32(ctx, prm)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var y2 = ctx.enqueue_create_buffer[DType.float32](rows * OC)
    var out = ctx.enqueue_create_buffer[DType.float32](rows * OC)
    launch[im2col_at](ctx, fp(dx), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * ckk)
    device_gemm(ctx, y2, cols, dw, rows, OC, ckk, OP_NT)
    launch[conv_out_at](ctx, fp(y2), fp(dbias), fp(out), fp(out), ip(dp), ip(dp), rows * OC)
    var result = download_f32(ctx, out, rows * OC)
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = out^
    _ = ctx^
    return result^


def conv2d_backward_device(
    x: List[Float32], w: List[Float32], dout: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    """[dx | dW | db], concatenated: dx is N*C*H*W, dW is OC*C*KH*KW, db is OC."""
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var ctx = DeviceContext()
    var dxin = upload_f32(ctx, x)
    var dw = upload_f32(ctx, w)
    var ddout = upload_f32(ctx, dout)
    var dp = upload_i32(ctx, prm)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var g = ctx.enqueue_create_buffer[DType.float32](rows * OC)
    var ones = ctx.enqueue_create_buffer[DType.float32](rows)
    var gw = ctx.enqueue_create_buffer[DType.float32](OC * ckk)
    var gb = ctx.enqueue_create_buffer[DType.float32](OC)
    var dcols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    launch[im2col_at](ctx, fp(dxin), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * ckk)
    launch[dout_rows_at](ctx, fp(ddout), fp(g), fp(g), fp(g), ip(dp), ip(dp), rows * OC)
    launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the weight gradient's reduction over the N*OH*OW rows is
    # the pinned GEMM's (leaves + balanced fold), never an atomic accumulation.
    device_gemm(ctx, gw, g, cols, OC, ckk, rows, OP_TN)
    device_gemm(ctx, gb, g, ones, OC, 1, rows, OP_TN)
    device_gemm(ctx, dcols, g, dw, rows, ckk, OC, OP_NN)
    launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    var rx = download_f32(ctx, gx, nx)
    var rw = download_f32(ctx, gw, OC * ckk)
    var rb = download_f32(ctx, gb, OC)
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
    var result = List[Float32](capacity=nx + OC * ckk + OC)
    result.extend(rx^)
    result.extend(rw^)
    result.extend(rb^)
    return result^


def _pool_sizes(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def maxpool2d_forward_device(x: List[Float32], prm: List[Int32], mut idx: List[Int32]) raises -> List[Float32]:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](no)
    var di = ctx.enqueue_create_buffer[DType.int32](no)
    launch[maxpool_fwd_at](ctx, fp(dx), fp(out), fp(out), fp(out), ip(di), ip(dp), no)
    var result = download_f32(ctx, out, no)
    idx = download_i32(ctx, di, no)
    _ = dx^
    _ = dp^
    _ = out^
    _ = di^
    _ = ctx^
    return result^


def maxpool2d_backward_device(dout: List[Float32], idx: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = DeviceContext()
    var dd = upload_f32(ctx, dout)
    var di = upload_i32(ctx, idx)
    var dp = upload_i32(ctx, prm)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    launch[maxpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(di), ip(dp), nx)
    var result = download_f32(ctx, gx, nx)
    _ = dd^
    _ = di^
    _ = dp^
    _ = gx^
    _ = ctx^
    return result^


def avgpool2d_forward_device(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](no)
    launch[avgpool_fwd_at](ctx, fp(dx), fp(out), fp(out), fp(out), ip(dp), ip(dp), no)
    var result = download_f32(ctx, out, no)
    _ = dx^
    _ = dp^
    _ = out^
    _ = ctx^
    return result^


def avgpool2d_backward_device(dout: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = DeviceContext()
    var dd = upload_f32(ctx, dout)
    var dp = upload_i32(ctx, prm)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    launch[avgpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    var result = download_f32(ctx, gx, nx)
    _ = dd^
    _ = dp^
    _ = gx^
    _ = ctx^
    return result^


def map2_device[f: ElemFn](a: List[Float32], b: List[Float32], n_out: Int, prm: List[Int32]) raises -> List[Float32]:
    """dst[i] = f(a, b) for i < n_out (slots a, b, dst)."""
    var ctx = DeviceContext()
    var da = upload_f32(ctx, a)
    var db = upload_f32(ctx, b)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out if n_out > 0 else 1)
    launch[f](ctx, fp(da), fp(db), fp(out), fp(out), ip(dp), ip(dp), n_out)
    var result = download_f32(ctx, out, n_out)
    _ = da^
    _ = db^
    _ = dp^
    _ = out^
    _ = ctx^
    return result^


def relu_forward_device(x: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_device[relu_fwd_at](x, x, len(x), prm)


def relu_backward_device(x: List[Float32], g: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_device[relu_bwd_at](x, g, len(x), prm)


def add_device(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_device[add_at](a, b, len(a), prm)


def linear_forward_device(x: List[Float32], w: List[Float32], bias: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    """y = x W^T + b: the pinned GEMM NT, then one add per element."""
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dw = upload_f32(ctx, w)
    var db = upload_f32(ctx, bias)
    var prm: List[Int32] = [Int32(n), Int32(d_in), Int32(d_out)]
    var dp = upload_i32(ctx, prm)
    var y = ctx.enqueue_create_buffer[DType.float32](n * d_out)
    var out = ctx.enqueue_create_buffer[DType.float32](n * d_out)
    device_gemm(ctx, y, dx, dw, n, d_out, d_in, OP_NT)
    launch[bias_rows_at](ctx, fp(y), fp(db), fp(out), fp(out), ip(dp), ip(dp), n * d_out)
    var result = download_f32(ctx, out, n * d_out)
    _ = dx^
    _ = dw^
    _ = db^
    _ = dp^
    _ = y^
    _ = out^
    _ = ctx^
    return result^


def linear_backward_device(x: List[Float32], w: List[Float32], g: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    """[dx | dW | db]: dW = G^T X and db = G^T 1 (GEMM TN over the rows,
    the pinned fold), dx = G W (GEMM NN)."""
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dw = upload_f32(ctx, w)
    var dg = upload_f32(ctx, g)
    var ones = List[Float32](length=n, fill=Float32(1))
    var dones = upload_f32(ctx, ones)
    var gx = ctx.enqueue_create_buffer[DType.float32](n * d_in)
    var gw = ctx.enqueue_create_buffer[DType.float32](d_out * d_in)
    var gb = ctx.enqueue_create_buffer[DType.float32](d_out)
    device_gemm(ctx, gw, dg, dx, d_out, d_in, n, OP_TN)
    device_gemm(ctx, gb, dg, dones, d_out, 1, n, OP_TN)
    device_gemm(ctx, gx, dg, dw, n, d_in, d_out, OP_NN)
    var rx = download_f32(ctx, gx, n * d_in)
    var rw = download_f32(ctx, gw, d_out * d_in)
    var rb = download_f32(ctx, gb, d_out)
    _ = dx^
    _ = dw^
    _ = dg^
    _ = dones^
    _ = gx^
    _ = gw^
    _ = gb^
    _ = ctx^
    rx.extend(rw^)
    rx.extend(rb^)
    return rx^


def softmax_xent_device(logits: List[Float32], labels: List[Int32], n: Int, k: Int) raises -> List[Float32]:
    """[grad (n*k) | proba (n*k) | mean loss (1)]."""
    var ctx = DeviceContext()
    var dl = upload_f32(ctx, logits)
    var dy = upload_i32(ctx, labels)
    var prm: List[Int32] = [Int32(n), Int32(k)]
    var dp = upload_i32(ctx, prm)
    var grad = ctx.enqueue_create_buffer[DType.float32](n * k)
    var proba = ctx.enqueue_create_buffer[DType.float32](n * k)
    var rl = ctx.enqueue_create_buffer[DType.float32](n)
    grad.enqueue_fill(Float32(0))
    rl.enqueue_fill(Float32(0))
    launch[softmax_xent_row_at](ctx, fp(dl), fp(grad), fp(proba), fp(rl), ip(dy), ip(dp), n)
    var rg = download_f32(ctx, grad, n * k)
    var rp = download_f32(ctx, proba, n * k)
    var rr = download_f32(ctx, rl, n)
    _ = dl^
    _ = dy^
    _ = dp^
    _ = grad^
    _ = proba^
    _ = rl^
    _ = ctx^
    rg.extend(rp^)
    rg.append(seq_mean(rr, n))
    return rg^


def sgd_device(w: List[Float32], g: List[Float32], v: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    """[w' | v']."""
    var n = len(w)
    var ctx = DeviceContext()
    var dw = upload_f32(ctx, w)
    var dg = upload_f32(ctx, g)
    var dv = upload_f32(ctx, v)
    var dh = upload_f32(ctx, hyper)
    var prm: List[Int32] = [Int32(n)]
    var dp = upload_i32(ctx, prm)
    launch[sgd_at](ctx, fp(dw), fp(dg), fp(dv), fp(dh), ip(dp), ip(dp), n)
    var rw = download_f32(ctx, dw, n)
    var rv = download_f32(ctx, dv, n)
    _ = dw^
    _ = dg^
    _ = dv^
    _ = dh^
    _ = dp^
    _ = ctx^
    rw.extend(rv^)
    return rw^


def batchnorm_forward_device(
    x: List[Float32], running: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    """[y | running' | aux'] (aux carries the statistics the backward reads)."""
    var total = len(x)
    var C = Int(prm[1])
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dr = upload_f32(ctx, running)
    var da = upload_f32(ctx, aux)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](total)
    if training:
        launch[bn_stats_at](ctx, fp(dx), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    else:
        launch[bn_eval_stats_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    launch[bn_apply_at](ctx, fp(dx), fp(da), fp(out), fp(out), ip(dp), ip(dp), total)
    if training:
        launch[bn_running_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    var ry = download_f32(ctx, out, total)
    var rr = download_f32(ctx, dr, len(running))
    var ra = download_f32(ctx, da, len(aux))
    _ = dx^
    _ = dr^
    _ = da^
    _ = dp^
    _ = out^
    _ = ctx^
    ry.extend(rr^)
    ry.extend(ra^)
    return ry^


def batchnorm_backward_device(
    x: List[Float32], g: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    """[dx | aux'] (aux' carries sum_g = dbeta and sum_gx = dgamma)."""
    var total = len(x)
    var C = Int(prm[1])
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dg = upload_f32(ctx, g)
    var da = upload_f32(ctx, aux)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](total)
    launch[bn_bwd_red_at](ctx, fp(dx), fp(dg), fp(da), fp(da), ip(dp), ip(dp), C)
    if training:
        launch[bn_bwd_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(out), ip(dp), ip(dp), total)
    else:
        launch[bn_bwd_eval_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(out), ip(dp), ip(dp), total)
    var ry = download_f32(ctx, out, total)
    var ra = download_f32(ctx, da, len(aux))
    _ = dx^
    _ = dg^
    _ = da^
    _ = dp^
    _ = out^
    _ = ctx^
    ry.extend(ra^)
    return ry^


def dropout2d_device(x: List[Float32], prm: List[Int32], hyper: List[Float32]) raises -> List[Float32]:
    """[y | mask]."""
    var n = len(x)
    var ctx = DeviceContext()
    var dx = upload_f32(ctx, x)
    var dp = upload_i32(ctx, prm)
    var dh = upload_f32(ctx, hyper)
    var mask = ctx.enqueue_create_buffer[DType.float32](n)
    var out = ctx.enqueue_create_buffer[DType.float32](n)
    launch[dropout2d_at](ctx, fp(dx), fp(mask), fp(out), fp(dh), ip(dp), ip(dp), n)
    var ry = download_f32(ctx, out, n)
    var rm = download_f32(ctx, mask, n)
    _ = dx^
    _ = dp^
    _ = dh^
    _ = mask^
    _ = out^
    _ = ctx^
    ry.extend(rm^)
    return ry^


def mul_device(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_device[mul_at](a, b, len(a), prm)


def spmm_device(vals: List[Float32], h: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var total = Int(prm[0]) * Int(prm[1])
    var ctx = DeviceContext()
    var dv = upload_f32(ctx, vals)
    var dh = upload_f32(ctx, h)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](total)
    launch[spmm_at](ctx, fp(dv), fp(dh), fp(out), fp(out), ip(dq), ip(dp), total)
    var result = download_f32(ctx, out, total)
    _ = dv^
    _ = dh^
    _ = dq^
    _ = dp^
    _ = out^
    _ = ctx^
    return result^


def gcn_norm_device(w: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var n = Int(prm[0])
    var nnz = Int(prm[2])
    var ctx = DeviceContext()
    var dw = upload_f32(ctx, w)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var dis = ctx.enqueue_create_buffer[DType.float32](n)
    var vals = ctx.enqueue_create_buffer[DType.float32](nnz)
    launch[gcn_deg_at](ctx, fp(dw), fp(dis), fp(dis), fp(dis), ip(dq), ip(dp), n)
    launch[gcn_norm_at](ctx, fp(dw), fp(dis), fp(vals), fp(vals), ip(dq), ip(dp), nnz)
    var result = download_f32(ctx, vals, nnz)
    _ = dw^
    _ = dq^
    _ = dp^
    _ = dis^
    _ = vals^
    _ = ctx^
    return result^
