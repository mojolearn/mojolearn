# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE ON THE DEVICE: x_cnn/ops.mojo's element functions, one
thread per element, and the pinned GEMM (`identical_gemm`, full FP32, never
the vendor route) for every contraction. The `*_into` entries take the
caller's host addresses in and out (the binding's path, DEVIATION 5716); the
List-returning forms wrap them for the seam check. Every entry runs on the
one process-lifetime context and synchronizes before it returns.

The host twin is x_cnn/host/ops_host.mojo: the same element functions in a
loop and `gemm_oracle` for the contractions."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
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
    pad_fwd_at, pad_bwd_at, adapt_avg_fwd_at, adapt_avg_bwd_at, adapt_max_fwd_at, adapt_max_bwd_at,
    sage_max_fwd_at, sage_max_bwd_at, l2norm_fwd_at, l2norm_bwd_at, adam_at,
)

comptime TPB = 256


struct _CnnContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_cnn entry. A context
    per call exhausted Metal's per-process command queues on the M2 Pro
    ("Failed to create Metal command queue for context") within one trainer
    fit (memory: METAL QUEUE LIMIT IS PER-PROCESS). Storage is
    `std.ffi._Global` (the pattern of the trees, RF and byte LM bindings),
    one slot per numeric tier so a FAST and an IDENTICAL .so in one process
    never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXCnnContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXCnnContextFast"
comptime X_CNN_CONTEXT = _Global[StorageType=_CnnContext, name=_CTX_NAME, init_fn=_CnnContext.__init__]


def cnn_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_CNN_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


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


# ------------------------------------------------------------------ host I/O
# DEVIATION 5716 (phase d, 2026-09-27): the binding hands the device entries
# the caller's HOST ADDRESSES and they copy straight between those and the
# device, once each way. Before, every array made five host copies around one
# transfer (read_f32 into a List, a copy inside upload_f32, the pinned
# staging buffer, an element-by-element append out of it, copy_f32 into the
# caller's array): measured on the RTX 4090 at N 256, 64x32x32, the
# forward's two transfers cost 30 ms against 2.9 ms of kernels. Copies only:
# no value is touched, so neither tier's bits move. Every `*_into` entry
# enqueues its uploads, kernels and downloads on the one in-order context and
# synchronizes ONCE before its buffers drop (`[[mojo-buffer-freed-at-last-use]]`:
# the trailing `_ = buf^` lines keep them alive past that wait). The caller's
# arrays stay alive for the whole call (the binding holds them).


def up(ctx: DeviceContext, src: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device copy of `n` floats at host address `src` (no staging copy)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
    return buf^


def up_i(ctx: DeviceContext, src: IP, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
    return buf^


def down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], dst: FP, n: Int) raises:
    """Enqueue the first `n` floats of `buf` into host address `dst`; the
    caller synchronizes (a sub-buffer view synchronizes here, since the view
    drops at return)."""
    if n <= 0:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
        ctx.synchronize()


def down_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], dst: IP, n: Int) raises:
    if n <= 0:
        return
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.int32](0, n)
        ctx.enqueue_copy(dst_ptr=dst, src_buf=view)
        ctx.synchronize()


@always_inline
def lp(mut l: List[Float32]) -> FP:
    """A List's storage as an entry's host address (the List must outlive the call)."""
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def lpi(mut l: List[Int32]) -> IP:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


# ------------------------------------------------------------------ entries


def gemm_into(a: FP, b: FP, c: FP, m: Int, n: Int, k: Int, op: Int) raises:
    var ctx = cnn_ctx()
    var da = up(ctx, a, m * k)
    var db = up(ctx, b, n * k)
    var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
    device_gemm(ctx, dc, da, db, m, n, k, op)
    down(ctx, dc, c, m * n)
    ctx.synchronize()
    _ = da^
    _ = db^
    _ = dc^
    _ = ctx^


def gemm_device(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, op: Int) raises -> List[Float32]:
    var sa = a.copy()
    var sb = b.copy()
    var out = List[Float32](length=m * n, fill=Float32(0))
    gemm_into(lp(sa), lp(sb), lp(out), m, n, k, op)
    _ = sa^
    _ = sb^
    return out^


def conv2d_forward_into(x: FP, w: FP, bias: FP, prm: List[Int32], dst: FP) raises:
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ctx = cnn_ctx()
    var dx = up(ctx, x, N * C * Int(prm[CP_H]) * Int(prm[CP_W]))
    var dw = up(ctx, w, OC * ckk)
    var dbias = up(ctx, bias, OC)
    var dp = upload_i32(ctx, prm)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var y2 = ctx.enqueue_create_buffer[DType.float32](rows * OC)
    var dout = ctx.enqueue_create_buffer[DType.float32](rows * OC)
    launch[im2col_at](ctx, fp(dx), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * ckk)
    device_gemm(ctx, y2, cols, dw, rows, OC, ckk, OP_NT)
    launch[conv_out_at](ctx, fp(y2), fp(dbias), fp(dout), fp(dout), ip(dp), ip(dp), rows * OC)
    down(ctx, dout, dst, rows * OC)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = dout^
    _ = ctx^


def conv2d_backward_into(x: FP, w: FP, dout: FP, prm: List[Int32], gx_out: FP, gw_out: FP, gb_out: FP) raises:
    """dx (N*C*H*W), dW (OC*C*KH*KW), db (OC)."""
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var ctx = cnn_ctx()
    var dxin = up(ctx, x, nx)
    var dw = up(ctx, w, OC * ckk)
    var ddout = up(ctx, dout, rows * OC)
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
    down(ctx, gx, gx_out, nx)
    down(ctx, gw, gw_out, OC * ckk)
    down(ctx, gb, gb_out, OC)
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


def _pool_sizes(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def maxpool2d_forward_into(x: FP, prm: List[Int32], dst: FP, idx_out: IP) raises:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = cnn_ctx()
    var dx = up(ctx, x, sizes[0])
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](no)
    var di = ctx.enqueue_create_buffer[DType.int32](no)
    launch[maxpool_fwd_at](ctx, fp(dx), fp(dout), fp(dout), fp(dout), ip(di), ip(dp), no)
    down(ctx, dout, dst, no)
    down_i(ctx, di, idx_out, no)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dout^
    _ = di^
    _ = ctx^


def maxpool2d_forward_device(x: List[Float32], prm: List[Int32], mut idx: List[Int32]) raises -> List[Float32]:
    var no = _pool_sizes(prm)[1]
    var sx = x.copy()
    var out = List[Float32](length=no, fill=Float32(0))
    idx = List[Int32](length=no, fill=Int32(0))
    maxpool2d_forward_into(lp(sx), prm, lp(out), lpi(idx))
    _ = sx^
    return out^


def maxpool2d_backward_into(dout: FP, idx: IP, prm: List[Int32], gx_out: FP) raises:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = cnn_ctx()
    var dd = up(ctx, dout, sizes[1])
    var di = up_i(ctx, idx, sizes[1])
    var dp = upload_i32(ctx, prm)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    launch[maxpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(di), ip(dp), nx)
    down(ctx, gx, gx_out, nx)
    ctx.synchronize()
    _ = dd^
    _ = di^
    _ = dp^
    _ = gx^
    _ = ctx^


def avgpool2d_forward_into(x: FP, prm: List[Int32], dst: FP) raises:
    var sizes = _pool_sizes(prm)
    var no = sizes[1]
    var ctx = cnn_ctx()
    var dx = up(ctx, x, sizes[0])
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](no)
    launch[avgpool_fwd_at](ctx, fp(dx), fp(dout), fp(dout), fp(dout), ip(dp), ip(dp), no)
    down(ctx, dout, dst, no)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dout^
    _ = ctx^


def avgpool2d_forward_device(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sx = x.copy()
    var out = List[Float32](length=_pool_sizes(prm)[1], fill=Float32(0))
    avgpool2d_forward_into(lp(sx), prm, lp(out))
    _ = sx^
    return out^


def avgpool2d_backward_into(dout: FP, prm: List[Int32], gx_out: FP) raises:
    var sizes = _pool_sizes(prm)
    var nx = sizes[0]
    var ctx = cnn_ctx()
    var dd = up(ctx, dout, sizes[1])
    var dp = upload_i32(ctx, prm)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx)
    launch[avgpool_bwd_at](ctx, fp(dd), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
    down(ctx, gx, gx_out, nx)
    ctx.synchronize()
    _ = dd^
    _ = dp^
    _ = gx^
    _ = ctx^


def map2_into[f: ElemFn](a: FP, na: Int, b: FP, nb: Int, n_out: Int, prm: List[Int32], dst: FP) raises:
    """dst[i] = f(a, b) for i < n_out (slots a, b, dst); `nb == 0` reuses a."""
    var ctx = cnn_ctx()
    var da = up(ctx, a, na)
    var db = up(ctx, b, nb) if nb > 0 else ctx.enqueue_create_buffer[DType.float32](1)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_out if n_out > 0 else 1)
    if nb > 0:
        launch[f](ctx, fp(da), fp(db), fp(dout), fp(dout), ip(dp), ip(dp), n_out)
    else:
        launch[f](ctx, fp(da), fp(da), fp(dout), fp(dout), ip(dp), ip(dp), n_out)
    down(ctx, dout, dst, n_out)
    ctx.synchronize()
    _ = da^
    _ = db^
    _ = dp^
    _ = dout^
    _ = ctx^


def relu_forward_into(x: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[relu_fwd_at](x, n, x, 0, n, prm, dst)


def relu_backward_into(x: FP, g: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[relu_bwd_at](x, n, g, n, n, prm, dst)


def add_into(a: FP, b: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[add_at](a, n, b, n, n, prm, dst)


def mul_into(a: FP, b: FP, n: Int, dst: FP) raises:
    var prm: List[Int32] = [0, 0, 0]
    map2_into[mul_at](a, n, b, n, n, prm, dst)


def linear_forward_into(x: FP, w: FP, bias: FP, n: Int, d_in: Int, d_out: Int, dst: FP) raises:
    """y = x W^T + b: the pinned GEMM NT, then one add per element."""
    var ctx = cnn_ctx()
    var dx = up(ctx, x, n * d_in)
    var dw = up(ctx, w, d_out * d_in)
    var db = up(ctx, bias, d_out)
    var prm: List[Int32] = [Int32(n), Int32(d_in), Int32(d_out)]
    var dp = upload_i32(ctx, prm)
    var y = ctx.enqueue_create_buffer[DType.float32](n * d_out)
    var dout = ctx.enqueue_create_buffer[DType.float32](n * d_out)
    device_gemm(ctx, y, dx, dw, n, d_out, d_in, OP_NT)
    launch[bias_rows_at](ctx, fp(y), fp(db), fp(dout), fp(dout), ip(dp), ip(dp), n * d_out)
    down(ctx, dout, dst, n * d_out)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = db^
    _ = dp^
    _ = y^
    _ = dout^
    _ = ctx^


def linear_backward_into(x: FP, w: FP, g: FP, n: Int, d_in: Int, d_out: Int, gx_out: FP, gw_out: FP, gb_out: FP) raises:
    """dW = G^T X and db = G^T 1 (GEMM TN over the rows, the pinned fold),
    dx = G W (GEMM NN)."""
    var ctx = cnn_ctx()
    var dx = up(ctx, x, n * d_in)
    var dw = up(ctx, w, d_out * d_in)
    var dg = up(ctx, g, n * d_out)
    var dones = ctx.enqueue_create_buffer[DType.float32](n)
    dones.enqueue_fill(Float32(1))
    var gx = ctx.enqueue_create_buffer[DType.float32](n * d_in)
    var gw = ctx.enqueue_create_buffer[DType.float32](d_out * d_in)
    var gb = ctx.enqueue_create_buffer[DType.float32](d_out)
    device_gemm(ctx, gw, dg, dx, d_out, d_in, n, OP_TN)
    device_gemm(ctx, gb, dg, dones, d_out, 1, n, OP_TN)
    device_gemm(ctx, gx, dg, dw, n, d_in, d_out, OP_NN)
    down(ctx, gx, gx_out, n * d_in)
    down(ctx, gw, gw_out, d_out * d_in)
    down(ctx, gb, gb_out, d_out)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dg^
    _ = dones^
    _ = gx^
    _ = gw^
    _ = gb^
    _ = ctx^


def softmax_xent_into(logits: FP, labels: IP, n: Int, k: Int, grad_out: FP, proba_out: FP) raises -> Float32:
    """grad and proba (n*k each) into the caller's arrays; returns the mean loss."""
    var ctx = cnn_ctx()
    var dl = up(ctx, logits, n * k)
    var dy = up_i(ctx, labels, n)
    var prm: List[Int32] = [Int32(n), Int32(k)]
    var dp = upload_i32(ctx, prm)
    var grad = ctx.enqueue_create_buffer[DType.float32](n * k)
    var proba = ctx.enqueue_create_buffer[DType.float32](n * k)
    var rl = ctx.enqueue_create_buffer[DType.float32](n)
    grad.enqueue_fill(Float32(0))
    rl.enqueue_fill(Float32(0))
    launch[softmax_xent_row_at](ctx, fp(dl), fp(grad), fp(proba), fp(rl), ip(dy), ip(dp), n)
    var rows = List[Float32](length=n, fill=Float32(0))
    down(ctx, grad, grad_out, n * k)
    down(ctx, proba, proba_out, n * k)
    down(ctx, rl, lp(rows), n)
    ctx.synchronize()
    _ = dl^
    _ = dy^
    _ = dp^
    _ = grad^
    _ = proba^
    _ = rl^
    _ = ctx^
    return seq_mean(rows, n)


def softmax_xent_device(logits: List[Float32], labels: List[Int32], n: Int, k: Int) raises -> List[Float32]:
    """[grad (n*k) | proba (n*k) | mean loss (1)] (the List form, for the seam check)."""
    var sl = logits.copy()
    var sy = labels.copy()
    var out = List[Float32](length=2 * n * k + 1, fill=Float32(0))
    var base = lp(out)
    var loss = softmax_xent_into(lp(sl), lpi(sy), n, k, base, base + n * k)
    out[2 * n * k] = loss
    _ = sl^
    _ = sy^
    return out^


def sgd_into(w: FP, g: FP, v: FP, hyper: List[Float32], n: Int) raises:
    """In place: w and the momentum buffer v."""
    var ctx = cnn_ctx()
    var dw = up(ctx, w, n)
    var dg = up(ctx, g, n)
    var dv = up(ctx, v, n)
    var dh = upload_f32(ctx, hyper)
    var prm: List[Int32] = [Int32(n)]
    var dp = upload_i32(ctx, prm)
    launch[sgd_at](ctx, fp(dw), fp(dg), fp(dv), fp(dh), ip(dp), ip(dp), n)
    down(ctx, dw, w, n)
    down(ctx, dv, v, n)
    ctx.synchronize()
    _ = dw^
    _ = dg^
    _ = dv^
    _ = dh^
    _ = dp^
    _ = ctx^


def sgd_device(w: List[Float32], g: List[Float32], v: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    """[w' | v'] (the List form, for the seam check)."""
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sv = v.copy()
    sgd_into(lp(sw), lp(sg), lp(sv), hyper, n)
    sw.extend(sv^)
    _ = sg^
    return sw^


def batchnorm_forward_into(x: FP, running: FP, aux: FP, prm: List[Int32], training: Bool, y_out: FP) raises:
    """y into y_out; running (2C) and aux (2 + 7C, the statistics the backward reads) in place."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var nr = 2 * C
    var na = 2 + 7 * C
    var ctx = cnn_ctx()
    var dx = up(ctx, x, total)
    var dr = up(ctx, running, nr)
    var da = up(ctx, aux, na)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    if training:
        launch[bn_stats_at](ctx, fp(dx), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    else:
        launch[bn_eval_stats_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    launch[bn_apply_at](ctx, fp(dx), fp(da), fp(dout), fp(dout), ip(dp), ip(dp), total)
    if training:
        launch[bn_running_at](ctx, fp(dr), fp(da), fp(da), fp(da), ip(dp), ip(dp), C)
    down(ctx, dout, y_out, total)
    down(ctx, dr, running, nr)
    down(ctx, da, aux, na)
    ctx.synchronize()
    _ = dx^
    _ = dr^
    _ = da^
    _ = dp^
    _ = dout^
    _ = ctx^


def batchnorm_forward_device(
    x: List[Float32], running: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    """[y | running' | aux'] (the List form, for the seam check)."""
    var sx = x.copy()
    var sr = running.copy()
    var sa = aux.copy()
    var out = List[Float32](length=len(x), fill=Float32(0))
    batchnorm_forward_into(lp(sx), lp(sr), lp(sa), prm, training, lp(out))
    out.extend(sr^)
    out.extend(sa^)
    _ = sx^
    return out^


def batchnorm_backward_into(x: FP, g: FP, aux: FP, prm: List[Int32], training: Bool, dx_out: FP) raises:
    """dx into dx_out; aux in place (sum_g = dbeta and sum_gx = dgamma)."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var na = 2 + 7 * C
    var ctx = cnn_ctx()
    var dx = up(ctx, x, total)
    var dg = up(ctx, g, total)
    var da = up(ctx, aux, na)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    launch[bn_bwd_red_at](ctx, fp(dx), fp(dg), fp(da), fp(da), ip(dp), ip(dp), C)
    if training:
        launch[bn_bwd_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(dout), ip(dp), ip(dp), total)
    else:
        launch[bn_bwd_eval_dx_at](ctx, fp(dx), fp(dg), fp(da), fp(dout), ip(dp), ip(dp), total)
    down(ctx, dout, dx_out, total)
    down(ctx, da, aux, na)
    ctx.synchronize()
    _ = dx^
    _ = dg^
    _ = da^
    _ = dp^
    _ = dout^
    _ = ctx^


def dropout2d_into(x: FP, n: Int, prm: List[Int32], hyper: List[Float32], y_out: FP, mask_out: FP) raises:
    var ctx = cnn_ctx()
    var dx = up(ctx, x, n)
    var dp = upload_i32(ctx, prm)
    var dh = upload_f32(ctx, hyper)
    var mask = ctx.enqueue_create_buffer[DType.float32](n)
    var dout = ctx.enqueue_create_buffer[DType.float32](n)
    launch[dropout2d_at](ctx, fp(dx), fp(mask), fp(dout), fp(dh), ip(dp), ip(dp), n)
    down(ctx, dout, y_out, n)
    down(ctx, mask, mask_out, n)
    ctx.synchronize()
    _ = dx^
    _ = dp^
    _ = dh^
    _ = mask^
    _ = dout^
    _ = ctx^


def dropout2d_device(x: List[Float32], prm: List[Int32], hyper: List[Float32]) raises -> List[Float32]:
    """[y | mask] (the List form, for the seam check)."""
    var n = len(x)
    var sx = x.copy()
    var out = List[Float32](length=2 * n, fill=Float32(0))
    var base = lp(out)
    dropout2d_into(lp(sx), n, prm, hyper, base, base + n)
    _ = sx^
    return out^


def spmm_into(vals: FP, nvals: Int, h: FP, csr: List[Int32], prm: List[Int32], dst: FP) raises:
    var total = Int(prm[0]) * Int(prm[1])
    var ctx = cnn_ctx()
    var dv = up(ctx, vals, nvals)
    var dh = up(ctx, h, total)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var dout = ctx.enqueue_create_buffer[DType.float32](total)
    launch[spmm_at](ctx, fp(dv), fp(dh), fp(dout), fp(dout), ip(dq), ip(dp), total)
    down(ctx, dout, dst, total)
    ctx.synchronize()
    _ = dv^
    _ = dh^
    _ = dq^
    _ = dp^
    _ = dout^
    _ = ctx^


def spmm_device(vals: List[Float32], h: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var sv = vals.copy()
    var sh = h.copy()
    var out = List[Float32](length=Int(prm[0]) * Int(prm[1]), fill=Float32(0))
    spmm_into(lp(sv), len(sv), lp(sh), csr, prm, lp(out))
    _ = sv^
    _ = sh^
    return out^


def gcn_norm_device(w: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var n = Int(prm[0])
    var nnz = Int(prm[2])
    var ctx = cnn_ctx()
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


def pad2d_forward_into(x: FP, prm: List[Int32], dst: FP) raises:
    var nc = Int(prm[0]) * Int(prm[1])
    var nx = nc * Int(prm[2]) * Int(prm[3])
    var n_out = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    map2_into[pad_fwd_at](x, nx, x, 0, n_out, prm, dst)


def pad2d_backward_into(g: FP, prm: List[Int32], dst: FP) raises:
    var nc = Int(prm[0]) * Int(prm[1])
    var ng = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    var n_out = nc * Int(prm[2]) * Int(prm[3])
    map2_into[pad_bwd_at](g, ng, g, 0, n_out, prm, dst)


def pad2d_backward_device(g: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var sg = g.copy()
    var out = List[Float32](length=Int(prm[0]) * Int(prm[1]) * Int(prm[2]) * Int(prm[3]), fill=Float32(0))
    pad2d_backward_into(lp(sg), prm, lp(out))
    _ = sg^
    return out^


def adaptive_device[f: ElemFn](a: List[Float32], idx: List[Int32], n_out: Int, prm: List[Int32], mut idx_out: List[Int32]) raises -> List[Float32]:
    """One adaptive-pool element function over n_out outputs; `idx` in, `idx_out` out (max pooling)."""
    var ctx = cnn_ctx()
    var da = upload_f32(ctx, a)
    var di = upload_i32(ctx, idx)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    launch[f](ctx, fp(da), fp(da), fp(out), fp(out), ip(di), ip(dp), n_out)
    var result = download_f32(ctx, out, n_out)
    idx_out = download_i32(ctx, di, len(idx))
    _ = da^
    _ = di^
    _ = dp^
    _ = out^
    _ = ctx^
    return result^


def adaptive_pool_device(x: List[Float32], idx: List[Int32], prm: List[Int32], kind: Int, mut idx_out: List[Int32]) raises -> List[Float32]:
    """kind 0 avg forward, 1 avg backward, 2 max forward (idx_out = winners), 3 max backward (idx = winners)."""
    var nc = Int(prm[0]) * Int(prm[1])
    var nin = nc * Int(prm[2]) * Int(prm[3])
    var nout = nc * Int(prm[4]) * Int(prm[5])
    if kind == 0:
        return adaptive_device[adapt_avg_fwd_at](x, idx, nout, prm, idx_out)
    if kind == 1:
        return adaptive_device[adapt_avg_bwd_at](x, idx, nin, prm, idx_out)
    if kind == 2:
        var slots = List[Int32](length=nout, fill=Int32(0))
        return adaptive_device[adapt_max_fwd_at](x, slots, nout, prm, idx_out)
    return adaptive_device[adapt_max_bwd_at](x, idx, nin, prm, idx_out)


def graph4_device[f: ElemFn](a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], total: Int, n_out: Int) raises -> List[Float32]:
    """[dst (n_out) | aux'] of one element function over `total` items (slots a, b, aux, dst)."""
    var ctx = cnn_ctx()
    var da = upload_f32(ctx, a)
    var db = upload_f32(ctx, b)
    var dx = upload_f32(ctx, aux)
    var dq = upload_i32(ctx, csr)
    var dp = upload_i32(ctx, prm)
    var out = ctx.enqueue_create_buffer[DType.float32](n_out)
    launch[f](ctx, fp(da), fp(db), fp(dx), fp(out), ip(dq), ip(dp), total)
    var result = download_f32(ctx, out, n_out)
    var raux = download_f32(ctx, dx, len(aux))
    _ = da^
    _ = db^
    _ = dx^
    _ = dq^
    _ = dp^
    _ = out^
    _ = ctx^
    result.extend(raux^)
    return result^


def graph_op_device(a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], kind: Int) raises -> List[Float32]:
    """kind 0 sage max forward, 1 sage max backward, 2 l2 normalize forward, 3 its backward; [dst | aux']."""
    var n = Int(prm[0])
    var F = Int(prm[1])
    if kind == 0:
        return graph4_device[sage_max_fwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 1:
        return graph4_device[sage_max_bwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 2:
        return graph4_device[l2norm_fwd_at](a, b, aux, csr, prm, n, n * F)
    return graph4_device[l2norm_bwd_at](a, b, aux, csr, prm, n, n * F)


def adam_into(w: FP, g: FP, mv: FP, hyper: List[Float32], n: Int) raises:
    """In place: w and mv = [m (n) | v (n)]."""
    var ctx = cnn_ctx()
    var dw = up(ctx, w, n)
    var dg = up(ctx, g, n)
    var dm = up(ctx, mv, 2 * n)
    var dh = upload_f32(ctx, hyper)
    var prm: List[Int32] = [Int32(n)]
    var dp = upload_i32(ctx, prm)
    launch[adam_at](ctx, fp(dw), fp(dg), fp(dm), fp(dh), ip(dp), ip(dp), n)
    down(ctx, dw, w, n)
    down(ctx, dm, mv, 2 * n)
    ctx.synchronize()
    _ = dw^
    _ = dg^
    _ = dm^
    _ = dh^
    _ = dp^
    _ = ctx^


def adam_device(w: List[Float32], g: List[Float32], mv: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    """[w' | mv'] (the List form, for the seam check)."""
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sm = mv.copy()
    adam_into(lp(sw), lp(sg), lp(sm), hyper, n)
    sw.extend(sm^)
    _ = sg^
    return sw^


# ------------------------------------------------------------ the conv block
# DEVIATION 5717 (phase d, 2026-09-27): CNNClassifier's block, Conv2d ->
# ReLU -> MaxPool2d, in ONE entry each way, so the activations between the
# three stay on the device. The same element functions and the same GEMMs,
# launched in the same order on the same values as the three separate layer
# entries: neither tier's bits move. The backward RECOMPUTES the conv output
# (im2col, GEMM NT, conv_out) from the input it uploads anyway for the
# weight gradient, instead of carrying it through the host; the recompute is
# the forward's own kernels on the forward's own inputs, so its bits are the
# forward's. `pool` False is Conv2d -> ReLU (the map is smaller than the
# window). `need_dx` False skips col2im and the NN GEMM (the first block's
# input gradient, which the trainer never reads).


def _conv_relu_on_device(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut dw: DeviceBuffer[DType.float32],
    mut dbias: DeviceBuffer[DType.float32], mut dp: DeviceBuffer[DType.int32], mut cols: DeviceBuffer[DType.float32],
    mut y2: DeviceBuffer[DType.float32], mut yconv: DeviceBuffer[DType.float32], rows: Int, OC: Int, ckk: Int,
) raises:
    """cols = im2col(x); y2 = cols . W^T; yconv = the NCHW conv output (+ bias)."""
    launch[im2col_at](ctx, fp(dx), fp(cols), fp(cols), fp(cols), ip(dp), ip(dp), rows * ckk)
    device_gemm(ctx, y2, cols, dw, rows, OC, ckk, OP_NT)
    launch[conv_out_at](ctx, fp(y2), fp(dbias), fp(yconv), fp(yconv), ip(dp), ip(dp), rows * OC)


def conv_block_forward_into(
    x: FP, w: FP, bias: FP, cprm: List[Int32], pprm: List[Int32], pool: Bool, dst: FP, idx_out: IP
) raises:
    """dst = maxpool(relu(conv(x))) (idx_out its winners), or relu(conv(x)) when `pool` is False."""
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = N * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ny = rows * OC
    var ctx = cnn_ctx()
    var dx = up(ctx, x, N * C * Int(cprm[CP_H]) * Int(cprm[CP_W]))
    var dw = up(ctx, w, OC * ckk)
    var dbias = up(ctx, bias, OC)
    var dp = upload_i32(ctx, cprm)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var y2 = ctx.enqueue_create_buffer[DType.float32](ny)
    var yconv = ctx.enqueue_create_buffer[DType.float32](ny)
    var r = ctx.enqueue_create_buffer[DType.float32](ny)
    _conv_relu_on_device(ctx, dx, dw, dbias, dp, cols, y2, yconv, rows, OC, ckk)
    launch[relu_fwd_at](ctx, fp(yconv), fp(yconv), fp(r), fp(r), ip(dp), ip(dp), ny)
    var no = _pool_sizes(pprm)[1] if pool else 1
    var dpp = upload_i32(ctx, pprm) if pool else upload_i32(ctx, cprm)
    var pout = ctx.enqueue_create_buffer[DType.float32](no)
    var di = ctx.enqueue_create_buffer[DType.int32](no)
    if pool:
        launch[maxpool_fwd_at](ctx, fp(r), fp(pout), fp(pout), fp(pout), ip(di), ip(dpp), no)
        down(ctx, pout, dst, no)
        down_i(ctx, di, idx_out, no)
    else:
        down(ctx, r, dst, ny)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dp^
    _ = cols^
    _ = y2^
    _ = yconv^
    _ = r^
    _ = dpp^
    _ = pout^
    _ = di^
    _ = ctx^


def conv_block_backward_into(
    x: FP, w: FP, bias: FP, g: FP, idx: IP, cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    gx_out: FP, gw_out: FP, gb_out: FP,
) raises:
    """From the gradient of the block's output `g`: dx (when `need_dx`), dW, db."""
    var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
    var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = N * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ny = rows * OC
    var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
    var no = _pool_sizes(pprm)[1] if pool else ny
    var ctx = cnn_ctx()
    var dx = up(ctx, x, nx)
    var dw = up(ctx, w, OC * ckk)
    var dbias = up(ctx, bias, OC)
    var dgo = up(ctx, g, no)
    var di = up_i(ctx, idx, no if pool else 0)
    var dp = upload_i32(ctx, cprm)
    var dpp = upload_i32(ctx, pprm) if pool else upload_i32(ctx, cprm)
    var cols = ctx.enqueue_create_buffer[DType.float32](rows * ckk)
    var y2 = ctx.enqueue_create_buffer[DType.float32](ny)
    var yconv = ctx.enqueue_create_buffer[DType.float32](ny)
    var gr = ctx.enqueue_create_buffer[DType.float32](ny)
    var gy = ctx.enqueue_create_buffer[DType.float32](ny)
    _conv_relu_on_device(ctx, dx, dw, dbias, dp, cols, y2, yconv, rows, OC, ckk)
    if pool:
        launch[maxpool_bwd_at](ctx, fp(dgo), fp(gr), fp(gr), fp(gr), ip(di), ip(dpp), ny)
        launch[relu_bwd_at](ctx, fp(yconv), fp(gr), fp(gy), fp(gy), ip(dp), ip(dp), ny)
    else:
        launch[relu_bwd_at](ctx, fp(yconv), fp(dgo), fp(gy), fp(gy), ip(dp), ip(dp), ny)
    # conv2d_backward_into from here, on the device-resident gy and cols
    var grow = ctx.enqueue_create_buffer[DType.float32](ny)
    var ones = ctx.enqueue_create_buffer[DType.float32](rows)
    var gw = ctx.enqueue_create_buffer[DType.float32](OC * ckk)
    var gb = ctx.enqueue_create_buffer[DType.float32](OC)
    launch[dout_rows_at](ctx, fp(gy), fp(grow), fp(grow), fp(grow), ip(dp), ip(dp), ny)
    launch[fill_one_at](ctx, fp(ones), fp(ones), fp(ones), fp(ones), ip(dp), ip(dp), rows)
    # DEVIATION 5701: the pinned GEMM's fold over the rows, never an atomic.
    device_gemm(ctx, gw, grow, cols, OC, ckk, rows, OP_TN)
    device_gemm(ctx, gb, grow, ones, OC, 1, rows, OP_TN)
    var dcols = ctx.enqueue_create_buffer[DType.float32](rows * ckk if need_dx else 1)
    var gx = ctx.enqueue_create_buffer[DType.float32](nx if need_dx else 1)
    if need_dx:
        device_gemm(ctx, dcols, grow, dw, rows, ckk, OC, OP_NN)
        launch[col2im_at](ctx, fp(dcols), fp(gx), fp(gx), fp(gx), ip(dp), ip(dp), nx)
        down(ctx, gx, gx_out, nx)
    down(ctx, gw, gw_out, OC * ckk)
    down(ctx, gb, gb_out, OC)
    ctx.synchronize()
    _ = dx^
    _ = dw^
    _ = dbias^
    _ = dgo^
    _ = di^
    _ = dp^
    _ = dpp^
    _ = cols^
    _ = y2^
    _ = yconv^
    _ = gr^
    _ = gy^
    _ = grow^
    _ = ones^
    _ = gw^
    _ = gb^
    _ = dcols^
    _ = gx^
    _ = ctx^
