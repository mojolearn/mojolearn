# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE ON THE HOST: x_cnn/device.mojo's entries with the same
names and the same results, bit for bit. The element functions are
x_cnn/ops.mojo's, called in a loop; every contraction is `gemm_oracle`, the
CPU implementation of mojolearn.identical.gemm.fp32.v1."""
from std.sys.compile import is_defined
from gemm.host.identical_gemm import gemm_oracle, OP_NN, OP_NT, OP_TN
from x_cnn.ops import (
    FP, IP, ElemFn, CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, CP_REV,
    im2col_at, conv_out_at, dout_rows_at, col2im_at,
    PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, PP_REV,
    maxpool_fwd_at, maxpool_bwd_at, avgpool_fwd_at, avgpool_bwd_at,
    relu_fwd_at, relu_bwd_at, add_at, bias_rows_at, softmax_xent_row_at, seq_mean, sgd_at,
    bn_stats_at, bn_eval_stats_at, bn_apply_at, bn_running_at, bn_bwd_red_at, bn_bwd_dx_at, bn_bwd_eval_dx_at,
    dropout2d_at, mul_at, spmm_at, gcn_deg_at, gcn_norm_at,
    pad_fwd_at, pad_bwd_at, adapt_avg_fwd_at, adapt_avg_bwd_at, adapt_max_fwd_at, adapt_max_bwd_at,
    sage_max_fwd_at, sage_max_bwd_at, l2norm_fwd_at, l2norm_bwd_at, adam_at,
)

#: The host family's negative control (host_surface sabotage_define): the
#: host col2im gathers in reversed (kh) order.
comptime X_CNN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


@always_inline
def hp(mut values: List[Float32]) -> FP:
    return values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def hi(mut values: List[Int32]) -> IP:
    return values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def run[f: ElemFn](a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int):
    for i in range(total):
        f(i, a, b, c, d, q, p)


def zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n if n > 0 else 1, fill=Float32(0))


def gemm_host(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, op: Int) raises -> List[Float32]:
    return gemm_oracle(a, b, op, m, n, k)


def conv2d_forward_host(
    x: List[Float32], w: List[Float32], bias: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var xs = x.copy()
    var bs = bias.copy()
    var ps = prm.copy()
    var cols = zeros(rows * ckk)
    run[im2col_at](hp(xs), hp(cols), hp(cols), hp(cols), hi(ps), hi(ps), rows * ckk)
    var y2 = gemm_oracle(cols, w, OP_NT, rows, OC, ckk)
    var out = zeros(rows * OC)
    run[conv_out_at](hp(y2), hp(bs), hp(out), hp(out), hi(ps), hi(ps), rows * OC)
    _ = xs^
    _ = bs^
    _ = ps^
    _ = cols^
    _ = y2^
    return out^


def conv2d_backward_host(
    x: List[Float32], w: List[Float32], dout: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var nx = N * C * Int(prm[CP_H]) * Int(prm[CP_W])
    var xs = x.copy()
    var ds = dout.copy()
    var ps = prm.copy()
    comptime if X_CNN_HOST_SABOTAGE:
        ps[CP_REV] = Int32(1)
    var cols = zeros(rows * ckk)
    var g = zeros(rows * OC)
    run[im2col_at](hp(xs), hp(cols), hp(cols), hp(cols), hi(ps), hi(ps), rows * ckk)
    run[dout_rows_at](hp(ds), hp(g), hp(g), hp(g), hi(ps), hi(ps), rows * OC)
    var ones = List[Float32](length=rows, fill=Float32(1))
    var gw = gemm_oracle(g, cols, OP_TN, OC, ckk, rows)
    var gb = gemm_oracle(g, ones, OP_TN, OC, 1, rows)
    var dcols = gemm_oracle(g, w, OP_NN, rows, ckk, OC)
    var gx = zeros(nx)
    run[col2im_at](hp(dcols), hp(gx), hp(gx), hp(gx), hi(ps), hi(ps), nx)
    _ = xs^
    _ = ds^
    _ = ps^
    _ = cols^
    _ = g^
    _ = dcols^
    var result = List[Float32](capacity=nx + OC * ckk + OC)
    for i in range(nx):
        result.append(gx[i])
    result.extend(gw^)
    result.extend(gb^)
    return result^


def _pool_sizes(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def _host_prm(prm: List[Int32], rev_slot: Int) -> List[Int32]:
    var ps = prm.copy()
    comptime if X_CNN_HOST_SABOTAGE:
        ps[rev_slot] = Int32(1)
    return ps^


def maxpool2d_forward_host(x: List[Float32], prm: List[Int32], mut idx: List[Int32]) raises -> List[Float32]:
    var no = _pool_sizes(prm)[1]
    var xs = x.copy()
    var ps = prm.copy()
    var out = zeros(no)
    idx = List[Int32](length=no if no > 0 else 1, fill=Int32(0))
    run[maxpool_fwd_at](hp(xs), hp(out), hp(out), hp(out), hi(idx), hi(ps), no)
    _ = xs^
    _ = ps^
    return out^


def maxpool2d_backward_host(dout: List[Float32], idx: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var nx = _pool_sizes(prm)[0]
    var ds = dout.copy()
    var ix = idx.copy()
    var ps = _host_prm(prm, PP_REV)
    var gx = zeros(nx)
    run[maxpool_bwd_at](hp(ds), hp(gx), hp(gx), hp(gx), hi(ix), hi(ps), nx)
    _ = ds^
    _ = ix^
    _ = ps^
    return gx^


def avgpool2d_forward_host(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var no = _pool_sizes(prm)[1]
    var xs = x.copy()
    var ps = prm.copy()
    var out = zeros(no)
    run[avgpool_fwd_at](hp(xs), hp(out), hp(out), hp(out), hi(ps), hi(ps), no)
    _ = xs^
    _ = ps^
    return out^


def avgpool2d_backward_host(dout: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var nx = _pool_sizes(prm)[0]
    var ds = dout.copy()
    var ps = _host_prm(prm, PP_REV)
    var gx = zeros(nx)
    run[avgpool_bwd_at](hp(ds), hp(gx), hp(gx), hp(gx), hi(ps), hi(ps), nx)
    _ = ds^
    _ = ps^
    return gx^


def map2_host[f: ElemFn](a: List[Float32], b: List[Float32], n_out: Int, prm: List[Int32]) raises -> List[Float32]:
    var sa = a.copy()
    var sb = b.copy()
    var ps = prm.copy()
    var out = zeros(n_out)
    run[f](hp(sa), hp(sb), hp(out), hp(out), hi(ps), hi(ps), n_out)
    _ = sa^
    _ = sb^
    _ = ps^
    return out^


def relu_forward_host(x: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_host[relu_fwd_at](x, x, len(x), prm)


def relu_backward_host(x: List[Float32], g: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_host[relu_bwd_at](x, g, len(x), prm)


def add_host(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_host[add_at](a, b, len(a), prm)


def linear_forward_host(x: List[Float32], w: List[Float32], bias: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    var y = gemm_oracle(x, w, OP_NT, n, d_out, d_in)
    var prm: List[Int32] = [Int32(n), Int32(d_in), Int32(d_out)]
    return map2_host[bias_rows_at](y, bias, n * d_out, prm)


def linear_backward_host(x: List[Float32], w: List[Float32], g: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    var ones = List[Float32](length=n, fill=Float32(1))
    var gw = gemm_oracle(g, x, OP_TN, d_out, d_in, n)
    var gb = gemm_oracle(g, ones, OP_TN, d_out, 1, n)
    var gx = gemm_oracle(g, w, OP_NN, n, d_in, d_out)
    gx.extend(gw^)
    gx.extend(gb^)
    return gx^


def softmax_xent_host(logits: List[Float32], labels: List[Int32], n: Int, k: Int) raises -> List[Float32]:
    var sl = logits.copy()
    var sy = labels.copy()
    var prm: List[Int32] = [Int32(n), Int32(k)]
    var grad = zeros(n * k)
    var proba = zeros(n * k)
    var rl = zeros(n)
    run[softmax_xent_row_at](hp(sl), hp(grad), hp(proba), hp(rl), hi(sy), hi(prm), n)
    _ = sl^
    _ = sy^
    _ = prm^
    var loss = seq_mean(rl, n)
    grad.extend(proba^)
    grad.append(loss)
    return grad^


def sgd_host(w: List[Float32], g: List[Float32], v: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sv = v.copy()
    var sh = hyper.copy()
    var prm: List[Int32] = [Int32(n)]
    run[sgd_at](hp(sw), hp(sg), hp(sv), hp(sh), hi(prm), hi(prm), n)
    _ = sg^
    _ = sh^
    _ = prm^
    sw.extend(sv^)
    return sw^


def batchnorm_forward_host(
    x: List[Float32], running: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    var total = len(x)
    var C = Int(prm[1])
    var sx = x.copy()
    var sr = running.copy()
    var sa = aux.copy()
    var ps = prm.copy()
    var out = zeros(total)
    if training:
        run[bn_stats_at](hp(sx), hp(sa), hp(sa), hp(sa), hi(ps), hi(ps), C)
    else:
        run[bn_eval_stats_at](hp(sr), hp(sa), hp(sa), hp(sa), hi(ps), hi(ps), C)
    run[bn_apply_at](hp(sx), hp(sa), hp(out), hp(out), hi(ps), hi(ps), total)
    if training:
        run[bn_running_at](hp(sr), hp(sa), hp(sa), hp(sa), hi(ps), hi(ps), C)
    _ = sx^
    _ = ps^
    out.extend(sr^)
    out.extend(sa^)
    return out^


def batchnorm_backward_host(
    x: List[Float32], g: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    var total = len(x)
    var C = Int(prm[1])
    var sx = x.copy()
    var sg = g.copy()
    var sa = aux.copy()
    var ps = prm.copy()
    var out = zeros(total)
    run[bn_bwd_red_at](hp(sx), hp(sg), hp(sa), hp(sa), hi(ps), hi(ps), C)
    if training:
        run[bn_bwd_dx_at](hp(sx), hp(sg), hp(sa), hp(out), hi(ps), hi(ps), total)
    else:
        run[bn_bwd_eval_dx_at](hp(sx), hp(sg), hp(sa), hp(out), hi(ps), hi(ps), total)
    _ = sx^
    _ = sg^
    _ = ps^
    out.extend(sa^)
    return out^


def dropout2d_host(x: List[Float32], prm: List[Int32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(x)
    var sx = x.copy()
    var ps = prm.copy()
    var sh = hyper.copy()
    var mask = zeros(n)
    var out = zeros(n)
    run[dropout2d_at](hp(sx), hp(mask), hp(out), hp(sh), hi(ps), hi(ps), n)
    _ = sx^
    _ = ps^
    _ = sh^
    out.extend(mask^)
    return out^


def mul_host(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    var prm: List[Int32] = [0, 0, 0]
    return map2_host[mul_at](a, b, len(a), prm)


def spmm_host(vals: List[Float32], h: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var total = Int(prm[0]) * Int(prm[1])
    var sv = vals.copy()
    var sh = h.copy()
    var sq = csr.copy()
    var ps = prm.copy()
    var out = zeros(total)
    run[spmm_at](hp(sv), hp(sh), hp(out), hp(out), hi(sq), hi(ps), total)
    _ = sv^
    _ = sh^
    _ = sq^
    _ = ps^
    return out^


def gcn_norm_host(w: List[Float32], csr: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var n = Int(prm[0])
    var nnz = Int(prm[2])
    var sw = w.copy()
    var sq = csr.copy()
    var ps = prm.copy()
    var dis = zeros(n)
    var vals = zeros(nnz)
    run[gcn_deg_at](hp(sw), hp(dis), hp(dis), hp(dis), hi(sq), hi(ps), n)
    run[gcn_norm_at](hp(sw), hp(dis), hp(vals), hp(vals), hi(sq), hi(ps), nnz)
    _ = sw^
    _ = sq^
    _ = ps^
    _ = dis^
    return vals^


def pad2d_forward_host(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var nc = Int(prm[0]) * Int(prm[1])
    var n_out = nc * (Int(prm[2]) + Int(prm[4]) + Int(prm[5])) * (Int(prm[3]) + Int(prm[6]) + Int(prm[7]))
    return map2_host[pad_fwd_at](x, x, n_out, prm)


def pad2d_backward_host(g: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var n_out = Int(prm[0]) * Int(prm[1]) * Int(prm[2]) * Int(prm[3])
    return map2_host[pad_bwd_at](g, g, n_out, prm)


def adaptive_host[f: ElemFn](a: List[Float32], idx: List[Int32], n_out: Int, prm: List[Int32], mut idx_out: List[Int32]) raises -> List[Float32]:
    var sa = a.copy()
    var ps = prm.copy()
    idx_out = idx.copy()
    var out = zeros(n_out)
    run[f](hp(sa), hp(sa), hp(out), hp(out), hi(idx_out), hi(ps), n_out)
    _ = sa^
    _ = ps^
    return out^


def adaptive_pool_host(x: List[Float32], idx: List[Int32], prm: List[Int32], kind: Int, mut idx_out: List[Int32]) raises -> List[Float32]:
    """kind 0 avg forward, 1 avg backward, 2 max forward (idx_out = winners), 3 max backward (idx = winners)."""
    var nc = Int(prm[0]) * Int(prm[1])
    var nin = nc * Int(prm[2]) * Int(prm[3])
    var nout = nc * Int(prm[4]) * Int(prm[5])
    if kind == 0:
        return adaptive_host[adapt_avg_fwd_at](x, idx, nout, prm, idx_out)
    if kind == 1:
        return adaptive_host[adapt_avg_bwd_at](x, idx, nin, prm, idx_out)
    if kind == 2:
        var slots = List[Int32](length=nout, fill=Int32(0))
        return adaptive_host[adapt_max_fwd_at](x, slots, nout, prm, idx_out)
    return adaptive_host[adapt_max_bwd_at](x, idx, nin, prm, idx_out)


def graph4_host[f: ElemFn](a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], total: Int, n_out: Int) raises -> List[Float32]:
    var sa = a.copy()
    var sb = b.copy()
    var sx = aux.copy()
    var sq = csr.copy()
    var ps = prm.copy()
    var out = zeros(n_out)
    run[f](hp(sa), hp(sb), hp(sx), hp(out), hi(sq), hi(ps), total)
    _ = sa^
    _ = sb^
    _ = sq^
    _ = ps^
    out.extend(sx^)
    return out^


def graph_op_host(a: List[Float32], b: List[Float32], aux: List[Float32], csr: List[Int32], prm: List[Int32], kind: Int) raises -> List[Float32]:
    """kind 0 sage max forward, 1 sage max backward, 2 l2 normalize forward, 3 its backward; [dst | aux']."""
    var n = Int(prm[0])
    var F = Int(prm[1])
    if kind == 0:
        return graph4_host[sage_max_fwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 1:
        return graph4_host[sage_max_bwd_at](a, b, aux, csr, prm, n * F, n * F)
    if kind == 2:
        return graph4_host[l2norm_fwd_at](a, b, aux, csr, prm, n, n * F)
    return graph4_host[l2norm_bwd_at](a, b, aux, csr, prm, n, n * F)


def adam_host(w: List[Float32], g: List[Float32], mv: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sm = mv.copy()
    var sh = hyper.copy()
    var prm: List[Int32] = [Int32(n)]
    run[adam_at](hp(sw), hp(sg), hp(sm), hp(sh), hi(prm), hi(prm), n)
    _ = sg^
    _ = sh^
    _ = prm^
    sw.extend(sm^)
    return sw^
