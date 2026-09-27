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
