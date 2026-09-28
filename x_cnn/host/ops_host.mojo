# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE ON THE HOST: x_cnn/device.mojo's entries with the same
names and the same results, bit for bit. The element functions are
x_cnn/ops.mojo's, called in a loop; every contraction is
mojolearn.identical.gemm.fp32.v1, computed by `x_cnn/host/gemm_host.mojo`
(`gemm_oracle`'s cells, bit for bit, threaded and vectorized).

PHASE 5 (CPU speed, DEVIATION 5719). `run` splits an element loop into
contiguous tasks (`core/host_predict_threads.mojo`, MOJOLEARN_CPU_THREADS):
every element function computes ONE output element from its inputs, the
contract the device relies on when it launches one thread per element, so
which thread runs an element moves no bit. The `*_into` entries take the
caller's addresses (inputs read in place, outputs written in place) and
allocate their scratch uninitialized; the List entries below are their
doors for the seam check."""
from std.memory import alloc
from std.sys.compile import is_defined
from core.host_parallel import host_parallelize
from gemm.host.identical_gemm import OP_NN, OP_NT, OP_TN
from x_cnn.host.gemm_host import gemm_host_into, parallel_tasks
from checks.numerics import ftz
from x_cnn.ops import (
    canon,
    FP, IP, ElemFn, CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, CP_REV,
    CP_SH, CP_SW, CP_PH, CP_PW, CP_DH, CP_DW, CP_BIAS,
    im2col_at, conv_out_at, dout_rows_at, col2im_at,
    PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW, PP_REV, PP_KH, PP_KW, PP_SH, PP_SW, PP_PH, PP_PW, PP_DH, PP_DW,
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


#: The fewest elements worth a second thread in an element loop.
comptime RUN_MIN_ELEMS = 16384


def run[f: ElemFn](a: FP, b: FP, c: FP, d: FP, q: IP, p: IP, total: Int):
    """f over elements [0, total), split into contiguous tasks."""
    var tasks = parallel_tasks(total, RUN_MIN_ELEMS)
    if tasks <= 1:
        for i in range(total):
            f(i, a, b, c, d, q, p)
        return
    var chunk = (total + tasks - 1) // tasks

    def _part(t: Int) {imm a, imm b, imm c, imm d, imm q, imm p, imm chunk, imm total}:
        var lo = t * chunk
        var hi = min(lo + chunk, total)
        for i in range(lo, hi):
            f(i, a, b, c, d, q, p)

    host_parallelize(_part, tasks)


def zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n if n > 0 else 1, fill=Float32(0))


@always_inline
def scratch(n: Int) -> FP:
    """n uninitialized words (at least one); the caller frees it."""
    return alloc[Float32](n if n > 0 else 1).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def scratch_i(n: Int) -> IP:
    return alloc[Int32](n if n > 0 else 1).unsafe_origin_cast[MutAnyOrigin]()


# ---------------------------------------------------------------- conv layout loops
# The element functions decode every output index with four or five integer
# divisions (x_cnn/ops.mojo im2col_at, conv_out_at, dout_rows_at, col2im_at):
# on the host that decoding cost several times the GEMM. These loops walk the
# same outputs with the indices advanced incrementally and store the SAME
# values: im2col and dout_rows are copies through `ftz`, conv_out is
# `canon(ftz(ftz(y) + ftz(bias)))`, col2im gathers each pixel's taps in
# (kh, kw) ascending order from +0.0 with `acc = ftz(acc + ftz(v))`
# (DEVIATION 5700; CP_REV reverses kh for the host sabotage build). A task
# owns whole rows or planes, so a thread count changes no value.


def _tasks_for(units: Int, per_unit: Int) -> Int:
    var t = parallel_tasks(units * per_unit, RUN_MIN_ELEMS)
    return t if t < units else (units if units > 0 else 1)


def _im2col_rows(x: FP, cols: FP, p: IP, lo: Int, hi: Int):
    """cols rows [lo, hi): im2col_at's words."""
    var C = Int(p[CP_C]); var H = Int(p[CP_H]); var W = Int(p[CP_W])
    var KH = Int(p[CP_KH]); var KW = Int(p[CP_KW])
    var OH = Int(p[CP_OH]); var OW = Int(p[CP_OW])
    var SH = Int(p[CP_SH]); var SW = Int(p[CP_SW])
    var PH = Int(p[CP_PH]); var PW = Int(p[CP_PW])
    var DH = Int(p[CP_DH]); var DW = Int(p[CP_DW])
    var ckk = C * KH * KW
    var n = lo // (OH * OW)
    var rem = lo - n * OH * OW
    var oh = rem // OW
    var ow = rem - oh * OW
    for r in range(lo, hi):
        var dst = cols + r * ckk
        var h0 = oh * SH - PH
        var w0 = ow * SW - PW
        var q = 0
        for c in range(C):
            var plane = x + (n * C + c) * H * W
            for kh in range(KH):
                var h = h0 + kh * DH
                if h < 0 or h >= H:
                    for _ in range(KW):
                        dst.unsafe_store(q, Float32(0))
                        q += 1
                    continue
                var row = plane + h * W
                for kw in range(KW):
                    var w = w0 + kw * DW
                    var v = Float32(0)
                    if w >= 0 and w < W:
                        v = ftz(row.unsafe_load(w))
                    dst.unsafe_store(q, v)
                    q += 1
        ow += 1
        if ow == OW:
            ow = 0
            oh += 1
            if oh == OH:
                oh = 0
                n += 1


def im2col_host(x: FP, cols: FP, mut prm: List[Int32]):
    """cols [N*OH*OW x C*KH*KW], im2col_at's words."""
    var rows = Int(prm[CP_N]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var tasks = _tasks_for(rows, Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW]))
    var chunk = (rows + tasks - 1) // tasks
    var p = hi(prm)

    def _part(t: Int) {imm x, imm cols, imm p, imm chunk, imm rows}:
        _im2col_rows(x, cols, p, t * chunk, min(t * chunk + chunk, rows))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)


def _conv_out_planes(y2: FP, bias: FP, dst: FP, p: IP, lo: Int, hi: Int):
    var OC = Int(p[CP_OC]); var HW = Int(p[CP_OH]) * Int(p[CP_OW])
    var has_bias = Int(p[CP_BIAS]) != 0
    for pl in range(lo, hi):
        var n = pl // OC
        var oc = pl - n * OC
        var src = y2 + n * HW * OC + oc
        var out = dst + pl * HW
        var b = ftz(bias.unsafe_load(oc)) if has_bias else Float32(0)
        for rem in range(HW):
            var v = ftz(src.unsafe_load(rem * OC))
            if has_bias:
                v = ftz(v + b)
            out.unsafe_store(rem, canon(v))


def conv_out_host(y2: FP, bias: FP, dst: FP, mut prm: List[Int32]):
    """dst (N, OC, OH, OW) from y2 [N*OH*OW x OC], conv_out_at's words."""
    var planes = Int(prm[CP_N]) * Int(prm[CP_OC])
    var tasks = _tasks_for(planes, Int(prm[CP_OH]) * Int(prm[CP_OW]))
    var chunk = (planes + tasks - 1) // tasks
    var p = hi(prm)

    def _part(t: Int) {imm y2, imm bias, imm dst, imm p, imm chunk, imm planes}:
        _conv_out_planes(y2, bias, dst, p, t * chunk, min(t * chunk + chunk, planes))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)


def _dout_rows_images(dout: FP, g: FP, p: IP, lo: Int, hi: Int):
    var OC = Int(p[CP_OC]); var HW = Int(p[CP_OH]) * Int(p[CP_OW])
    for n in range(lo, hi):
        for oc in range(OC):
            var src = dout + (n * OC + oc) * HW
            var out = g + n * HW * OC + oc
            for rem in range(HW):
                out.unsafe_store(rem * OC, ftz(src.unsafe_load(rem)))


def dout_rows_host(dout: FP, g: FP, mut prm: List[Int32]):
    """g [N*OH*OW x OC] from dout (N, OC, OH, OW), dout_rows_at's words."""
    var N = Int(prm[CP_N])
    var tasks = _tasks_for(N, Int(prm[CP_OC]) * Int(prm[CP_OH]) * Int(prm[CP_OW]))
    var chunk = (N + tasks - 1) // tasks
    var p = hi(prm)

    def _part(t: Int) {imm dout, imm g, imm p, imm chunk, imm N}:
        _dout_rows_images(dout, g, p, t * chunk, min(t * chunk + chunk, N))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)


def _col2im_planes(dcols: FP, dx: FP, p: IP, po: IP, pw: IP, lo: Int, hi: Int):
    var C = Int(p[CP_C]); var H = Int(p[CP_H]); var W = Int(p[CP_W])
    var KH = Int(p[CP_KH]); var KW = Int(p[CP_KW])
    var OH = Int(p[CP_OH]); var OW = Int(p[CP_OW])
    var rev = Int(p[CP_REV]) != 0
    var ckk = C * KH * KW
    for pl in range(lo, hi):
        var n = pl // C
        var c = pl - n * C
        var out = dx + pl * H * W
        for h in range(H):
            for w in range(W):
                var acc = Float32(0)
                for a in range(KH):
                    var oh = Int(po[a * H + h])
                    if oh < 0:
                        continue
                    var kh = KH - 1 - a if rev else a
                    var rbase = (n * OH + oh) * OW
                    var qbase = (c * KH + kh) * KW
                    for kw in range(KW):
                        var ow = Int(pw[kw * W + w])
                        if ow < 0:
                            continue
                        acc = ftz(acc + ftz(dcols.unsafe_load((rbase + ow) * ckk + qbase + kw)))
                out.unsafe_store(h * W + w, acc)


def col2im_host(dcols: FP, dx: FP, mut prm: List[Int32]):
    """dx (N, C, H, W) from dcols [N*OH*OW x C*KH*KW], col2im_at's words:
    per pixel the taps in (kh, kw) ascending order (kh descending under
    CP_REV), acc = ftz(acc + ftz(v)) from +0.0. The tests th % SH == 0 and
    tw % SW == 0 (and the output bounds) are tabulated once per call."""
    var H = Int(prm[CP_H]); var W = Int(prm[CP_W])
    var KH = Int(prm[CP_KH]); var KW = Int(prm[CP_KW])
    var OH = Int(prm[CP_OH]); var OW = Int(prm[CP_OW])
    var SH = Int(prm[CP_SH]); var SW = Int(prm[CP_SW])
    var PH = Int(prm[CP_PH]); var PW = Int(prm[CP_PW])
    var DH = Int(prm[CP_DH]); var DW = Int(prm[CP_DW])
    var rev = Int(prm[CP_REV]) != 0
    # ohs[a * H + h]: the output row gather tap a reads for input row h, or -1
    var ohs = List[Int32](length=KH * H, fill=Int32(-1))
    for a in range(KH):
        var kh = KH - 1 - a if rev else a
        for h in range(H):
            var th = h + PH - kh * DH
            if th >= 0 and th % SH == 0 and th // SH < OH:
                ohs[a * H + h] = Int32(th // SH)
    var ows = List[Int32](length=KW * W, fill=Int32(-1))
    for kw in range(KW):
        for w in range(W):
            var tw = w + PW - kw * DW
            if tw >= 0 and tw % SW == 0 and tw // SW < OW:
                ows[kw * W + w] = Int32(tw // SW)
    var planes = Int(prm[CP_N]) * Int(prm[CP_C])
    var tasks = _tasks_for(planes, H * W * KH * KW)
    var chunk = (planes + tasks - 1) // tasks
    var p = hi(prm)
    var po = hi(ohs)
    var pw = hi(ows)

    def _part(t: Int) {imm dcols, imm dx, imm p, imm po, imm pw, imm chunk, imm planes}:
        _col2im_planes(dcols, dx, p, po, pw, t * chunk, min(t * chunk + chunk, planes))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)
    _ = ohs^
    _ = ows^


def gemm_host(a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, op: Int) raises -> List[Float32]:
    var sa = a.copy()
    var sb = b.copy()
    var c = zeros(m * n)
    gemm_host_into(hp(sa), hp(sb), hp(c), op, m, n, k)
    _ = sa^
    _ = sb^
    return c^


def conv2d_forward_into(x: FP, w: FP, bias: FP, dst: FP, prm: List[Int32]) raises:
    """dst (N, OC, OH, OW) = conv(x) + bias: im2col, the pinned NT GEMM,
    the NCHW layout and bias (conv_out_at)."""
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = Int(prm[CP_N]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var cols = scratch(rows * ckk)
    conv2d_forward_cols(x, w, bias, dst, prm, cols)
    cols.free()


def conv2d_forward_cols(x: FP, w: FP, bias: FP, dst: FP, prm: List[Int32], cols: FP) raises:
    """conv2d_forward_into with the caller's im2col buffer, left holding
    x's im2col matrix (the CPU twin's saved array, DEVIATION 5719)."""
    var OC = Int(prm[CP_OC])
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = Int(prm[CP_N]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ps = prm.copy()
    var y2 = scratch(rows * OC)
    im2col_host(x, cols, ps)
    gemm_host_into(cols, w, y2, OP_NT, rows, OC, ckk)
    conv_out_host(y2, bias, dst, ps)
    y2.free()
    _ = ps^


def conv2d_backward_into(x: FP, w: FP, dout: FP, gx: FP, gw: FP, gb: FP, prm: List[Int32], need_dx: Bool) raises:
    """gx (when need_dx), gw [OC x ckk] and gb [OC] of a conv from its
    output gradient: dW = TN over the N*OH*OW rows (5701), db = TN against
    ones, dcols = NN, col2im as a gather (5700)."""
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = Int(prm[CP_N]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ps = prm.copy()
    var cols = scratch(rows * ckk)
    im2col_host(x, cols, ps)
    conv2d_backward_cols(cols, w, dout, gx, gw, gb, prm, need_dx)
    cols.free()
    _ = ps^


def conv2d_backward_cols(cols: FP, w: FP, dout: FP, gx: FP, gw: FP, gb: FP, prm: List[Int32], need_dx: Bool) raises:
    """conv2d_backward_into from the input's im2col matrix `cols`."""
    var N = Int(prm[CP_N]); var C = Int(prm[CP_C]); var OC = Int(prm[CP_OC])
    var ckk = C * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var rows = N * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var ps = prm.copy()
    comptime if X_CNN_HOST_SABOTAGE:
        ps[CP_REV] = Int32(1)
    var g = scratch(rows * OC)
    dout_rows_host(dout, g, ps)
    var ones = List[Float32](length=rows, fill=Float32(1))
    gemm_host_into(g, cols, gw, OP_TN, OC, ckk, rows)
    gemm_host_into(g, hp(ones), gb, OP_TN, OC, 1, rows)
    if need_dx:
        var dcols = scratch(rows * ckk)
        gemm_host_into(g, w, dcols, OP_NN, rows, ckk, OC)
        col2im_host(dcols, gx, ps)
        dcols.free()
    g.free()
    _ = ones^
    _ = ps^


def conv2d_forward_host(
    x: List[Float32], w: List[Float32], bias: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    var rows = Int(prm[CP_N]) * Int(prm[CP_OH]) * Int(prm[CP_OW])
    var xs = x.copy()
    var ws = w.copy()
    var bs = bias.copy()
    var out = zeros(rows * Int(prm[CP_OC]))
    conv2d_forward_into(hp(xs), hp(ws), hp(bs), hp(out), prm)
    _ = xs^
    _ = ws^
    _ = bs^
    return out^


def conv2d_backward_host(
    x: List[Float32], w: List[Float32], dout: List[Float32], prm: List[Int32]
) raises -> List[Float32]:
    """[gx | gw | gb]."""
    var OC = Int(prm[CP_OC])
    var ckk = Int(prm[CP_C]) * Int(prm[CP_KH]) * Int(prm[CP_KW])
    var nx = Int(prm[CP_N]) * Int(prm[CP_C]) * Int(prm[CP_H]) * Int(prm[CP_W])
    var xs = x.copy()
    var ws = w.copy()
    var ds = dout.copy()
    var result = zeros(nx + OC * ckk + OC)
    var rp = hp(result)
    conv2d_backward_into(hp(xs), hp(ws), hp(ds), rp, rp + nx, rp + nx + OC * ckk, prm, True)
    _ = xs^
    _ = ws^
    _ = ds^
    return result^


def conv_block_forward_into(
    x: FP, w: FP, bias: FP, dst: FP, idx: IP, cprm: List[Int32], pprm: List[Int32], pool: Bool,
    keep: Bool, kcols: FP, ky: FP,
) raises:
    """CNNClassifier's Conv2d -> ReLU -> MaxPool2d block (DEVIATION 5717's
    host twin): dst is the pooled output (idx its winners) or, without a
    pool, the ReLU output. With `keep`, kcols receives x's im2col matrix and
    ky the conv output (bias added, before the ReLU): the fit's saved
    arrays, which its backward reads instead of recomputing them."""
    var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ckk = Int(cprm[CP_C]) * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = Int(cprm[CP_N]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var zp: List[Int32] = [0, 0, 0]
    var cols = kcols if keep else scratch(rows * ckk)
    var y = ky if keep else (dst if not pool else scratch(ny))
    conv2d_forward_cols(x, w, bias, y, cprm, cols)
    if not keep:
        cols.free()
    if not pool:
        run[relu_fwd_at](y, y, dst, dst, hi(zp), hi(zp), ny)
    elif keep:
        var r = scratch(ny)
        run[relu_fwd_at](y, y, r, r, hi(zp), hi(zp), ny)
        maxpool_fwd_host(r, dst, idx, pprm)
        r.free()
    else:
        run[relu_fwd_at](y, y, y, y, hi(zp), hi(zp), ny)
        maxpool_fwd_host(y, dst, idx, pprm)
        y.free()
    _ = zp^


def conv_block_backward_into(
    x: FP, w: FP, bias: FP, g: FP, idx: IP, gx: FP, gw: FP, gb: FP,
    cprm: List[Int32], pprm: List[Int32], pool: Bool, need_dx: Bool,
    keep: Bool, kcols: FP, ky: FP,
) raises:
    """The block's backward from its output gradient g. The conv output and
    x's im2col matrix are the forward's saved arrays (`keep`) or recomputed
    from x: the forward's kernels on the forward's inputs, the same words."""
    var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var ckk = Int(cprm[CP_C]) * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
    var rows = Int(cprm[CP_N]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
    var zp: List[Int32] = [0, 0, 0]
    var cols = kcols
    var yconv = ky
    if not keep:
        cols = scratch(rows * ckk)
        yconv = scratch(ny)
        conv2d_forward_cols(x, w, bias, yconv, cprm, cols)
    var gy = scratch(ny)
    if pool:
        var ps = _host_prm(pprm, PP_REV)
        maxpool_bwd_host(g, idx, gy, ps)
        run[relu_bwd_at](yconv, gy, gy, gy, hi(zp), hi(zp), ny)
        _ = ps^
    else:
        run[relu_bwd_at](yconv, g, gy, gy, hi(zp), hi(zp), ny)
    if not keep:
        yconv.free()
    conv2d_backward_cols(cols, w, gy, gx, gw, gb, cprm, need_dx)
    if not keep:
        cols.free()
    gy.free()
    _ = zp^


def _pool_sizes(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def _host_prm(prm: List[Int32], rev_slot: Int) -> List[Int32]:
    var ps = prm.copy()
    comptime if X_CNN_HOST_SABOTAGE:
        ps[rev_slot] = Int32(1)
    return ps^


def _maxpool_fwd_planes(x: FP, dst: FP, idx: IP, p: IP, lo: Int, hi: Int):
    """maxpool_fwd_at's words for planes [lo, hi): the taps in (kh, kw)
    ascending order, strict > (the first maximum wins), a NaN wins."""
    var H = Int(p[PP_H]); var W = Int(p[PP_W])
    var KH = Int(p[PP_KH]); var KW = Int(p[PP_KW])
    var OH = Int(p[PP_OH]); var OW = Int(p[PP_OW])
    var SH = Int(p[PP_SH]); var SW = Int(p[PP_SW])
    var PH = Int(p[PP_PH]); var PW = Int(p[PP_PW])
    var DH = Int(p[PP_DH]); var DW = Int(p[PP_DW])
    for nc in range(lo, hi):
        var base = x + nc * H * W
        var o = nc * OH * OW
        for oh in range(OH):
            for ow in range(OW):
                var best = Float32(0)
                var bi = -1
                for kh in range(KH):
                    var h = oh * SH - PH + kh * DH
                    if h < 0 or h >= H:
                        continue
                    for kw in range(KW):
                        var w = ow * SW - PW + kw * DW
                        if w < 0 or w >= W:
                            continue
                        var v = ftz(base.unsafe_load(h * W + w))
                        if bi < 0 or v > best or v != v:
                            best = v
                            bi = h * W + w
                dst.unsafe_store(o, best)
                idx.unsafe_store(o, Int32(bi))
                o += 1


def maxpool_fwd_host(x: FP, dst: FP, idx: IP, prm: List[Int32]):
    var ps = prm.copy()
    var planes = Int(prm[PP_N]) * Int(prm[PP_C])
    var tasks = _tasks_for(planes, Int(prm[PP_OH]) * Int(prm[PP_OW]) * Int(prm[PP_KH]) * Int(prm[PP_KW]))
    var chunk = (planes + tasks - 1) // tasks
    var p = hi(ps)

    def _part(t: Int) {imm x, imm dst, imm idx, imm p, imm chunk, imm planes}:
        _maxpool_fwd_planes(x, dst, idx, p, t * chunk, min(t * chunk + chunk, planes))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)
    _ = ps^


def _maxpool_bwd_planes(dout: FP, idx: IP, dx: FP, p: IP, po: IP, pw: IP, lo: Int, hi: Int):
    """maxpool_bwd_at's words for planes [lo, hi)."""
    var H = Int(p[PP_H]); var W = Int(p[PP_W])
    var KH = Int(p[PP_KH]); var KW = Int(p[PP_KW])
    var OH = Int(p[PP_OH]); var OW = Int(p[PP_OW])
    for nc in range(lo, hi):
        var out = dx + nc * H * W
        for h in range(H):
            for w in range(W):
                var me = Int32(h * W + w)
                var acc = Float32(0)
                for a in range(KH):
                    var oh = Int(po[a * H + h])
                    if oh < 0:
                        continue
                    for kw in range(KW):
                        var ow = Int(pw[kw * W + w])
                        if ow < 0:
                            continue
                        var o = (nc * OH + oh) * OW + ow
                        if idx.unsafe_load(o) == me:
                            acc = ftz(acc + ftz(dout.unsafe_load(o)))
                out.unsafe_store(h * W + w, acc)


def maxpool_bwd_host(dout: FP, idx: IP, dx: FP, prm: List[Int32]):
    """dx from the pooled gradient: maxpool_bwd_at's gather order (kh
    descending under PP_REV), the stride and bounds tests tabulated."""
    var H = Int(prm[PP_H]); var W = Int(prm[PP_W])
    var KH = Int(prm[PP_KH]); var KW = Int(prm[PP_KW])
    var OH = Int(prm[PP_OH]); var OW = Int(prm[PP_OW])
    var SH = Int(prm[PP_SH]); var SW = Int(prm[PP_SW])
    var PH = Int(prm[PP_PH]); var PW = Int(prm[PP_PW])
    var DH = Int(prm[PP_DH]); var DW = Int(prm[PP_DW])
    var rev = Int(prm[PP_REV]) != 0
    var ohs = List[Int32](length=KH * H, fill=Int32(-1))
    for a in range(KH):
        var kh = KH - 1 - a if rev else a
        for h in range(H):
            var th = h + PH - kh * DH
            if th >= 0 and th % SH == 0 and th // SH < OH:
                ohs[a * H + h] = Int32(th // SH)
    var ows = List[Int32](length=KW * W, fill=Int32(-1))
    for kw in range(KW):
        for w in range(W):
            var tw = w + PW - kw * DW
            if tw >= 0 and tw % SW == 0 and tw // SW < OW:
                ows[kw * W + w] = Int32(tw // SW)
    var ps = prm.copy()
    var planes = Int(prm[PP_N]) * Int(prm[PP_C])
    var tasks = _tasks_for(planes, H * W * KH * KW)
    var chunk = (planes + tasks - 1) // tasks
    var p = hi(ps)
    var po = hi(ohs)
    var pw = hi(ows)

    def _part(t: Int) {imm dout, imm idx, imm dx, imm p, imm po, imm pw, imm chunk, imm planes}:
        _maxpool_bwd_planes(dout, idx, dx, p, po, pw, t * chunk, min(t * chunk + chunk, planes))

    if tasks <= 1:
        _part(0)
    else:
        host_parallelize(_part, tasks)
    _ = ps^
    _ = ohs^
    _ = ows^


def maxpool2d_forward_into(x: FP, dst: FP, idx: IP, prm: List[Int32]):
    maxpool_fwd_host(x, dst, idx, prm)


def maxpool2d_backward_into(dout: FP, idx: IP, gx: FP, prm: List[Int32]):
    var ps = _host_prm(prm, PP_REV)
    maxpool_bwd_host(dout, idx, gx, ps)
    _ = ps^


def avgpool2d_forward_into(x: FP, dst: FP, prm: List[Int32]):
    var no = _pool_sizes(prm)[1]
    var ps = prm.copy()
    run[avgpool_fwd_at](x, dst, dst, dst, hi(ps), hi(ps), no)
    _ = ps^


def avgpool2d_backward_into(dout: FP, gx: FP, prm: List[Int32]):
    var nx = _pool_sizes(prm)[0]
    var ps = _host_prm(prm, PP_REV)
    run[avgpool_bwd_at](dout, gx, gx, gx, hi(ps), hi(ps), nx)
    _ = ps^


def map2_into[f: ElemFn](a: FP, b: FP, dst: FP, n_out: Int):
    """f over [0, n_out) with the zero parameter block (relu, add, mul)."""
    var zp: List[Int32] = [0, 0, 0]
    run[f](a, b, dst, dst, hi(zp), hi(zp), n_out)
    _ = zp^


def maxpool2d_forward_host(x: List[Float32], prm: List[Int32], mut idx: List[Int32]) raises -> List[Float32]:
    var no = _pool_sizes(prm)[1]
    var xs = x.copy()
    var out = zeros(no)
    idx = List[Int32](length=no if no > 0 else 1, fill=Int32(0))
    maxpool2d_forward_into(hp(xs), hp(out), hi(idx), prm)
    _ = xs^
    return out^


def maxpool2d_backward_host(dout: List[Float32], idx: List[Int32], prm: List[Int32]) raises -> List[Float32]:
    var ds = dout.copy()
    var ix = idx.copy()
    var gx = zeros(_pool_sizes(prm)[0])
    maxpool2d_backward_into(hp(ds), hi(ix), hp(gx), prm)
    _ = ds^
    _ = ix^
    return gx^


def avgpool2d_forward_host(x: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var xs = x.copy()
    var out = zeros(_pool_sizes(prm)[1])
    avgpool2d_forward_into(hp(xs), hp(out), prm)
    _ = xs^
    return out^


def avgpool2d_backward_host(dout: List[Float32], prm: List[Int32]) raises -> List[Float32]:
    var ds = dout.copy()
    var gx = zeros(_pool_sizes(prm)[0])
    avgpool2d_backward_into(hp(ds), hp(gx), prm)
    _ = ds^
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


def linear_forward_into(x: FP, w: FP, bias: FP, dst: FP, n: Int, d_in: Int, d_out: Int):
    """dst [n x d_out] = x . w^T + bias (the pinned NT GEMM, bias_rows_at)."""
    var y = scratch(n * d_out)
    gemm_host_into(x, w, y, OP_NT, n, d_out, d_in)
    var prm: List[Int32] = [Int32(n), Int32(d_in), Int32(d_out)]
    run[bias_rows_at](y, bias, dst, dst, hi(prm), hi(prm), n * d_out)
    y.free()
    _ = prm^


def linear_backward_into(x: FP, w: FP, g: FP, gx: FP, gw: FP, gb: FP, n: Int, d_in: Int, d_out: Int):
    var ones = List[Float32](length=n, fill=Float32(1))
    gemm_host_into(g, x, gw, OP_TN, d_out, d_in, n)
    gemm_host_into(g, hp(ones), gb, OP_TN, d_out, 1, n)
    gemm_host_into(g, w, gx, OP_NN, n, d_in, d_out)
    _ = ones^


def linear_forward_host(x: List[Float32], w: List[Float32], bias: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    var xs = x.copy()
    var ws = w.copy()
    var bs = bias.copy()
    var out = zeros(n * d_out)
    linear_forward_into(hp(xs), hp(ws), hp(bs), hp(out), n, d_in, d_out)
    _ = xs^
    _ = ws^
    _ = bs^
    return out^


def linear_backward_host(x: List[Float32], w: List[Float32], g: List[Float32], n: Int, d_in: Int, d_out: Int) raises -> List[Float32]:
    """[gx | gw | gb]."""
    var xs = x.copy()
    var ws = w.copy()
    var gs = g.copy()
    var r = zeros(n * d_in + d_out * d_in + d_out)
    var rp = hp(r)
    linear_backward_into(hp(xs), hp(ws), hp(gs), rp, rp + n * d_in, rp + n * d_in + d_out * d_in, n, d_in, d_out)
    _ = xs^
    _ = ws^
    _ = gs^
    return r^


def softmax_xent_into(logits: FP, labels: IP, grad: FP, proba: FP, n: Int, k: Int) -> Float32:
    """grad and proba [n x k] written in place; returns the mean loss."""
    var prm: List[Int32] = [Int32(n), Int32(k)]
    var rl = zeros(n)
    run[softmax_xent_row_at](logits, grad, proba, hp(rl), labels, hi(prm), n)
    _ = prm^
    return seq_mean(rl, n)


def softmax_xent_host(logits: List[Float32], labels: List[Int32], n: Int, k: Int) raises -> List[Float32]:
    var sl = logits.copy()
    var sy = labels.copy()
    var grad = zeros(n * k)
    var proba = zeros(n * k)
    var loss = softmax_xent_into(hp(sl), hi(sy), hp(grad), hp(proba), n, k)
    _ = sl^
    _ = sy^
    grad.extend(proba^)
    grad.append(loss)
    return grad^


def sgd_into(w: FP, g: FP, v: FP, hyper: List[Float32], n: Int):
    """w and the momentum buffer v updated in place (sgd_at reads and writes
    only its own element)."""
    var sh = hyper.copy()
    var prm: List[Int32] = [Int32(n)]
    run[sgd_at](w, g, v, hp(sh), hi(prm), hi(prm), n)
    _ = sh^
    _ = prm^


def sgd_host(w: List[Float32], g: List[Float32], v: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sv = v.copy()
    sgd_into(hp(sw), hp(sg), hp(sv), hyper, n)
    _ = sg^
    sw.extend(sv^)
    return sw^


def batchnorm_forward_into(x: FP, running: FP, aux: FP, dst: FP, prm: List[Int32], training: Bool):
    """dst = the normalized x; running (2C) and aux (2 + 7C) updated in place."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var ps = prm.copy()
    if training:
        run[bn_stats_at](x, aux, aux, aux, hi(ps), hi(ps), C)
    else:
        run[bn_eval_stats_at](running, aux, aux, aux, hi(ps), hi(ps), C)
    run[bn_apply_at](x, aux, dst, dst, hi(ps), hi(ps), total)
    if training:
        run[bn_running_at](running, aux, aux, aux, hi(ps), hi(ps), C)
    _ = ps^


def batchnorm_backward_into(x: FP, g: FP, aux: FP, dst: FP, prm: List[Int32], training: Bool):
    """dst = dx; aux updated in place (sum_g, sum_gx)."""
    var C = Int(prm[1])
    var total = Int(prm[0]) * C * Int(prm[2])
    var ps = prm.copy()
    run[bn_bwd_red_at](x, g, aux, aux, hi(ps), hi(ps), C)
    if training:
        run[bn_bwd_dx_at](x, g, aux, dst, hi(ps), hi(ps), total)
    else:
        run[bn_bwd_eval_dx_at](x, g, aux, dst, hi(ps), hi(ps), total)
    _ = ps^


def batchnorm_forward_host(
    x: List[Float32], running: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    var sx = x.copy()
    var sr = running.copy()
    var sa = aux.copy()
    var out = zeros(len(x))
    batchnorm_forward_into(hp(sx), hp(sr), hp(sa), hp(out), prm, training)
    _ = sx^
    out.extend(sr^)
    out.extend(sa^)
    return out^


def batchnorm_backward_host(
    x: List[Float32], g: List[Float32], aux: List[Float32], prm: List[Int32], training: Bool
) raises -> List[Float32]:
    var sx = x.copy()
    var sg = g.copy()
    var sa = aux.copy()
    var out = zeros(len(x))
    batchnorm_backward_into(hp(sx), hp(sg), hp(sa), hp(out), prm, training)
    _ = sx^
    _ = sg^
    out.extend(sa^)
    return out^


def dropout2d_into(x: FP, dst: FP, mask: FP, prm: List[Int32], hyper: List[Float32], n: Int):
    var ps = prm.copy()
    var sh = hyper.copy()
    run[dropout2d_at](x, mask, dst, hp(sh), hi(ps), hi(ps), n)
    _ = ps^
    _ = sh^


def dropout2d_host(x: List[Float32], prm: List[Int32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(x)
    var sx = x.copy()
    var mask = zeros(n)
    var out = zeros(n)
    dropout2d_into(hp(sx), hp(out), hp(mask), prm, hyper, n)
    _ = sx^
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


def adam_into(w: FP, g: FP, mv: FP, hyper: List[Float32], n: Int):
    """w and mv = [m (n) | v (n)] updated in place (adam_at reads and writes
    only its own element)."""
    var sh = hyper.copy()
    var prm: List[Int32] = [Int32(n)]
    run[adam_at](w, g, mv, hp(sh), hi(prm), hi(prm), n)
    _ = sh^
    _ = prm^


def adam_host(w: List[Float32], g: List[Float32], mv: List[Float32], hyper: List[Float32]) raises -> List[Float32]:
    var n = len(w)
    var sw = w.copy()
    var sg = g.copy()
    var sm = mv.copy()
    adam_into(hp(sw), hp(sg), hp(sm), hyper, n)
    _ = sg^
    sw.extend(sm^)
    return sw^
