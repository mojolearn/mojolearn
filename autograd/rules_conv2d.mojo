# SPDX-License-Identifier: Apache-2.0
# The backward rules of the tape's rank-4 conv2d and pools (lane
# ag-rankn-conv, docs/plans/AUTOGRAD_RANKN.md section 2.8). The forwards are
# autograd/rankn_conv.mojo; the kernels autograd/rankn_conv_ops.mojo.
#
# Custom tags 176..191 are this lane's block (RN_CONV2D 176, RN_MAXPOOL2D
# 177, RN_AVGPOOL2D 178; 179..191 reserved). The x_cnn rules (99..111) run on
# cnn_ctx; these are tape-executor rules on any Exec.
#
# SAVED ARGS (written by the forwards):
#   RN_CONV2D   p0 cols [rows, C KH KW] (the forward's im2col, kept), p2 the
#               GEMM scratch y2 [rows, OC] (reused as grow: the forward no
#               longer reads it), p4 dcols [rows, C KH KW] (when x needs a
#               gradient), p5 the bias fold's partials (when b needs one),
#               p11 the CP_* descriptor; i0 1 with a bias, i1 rows = N OH OW,
#               i2 C KH KW, i3 OC, i4 x's element count, i5 / i6 / i7 x / w /
#               b need a gradient, i8 groups (1 for an ungrouped
#               convolution); with groups > 1, p6 the dW product's OP_RN_BMM
#               descriptor (when w needs a gradient) and p7 the dcols
#               product's (when x needs one).
#   RN_MAXPOOL2D p10 the winners (int32 patterns), p11 the PP_* descriptor;
#               i4 x's element count, i5 x needs a gradient.
#   RN_AVGPOOL2D p11 the PP_* descriptor; i4, i5 as the max pool.
#
# VJP LAUNCHES (every destination word written once, so no zero fill:
# zeroed_destination False everywhere; an input that needs no gradient is
# never touched).
#   conv2d: OP_RNV_CONV_GROWS (grow = dout in rows layout, a copy), then
#     dW [OC, C KH KW] = grow^T @ cols (OP_GEMM), db = the tape's blocked
#     fold of grow over its rows (AXIS_ROWS, factor 1: the tape's column-sum
#     order), dcols = grow @ W (OP_GEMM), dx = OP_RNV_COL2IM (each pixel's
#     taps in (kh, kw) ascending order). Up to 5 launches plus the fold's.
#     groups > 1: the same launches with the two OP_GEMMs replaced by one
#     OP_RN_BMM each over every group (dW block g = grow's block g ^T @
#     cols's block g, k = rows ascending; dcols block g = grow's block g @
#     W block g, k = OC / groups ascending), each word one fma chain from
#     +0.0 (op_gemm's chain over that group's terms).
#   max_pool2d: OP_RNV_MAXPOOL_VJP (one thread an x word). 1 launch.
#   avg_pool2d: OP_RNV_AVGPOOL_VJP. 1 launch.
from autograd.tape import BackwardRules
from autograd.fold import blocked_fold
from autograd.ops import AXIS_ROWS
from autograd.rankn_conv_ops import (
    OP_RNV_CONV_GROWS, OP_RNV_COL2IM, OP_RNV_MAXPOOL_VJP, OP_RNV_AVGPOOL_VJP,
)
from sequence.exec_trait import Exec
from autograd.rankn_bmm_ops import OP_RN_BMM
from sequence.ops import FP, Args, OP_GEMM

comptime RN_CONV2D = 176
comptime RN_MAXPOOL2D = 177
comptime RN_AVGPOOL2D = 178
comptime RN_CONV_TAG_FIRST = 176
comptime RN_CONV_TAG_LAST = 191

# the three grouped products of a grouped convolution (rn_conv_group_words)
comptime RNG_FWD = 0
comptime RNG_DW = 1
comptime RNG_DCOLS = 2


def rn_conv_gemm[E: Exec](
    mut ex: E, dst: FP, a: FP, b: FP, m: Int, n: Int, k: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int,
) raises:
    """dst[m, n] (dense, row stride n) = sum over k ascending of A(m, k) B(k, n),
    A(m, k) = a[m sam + k sak], B(k, n) = b[k sbk + n sbn]: one OP_GEMM
    launch, the tape's GEMM word (tape.mojo _gemm's arguments)."""
    var args = Args()
    args.p0 = a
    args.p1 = b
    args.p2 = dst
    args.i0 = m
    args.i1 = n
    args.i2 = k
    args.i3 = sam
    args.i4 = sak
    args.i5 = sbk
    args.i6 = sbn
    args.i8 = n
    ex.launch[OP_GEMM](args, m * n)


def rn_conv_group_words(which: Int, rows: Int, C: Int, OC: Int, KK: Int, groups: Int) raises -> List[Int]:
    """The OP_RN_BMM descriptor (autograd/rankn_bmm_ops.mojo header: nf,
    nc, K, sak, sbk, then extent, A stride, B stride per free axis) of one
    of a grouped convolution's three products, over G = groups blocks of
    Cg = C / G input channels and OCg = OC / G outputs; cols is
    [rows, C KK] (group g's taps the columns g Cg KK ..), W [OC, Cg KK],
    grow [rows, OC]. Shape words only.

    RNG_FWD   y2[r, g, o]   = sum_q cols[r, g Cg KK + q] W[g OCg + o, q]
    RNG_DW    dW[g, o, q]   = sum_r grow[r, g OCg + o] cols[r, g Cg KK + q]
    RNG_DCOLS dcols[r, g, q] = sum_o grow[r, g OCg + o] W[g OCg + o, q]
    """
    if groups < 1 or C % groups != 0 or OC % groups != 0:
        raise Error("conv2d: groups must divide the input channels and out_channels")
    var G = groups
    var ckk = C * KK
    var cgkk = (C // G) * KK
    var ocg = OC // G
    var w = List[Int]()
    w.append(3)
    w.append(0)
    if which == RNG_FWD:
        # A = cols, B = W; K = Cg KK
        w.append(cgkk); w.append(1); w.append(1)
        w.append(rows); w.append(ckk); w.append(0)
        w.append(G); w.append(cgkk); w.append(ocg * cgkk)
        w.append(ocg); w.append(0); w.append(cgkk)
    elif which == RNG_DW:
        # A = grow, B = cols; K = rows
        w.append(rows); w.append(OC); w.append(ckk)
        w.append(G); w.append(ocg); w.append(cgkk)
        w.append(ocg); w.append(1); w.append(0)
        w.append(cgkk); w.append(0); w.append(1)
    elif which == RNG_DCOLS:
        # A = grow, B = W; K = OCg
        w.append(ocg); w.append(1); w.append(cgkk)
        w.append(rows); w.append(OC); w.append(0)
        w.append(G); w.append(ocg); w.append(ocg * cgkk)
        w.append(cgkk); w.append(0); w.append(1)
    else:
        raise Error("conv2d: unknown grouped product")
    return w^


def rn_conv_group_gemm[E: Exec](mut ex: E, dst: FP, a: FP, b: FP, desc: FP, count: Int) raises:
    """dst (dense, count words) = the grouped product `desc` describes
    (rn_conv_group_words), A from a, B from b: one OP_RN_BMM launch, one
    thread a word."""
    var args = Args()
    args.p0 = a
    args.p1 = b
    args.p2 = dst
    args.p11 = desc
    args.i0 = 0
    args.i1 = 0
    args.i10 = 0
    ex.launch[OP_RN_BMM](args, count)


@fieldwise_init
struct Conv2dRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of conv2d, max_pool2d and avg_pool2d recorded on the tape."""

    def supports(self, tag: Int) -> Bool:
        return tag == RN_CONV2D or tag == RN_MAXPOOL2D or tag == RN_AVGPOOL2D

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # every destination this rule touches is written word for word; the
        # others belong to inputs that need no gradient
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag == RN_CONV2D:
            _conv2d_vjp(ex, saved, upstream, inputs, gradients)
        elif tag == RN_MAXPOOL2D:
            if saved.i5 != 0:
                var a = Args()
                a.p0 = upstream
                a.p1 = saved.p10
                a.p2 = gradients[0]
                a.p11 = saved.p11
                ex.launch[OP_RNV_MAXPOOL_VJP](a, saved.i4)
        elif tag == RN_AVGPOOL2D:
            if saved.i5 != 0:
                var a = Args()
                a.p0 = upstream
                a.p1 = gradients[0]
                a.p11 = saved.p11
                ex.launch[OP_RNV_AVGPOOL_VJP](a, saved.i4)
        else:
            raise Error("conv2d rules: unknown tag")


def _conv2d_vjp[E: Exec](
    mut ex: E, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP],
) raises:
    """conv2d's VJP (the launches in the header). inputs: x, w[, b]."""
    var rows = saved.i1
    var ckk = saved.i2
    var oc = saved.i3
    var grow = saved.p2
    var grouped = saved.i8 > 1
    var g = Args()
    g.p0 = upstream
    g.p1 = grow
    g.p11 = saved.p11
    ex.launch[OP_RNV_CONV_GROWS](g, rows * oc)
    if saved.i6 != 0:
        if grouped:
            # dW[OC, Cg KK], group by group in one launch
            rn_conv_group_gemm(ex, gradients[1], grow, saved.p0, saved.p6, oc * (ckk // saved.i8))
        else:
            # dW(oc, q) = sum over r of grow[r, oc] cols[r, q]
            rn_conv_gemm(ex, gradients[1], grow, saved.p0, oc, ckk, rows, 1, oc, ckk, 1)
    if saved.i0 != 0 and saved.i7 != 0:
        blocked_fold(ex, gradients[2], grow, rows, oc, AXIS_ROWS, Float32(1), saved.p5)
    if saved.i5 != 0:
        if grouped:
            # dcols[rows, C KK], every group's block in one launch
            rn_conv_group_gemm(ex, saved.p4, grow, inputs[1], saved.p7, rows * ckk)
        else:
            # dcols(r, q) = sum over oc of grow[r, oc] W[oc, q]
            rn_conv_gemm(ex, saved.p4, grow, inputs[1], rows, ckk, oc, oc, 1, ckk, 1)
        var c = Args()
        c.p0 = saved.p4
        c.p1 = gradients[0]
        c.p11 = saved.p11
        ex.launch[OP_RNV_COL2IM](c, saved.i4)
