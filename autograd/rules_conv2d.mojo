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
#               b need a gradient.
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
#   max_pool2d: OP_RNV_MAXPOOL_VJP (one thread an x word). 1 launch.
#   avg_pool2d: OP_RNV_AVGPOOL_VJP. 1 launch.
from autograd.tape import BackwardRules
from autograd.fold import blocked_fold
from autograd.ops import AXIS_ROWS
from autograd.rankn_conv_ops import (
    OP_RNV_CONV_GROWS, OP_RNV_COL2IM, OP_RNV_MAXPOOL_VJP, OP_RNV_AVGPOOL_VJP,
)
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_GEMM

comptime RN_CONV2D = 176
comptime RN_MAXPOOL2D = 177
comptime RN_AVGPOOL2D = 178
comptime RN_CONV_TAG_FIRST = 176
comptime RN_CONV_TAG_LAST = 191


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
    var g = Args()
    g.p0 = upstream
    g.p1 = grow
    g.p11 = saved.p11
    ex.launch[OP_RNV_CONV_GROWS](g, rows * oc)
    if saved.i6 != 0:
        # dW(oc, q) = sum over r of grow[r, oc] cols[r, q]
        rn_conv_gemm(ex, gradients[1], grow, saved.p0, oc, ckk, rows, 1, oc, ckk, 1)
    if saved.i0 != 0 and saved.i7 != 0:
        blocked_fold(ex, gradients[2], grow, rows, oc, AXIS_ROWS, Float32(1), saved.p5)
    if saved.i5 != 0:
        # dcols(r, q) = sum over oc of grow[r, oc] W[oc, q]
        rn_conv_gemm(ex, saved.p4, grow, inputs[1], rows, ckk, oc, oc, 1, ckk, 1)
        var c = Args()
        c.p0 = saved.p4
        c.p1 = gradients[0]
        c.p11 = saved.p11
        ex.launch[OP_RNV_COL2IM](c, saved.i4)
