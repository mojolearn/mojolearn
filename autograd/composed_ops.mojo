# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# Lane ag-pass5-composed (2026-10-19, PASS5_IDEAS ideas 4 and 5): element
# bodies that run several existing bodies in one launch, each on its own
# index range, so a step issues fewer launches with the same words.
#
#   OP_AGC_OPT_FINISH (2000): the composed tape's gradient finish and the
#     optimizer step in one launch. Thread t of the flat parameter vector
#     first runs op_l2grad (a weight slot: (g + alpha w) / divisor) or
#     op_divs (a bias slot: g / divisor) on its own word, stores it, then
#     runs op_opt, which reads that word back. The two finish launches a
#     layer (OP_L2GRAD, OP_DIVS) are gone; the words are those bodies'
#     words, so the step is the same.
#   OP_AGC_LINEAR_PAIR (2001): both GEMMs of one linear (or matmul) node's
#     backward, and its bias column sums, in one launch: threads
#     [0, rows fi) are dX's cells, [rows fi, rows fi + fi fo) dW's cells,
#     then fo bias columns. Each range runs the body the separate launch
#     ran on a launch where the GEMM takes one thread a cell (op_gemm,
#     op_agk_gemm_epi, op_colsum_div): the same chain per word (k ascending
#     from +0.0, one fma a term, the same epilogue), so the same bits as
#     the tiled and thin routes, which keep that chain (sequence/
#     gemm_tiled.mojo seq_gemm_tile_kernel's contract).
#
# Codes OP_AGC_FIRST..OP_AGC_LAST are this family; sequence/dispatch.mojo
# routes the range and asserts it overlaps no other family. This module
# imports sequence.mlp (for the bodies it reuses), which is why the bodies
# are not in autograd/ops.mojo: sequence.mlp imports autograd.ops.
#
# Metal: every function a body reaches here is @always_inline; each range
# builds its own Args with fixed slots and loads through branches, never
# through a pointer chosen by a conditional expression (autograd/ops.mojo
# _ld_right).
from sequence.ops import Args, FP, op_gemm, op_opt
from sequence.mlp import EPI_ACT_BWD, EPI_L2GRAD, op_l2grad
from autograd.kernels.mlp import op_divs
from autograd.kernels.linear import op_agk_gemm_epi, op_colsum_div

comptime OP_AGC_OPT_FINISH = 2000
comptime OP_AGC_LINEAR_PAIR = 2001
comptime OP_AGC_FIRST = OP_AGC_OPT_FINISH
comptime OP_AGC_LAST = OP_AGC_LINEAR_PAIR

# Parameter tensors one OP_AGC_OPT_FINISH launch covers: two bounds each in
# the integer slots i4..i11 (op_opt reads i0..i2). A trainer with more
# tensors issues one launch per group of four (autograd/optim.mojo).
comptime AGC_FINISH_TENSORS = 4


@always_inline
def agc_bound(a: Args, j: Int) -> Int:
    """Integer slot i4 + j (j in 0..7) of an OP_AGC_OPT_FINISH launch."""
    if j == 0:
        return a.i4
    if j == 1:
        return a.i5
    if j == 2:
        return a.i6
    if j == 3:
        return a.i7
    if j == 4:
        return a.i8
    if j == 5:
        return a.i9
    if j == 6:
        return a.i10
    return a.i11


def agc_set_bound(mut a: Args, j: Int, v: Int):
    """Host side: write integer slot i4 + j (j in 0..7)."""
    if j == 0:
        a.i4 = v
    elif j == 1:
        a.i5 = v
    elif j == 2:
        a.i6 = v
    elif j == 3:
        a.i7 = v
    elif j == 4:
        a.i8 = v
    elif j == 5:
        a.i9 = v
    elif j == 6:
        a.i10 = v
    else:
        a.i11 = v


@always_inline
def _agc_l2grad(t: Int, a: Args):
    # OP_L2GRAD's arguments as mlp_backward_autograd's export launched them
    var e = Args()
    e.p0 = a.p1
    e.p1 = a.p0
    e.f0 = a.f6
    e.f1 = a.f7
    op_l2grad(t, e)


@always_inline
def _agc_divs(t: Int, a: Args):
    # OP_DIVS's arguments as the export launched them
    var e = Args()
    e.p0 = a.p1
    e.f0 = a.f7
    op_divs(t, e)


@always_inline
def op_agc_opt_finish(t: Int, a: Args):
    """Finish gradient word t, then op_opt on it.

    op_opt's slots (p0 params, p1 grads, p2..p4 state, i0 kind, i1 step,
    i2 flags, f0..f5) plus: i3 the tensor count (1..AGC_FINISH_TENSORS),
    i4 + 2 l and i5 + 2 l tensor l's bias start and end (relative to p0;
    its weights run from the previous tensor's end, or 0, to its bias
    start), f6 alpha, f7 the divisor. Only the sklearn kinds (OPT_SK_ADAM,
    OPT_SK_SGD), which read neither f6 nor f7, run here (the launcher
    refuses the others). Weight words take op_l2grad's word, bias words
    op_divs's; a word past the last tensor is never launched."""
    var done = False
    comptime for l in range(AGC_FINISH_TENSORS):
        if not done and l < a.i3:
            if t < agc_bound(a, 2 * l):
                _agc_l2grad(t, a)
                done = True
            elif t < agc_bound(a, 2 * l + 1):
                _agc_divs(t, a)
                done = True
    op_opt(t, a)


@always_inline
def _agc_dx(t: Int, a: Args):
    # dX = dz W^T: _gemm(dst, dz, w, rows, fi, fo, fo, 1, 1, fo)
    var s = Args()
    s.p0 = a.p2
    s.p1 = a.p1
    s.p2 = a.p3
    s.i0 = a.i0
    s.i1 = a.i1
    s.i2 = a.i2
    s.i3 = a.i2
    s.i4 = 1
    s.i5 = 1
    s.i6 = a.i2
    s.i8 = a.i1
    op_gemm(t, s)


@always_inline
def _agc_dx_act(t: Int, a: Args):
    # dX = (dz W^T) act'(x): _gemm_epi(dz, w, dst, rows, fi, fo, fo, 1, 1, fo,
    # EPI_ACT_BWD, x, x_act, 0, 0)
    var s = Args()
    s.p0 = a.p2
    s.p1 = a.p1
    s.p2 = a.p3
    s.p3 = a.p0
    s.i0 = a.i0
    s.i1 = a.i1
    s.i2 = a.i2
    s.i3 = a.i2
    s.i4 = 1
    s.i5 = 1
    s.i6 = a.i2
    s.i8 = a.i1
    s.i9 = EPI_ACT_BWD
    s.i10 = a.i4
    op_agk_gemm_epi(t, s)


@always_inline
def _agc_dw(t: Int, a: Args):
    # dW = x^T dz: _gemm(dst, x, dz, fi, fo, rows, 1, fi, fo, 1)
    var s = Args()
    s.p0 = a.p0
    s.p1 = a.p2
    s.p2 = a.p4
    s.i0 = a.i1
    s.i1 = a.i2
    s.i2 = a.i0
    s.i3 = 1
    s.i4 = a.i1
    s.i5 = a.i2
    s.i6 = 1
    s.i8 = a.i2
    op_gemm(t, s)


@always_inline
def _agc_dw_l2(t: Int, a: Args):
    # dW = (x^T dz + alpha W) / divisor: _gemm_epi(x, dz, dst, fi, fo, rows,
    # 1, fi, fo, 1, EPI_L2GRAD, w, 0, alpha, divisor)
    var s = Args()
    s.p0 = a.p0
    s.p1 = a.p2
    s.p2 = a.p4
    s.p3 = a.p1
    s.i0 = a.i1
    s.i1 = a.i2
    s.i2 = a.i0
    s.i3 = 1
    s.i4 = a.i1
    s.i5 = a.i2
    s.i6 = 1
    s.i8 = a.i2
    s.i9 = EPI_L2GRAD
    s.f0 = a.f0
    s.f1 = a.f1
    op_agk_gemm_epi(t, s)


@always_inline
def _agc_db(t: Int, a: Args):
    # db = colsum(dz) / divisor: OP_COLSUM_DIV(dz, dst, rows, fo, divisor)
    var c = Args()
    c.p0 = a.p2
    c.p1 = a.p5
    c.i0 = a.i0
    c.i2 = a.i2
    c.f0 = a.f1
    op_colsum_div(t, c)


@always_inline
def op_agc_linear_pair(t: Int, a: Args):
    """One linear node's backward GEMMs and bias sums in one launch.

    p0 x (rows, fi) values, p1 W (fi, fo), p2 dz (rows, fo), p3 dX
    destination (rows, fi), p4 dW destination (fi, fo), p5 db destination
    (fo); i0 rows, i1 fi, i2 fo; i3 EPI_ACT_BWD for dX's act'(x) epilogue
    (i4 the activation), else 0; i5 EPI_L2GRAD for dW's (+ alpha W) /
    divisor epilogue, else 0; i6 = 1 adds the fo bias columns (colsum(dz) /
    divisor); f0 alpha, f1 divisor. Launched over rows fi + fi fo
    (+ fo) threads. No range reads a word another range writes."""
    var n_dx = a.i0 * a.i1
    var n_dw = a.i1 * a.i2
    if t < n_dx:
        if a.i3 == EPI_ACT_BWD:
            _agc_dx_act(t, a)
        else:
            _agc_dx(t, a)
        return
    var u = t - n_dx
    if u < n_dw:
        if a.i5 == EPI_L2GRAD:
            _agc_dw_l2(u, a)
        else:
            _agc_dw(u, a)
        return
    _agc_db(u - n_dw, a)


@always_inline
def apply_composed[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AGC_OPT_FINISH:
        op_agc_opt_finish(t, a)
    elif OP == OP_AGC_LINEAR_PAIR:
        op_agc_linear_pair(t, a)
