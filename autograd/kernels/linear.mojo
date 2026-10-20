# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tape's fused linear kernels: OP_AGK_GEMM_EPI's one-thread body and
OP_COLSUM_DIV (the linear node's bias gradient). Moved here unchanged from
sequence/mlp.mojo; the in-register epilogue they share with the forward
(gemm_epi_value) stays there."""
from sequence.ops import Args, gemm_dot, ld, op_colsum, st
from sequence.mlp import gemm_epi_value, op_divs


@always_inline
def op_agk_gemm_epi(t: Int, a: Args):
    """OP_AGK_GEMM_EPI's one-thread body (the launch where the tile is off:
    K under GT_MIN_K, a non-dense C, Apple, FAST; under
    MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1 also an edge under GT_MIN_EDGE;
    sequence/gemm_tiled.mojo seq_gemm_route): op_gemm's chain for cell t, then
    gemm_epi_value, one store. The same word as op_gemm_epi. Dense C
    (i8 = i1), no accumulate (i7 = 0), as the tape launches it."""
    var n_cols = a.i1
    var m = t // n_cols
    var n = t - m * n_cols
    var ci = m * a.i8 + n
    var acc = Float32(0.0)
    if a.i7 != 0:
        acc = ld(a.p2, ci)
    acc = gemm_dot(a.p0, m * a.i3, a.i4, a.p1, n * a.i6, a.i5, a.i2, acc)
    st(a.p2, ci, gemm_epi_value(acc, ci, n, a.i9, a.i10, a.p3, a.f0, a.f1))


def op_colsum_div(t: Int, a: Args):
    """`op_colsum` then `op_divs` of the same element by f0: one launch."""
    op_colsum(t, a)
    var e = Args()
    e.p0 = a.p1
    e.f0 = a.f0
    op_divs(t, e)
