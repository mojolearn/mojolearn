# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MLP backward body left in sequence/mlp.mojo: OP_DIVS, the gradient
divide (biases g / divisor, OP_COLSUM_DIV's second half). Moved unchanged;
the fused GEMM epilogue cells (act_bwd_value, op_l2grad, gemm_epi_value) stay
in sequence/mlp.mojo because the forward epilogue and the tiled GEMM share
them."""

from sequence.ops import Args, ld, st
from sequence.mlp import div


def op_divs(t: Int, a: Args):
    """p0[t] = p0[t] / f0."""
    st(a.p0, t, div(ld(a.p0, t), a.f0))
