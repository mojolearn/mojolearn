# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`Tikhonov` and `RegularizedGLM`: the l2 penalty and how it wraps a loss.

PORT OF `cuml/cpp/src/glm/qn/glm_regularizer.cuh` at cuML `00094f7`. Whole
file. Do not improve.

`Tikhonov::reg_grad` (`glm_regularizer.cuh:40-54`): the BIAS IS NOT
PENALIZED ("scikit generally does not penalize biases") -- the gradient
slice is `G[:, 0:n_param - has_bias]` and

    Gweights = l2 * Wweights                         (`ax`)
    reg_val  = sum_j 0.5 * l2 * w_j * w_j            (`mapThenSumReduce`)

`RegularizedGLM::loss_grad` (`:68-88`): `G.fill(0)`, then `reg_grad` writes
the penalty's gradient INTO G and its value to the scalar, then the loss's
`loss_grad` with `initGradZero = false` ADDS to G (its gemm runs with
`beta = 1`), and the objective is `loss_host + reg_host` -- TWO device
scalars read back and added ON THE HOST in Float32, which is how the value
the line search compares is assembled and is reproduced exactly.

THE ONE KERNEL BELOW IS BOTH HALVES OF `reg_grad`: one block of
`STATS_TPB` threads strides the D weights, stores `l2 * w_j` (the `ax`) and
accumulates `0.5 * l2 * w_j * w_j` left to right, then folds through
`pinned_block_sum`. Their `mapThenSumReduce` is the float-atomic fold
DEVIATION 547 replaces (`simple_mat/dense.mojo`); the arithmetic per
weight is theirs.
"""

from std.gpu import thread_idx

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from mojo_only.numerics import ftz


def tikhonov_reg_grad_kernel(
    reg_val: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_weights_in: Int32,
    l2: Float32,
):
    """`Tikhonov::reg_grad` over the first `n_weights` entries (the bias,
    if any, is at index `n_weights` and is left alone). Launch with
    `grid = 1, block = STATS_TPB`."""
    var n = Int(n_weights_in)
    var tid = Int(thread_idx.x)
    var half_l2 = ftz(Float32(0.5) * l2)
    var acc = Float32(0.0)
    var j = tid
    while j < n:
        var wj = w.unsafe_load(j)
        g.unsafe_store(j, ftz(l2 * wj))
        var t = ftz(half_l2 * wj)
        acc = ftz(acc + ftz(t * wj))
        j += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        reg_val.unsafe_store(0, s0)
